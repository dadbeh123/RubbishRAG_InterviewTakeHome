"""Query entities and slot extraction for Credit-Scoring QA.

Detects:
BANK: Known Iranian banks (ملی, صادرات, ملت, تجارت, سپه, پاسارگاد, ...)
RANK: Credit score grade A1..E3
PERSONA: حقیقی (individual) vs حقوقی (corporate entity)
DEBT: Category of debt (چک, مالیاتی, تسهیلاتی, محکومیت)
AMOUNT: Numerical figures and threshold amounts
TIME: Temporal indicators (ساعت, روز, ماه, سال, مدت)
NEGATED: Boolean negation indicator

Implementation notes (see tests/test_slots.py for the regression cases):

Matching is token/phrase based, never raw substring: کد ملی is not a bank,
چکیده is not a check, منحصر به فرد is not a person.
Longest match wins (توسعه صادرات is not shadowed by صادرات).
Multi-valued: BANK_ALL / DEBT_ALL / RANK_ALL keep every mention in text
order; the legacy scalar keys (BANK, DEBT, RANK) hold the *first mention*.
Ambiguous bank names (ملی, ملت, تجارت, صادرات, سپه, تارا are also ordinary words)
need the word بانک immediately before them.
Numbers keep their semantics: 1,000,000 -> 1000000, 3.5 stays 3.5,
350 هزار -> AMOUNT_VALUE 350000, and the digit of a rank (C1) is not a number.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .normalize_fa import ZWNJ, normalize_fa

# Public constants kept for backward compatibility ------------------------------------
BANKS = [
    "ملی", "صادرات", "تجارت", "توسعه صادرات", "ملت", "سپه",
    "پاسارگاد", "دیجی پی", "دیجی‌پی", "اسنپ پی", "اسنپ‌پی", "تارا",
    "دیجیکالا", "دیجی‌کالا",
]

RANKS = [f"{L}{N}" for L in "ABCDE" for N in "123"]

DEBT_PATTERNS = [
    ("چک", ["چک", "صیادی", "برگشتی", "رفع سوءاثر", "رفع سواثر"]),
    ("مالیاتی", ["مالیات", "جریمه مالیاتی", "امور مالیاتی"]),
    ("تسهیلاتی", ["تسهیلات", "وام", "قسط", "اقساط", "معوق"]),
    ("محکومیت", ["محکومیت", "ورشکستگی", "دادگاه", "قضایی"]),
]

# Internals -----------------------------------------------------------------------------
_TOKEN = re.compile(r"[\w\u200c]+")
_SUFFIXES = ("ها", "های", "هایی", "ی", "ای", "ام", "م", "ت", "ش", "مان", "تان", "شان")


def _key(tok: str) -> str:
    return tok.replace(ZWNJ, "").lower()


def _phrase(text: str) -> Tuple[str, ...]:
    return tuple(_key(t) for t in _TOKEN.findall(normalize_fa(text)))


def _tokens(norm: str) -> List[str]:
    return [_key(m.group()) for m in _TOKEN.finditer(norm)]


_PLURALS = ("ها", "های", "هایی")


def _tok_is(tok: str, kw: str, suffixes: Sequence[str] = _SUFFIXES) -> bool:
    """Token equals the keyword or the keyword plus one allowed suffix."""
    return tok == kw or (tok.startswith(kw) and tok[len(kw):] in suffixes)


# Banks: (canonical, phrase variants, needs_prefix "بانک")
_AMBIGUOUS_BANKS = {"ملی", "ملت", "تجارت", "صادرات", "سپه", "تارا"}
_BANK_SPECS: List[Tuple[str, Tuple[str, ...], bool]] = []
for _b in BANKS:
    _canon = _b
    _BANK_SPECS.append((_canon, _phrase(_b), _canon in _AMBIGUOUS_BANKS))
    _BANK_SPECS.append((_canon, (_key(_b.replace(" ", "")),), _canon in _AMBIGUOUS_BANKS))
# canonical value for variants that differ only by ZWNJ/space
_BANK_CANON = {"دیجی پی": "دیجی‌پی", "اسنپ پی": "اسنپ‌پی", "دیجیکالا": "دیجی‌کالا"}

_PERSONA_INDIVIDUAL = {"حقیقی", "شخصی", "فرد", "افراد", "شخص", "اشخاص", "خودم"}
_PERSONA_LEGAL = {"حقوقی", "شرکت", "سازمان", "کسبوکار"}
# phrases that decide persona outright / idioms that must not trigger anything
_PERSONA_PHRASES: List[Tuple[Tuple[str, ...], Optional[str]]] = [
    (_phrase("شخص حقوقی"), "حقوقی"), (_phrase("اشخاص حقوقی"), "حقوقی"),
    (_phrase("شخص حقیقی"), "حقیقی"), (_phrase("اشخاص حقیقی"), "حقیقی"),
    (_phrase("کسب و کار"), "حقوقی"), (_phrase("کسب و کارها"), "حقوقی"),
    (_phrase("منحصر به فرد"), None), (_phrase("منحصربهفرد"), None),
]

_DEBT_SPECS: List[Tuple[str, Tuple[str, ...], bool]] = []   # (category, phrase, inflect?)
for _cat, _kws in DEBT_PATTERNS:
    for _kw in _kws:
        _ph = _phrase(_kw)
        _DEBT_SPECS.append((_cat, _ph, len(_ph) == 1))
_DEBT_SPECS.append(("چک", _phrase("رفع سو اثر"), False))

_TIME_WORDS = ("سال", "ماه", "روز", "ساعت", "مدت", "هفته")
_NEG_EXACT = {"نیست", "نیستم", "نیستند", "نیستید", "فاقد", "بدون", "ندارم", "ندارد", "ندارند",
              "معاف", "عدم", "هرگز", "نشده", "نشود", "نباشد", "نکرده", "نبوده"}
_MULTIPLIERS = {"هزار": 10 ** 3, "میلیون": 10 ** 6, "میلیارد": 10 ** 9}

_RANK_RE = re.compile(r"(?<![A-Za-z0-9])([A-Ea-e])( ?)([1-3])(?![A-Za-z0-9])")
_RANK_CONTEXT = {"رتبه", "گروه", "درجه", "رده", "سطح"}
_NUM_RE = re.compile(r"(?<![\w.,])(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")


def _match_at(toks: Sequence[str], i: int, phrase: Tuple[str, ...], inflect: bool) -> int:
    """Length of phrase matched at token i (0 if none). Single-word phrases may
    carry one clitic/plural suffix when inflect is set; multi-word phrases match exactly."""
    n = len(phrase)
    if i + n > len(toks):
        return 0
    if n == 1 and inflect:
        return 1 if _tok_is(toks[i], phrase[0]) else 0
    return n if tuple(toks[i:i + n]) == phrase else 0


def _ranks(norm: str) -> List[Tuple[int, int, str]]:
    out = []
    for m in _RANK_RE.finditer(norm):
        letter, space, digit = m.groups()
        if space and letter.islower():
            prev = _tokens(norm[:m.start()])
            if not (prev and prev[-1] in _RANK_CONTEXT):
                continue                      # "a 1" in running text is not a rank
        out.append((m.start(), m.end(), f"{letter.upper()}{digit}"))
    return out


def extract_slots(query: str) -> Dict[str, Any]:
    """Extract domain entity slots from query."""
    norm = normalize_fa(query or "")
    toks = _tokens(norm)
    slots: Dict[str, Any] = {
        "BANK": None, "RANK": None, "PERSONA": None, "DEBT": None, "AMOUNT": None,
        "TIME": False, "NUMS": [], "NEGATED": False,
        "BANK_ALL": [], "RANK_ALL": [], "DEBT_ALL": [], "AMOUNT_VALUE": None,
    }

    # Banks (longest match first, text order)
    i = 0
    while i < len(toks):
        best = None
        for canon, ph, needs_prefix in _BANK_SPECS:
            n = _match_at(toks, i, ph, False)
            if n and (not needs_prefix or (i > 0 and toks[i - 1] == "بانک")):
                if best is None or n > best[0]:
                    best = (n, _BANK_CANON.get(canon, canon))
        if best:
            if best[1] not in slots["BANK_ALL"]:
                slots["BANK_ALL"].append(best[1])
            i += best[0]
        else:
            i += 1
    slots["BANK"] = slots["BANK_ALL"][0] if slots["BANK_ALL"] else None

    # Ranks
    ranks = _ranks(norm)
    for _, _, r in ranks:
        if r not in slots["RANK_ALL"]:
            slots["RANK_ALL"].append(r)
    slots["RANK"] = slots["RANK_ALL"][0] if slots["RANK_ALL"] else None

    # Persona (phrases first, then single tokens)
    flags = set()
    i = 0
    while i < len(toks):
        hit = None
        for ph, val in _PERSONA_PHRASES:
            if toks[i:i + len(ph)] == list(ph):
                hit = (len(ph), val)
                break
        if hit:
            if hit[1]:
                flags.add(hit[1])
            i += hit[0]
            continue
        t = toks[i]
        if any(_tok_is(t, w, _PLURALS) for w in _PERSONA_INDIVIDUAL):
            flags.add("حقیقی")
        if any(_tok_is(t, w, _PLURALS) for w in _PERSONA_LEGAL):
            flags.add("حقوقی")
        i += 1
    if len(flags) == 2:
        slots["PERSONA"] = "both"
    elif flags:
        slots["PERSONA"] = next(iter(flags))

    # Debt types, in text order (first mention wins the scalar slot)
    i = 0
    while i < len(toks):
        best = None
        for cat, ph, inflect in _DEBT_SPECS:
            n = _match_at(toks, i, ph, inflect)
            if n and (best is None or n > best[0]):
                best = (n, cat)
        if best:
            if best[1] not in slots["DEBT_ALL"]:
                slots["DEBT_ALL"].append(best[1])
            i += best[0]
        else:
            i += 1
    slots["DEBT"] = slots["DEBT_ALL"][0] if slots["DEBT_ALL"] else None

    # Numbers: mask rank spans so the "1" of "C1" is not a quantity
    masked = list(norm)
    for s, e, _ in ranks:
        masked[s:e] = " " * (e - s)
    masked = "".join(masked)
    for m in _NUM_RE.finditer(masked):
        slots["NUMS"].append(m.group().replace(",", ""))
    if slots["NUMS"]:
        slots["AMOUNT"] = slots["NUMS"][0]
        first = next(_NUM_RE.finditer(masked))
        val = float(slots["AMOUNT"])
        nxt = _tokens(masked[first.end():])[:1]
        if nxt and nxt[0] in _MULTIPLIERS:
            val *= _MULTIPLIERS[nxt[0]]
        slots["AMOUNT_VALUE"] = int(val) if val == int(val) else val

    # Time frame ("بازه" = range, not time; removed)
    slots["TIME"] = any(_tok_is(t, w, _PLURALS) for t in toks for w in _TIME_WORDS)

    # Negation: exact negative words or a negated imperfective verb (نمی‌X; keys are ZWNJ-free)
    slots["NEGATED"] = any(t in _NEG_EXACT or (t.startswith("نمی") and len(t) > 4) for t in toks)
    return slots


def extract_slots_broken(query: str) -> dict:
    """Starter stub (detects nothing). Used to demonstrate baseline broken behavior."""
    return {"BANK": None, "RANK": None, "PERSONA": None, "DEBT": None,
            "AMOUNT": None, "TIME": None}
