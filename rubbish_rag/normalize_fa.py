"""Persian text normalization for Credit-Scoring QA.

Design goals (verified in tests/test_normalization.py and docs/PROOF.md):

Idempotent – normalize_fa(normalize_fa(x)) == normalize_fa(x).
Semantics-preserving for numbers – digits are mapped to ASCII, Arabic
thousands/decimal separators become , / .; digits are never dropped.
Corpus-native half-spaces – the indexed corpus uses ZWNJ ~3x more often
than a plain space, and the remote index is ZWNJ-sensitive (see PROOF.md,
ZWNJ A/B).  So ZWNJ is the canonical form: stray/duplicate ZWNJs are
cleaned and ZWNJs that users typed as spaces are restored
(می شود -> می‌شود, چک های -> چک‌های).
Careful letter folding – ي ى ك ة أ إ ٱ ۀ are folded; legitimate Persian
letters such as ئ and ؤ are left alone.
Colloquial -> formal at the token level – no regex \\b (which does not
understand ZWNJ) and no dead keys: lookups ignore ZWNJ.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, Optional, Tuple

ZWNJ = "\u200c"

FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"

_CHAR_MAP = {ord(a): b for a, b in zip(FA_DIGITS + AR_DIGITS, EN_DIGITS * 2)}
_CHAR_MAP.update({
    ord("ي"): "ی", ord("ى"): "ی",          # Arabic yeh / alef 
    ord("ك"): "ک",                          # Arabic kaf
    ord("ة"): "ه",                          # teh 
    ord("أ"): "ا", ord("إ"): "ا", ord("ٱ"): "ا",
    ord("ۀ"): "ه" + ZWNJ + "ی",             # heh+yeh-above -> corpus-native form
})
_STRIP = re.compile("[\u0640\u064b-\u065f\u0670\u06d6-\u06ed"
                    "\u200b\u200d\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
_WS = re.compile(r"[^\S\u200c]+")                       # any whitespace except ZWNJ
# (The Persian comma "،" is deliberately untouched: "250،300" is more often a list than 250,300.)
_THOUSANDS_SEP = re.compile(r"(?<=[0-9])\u066c(?=[0-9])")
_DECIMAL_SEP = re.compile(r"(?<=[0-9])\u066b(?=[0-9])")
_ZWNJ_RUN = re.compile(ZWNJ + "{2,}")

# Rule-based half-space restoration (language level not corpus specific)
_MI_PREFIX = re.compile(rf"(?<![\w{ZWNJ}])(نمی|می) (?=[^\W\d_])")
_PLURAL_SUFFIX = re.compile(rf"(?<=[^\W\d_]) (ها|های|هایی)(?![\w{ZWNJ}])")

# ----------------------------------------------------------------------------
# Colloquial -> formal.  Keys are ZWNJ-free; values are canonical (ZWNJ form).
# ----------------------------------------------------------------------------
_COLLOQUIAL_VERBS: Dict[str, str] = {
    "میشه": "می‌شود", "نمیشه": "نمی‌شود", "میشن": "می‌شوند", "نمیشن": "نمی‌شوند",
    "میتونم": "می‌توانم", "نمیتونم": "نمی‌توانم", "میتونه": "می‌تواند", "نمیتونه": "نمی‌تواند",
    "میتونی": "می‌توانی", "نمیتونی": "نمی‌توانی", "میخوام": "می‌خواهم", "نمیخوام": "نمی‌خواهم",
    "میخواد": "می‌خواهد", "نمیخواد": "نمی‌خواهد", "میدونم": "می‌دانم", "نمیدونم": "نمی‌دانم",
    "میکنه": "می‌کند", "نمیکنه": "نمی‌کند", "میکنم": "می‌کنم", "نمیکنم": "نمی‌کنم",
    "میره": "می‌رود", "میاد": "می‌آید", "نمیاد": "نمی‌آید", "میگیره": "می‌گیرد",
    "نمیگیره": "نمی‌گیرد", "میده": "می‌دهد", "نمیده": "نمی‌دهد", "میگن": "می‌گویند",
    "میگه": "می‌گوید", "میفته": "می‌افتد", "میمونه": "می‌ماند", "میشم": "می‌شوم",
    "بشه": "شود", "نشه": "نشود", "باشه": "باشد", "نباشه": "نباشد",
    "داره": "دارد", "نداره": "ندارد", "دارن": "دارند", "ندارن": "ندارند",
}
_COLLOQUIAL_OTHER: Dict[str, str] = {
    "چیه": "چیست", "چنده": "چند است", "چقد": "چقدر", "چطوره": "چگونه است", "کیه": "کیست",
    "اینه": "این است", "چی": "چه", "چیکار": "چه کار", "چجوری": "چگونه", "اگه": "اگر",
    "دیگه": "دیگر", "اون": "آن", "اونجا": "آنجا", "واسه": "برای", "بخاطر": "به خاطر",
    "الان": "اکنون",
}
_POSSESSIVE_NOUNS = ("رتبه", "امتیاز", "سابقه", "شرکت", "چک", "وام", "گزارش", "مالیات",
                     "حساب", "تسهیلات", "اقساط", "قسط", "کارت", "ضامن", "ضمانت")


def _build_colloquial() -> Dict[str, str]:
    m: Dict[str, str] = {}
    m.update(_COLLOQUIAL_VERBS)
    m.update(_COLLOQUIAL_OTHER)
    for n in _POSSESSIVE_NOUNS:
        stems = {n}
        if n.endswith("ه"):
            stems.add(n[:-1])               # رتبم (final heh dropped in speech)
        for s in stems:
            m[s + "م"] = n + " من"
            m[s + "مو"] = n + " من را"
            m[s + "ام"] = n + " من"
    return {k: v for k, v in m.items() if k != v}


COLLOQUIAL_MAP: Dict[str, str] = _build_colloquial()

# ----------------------------------------------------------------------------
# Corpus-derived ZWNJ lexicon (built by scripts/build_zwnj_lexicon.py)
# ----------------------------------------------------------------------------
_LEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "zwnj_lexicon.json")
_LEX_RE: Optional[re.Pattern] = None
_LEX_MAP: Dict[str, str] = {}


def _lexicon() -> Tuple[Optional[re.Pattern], Dict[str, str]]:
    """Lazy-load the lexicon.  Each corpus pair ``A B`` yields two surface keys that both
    restore to ``A\u200cB``: the spaced form (``می شود``) and the fully joined one (``میشود``)."""
    global _LEX_RE, _LEX_MAP
    if _LEX_RE is None:
        try:
            with open(_LEX_PATH, encoding="utf-8") as f:
                pairs = json.load(f)["pairs"]
        except (OSError, ValueError, KeyError):
            pairs = []
        mapping: Dict[str, str] = {}
        for p in sorted(set(pairs), key=len):
            a, b = p.split(" ", 1)
            canon = a + ZWNJ + b
            mapping[p] = canon
            mapping.setdefault(a + b, canon)       # unspaced typing; first (shortest) pair wins
        keys = sorted(mapping, key=len, reverse=True)
        body = "|".join(re.escape(k) for k in keys) or r"(?!x)x"
        _LEX_RE = re.compile(rf"(?<![\w{ZWNJ}])({body})(?![\w{ZWNJ}])")
        _LEX_MAP = mapping
    return _LEX_RE, _LEX_MAP


def _canon_chars(text: str) -> str:
    """Character-level canonicalisation (no half-space restoration)."""
    s = unicodedata.normalize("NFKC", text)
    s = _STRIP.sub("", s)
    s = s.translate(_CHAR_MAP)
    s = _THOUSANDS_SEP.sub(",", s)
    s = _DECIMAL_SEP.sub(".", s)
    s = _ZWNJ_RUN.sub(ZWNJ, s)
    s = _WS.sub(" ", s)
    # a ZWNJ is only meaningful between two letters
    s = re.sub(rf"(?<![^\W\d_]){ZWNJ}|{ZWNJ}(?![^\W\d_])", "", s)
    return s.strip()


def _restore_half_spaces(s: str) -> str:
    """Join ``می شود`` / ``چک های`` / lexicon pairs with ZWNJ until a fixpoint."""
    lex, mapping = _lexicon()
    for _ in range(8):                       # each pass strictly removes spaces -> terminates
        prev = s
        s = _MI_PREFIX.sub(lambda m: m.group(1) + ZWNJ, s)
        s = _PLURAL_SUFFIX.sub(lambda m: ZWNJ + m.group(1), s)
        s = lex.sub(lambda m: mapping[m.group(1)], s)
        if s == prev:
            break
    return s


def _colloquial(s: str) -> str:
    out = []
    for tok in s.split(" "):
        key = tok.replace(ZWNJ, "")
        # keep leading/trailing punctuation outside the lookup
        m = re.fullmatch(r"([^\w]*)(.*?)([^\w]*)", key)
        pre, core, post = m.group(1), m.group(2), m.group(3)
        rep = COLLOQUIAL_MAP.get(core)
        out.append(f"{pre}{rep}{post}" if rep else tok)
    return " ".join(out)


def normalize_fa(text: Optional[str]) -> str:
    """Normalize Persian text to its canonical, idempotent form."""
    if not text:
        return ""
    s = _canon_chars(str(text))
    s = _MI_PREFIX.sub(lambda m: m.group(1) + ZWNJ, s)     # می شه -> می‌شه  (so lookup works)
    s = _colloquial(s)
    s = _restore_half_spaces(s)
    s = _canon_chars(s)                                    # tidy anything the steps introduced
    return s
