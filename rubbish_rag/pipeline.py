"""RubbishRAG pipeline — the 1-minute version.

Stages (override any of them with @rubbish_rag.* decorators —
see docs/DECORATORS.md for the mechanism):
  chunk -> retrieve (remote) -> rerank -> resolve
"""
import re
from functools import lru_cache

import rubbish_rag as R
from .grounding import ground_filter
from .normalize_fa import normalize_fa
from .slots import extract_slots
from .orchestrator import CorpusStore


MAX_CHUNK_CHARS = 450
_SENTENCE_BREAK = re.compile(r"(?<=[.!؟؛])\s+|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=[،,])\s+")
_NUMBER = re.compile(r"^[0-9۰-۹٠-٩]+(?:[٫.][0-9۰-۹٠-٩]+)?$")
_NUMBER_LINK = {"تا", "الی", "و"}
_UNITS = {
    "درصد", "تومان", "ریال",
    "امتیاز", "ماه", "سال"
}


def _as_text(value):
    """None / NaN / missing -> "" (never the literal strings "None" or "nan")."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _clean_metadata(value):
    return " ".join(_as_text(value).split())


def _safe_word_break(text, position):
    """Avoid separating a number from its range or unit."""
    before = text[:position].split()
    after = text[position:].split()
    if not before or not after:
        return True

    left = before[-1]
    right = after[0]

    if _NUMBER.fullmatch(left) and right in (_NUMBER_LINK | _UNITS):
        return False
    if left in _NUMBER_LINK and _NUMBER.fullmatch(right):
        return False
    return True


def _split_long_text(text, limit):
    """Split oversized text at safe word boundaries."""
    while len(text) > limit:
        spaces = [
            m.start()
            for m in re.finditer(r"\s+", text[:limit + 1])
        ]
        safe = [
            p for p in spaces
            if _safe_word_break(text, p)
        ]

        cut = (safe or spaces)[-1] if spaces else limit
        if cut == 0:
            cut = limit

        yield text[:cut].strip()
        text = text[cut:].strip()

    if text:
        yield text


def _iter_pieces(raw_text, limit):
    """Prefer sentence, then clause, then word boundaries."""
    for sentence in _SENTENCE_BREAK.split(raw_text):
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        if len(sentence) <= limit:
            yield sentence
            continue
        for clause in _CLAUSE_BREAK.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= limit:
                yield clause
            else:
                yield from _split_long_text(clause, limit)


@R.chunker
def chunk(docs):
    """
    Context-preserving Persian chunking:

    Respect sentence boundaries.
    Prefer clause or word boundaries for long sentences.
    Preserve Category and Keyword in every chunk.
    Bound total chunk length.
    Keep stable doc_id and chunk_id.
    """
    chunks = []

    for doc in docs:
        # Answer is the body; fall back to BriefAnswer for rows that only have the short form.
        raw_text = _as_text(doc.get("Answer")).strip() or _as_text(doc.get("BriefAnswer")).strip()
        if not raw_text:
            continue

        category = _clean_metadata(doc.get("Category"))
        keyword = _clean_metadata(doc.get("Keyword"))

        # Construct a context header for every chunk.
        metadata = " | ".join(value for value in (category, keyword) if value)

        # Reserve at least 120 characters for answer content.
        max_header = MAX_CHUNK_CHARS - 200
        if metadata and len(metadata) + 3 > max_header:
            shortened = metadata[:max_header - 4].rsplit(" ", 1)[0]
            metadata = shortened.rstrip() + "…"

        header = f"[{metadata}]\n" if metadata else ""
        body_limit = MAX_CHUNK_CHARS - len(header)
        bodies = []
        current = ""

        for piece in _iter_pieces(raw_text, body_limit):
            if (current and len(current) + 1 + len(piece) > body_limit):
                bodies.append(current)
                current = ""
            current = (f"{current} {piece}" if current else piece)

        if current:
            bodies.append(current)

        # Attach document metadata to every chunk.
        for index, body in enumerate(bodies):
            chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f'{doc["doc_id"]}:{index}',
                "category": category,
                "keyword": keyword,
                "text": header + body,
            })

    return chunks


# Retrieval: over-fetch -> ground (poison defence) -> fuse
POOL_DEPTH = 55          # candidates requested from the server (it only returns 5 by default)
RERANK_DEPTH = 55        # hits handed to the reranker (>= topk), so dedup/MMR have room to work
FUSION = "lencomp"       # "lencomp" | "rrf" | "server"   (evidence: docs/PROOF.md, Task B)
LENGTH_GAMMA = 0.15      # BM25 length-compensation exponent (best dev MRR; docs/PROOF.md)
DENSE_WEIGHT = 0.30       
RRF_K = 100


def calculate_ranks(hits, score_key):
    """Rank hits by one score, giving tied scores the same rank."""
    order = sorted(range(len(hits)), key=lambda i: float(hits[i].get(score_key, 0.0)), reverse=True)
    ranks, previous, current = {}, None, 0
    for position, index in enumerate(order, start=1):
        score = float(hits[index].get(score_key, 0.0))
        if position == 1 or score != previous:
            current = position
        ranks[index] = current
        previous = score
    return ranks


def _minmax(values):
    lo, hi = min(values), max(values)
    span = hi - lo
    return [1.0 if span <= 0 else (v - lo) / span for v in values]


def _finish(hits, scores, name):
    """Attach the new score. Fused stays on a 0 to 1 scale. The server's own value is kept."""
    out = []
    top = max(scores) if scores else 1.0
    for hit, s in zip(hits, scores):
        h = dict(hit)
        h.setdefault("server_fused", hit.get("fused", 0.0))
        h[name] = s
        h["fused"] = s / top if top > 0 else 0.0
        out.append(h)
    return sorted(out, key=lambda h: h["fused"], reverse=True)


def rrf_fuse(hits, k=RRF_K, weights=(1.0, 0.5)):
    """Reciprocal Rank Fusion: sum_i w_i / (k + rank_i(d)).

    On this corpus equal-weight RRF *loses* to the server's
    own blend (see docs/PROOF.md). The dense ranker is the weaker signal and RRF throws
    away BM25's magnitude, which for exact lexical matches is considered information."""
    if not hits:
        return []
    rb, rd = calculate_ranks(hits, "bm25"), calculate_ranks(hits, "dense")
    scores = [weights[0] / (k + rb[i]) + weights[1] / (k + rd[i]) for i in range(len(hits))]
    return _finish(hits, scores, "rrf_score")


def length_compensated_fuse(hits, gamma=LENGTH_GAMMA, dense_weight=DENSE_WEIGHT):
    """
    Counter BM25's length normalisation (b=0.9 punishes thorough chunks):
    bm25' = bm25 * len(text)**gamma     (gamma=0 -> plain BM25)
    score = (1-w) * minmax(bm25') + w * minmax(dense)
    The constant reference length cancels inside min-max, so only gamma matters.
    """
    if not hits:
        return []
    adj = [float(h.get("bm25", 0.0)) * (max(len(h.get("text", "")), 1) ** gamma) for h in hits]
    dense = [float(h.get("dense", 0.0)) for h in hits]
    b, d = _minmax(adj), _minmax(dense)
    scores = [(1.0 - dense_weight) * b[i] + dense_weight * d[i] for i in range(len(hits))]
    return _finish(hits, scores, "lencomp_score")


def fuse(hits, strategy=None, **kw):
    strategy = strategy or FUSION
    if strategy == "rrf":
        return rrf_fuse(hits, **kw)
    if strategy == "lencomp":
        return length_compensated_fuse(hits, **kw)
    if strategy == "server":
        return [dict(h) for h in hits]
    raise ValueError(f"unknown fusion strategy: {strategy!r}")


class HitList(list):
    """A plain list that remembers the topk the caller asked for, so the reranker can
    trim to it *after* diversifying (the CLI contract is: retrieve(topk) -> rerank -> topk hits)."""
    topk = None


def process_retrieval(raw_hits, topk=5, strategy=None, depth=None, ground=True, **kw):
    """Pure post-processing of the server's candidate list (shared with the offline eval)."""
    hits = ground_filter(raw_hits) if ground else list(raw_hits)
    hits = fuse(hits, strategy, **kw)
    return hits[: max(topk, RERANK_DEPTH if depth is None else depth)]


@R.retriever
def retrieve(query, topk=5, api_key=None):
    """
    Normalize --> over-fetch --> drop ungrounded (poison) chunks --> length-compensated fusion

    Returns at least topk hits (RERANK_DEPTH) so the reranker can diversify.
    Callers slice [:topk] after reranking.
    """
    from server._hidden_retriever import remote_retrieve

    raw = remote_retrieve(normalize_fa(query), topk=max(POOL_DEPTH, topk), api_key=api_key)
    out = HitList(process_retrieval(raw, topk=topk))
    out.topk = topk
    return out


# Reranking: slot alignment, document-level dedup, MMR
MMR_LAMBDA = 0.5
SLOT_BETA = 1.5


@lru_cache(maxsize=2048)
def _mmr_tokens_cached(text):
    return frozenset(re.findall(r"\w+", normalize_fa(text).lower()))


def _mmr_tokens(text):
    """Tokenize text for lightweight diversity scoring."""
    return _mmr_tokens_cached(str(text))


def _text_similarity(left, right):
    """Cosine-like similarity over token presence (dependency-free)."""
    a, b = _mmr_tokens(left), _mmr_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def _squash_text(text):
    return re.sub(r"[\s\u200c]+", "", normalize_fa(str(text)))


def _text_slots(text):
    return extract_slots(text)


def slot_alignment(query_slots, text):
    """
    +1 if the chunk mentions the query's rank/bank, -1 if it names only *other*
    ranks/banks, 0 when it says nothing about them.
    """
    if not (query_slots.get("RANK_ALL") or query_slots.get("BANK_ALL")):
        return 0.0
    ts = _text_slots(str(text))
    score = 0.0
    for key in ("RANK_ALL", "BANK_ALL"):
        want, have = set(query_slots.get(key) or []), set(ts.get(key) or [])
        if want and have:
            score += 1.0 if want & have else -1.0
    return score


def maximal_marginal_relevance(hits, lambda_mult=MMR_LAMBDA, top_k=None):
    """
    MMR = lambda * relevance - (1 - lambda) * max similarity to already selected.

    Relevance is min-max normalised to [0, 1] *within the candidate list* so it is on the
    same scale as the 0..1 similarity (raw RRF scores live in ~0.03 and were dominated
    by the redundancy term).
    """
    if not hits:
        return []
    top_k = len(hits) if top_k is None else top_k
    cands = [dict(h) for h in hits]
    rel = _minmax([float(h.get("rerank_base", h.get("fused", 0.0))) for h in cands])
    pool = list(zip(cands, rel))
    selected, sel_texts = [], []
    while pool and len(selected) < top_k:
        best_i, best_s = 0, float("-inf")
        for i, (h, r) in enumerate(pool):
            red = max((_text_similarity(h.get("text", ""), t) for t in sel_texts), default=0.0)
            s = lambda_mult * r - (1.0 - lambda_mult) * red
            if s > best_s:
                best_i, best_s = i, s
        h, _ = pool.pop(best_i)
        h["mmr_score"] = h["rerank_score"] = best_s
        selected.append(h)
        sel_texts.append(h.get("text", ""))
    return selected


def rerank_hits(query, hits, mmr_lambda=None, slot_beta=None, dedup=True, topk=None):
    """
    Evidence reranking: (1) slot alignment bonus, (2) push same-document and
    same-content duplicates behind distinct evidence, (3) MMR over the distinct set.
    """
    topk = topk or getattr(hits, "topk", None)
    mmr_lambda = MMR_LAMBDA if mmr_lambda is None else mmr_lambda
    slot_beta = SLOT_BETA if slot_beta is None else slot_beta
    if not hits:
        return []
    qs = extract_slots(query)
    faq_scores = dict(
        CorpusStore.get().find_similar_questions(query, top_n=330)
    )
    rel = _minmax([float(h.get("fused", 0.0)) for h in hits])
    scored = []
    for h, r in zip(hits, rel):
        n = dict(h)
        n["slot_align"] = slot_alignment(qs, h.get("text", "")) if slot_beta else 0.0
        n["rerank_base"] = (
            r
            + slot_beta * n["slot_align"]
            + 1.5 * faq_scores.get(h["doc_id"], 0.0)
        )
        scored.append(n)
    scored.sort(key=lambda h: h["rerank_base"], reverse=True)

    distinct, dups, seen_docs, seen_text = [], [], set(), set()
    for h in scored:
        key = _squash_text(h.get("text", ""))
        if dedup and (h.get("doc_id") in seen_docs or key in seen_text):
            dups.append(h)
            continue
        seen_docs.add(h.get("doc_id"))
        seen_text.add(key)
        distinct.append(h)

    ordered = maximal_marginal_relevance(distinct, mmr_lambda) if mmr_lambda < 1.0 else distinct
    for h in ordered:
        h.setdefault("rerank_score", h["rerank_base"])
    for h in dups:                       # kept (list length is part of the contract), but last
        h["rerank_score"] = -1.0 + h["rerank_base"] * 1e-3
        h["duplicate"] = True
    result = ordered + dups
    return result[:topk] if topk else result


@R.reranker
def rerank(query, hits):
    return rerank_hits(query, hits)


@R.resolver
def resolve(query, hits):
    """Current behavior: always returns an answer built from the top hits.

    Cites are the first three *distinct* documents (the reranker already pushed duplicates
    to the back).  The clarify-on-ambiguity behaviour is out of scope for Tasks A-F.
    """
    top = hits[0] if hits else {"text": "", "doc_id": -1}
    cites = []
    for h in hits:
        if h["doc_id"] not in cites:
            cites.append(h["doc_id"])
        if len(cites) == 3:
            break
    return {"status": "answer", "brief": top["text"][:200], "cites": cites or [top["doc_id"]]}
