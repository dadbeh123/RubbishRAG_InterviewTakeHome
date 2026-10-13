"""Poison / noise-document defence: ground retrieved chunks against the verified corpus.

The index may contain injected documents (IDs far outside the corpus).  Two kinds were
observed empirically (see docs/PROOF.md, Track F):

keyword-stuffed spam           ("رتبه A برای همه! (پاسخ جعلی) رتبه A1 A2 ...")
well-formed corrupted copies of real answers with altered facts (wrong score band,
  wrong fee).  These look perfectly natural, so a repetition detector cannot find them.

A retrieved chunk is grounded iff its text is a contiguous substring (ignoring
whitespace/ZWNJ and script variants) of the verified record that its own doc_id
points to (doc_id is the 0-based row of corpus/test.csv. the server builds its
chunks from BriefAnswer + Answer).  There is no ID threshold: an id that does not
index a corpus row simply has nothing to be grounded in, and an id that does must prove
that the text really comes from that record.

repetition_score is an independent, label-free diagnostic (bigram repetition rate).
"""
from __future__ import annotations

import csv
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from .normalize_fa import _canon_chars

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(BASE, "corpus", "test.csv")
_STRIP_WS = re.compile(r"[\s\u200c]+")


def _squash(text: Optional[str]) -> str:
    return _STRIP_WS.sub("", _canon_chars(text or ""))


def repetition_score(text: str) -> float:
    """Fraction of word bigrams that are repeats (0 = no repetition, -> 1 = stuffed)."""
    toks = re.findall(r"\w+", _canon_chars(text or ""))
    bigrams = list(zip(toks, toks[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


class GroundingIndex:
    """Verified-record store keyed by doc_id (0-based corpus row)."""

    def __init__(self, rows: Optional[Iterable[Dict[str, str]]] = None, path: Optional[str] = None):
        self.records: List[str] = []
        if rows is None:
            path = path or DEFAULT_CORPUS
            try:
                with open(path, encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
            except OSError:
                rows = []
        for r in rows:
            self.records.append(_squash(r.get("BriefAnswer")) + _squash(r.get("Answer")))

    @property
    def available(self) -> bool:
        return bool(self.records)

    def is_grounded(self, hit: Dict) -> bool:
        d = hit.get("doc_id")
        if not isinstance(d, int) or not (0 <= d < len(self.records)):
            return False
        key = _squash(hit.get("text"))
        return bool(key) and key in self.records[d]

    def split(self, hits: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        kept, dropped = [], []
        for h in hits:
            (kept if self.is_grounded(h) else dropped).append(h)
        return kept, dropped


_DEFAULT: Optional[GroundingIndex] = None
_WARNED = False


def default_index() -> GroundingIndex:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GroundingIndex()
    return _DEFAULT


def ground_filter(hits: List[Dict], index: Optional[GroundingIndex] = None) -> List[Dict]:
    """Drop every hit that is not grounded.  If the corpus is unavailable we fail open
    (with a one-time warning) rather than return nothing."""
    global _WARNED
    index = index or default_index()
    if not index.available:
        if not _WARNED:
            print("[grounding] corpus/test.csv not found - poison defence DISABLED", file=sys.stderr)
            _WARNED = True
        return list(hits)
    return index.split(hits)[0]
