"""Shared helpers for the offline evaluation harness (honor-code clean:
only corpus/test.csv + /retrieve responses are used)."""
import csv, json, os, re, sys, time
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from rubbish_rag.normalize_fa import normalize_fa


def load_corpus():
    with open(os.path.join(BASE, "corpus", "test.csv"), encoding="utf-8-sig") as f:
        return [{k: v for k, v in r.items() if k} for r in csv.DictReader(f)]


def squash(s):
    """Whitespace/ZWNJ-insensitive comparison key."""
    return re.sub(r"[\s\u200c]+", "", normalize_fa(s or ""))


def gold_classes(rows):
    """doc_id (== 0-based row index) -> set of row ids with identical BriefAnswer."""
    by = {}
    for i, r in enumerate(rows):
        k = squash(r.get("BriefAnswer"))
        if k:
            by.setdefault(k, set()).add(i)
    return {i: by[squash(r["BriefAnswer"])] for i, r in enumerate(rows) if squash(r.get("BriefAnswer"))}


def cached(path, fn):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    out = fn()
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return out
