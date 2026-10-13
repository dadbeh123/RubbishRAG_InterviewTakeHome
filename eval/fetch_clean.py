"""Cache raw top-20 server hits for every corpus Question (1 call each)."""
import sys, json, os
from common import *
from server._hidden_retriever import remote_retrieve


rows = load_corpus(); gc = gold_classes(rows)
qs = [(i, rows[i]["Question"]) for i in sorted(gc) if (rows[i].get("Question") or "").strip()]
print("labelled queries:", len(qs), "of", len(rows))
out = {}

for i, q in qs:
    out[str(i)] = {"q": q, "hits": remote_retrieve(q, topk=20, api_key="fix-eval-clean")}
    
json.dump(out, open(os.path.join(BASE, "eval", "cache_clean.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("saved", len(out))
