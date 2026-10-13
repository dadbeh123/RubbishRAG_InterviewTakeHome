"""Replay the ORIGINAL candidate pipeline (pristine copy) on cached raw hits: RRF over the
server's top-5 followed by the original MMR. Run with sys.path pointing at /home/claude/orig."""
import sys, json
sys.path.insert(0, "/home/claude/orig")
import rubbish_rag.pipeline as OP


assert OP.__file__.startswith("/home/claude/orig"), OP.__file__

cache = json.load(open("/home/claude/work/eval/cache_clean.json", encoding="utf-8"))
out = {}
for k, e in cache.items():
    hits = OP.rrf_fuse(e["hits"][:5])                       # original retrieve(): topk=5 then RRF
    hits = OP.maximal_marginal_relevance(hits, 0.7)          # original rerank()
    out[k] = [{"doc_id": h["doc_id"], "text": h["text"]} for h in hits]

json.dump(out, open("/home/claude/work/eval/orig_orders.json", "w"), ensure_ascii=False)
print("replayed", len(out))
