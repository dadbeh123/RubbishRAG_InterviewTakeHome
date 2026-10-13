import zlib, itertools, sys
from metrics import *
import rubbish_rag.pipeline as P
rows = load_corpus(); gc = build_gold(rows)
cache = json.load(open("cache_clean.json", encoding="utf-8"))
dev = {k: v for k, v in cache.items() if zlib.crc32(k.encode()) % 2 == 0}
tst = {k: v for k, v in cache.items() if k not in dev}
_FUSED = {}
def run(E, fuse_kw, rr_kw, rerank=True):
    def fn(q, raw):
        key = (q, tuple(sorted(fuse_kw.items())))
        if key not in _FUSED:
            _FUSED[key] = P.process_retrieval(raw, topk=5, **fuse_kw)
        hs = _FUSED[key]
        return P.rerank_hits(q, hs, **rr_kw) if rerank else hs
    return fn
if __name__ == "__main__":
    print("== stage 1: fusion (grounded, no rerank), DEV")
    res = []
    for g, w in itertools.product((0, .25, .5, .75, 1.0), (0.0, .25, .5)):
        m = evaluate(run(dev, dict(strategy="lencomp", gamma=g, dense_weight=w), {}, False), dev, gc, rows, f"lencomp g={g} w={w}")
        res.append((m["mrr"], m["hit3"], g, w)); print(fmt(m)[:104])
    m = evaluate(run(dev, dict(strategy="server"), {}, False), dev, gc, rows, "server order (grounded)"); print(fmt(m)[:104])
    print("best dev by MRR:", sorted(res, reverse=True)[:3])
