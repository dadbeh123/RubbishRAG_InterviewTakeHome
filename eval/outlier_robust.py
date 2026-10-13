import numpy as np, json
from tune import *
import rubbish_rag.pipeline as P


full = [squash(r.get("BriefAnswer")) + squash(r.get("Answer")) for r in rows]
g = lambda h: h["doc_id"] < len(rows) and squash(h["text"]) in full[h["doc_id"]]

def comp(hits):
    b = sorted((h["bm25"] for h in hits), reverse=True)
    return (b[0] - b[1]) / (b[0] - b[-1] + 1e-9)          # share of min-max range owned by top1-vs-runner-up gap

srv = lambda q, h: h
for pool_name, filt in (("raw pool (as served)", lambda hs: hs), ("grounded pool", lambda hs: [x for x in hs if g(x)])):
    C = {k: {"q": e["q"], "hits": filt(e["hits"])} for k, e in cache.items()}
    score = {k: comp(v["hits"]) for k, v in C.items()}
    thr = np.percentile(list(score.values()), 75)
    out = {k: v for k, v in C.items() if score[k] >= thr}; non = {k: v for k, v in C.items() if score[k] < thr}
    rr5 = lambda q, h: P.rrf_fuse(h[:5])
    print(f"== {pool_name}: outlier quartile threshold = {thr:.3f} (top1-runner-up gap owns >= {thr:.0%} of the BM25 range)")
    for nm, E in ((f"outlier quartile n={len(out)}", out), (f"others n={len(non)}", non)):
        a, b = evaluate(srv, E, gc, rows), evaluate(rr5, E, gc, rows)
        print(f"   {nm:24s} server H@3={a['hit3']:.3f} MRR={a['mrr']:.3f} | RRF@5 H@3={b['hit3']:.3f} MRR={b['mrr']:.3f}")
    # trend by quintile
    qs = np.quantile(list(score.values()), np.linspace(0, 1, 6)); tr = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        ks = [k for k in C if lo <= score[k] <= hi]; tr.append((round(float(np.mean([score[k] for k in ks])), 3), round(evaluate(srv, {k: C[k] for k in ks}, gc, rows)['hit3'], 3)))
    print("   server Hit@3 by outlier quintile:", tr)
    json.dump({"thr": float(thr), "trend": tr}, open(f"outlier_{'raw' if 'raw' in pool_name else 'grounded'}.json", "w"))
