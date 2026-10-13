from tune import *
import random, numpy as np
def per_query(fn, E):
    out = {}
    for k, e in E.items():
        g = gc[int(k)]; ids = [h["doc_id"] for h in fn(e["q"], e["hits"])]
        r = next((j + 1 for j, d in enumerate(ids) if d in g), None)
        out[k] = (1.0 / r if r else 0.0, 1.0 if (r and r <= 3) else 0.0)
    return out
def boot(a, b, B=4000, seed=1):
    ks = list(a); rng = np.random.default_rng(seed); n = len(ks)
    dm = np.array([a[k][0] - b[k][0] for k in ks]); dh = np.array([a[k][1] - b[k][1] for k in ks])
    idx = rng.integers(0, n, size=(B, n)); f = lambda x: (round(float(x.mean()), 4), round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4))
    return f(dm[idx].mean(1)), f(dh[idx].mean(1))
if __name__ == "__main__":
    base = run(None, dict(strategy="server"), {}, False)
    arms = {"lencomp g=0.75 w=0 (SELECTED)": dict(strategy="lencomp", gamma=0.75, dense_weight=0.0),
            "lencomp g=0.5 w=0.25": dict(strategy="lencomp", gamma=0.5, dense_weight=0.25),
            "RRF equal k=60": dict(strategy="rrf")}
    for tag, E in (("DEV", dev), ("TEST", tst), ("ALL", cache)):
        print("==", tag, "n=", len(E)); print(fmt(evaluate(base, E, gc, rows, "server order, grounded"))[:100])
        for nm, kw in arms.items():
            fn = run(None, kw, {}, False); print(fmt(evaluate(fn, E, gc, rows, nm))[:100])
            m, h = boot(per_query(fn, E), per_query(base, E)); print(f"      paired vs server: dMRR={m}  dHit@3={h}")
