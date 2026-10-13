from tune import *
import itertools
FK = dict(strategy="lencomp")   # pipeline defaults (gamma=.75, w=0)
rows_out = []
print("== stage 2: rerank on DEV (fusion = pipeline default)")
print(fmt(evaluate(run(dev, FK, {}, False), dev, gc, rows, "no rerank"))[:150])
for dedup, lam, beta in itertools.product((False, True), (1.0, 0.9, 0.8, 0.7, 0.5), (0.0, 0.1, 0.25, 0.5, 1.0)):
    if not dedup and lam == 1.0 and beta == 0.0: continue
    m = evaluate(run(dev, FK, dict(dedup=dedup, mmr_lambda=lam, slot_beta=beta)), dev, gc, rows, f"dedup={int(dedup)} lam={lam} beta={beta}")
    rows_out.append(m)
for m in sorted(rows_out, key=lambda m: (-m["mrr"], -m["hit3"]))[:12]: print(fmt(m)[:150])
print("-- dedup only / MMR only / slot only")
for m in rows_out:
    if (m["label"] in ("dedup=1 lam=1.0 beta=0.0", "dedup=0 lam=0.7 beta=0.0", "dedup=0 lam=1.0 beta=0.5")): print(fmt(m)[:150])
