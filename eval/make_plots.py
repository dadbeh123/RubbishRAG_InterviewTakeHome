import json, os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from confirm_fusion import *
import rubbish_rag.pipeline as P


PL = os.path.join(BASE, "plots")
FR = json.load(open("final_eval_results.json")); NA = json.load(open("norm_ab_results.json"))
full = [squash(r.get("BriefAnswer")) + squash(r.get("Answer")) for r in rows]
grounded = lambda h: h["doc_id"] < len(rows) and squash(h["text"]) in full[h["doc_id"]]
wilson = lambda p, n, z=1.96: ((p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n), (p + z*z/(2*n) + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))

# ---------- Figure 1: Task B ablation (plots/length_vs_bm25.png) ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
L_g, B_g, L_o, B_o = [], [], [], []
for k, e in cache.items():
    g = gc[int(k)]
    for h in e["hits"]:
        if not grounded(h): continue
        (L_g, B_g) if h["doc_id"] in g else (L_o, B_o)
        (L_g if h["doc_id"] in g else L_o).append(len(h["text"])); (B_g if h["doc_id"] in g else B_o).append(h["bm25"])

ax[0].scatter(L_o, B_o, s=5, alpha=.15, label=f"non-gold (n={len(L_o)})", color="#7f7f7f")
ax[0].scatter(L_g, B_g, s=9, alpha=.55, label=f"gold (n={len(L_g)})", color="#d62728")

for arr, c, nm in ((np.array([L_o, B_o]), "#444", "non-gold"), (np.array([L_g, B_g]), "#a00", "gold")):
    z = np.polyfit(arr[0], arr[1], 1); xs = np.linspace(40, 300, 50); ax[0].plot(xs, np.polyval(z, xs), c, lw=2)

ax[0].set_xlabel("retrieved chunk length (chars)"); ax[0].set_ylabel("raw server BM25")
ax[0].set_title("A. BM25 vs length: gold vs non-gold (330 queries, grounded)"); ax[0].legend(fontsize=8)
arms = [("server order\n(raw)", "0 server order (raw, topk=20)"), ("ORIGINAL\nRRF@5+MMR", "1 ORIGINAL pipeline (RRF@5 + MMR)"),
        ("+ grounding", "2 + grounding only (server order)"), ("+ length-comp\nfusion", "3 + length-comp fusion"),
        ("+ dedup", "4 + doc/content dedup"), ("FINAL", "5 + slot bonus = FINAL pipeline")]
n = FR["ALL"][arms[0][1]]["n"]; xs = np.arange(len(arms)); h3 = [FR["ALL"][a]["hit3"] for _, a in arms]
err = np.array([[p - wilson(p, n)[0], wilson(p, n)[1] - p] for p in h3]).T
cols = ["#7f7f7f", "#d62728", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
ax[1].bar(xs, h3, yerr=err, color=cols, capsize=3)

for i, v in enumerate(h3):
    ax[1].text(i, .02, f"{v:.1%}", ha="center", fontsize=9, color="white", fontweight="bold")

ax[1].set_xticks(xs); ax[1].set_xticklabels([a for a, _ in arms], fontsize=7.5); ax[1].set_ylim(0, .6)
ax[1].set_ylabel("Hit@3 (Wilson 95% CI)"); ax[1].set_title("B. Paired ablation ladder, n=330")
base = per_query(run(None, dict(strategy="server"), {}, False), cache)
final = per_query(run(None, dict(strategy="lencomp"), {}), cache)
rrf_pq = per_query(run(None, dict(strategy="rrf"), {}, False), cache)
rank = lambda fn, k: next((j + 1 for j, h in enumerate(fn(cache[k]["q"], cache[k]["hits"])) if h["doc_id"] in gc[int(k)]), None)
fs, fr, fl = (run(None, dict(strategy="server"), {}, False), run(None, dict(strategy="rrf"), {}, False), run(None, dict(strategy="lencomp"), {}))
gains_rrf, gains_fin, glen = [], [], []

for k, e in cache.items():
    r0, r1, r2 = rank(fs, k), rank(fr, k), rank(fl, k)
    if r0 and r1 and r2:
        glen.append(np.mean([len(h["text"]) for h in e["hits"] if h["doc_id"] in gc[int(k)] and grounded(h)] or [0])); gains_rrf.append(r0 - r1); gains_fin.append(r0 - r2)

rng = np.random.default_rng(0); j = lambda n: rng.uniform(-.12, .12, n)
ax[2].scatter(np.array(glen) + j(len(glen)) * 8, np.array(gains_rrf) + j(len(glen)), s=10, alpha=.4, color="#d62728", label=f"RRF equal-weight (mean {np.mean(gains_rrf):+.2f})")
ax[2].scatter(np.array(glen) + j(len(glen)) * 8, np.array(gains_fin) + j(len(glen)), s=10, alpha=.4, color="#08519c", label=f"length-comp (mean {np.mean(gains_fin):+.2f})")
ax[2].axhline(0, color="k", lw=.8, ls="--"); ax[2].set_xlabel("gold chunk length (chars)"); ax[2].set_ylabel("gold rank gain vs server (+ = better)")
ax[2].set_title("C. Per-query gold-rank change (gold in pool)"); ax[2].legend(fontsize=8)
fig.suptitle("Task B - length_vs_bm25: fusion ablation on 330 corpus-derived labelled queries", fontsize=12); fig.tight_layout()
fig.savefig(os.path.join(PL, "length_vs_bm25.png"), dpi=140); plt.close(fig)

# ---------- Figure 2: outlier hypothesis + score spreads (Task A distributions) ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
gaps = [e["hits"][0]["fused"] - e["hits"][4]["fused"] for e in cache.values()]
ax[0].hist(gaps, bins=20, color="#3182bd"); ax[0].set_title("Top1-Top5 server fused gap (330 queries)"); ax[0].set_xlabel("gap")

def _comp(hits):
    b = sorted((h["bm25"] for h in hits), reverse=True)
    return (b[0] - b[1]) / (b[0] - b[-1] + 1e-9)      # share of the min-max BM25 range owned by the top1-vs-runner-up gap

ratio = {k: _comp(e["hits"]) for k, e in cache.items()}
thr = np.percentile(list(ratio.values()), 75); out = {k: v for k, v in cache.items() if ratio[k] >= thr}; non = {k: v for k, v in cache.items() if ratio[k] < thr}
srv = lambda q, h: h; rr5 = lambda q, h: P.rrf_fuse(h[:5])
bars = []

for nm, E in ((f"outlier quartile\n(n={len(out)})", out), (f"others\n(n={len(non)})", non)):
    bars.append((nm, evaluate(srv, E, gc, rows)["hit3"], evaluate(rr5, E, gc, rows)["hit3"]))

x = np.arange(2); ax[1].bar(x - .18, [b[1] for b in bars], .36, label="server fused", color="#7f7f7f"); ax[1].bar(x + .18, [b[2] for b in bars], .36, label="RRF@5", color="#d62728")
for i, b in enumerate(bars): 
    ax[1].text(i - .18, b[1] + .01, f"{b[1]:.1%}", ha="center", fontsize=8); ax[1].text(i + .18, b[2] + .01, f"{b[2]:.1%}", ha="center", fontsize=8)

ax[1].set_xticks(x); ax[1].set_xticklabels([b[0] for b in bars]); ax[1].set_ylabel("Hit@3"); ax[1].legend(); ax[1].set_title("Outlier hypothesis falsified:\nRRF does not help where BM25 has an outlier")
ax[2].scatter([ratio[k] for k in cache], [base[k][1] for k in cache], s=6, alpha=.0)
bins = np.quantile(list(ratio.values()), np.linspace(0, 1, 6)); xs_, ys_ = [], []

for a, b in zip(bins[:-1], bins[1:]):
    ks = [k for k in cache if a <= ratio[k] <= b]; xs_.append(np.mean([ratio[k] for k in ks])); ys_.append(np.mean([base[k][1] for k in ks]))

ax[2].plot(xs_, ys_, "o-", color="#08519c"); ax[2].set_xlabel("outlier strength: top1-runner-up gap / BM25 range"); ax[2].set_ylabel("server Hit@3")
ax[2].set_title("Larger BM25 outliers = *easier* queries\n(outliers are exact lexical matches = signal)")
fig.tight_layout(); fig.savefig(os.path.join(PL, "outlier_hypothesis.png"), dpi=140); plt.close(fig)

# ---------- Figure 3: poison + redundancy ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
names = [("server order", "0 server order (raw, topk=20)"), ("original\npipeline", "1 ORIGINAL pipeline (RRF@5 + MMR)"), ("repetition\nfilter only", "x repetition-filter only (corpus-max thr.)"), ("grounding", "2 + grounding only (server order)"), ("FINAL", "5 + slot bonus = FINAL pipeline")]
v = [FR["ALL"][a]["poison_q@3"] for _, a in names]; ax[0].bar(range(len(v)), v, color=["#7f7f7f", "#d62728", "#fd8d3c", "#31a354", "#08519c"])
for i, t in enumerate(v):
    ax[0].text(i, t + .004, f"{t:.1%}", ha="center", fontsize=9)

ax[0].set_xticks(range(len(v))); ax[0].set_xticklabels([a for a, _ in names], fontsize=8); ax[0].set_ylabel("queries with a poison doc in top-3"); ax[0].set_title("Track F: poison citation rate (330 queries)")
names2 = [("server\norder", "0 server order (raw, topk=20)"), ("+ grounding", "2 + grounding only (server order)"), ("+ length-comp", "3 + length-comp fusion"), ("+ dedup", "4 + doc/content dedup")]
d1 = [FR["ALL"][a]["dup_docid@5"] for _, a in names2]; d2 = [FR["ALL"][a]["dup_text@5"] for _, a in names2]
x = np.arange(len(names2)); ax[1].bar(x - .18, d1, .36, label="same doc_id", color="#6baed6"); ax[1].bar(x + .18, d2, .36, label="identical text", color="#08519c")
ax[1].set_xticks(x); ax[1].set_xticklabels([a for a, _ in names2], fontsize=8); ax[1].set_ylabel("redundancy rate in top-5"); ax[1].legend(); ax[1].set_title("Track E: document redundancy in top-5")
fig.tight_layout(); fig.savefig(os.path.join(PL, "poison_and_redundancy.png"), dpi=140); plt.close(fig)

# ---------- Figure 4: normalization A/B ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); arm_names = ["clean (upper bound)", "raw perturbed", "OLD normalize(perturbed)", "NEW normalize(perturbed)"]
for a, (cond, res) in zip(ax, NA.items()):
    v = [res[x]["mrr"] for x in arm_names]; a.bar(range(4), v, color=["#bdbdbd", "#d62728", "#fd8d3c", "#08519c"])
    for i, t in enumerate(v): a.text(i, t + .005, f"{t:.3f}", ha="center", fontsize=8)
    a.set_xticks(range(4)); a.set_xticklabels(["clean", "raw", "OLD\nnorm", "NEW\nnorm"]); a.set_title(f"{cond} (n={res[arm_names[0]]['n']})", fontsize=9); a.set_ylabel("MRR"); a.set_ylim(0, .48)

fig.suptitle("Task C: normalization A/B against the real retriever (MRR)"); fig.tight_layout(); fig.savefig(os.path.join(PL, "normalization_ab.png"), dpi=140); plt.close(fig)
print("plots written; mean rank gain RRF %.3f lencomp %.3f (n=%d)" % (np.mean(gains_rrf), np.mean(gains_fin), len(gains_rrf)))
print("RRF worse/same/better:", sum(g < 0 for g in gains_rrf), sum(g == 0 for g in gains_rrf), sum(g > 0 for g in gains_rrf), "| lencomp:", sum(g < 0 for g in gains_fin), sum(g == 0 for g in gains_fin), sum(g > 0 for g in gains_fin))
