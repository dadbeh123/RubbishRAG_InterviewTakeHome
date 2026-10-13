from confirm_fusion import *
from rubbish_rag.grounding import repetition_score


orig = json.load(open("orig_orders.json", encoding="utf-8"))
RAW = lambda q, h: h
ORIG = None

def mk_orig(E):
    qid = {e["q"]: k for k, e in E.items()}
    return lambda q, h: orig[qid[q]]

def ladder(E):
    arms = {}
    arms["0 server order (raw, topk=20)"] = RAW
    arms["1 ORIGINAL pipeline (RRF@5 + MMR)"] = mk_orig(E)
    arms["2 + grounding only (server order)"] = run(None, dict(strategy="server"), {}, False)
    arms["3 + length-comp fusion"] = run(None, dict(strategy="lencomp"), {}, False)
    arms["4 + doc/content dedup"] = run(None, dict(strategy="lencomp"), dict(dedup=True, mmr_lambda=1.0, slot_beta=0.0))
    arms["5 + slot bonus = FINAL pipeline"] = run(None, dict(strategy="lencomp"), {})
    arms["x repetition-filter only (corpus-max thr.)"] = None
    return arms


import rubbish_rag.pipeline as P

# repetition-only defence threshold = max score among verified corpus records (data-derived)
maxrep = max(repetition_score((r.get("BriefAnswer") or "") + "\n" + (r.get("Answer") or "")) for r in rows)

def rep_only(q, h): 
    return [x for x in h if repetition_score(x["text"]) <= maxrep]

results = {"maxrep_threshold": maxrep}
for tag, E in (("TEST", tst), ("ALL", cache)):
    print(f"\n=== {tag} n={len(E)}  (repetition threshold from corpus = {maxrep:.3f})"); results[tag] = {}
    arms = ladder(E); arms["x repetition-filter only (corpus-max thr.)"] = rep_only
    for nm, fn in arms.items():
        m = evaluate(fn, E, gc, rows, nm); results[tag][nm] = m; print(fmt(m)[:165])
    if tag == "ALL":
        base, fin = per_query(arms["0 server order (raw, topk=20)"], E), per_query(arms["5 + slot bonus = FINAL pipeline"], E)
        orgq = per_query(arms["1 ORIGINAL pipeline (RRF@5 + MMR)"], E)
        print("FINAL vs server  (dMRR, dHit@3):", boot(fin, base)); print("FINAL vs ORIGINAL (dMRR, dHit@3):", boot(fin, orgq))
        results["boot_final_vs_server"] = boot(fin, base); results["boot_final_vs_orig"] = boot(fin, orgq)

json.dump(results, open("final_eval_results.json", "w"), indent=1)
# poison ids that repetition-only still lets through
leak = set(); caught = set()
for e in cache.values():
    for h in e["hits"]:
        if h["doc_id"] >= 9000: (leak if repetition_score(h["text"]) <= maxrep else caught).add(h["doc_id"])
        
print("\nrepetition-only: poison ids caught:", sorted(caught - leak), "| still leaking:", sorted(leak))
