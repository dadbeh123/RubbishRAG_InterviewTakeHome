from tune import *
from rubbish_rag.slots import extract_slots
FK = dict(strategy="lencomp")
sub = {k: v for k, v in cache.items() if (lambda s: s["RANK_ALL"] or s["BANK_ALL"])(extract_slots(v["q"]))}
print("queries with a RANK/BANK slot:", len(sub), "of", len(cache), "| ranks:", sum(bool(extract_slots(v['q'])['RANK_ALL']) for v in sub.values()), "banks:", sum(bool(extract_slots(v['q'])['BANK_ALL']) for v in sub.values()))
for beta in (0.0, 0.1, 0.25, 0.5, 1.0):
    m = evaluate(run(None, FK, dict(dedup=True, mmr_lambda=1.0, slot_beta=beta)), sub, gc, rows, f"slot subset beta={beta}")
    print(fmt(m)[:110])
print("-- baseline server order on the same subset")
print(fmt(evaluate(run(None, dict(strategy='server'), {}, False), sub, gc, rows, "server (grounded)"))[:110])
