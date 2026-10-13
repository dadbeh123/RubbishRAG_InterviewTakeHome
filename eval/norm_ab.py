"""Normalization A/B on perturbed corpus questions (real retriever, grounded pool)."""
import importlib.util, re, sys, json
from common import *
import qcache
from metrics import evaluate, fmt, build_gold
import rubbish_rag.normalize_fa as NEW

spec = importlib.util.spec_from_file_location("old_norm", "/home/claude/orig/rubbish_rag/normalize_fa.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)

rows = load_corpus(); gc = build_gold(rows)
full = [squash(r.get("BriefAnswer")) + squash(r.get("Answer")) for r in rows]
grounded = lambda h: h["doc_id"] < len(rows) and squash(h["text"]) in full[h["doc_id"]]
clean = json.load(open("cache_clean.json", encoding="utf-8"))
qcache.seed()
AR = {ord(a): b for a, b in zip("0123456789۰۱۲۳۴۵۶۷۸۹", "٠١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٧٨٩")}

def drift_script(q):
    return q.translate({**AR, ord("ی"): "ي", ord("ک"): "ك"})

def drift_zwnj(q):
    return q.replace("\u200c", " ")

COLLOQ = {"می‌شود": "میشه", "نمی‌شود": "نمیشه", "چیست": "چیه", "می‌توانم": "میتونم", "نمی‌توانم": "نمیتونم",
          "می‌خواهم": "میخوام", "نمی‌کند": "نمیکنه", "می‌کند": "میکنه", "دارد": "داره", "ندارد": "نداره",
          "باشد": "باشه", "چه": "چی", "اگر": "اگه", "آن": "اون"}

def drift_colloq(q):
    for f, c in COLLOQ.items():
        q = re.sub(rf"(?<![\w\u200c]){re.escape(f)}(?![\w\u200c])", c, q)
    return q

CONDS = {"S script-drift (ي ك ٠-٩)": (drift_script, "fix-eval-norm-S"),
         "Z ZWNJ typed as space": (drift_zwnj, "fix-eval-norm-Z"),
         "C colloquial": (drift_colloq, "fix-eval-norm-C")}
         
results = {}
for name, (fn, key) in CONDS.items():
    ids = [k for k, e in clean.items() if fn(e["q"]) != e["q"]]
    arms = {"clean (upper bound)": lambda q: q,
            "raw perturbed": lambda q: fn(q),
            "OLD normalize(perturbed)": lambda q: OLD.normalize_fa(fn(q)),
            "NEW normalize(perturbed)": lambda q: NEW.normalize_fa(fn(q))}
    print(f"\n=== {name}: n={len(ids)}")
    results[name] = {}
    for arm, tf in arms.items():
        E = {k: {"q": clean[k]["q"], "hits": [h for h in qcache.get(tf(clean[k]["q"]), key) if grounded(h)]} for k in ids}
        m = evaluate(lambda q, h: h, E, gc, rows, arm); results[name][arm] = m
        print(fmt(m)[:118])
        
qcache.save(); print("calls per key:", qcache.calls)
json.dump(results, open("norm_ab_results.json", "w"), indent=1)
