"""A/B: how does the remote index treat ZWNJ?  Variants of ZWNJ-containing corpus questions."""
import json, os
from common import *
from server._hidden_retriever import remote_retrieve


rows = load_corpus(); gc = gold_classes(rows)
clean = json.load(open(os.path.join(BASE, "eval", "cache_clean.json"), encoding="utf-8"))
ids = [k for k, e in clean.items() if "\u200c" in e["q"]]
print("questions containing ZWNJ:", len(ids), "of", len(clean))
out = {}

for k in ids:
    q = clean[k]["q"]
    out[k] = {"orig": clean[k]["hits"],
              "space": remote_retrieve(q.replace("\u200c", " "), topk=20, api_key="fix-eval-zwnj"),
              "join": remote_retrieve(q.replace("\u200c", ""), topk=20, api_key="fix-eval-zwnj")}
    
json.dump(out, open(os.path.join(BASE, "eval", "cache_zwnj.json"), "w", encoding="utf-8"), ensure_ascii=False)
