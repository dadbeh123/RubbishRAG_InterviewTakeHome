"""Disk-backed query -> hits cache so each distinct query string hits /retrieve once."""
import json, os
from common import *
from server._hidden_retriever import remote_retrieve
PATH = os.path.join(BASE, "eval", "qcache.json")
_C = json.load(open(PATH, encoding="utf-8")) if os.path.exists(PATH) else {}
calls = {}
def seed():
    cl = json.load(open(os.path.join(BASE, "eval", "cache_clean.json"), encoding="utf-8"))
    zz = json.load(open(os.path.join(BASE, "eval", "cache_zwnj.json"), encoding="utf-8"))
    for k, e in cl.items():
        _C.setdefault(e["q"], e["hits"])
        if k in zz:
            _C.setdefault(e["q"].replace("\u200c", " "), zz[k]["space"])
            _C.setdefault(e["q"].replace("\u200c", ""), zz[k]["join"])
def get(q, key):
    if q not in _C:
        _C[q] = remote_retrieve(q, topk=20, api_key=key); calls[key] = calls.get(key, 0) + 1
    return _C[q]
def save():
    json.dump(_C, open(PATH, "w", encoding="utf-8"), ensure_ascii=False)
