"""IR metrics over cached /retrieve pools."""
import math
from common import *

def build_gold(rows):
    return gold_classes(rows)

def evaluate(order_fn, cache, gc, rows, label=""):
    """order_fn(query, hits)-> ordered list of hits. Returns dict of metrics."""
    n = h1 = h3 = h5 = 0; rr = 0.0; pois3 = 0; pois_cites = 0; cites = 0
    dup_doc5 = 0.0; dup_txt5 = 0.0; uniq3 = 0.0
    for k, e in cache.items():
        g = gc[int(k)]
        hs = order_fn(e["q"], e["hits"])
        ids = [h["doc_id"] for h in hs]
        n += 1
        rank = next((j + 1 for j, d in enumerate(ids) if d in g), None)
        h1 += rank == 1; h3 += bool(rank and rank <= 3); h5 += bool(rank and rank <= 5)
        rr += 1.0 / rank if rank else 0.0
        top3 = ids[:3]
        pois3 += any(d >= 9000 for d in top3)
        pois_cites += sum(d >= 9000 for d in top3); cites += len(top3)
        top5 = hs[:5]
        dup_doc5 += 1 - len({h["doc_id"] for h in top5}) / max(len(top5), 1)
        dup_txt5 += 1 - len({squash(h["text"]) for h in top5}) / max(len(top5), 1)
        uniq3 += len({squash(h["text"]) for h in hs[:3]}) / max(len(hs[:3]), 1)
    return {"label": label, "n": n, "hit1": h1 / n, "hit3": h3 / n, "hit5": h5 / n, "mrr": rr / n,
            "poison_q@3": pois3 / n, "poison_cite@3": pois_cites / max(cites, 1),
            "dup_docid@5": dup_doc5 / n, "dup_text@5": dup_txt5 / n, "uniq_text@3": uniq3 / n}

def fmt(m):
    return (f'{m["label"]:34s} H@1={m["hit1"]:.3f} H@3={m["hit3"]:.3f} H@5={m["hit5"]:.3f} MRR={m["mrr"]:.3f} '
            f'poisonQ@3={m["poison_q@3"]:.3f} dupDoc@5={m["dup_docid@5"]:.3f} dupTxt@5={m["dup_text@5"]:.3f} uniqTxt@3={m["uniq_text@3"]:.3f}')
