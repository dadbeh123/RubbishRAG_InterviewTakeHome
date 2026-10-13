#!/usr/bin/env python3
"""Random/grid-style hyperparameter tuning for the ORIGINAL RubbishRAG pipeline.

Uses ONLY the published visible benchmark/corpus and recorded /retrieve responses.
All ranking uses the existing pipeline.process_retrieval() and rerank_hits().
Never reads hidden tests or sealed server internals.

Usage (from the project root, with this file copied there):
    python tune_bench.py --trials 1200
    python tune_bench.py --trials 600 --objective corpus-dev
    python3.11 tune_bench.py --source live --trials 1200
    python3.11 tune_bench.py --official-best
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rubbish_rag import pipeline as P
from rubbish_rag.normalize_fa import normalize_fa

# Match the original, submitted pipeline. This must run as trial #0.
BASELINE = {
    "fusion": "lencomp", "pool_depth": 20, "rerank_depth": 10,
    "gamma": 0.75, "dense_weight": 0.0,
    "rrf_k": 60, "rrf_dense_weight": 1.0,
    "mmr_lambda": 1.0, "slot_beta": 0.5, "dedup": True,
}

# Tunable ranges. We keep the original algorithms, NOT the earlier proposed reranker.
SPACE = {
    "pool_depth": [5, 10, 15, 20],
    "rerank_depth": [5, 10, 15, 20],
    "gamma": [round(0.15 * i, 2) for i in range(11)],         # 0 .. 1.5
    "dense_weight": [round(0.1 * i, 2) for i in range(11)], # 0 .. 1
    "rrf_k": [10, 30, 60, 100],
    "rrf_dense_weight": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "mmr_lambda": [0.5, 0.65, 0.8, 0.9, 1.0],
    "slot_beta": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "dedup": [True, False],
}
COLUMNS = list(BASELINE) + ["accuracy", "correct", "mrr3", "hit1", "poison_at3", "trials_seen", "phase"]


def json_read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def bench_items():
    items = json_read(ROOT / "server/visible_bench.json")
    if not items or any(it.get("expect") != "answer" for it in items):
        raise ValueError("This script's documented Hit@3 fallback only covers answer queries")
    return items


def load_cached_pool(path):
    """Cache uses corpus row IDs as keys; map by normalized *query*, not row ID."""
    cache = json_read(path)
    pools = {}
    for row in cache.values() if isinstance(cache, dict) else cache:
        key = normalize_fa(row["q"])
        if key in pools and row["hits"] != pools[key]:
            raise ValueError(f"Ambiguous normalized query in cache: {key}")
        pools[key] = row["hits"]
    return pools


def live_pools(items, path, key, depth, refresh=False):
    """Fetch once through the sanctioned remote_retrieve API, then reuse the results."""
    if path.exists() and not refresh:
        saved = json_read(path)
        if saved.get("depth", 0) < depth:
            raise ValueError("Existing live cache is too shallow; pass --refresh")
        return {normalize_fa(row["q"]): row["hits"] for row in saved["data"]}
    try:
        from server._hidden_retriever import remote_retrieve
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Live retrieval requires the challenge's Python 3.11 environment") from exc
    data = []
    for index, it in enumerate(items, 1):
        query = it["query"]
        hits = remote_retrieve(normalize_fa(query), topk=depth, api_key=key)
        data.append({"q": query, "hits": hits})
        print(f"\rCollecting official /retrieve pools: {index}/{len(items)}", end="", flush=True)
    print()
    json_write(path, {"depth": depth, "data": data})
    return {normalize_fa(row["q"]): row["hits"] for row in data}


def config_key(cfg):
    # Collapse irrelevant params (otherwise random trials often duplicate each other).
    fields = ["fusion", "pool_depth", "rerank_depth", "mmr_lambda", "slot_beta", "dedup"]
    fields += (["gamma", "dense_weight"] if cfg["fusion"] == "lencomp" else
               ["rrf_k", "rrf_dense_weight"] if cfg["fusion"] == "rrf" else [])
    return tuple((k, cfg[k]) for k in fields)


def candidates(seed, count):
    """Baseline + controlled one-at-a-time probes + seeded random configurations."""
    rng = random.Random(seed)
    seen = set()
    def unique(c):
        key = config_key(c)
        if key in seen:
            return False
        seen.add(key)
        return True

    if unique(BASELINE):
        yield dict(BASELINE)
    n = 1
    # Build reliable ablations before sampling parameter interactions.
    for fusion in ("lencomp", "rrf", "server"):
        c = {**BASELINE, "fusion": fusion}
        if unique(c):
            yield c
            n += 1
            if n >= count:
                return
    for name, vals in SPACE.items():
        for value in vals:
            c = {**BASELINE, name: value}
            if unique(c):
                yield c
                n += 1
                if n >= count:
                    return
    # Sample all three strategies, but sample lencomp more often.
    attempts = 0
    while n < count:
        attempts += 1
        if attempts > 100 * count:
            break
        c = {**BASELINE, "fusion": rng.choices(["lencomp", "rrf", "server"], [0.6, 0.3, 0.1])[0]}
        for name, vals in SPACE.items():
            c[name] = rng.choice(vals)
        if c["rerank_depth"] > c["pool_depth"]:
            c["rerank_depth"] = c["pool_depth"]
        if unique(c):
            yield c
            n += 1


def one_parameter_neighbors(cfg):
    """Coordinate-descent refinement after broad seeded search."""
    names = ["fusion", "pool_depth", "rerank_depth", "mmr_lambda", "slot_beta", "dedup"]
    names += (["gamma", "dense_weight"] if cfg["fusion"] == "lencomp" else
              ["rrf_k", "rrf_dense_weight"] if cfg["fusion"] == "rrf" else [])
    for name in names:
        values = ["lencomp", "rrf", "server"] if name == "fusion" else SPACE[name]
        for value in values:
            if value == cfg[name]:
                continue
            candidate = {**cfg, name: value}
            if candidate["rerank_depth"] > candidate["pool_depth"]:
                # Rerank depth > pool size is redundant; do not spend experiments on it.
                continue
            yield candidate


def ranking_key(m):
    """Primary objective is benchmark Hit@3; tie-break with MRR@3 and Hit@1."""
    return (m["accuracy"], m["mrr3"], m["hit1"], -m["poison_at3"])


def process_one(query, raw, config):
    """Use EXACTLY the original fusion+rerank+resolve functions, with explicit params."""
    # Slice BEFORE score normalization: simulates a smaller remote topk call.
    raw = raw[:config["pool_depth"]]
    strategy = config["fusion"]
    if strategy == "lencomp":
        kw = {"gamma": config["gamma"], "dense_weight": config["dense_weight"]}
    elif strategy == "rrf":
        kw = {"k": config["rrf_k"], "weights": (1.0, config["rrf_dense_weight"])}
    else:
        kw = {}
    hits = P.process_retrieval(
        raw, topk=5, strategy=strategy, depth=config["rerank_depth"], ground=True, **kw
    )
    # The official bench's resolver is fed only the requested 5 reranked hits.
    ranked = P.rerank_hits(
        query, hits, mmr_lambda=config["mmr_lambda"],
        slot_beta=config["slot_beta"], dedup=config["dedup"], topk=5,
    )
    prediction = P.resolve(query, ranked)
    return prediction, ranked


def gold_equivalents():
    """Optional: accept corpus records with the exact same normalized BriefAnswer."""
    with open(ROOT / "corpus/test.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_answer = {}
    for i, row in enumerate(rows):
        answer = normalize_fa(row.get("BriefAnswer", "")).replace(" ", "").replace("\u200c", "")
        if answer:
            by_answer.setdefault(answer, set()).add(i)
    equivalents = {}
    for values in by_answer.values():
        for i in values:
            equivalents[i] = values
    return equivalents


def evaluate(items, pools, config, use_official=False, eq=None, predictions=False):
    correct = top1 = poison = 0
    rr3 = 0.0
    details = []
    score_one = None
    if use_official:
        try:
            from server.evaluator import score_one as official_score_one
            score_one = official_score_one
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Official scoring requires the challenge's Python 3.11 environment") from exc
    for it in items:
        q = it["query"]
        raw = pools.get(normalize_fa(q))
        if raw is None:
            raise ValueError(f"Query is missing from retrieval cache: {q!r}")
        pred, ranked = process_one(q, raw, config)
        ids = list(pred.get("cites", []))[:3]
        gold = int(it["gold_doc_id"])
        acceptable = eq.get(gold, {gold}) if eq is not None else {gold}
        ranks = [j + 1 for j, docid in enumerate(ids) if docid in acceptable]
        rank = min(ranks) if ranks else None
        if score_one is not None:
            passed = bool(score_one(pred, it).get("pass", False))
        else:
            passed = bool(pred.get("status") == "answer" and rank is not None)
        correct += passed
        top1 += (rank == 1)
        poison += any(int(doc) >= 9000 for doc in ids)
        rr3 += 1.0 / rank if rank else 0.0
        if predictions:
            details.append({"id": it.get("id"), "query": q, "expected": gold,
                            "cites": ids, "pass": passed, "rank": rank})
    n = len(items)
    result = {"accuracy": correct / n, "correct": correct, "n": n,
              "mrr3": rr3 / n, "hit1": top1 / n, "poison_at3": poison / n}
    return result, details


def corpus_items_and_split(seed, cache_path):
    """Tuning on 75% of corpus QAs, keeping 25% untouched for a validation check."""
    with open(ROOT / "corpus/test.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    entries = json_read(cache_path)
    all_items = []
    for i, row in enumerate(rows):
        q = row.get("Question", "")
        if not q or str(i) not in entries:
            continue
        all_items.append({"id": f"corpus-{i}", "query": q, "gold_doc_id": i, "expect": "answer"})
    # Stable split, independent of hyperparameters and random-search sampling.
    dev, hold = [], []
    for it in all_items:
        hashvalue = zlib.crc32(f"{seed}:{it['id']}".encode("utf-8")) % 100
        (dev if hashvalue < 75 else hold).append(it)
    return dev, hold


def write_results(outdir, records, best, selected_details, meta):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "all_trials.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    json_write(outdir / "best_config.json", {k: best[k] for k in BASELINE})
    json_write(outdir / "top20.json", sorted(records,
               key=lambda r: (r["accuracy"], r["mrr3"], r["hit1"], -r["poison_at3"]),
               reverse=True)[:20])
    json_write(outdir / "best_predictions.json", selected_details)
    json_write(outdir / "summary.json", meta)


def run_search(args):
    start = time.perf_counter()
    visible = bench_items()
    eq = gold_equivalents() if args.accept_equivalent_briefs else None
    if args.source == "cache":
        pools = load_cached_pool(ROOT / args.cache)
        source_name = args.cache
    else:
        live_cache = ROOT / args.live_cache
        pools = live_pools(visible, live_cache, args.key, depth=20, refresh=args.refresh)
        source_name = args.live_cache
    if args.objective == "bench":
        train = visible
        validation = []
        official = args.official_score
    else:
        train, validation = corpus_items_and_split(args.seed, ROOT / args.cache)
        official = False
    if args.official_score and args.objective != "bench":
        raise ValueError("--official-score is only available with --objective bench")
    if not train:
        raise ValueError("No tuning questions found")
    # Require pool availability at startup rather than dropping inconvenient queries.
    missing = [it["id"] for it in train if normalize_fa(it["query"]) not in pools]
    if missing:
        raise ValueError(f"Missing training query pools: {missing[:10]}")
    if validation and any(normalize_fa(it["query"]) not in pools for it in validation):
        raise ValueError("Validation queries missing from cached pool")

    results = []
    visited = {}
    best_cfg = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    baseline, _ = evaluate(train, pools, BASELINE, official, eq)
    print(f"Baseline: {baseline['correct']}/{baseline['n']} ({baseline['accuracy']:.1%}) ")

    def try_config(cfg, phase="random"):
        nonlocal best_score, best_cfg
        key = config_key(cfg)
        if key in visited:
            return visited[key]
        metrics, _ = evaluate(train, pools, cfg, official, eq)
        row = {**cfg, **metrics, "trials_seen": len(results) + 1, "phase": phase}
        results.append(row)
        visited[key] = row
        score = ranking_key(metrics)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            print(f"Trial {len(results):4d} | NEW BEST {metrics['correct']}/{metrics['n']}"
                  f" ({metrics['accuracy']:.1%}) MRR@3={metrics['mrr3']:.3f}"
                  f"  {cfg}", flush=True)
        return row

    for cfg in candidates(args.seed, args.trials):
        try_config(cfg)
        if len(results) % args.progress == 0:
            print(f"Trial {len(results)}/{args.trials} | best Hit@3={best_score[0]:.1%}", flush=True)

    # Starting from several distinct strong trials, locally refine the parameters.
    # This can improve over blind random search while keeping all experiments logged.
    if args.refine_rounds > 0:
        seeds = sorted(results, key=ranking_key, reverse=True)[:args.refine_seeds]
        for seed_num, seed in enumerate(seeds, 1):
            current = {k: seed[k] for k in BASELINE}
            for _ in range(args.refine_rounds):
                current_metric = visited[config_key(current)]
                improved = current
                best_local = ranking_key(current_metric)
                for candidate in one_parameter_neighbors(current):
                    result = try_config(candidate, phase=f"refine_{seed_num}")
                    if ranking_key(result) > best_local:
                        improved, best_local = candidate, ranking_key(result)
                if improved == current:
                    break
                current = improved

    if best_cfg is None:
        raise RuntimeError("No trials evaluated")

    selected, details = evaluate(train, pools, best_cfg, official, eq, predictions=True)
    summary = {"source": source_name, "objective": args.objective,
               "scoring": "official server.evaluator.score_one" if official else "published Hit@3",
               "trials": len(results), "seed": args.seed, "baseline": baseline,
               "best": selected, "best_config": best_cfg,
               "seconds": round(time.perf_counter() - start, 2)}
    if validation:
        val, val_details = evaluate(validation, pools, best_cfg, False, eq, predictions=True)
        base_val, _ = evaluate(validation, pools, BASELINE, False, eq)
        summary.update({"validation": val, "validation_baseline": base_val})
        # Visible bench is for reporting only in this split (not part of objective).
        bench_best, _ = evaluate(visible, pools, best_cfg, False, eq)
        bench_base, _ = evaluate(visible, pools, BASELINE, False, eq)
        summary.update({"visible_bench_best": bench_best, "visible_bench_baseline": bench_base})
        json_write(ROOT / args.out / "validation_predictions.json", val_details)
    write_results(ROOT / args.out, results, best_cfg, details, summary)
    print("\nRESULT")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved CSV / best configuration / predictions to: {ROOT / args.out}")
    print("To run actual official bench: python3.11 tune_bench.py --official-best"
          f" --out {args.out}")


def official_best(args):
    """Executes the EXACT challenge benchmark, no approximation or metric proxy."""
    config = json_read(ROOT / args.out / "best_config.json")
    from rubbish_rag.tuning_override import install
    install(config)
    from rubbish import _run_bench
    agg, results = _run_bench(args.key)
    out = ROOT / args.out
    json_write(out / "official_metrics.json", {"agg": agg, "results": results,
                                                 "config": config})
    print("Official visible bench with tuned parameters:")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"Saved: {out / 'official_metrics.json'}")
    print("Note: the standard bench also writes traces/metrics.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=800, help="Max candidate configs including baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", choices=["cache", "live"], default="cache",
                    help="Cached 330-query /retrieve responses (fast), or refresh via API")
    ap.add_argument("--cache", default="eval/cache_clean.json")
    ap.add_argument("--live-cache", default="eval/live_visible_tuning_cache.json")
    ap.add_argument("--refresh", action="store_true", help="Refetch live API pools")
    ap.add_argument("--key", default="demo-key", help="API key forwarded to the official bench")
    ap.add_argument("--objective", choices=["bench", "corpus-dev"], default="bench")
    ap.add_argument("--official-score", action="store_true", help="Use the official score_one each trial (Python 3.11)")
    ap.add_argument("--accept-equivalent-briefs", action="store_true",
                    help="For published-Hit@3 proxy only, count identical BriefAnswer records too")
    ap.add_argument("--out", default="tuning_results")
    ap.add_argument("--progress", type=int, default=100)
    ap.add_argument("--refine-rounds", type=int, default=2,
                    help="Coordinate-descent passes after random search (0 to disable)")
    ap.add_argument("--refine-seeds", type=int, default=3,
                    help="Number of strongest random trials to refine")
    ap.add_argument("--official-best", action="store_true",
                    help="Use best_config.json to execute the EXACT rubbish.py bench")
    args = ap.parse_args()
    if args.trials < 1 or args.progress < 1 or args.refine_rounds < 0 or args.refine_seeds < 1:
        ap.error("--trials, --progress, and --refine-seeds must be positive; --refine-rounds >= 0")
    if args.official_best:
        official_best(args)
    else:
        run_search(args)


if __name__ == "__main__":
    main()
