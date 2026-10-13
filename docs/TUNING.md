# Hyperparameter tuner for ORIGINAL RubbishRAG pipeline

This patch evaluates `process_retrieval`, `rerank_hits`, and `resolve` functions under different hyperparameters.

## Install

1. Copy `tune_bench.py` into the **root** of your original project, next to `rubbish.py`.
2. Copy `tuning_override.py` into `rubbish_rag/`.

## Search 800 configs with existing 20 visible bench retrieval pools

```bash
python tune_bench.py --trials 800 --seed 42
```

- Reads the actual 20 questions from `server/visible_bench.json`.
- Reuses recorded `/retrieve` results from `eval/cache_clean.json` (recorded previously by the original solution).
- Re-scores all 20 using the original stages, with the answer-query Hit@3 metric documented by the challenge.
- Does not re-fetch candidates or read hidden evaluation data.
- Prints every improvement and saves `tuning_results/all_trials.csv`, `best_config.json`, `best_predictions.json`, and `summary.json`.
- The first trial is **the original config** and should reproduce **40%**. If not, confirm you used the original `pipeline.py` and original cache.

## Refresh visible candidate pools through the live `/retrieve` API

```bash
python tune_bench.py --source live --trials 800 --refresh --out tuning_live
```

It calls `/retrieve` **once per visible question**, caches the responses, then tunes without repeated remote API calls. If available, `--official-score` uses the server's public evaluator for each trial:

```bash
python tune_bench.py --source live --official-score --trials 800 --out tuning_live
```

Do not use `--refresh` when unnecessary: it makes remote calls.

## Verify the winning config with the EXACT official benchmark

```bash
python tune_bench.py --official-best
```

## Hyperparameters searched

- Fusion: `lencomp`, `rrf`, `server`.
- Length compensation exponent: `gamma = 0.0...1.5`.
- Dense/lexical fusion weight: `dense_weight = 0.0...1.0`.
- RRF smoothing and dense rank weight: `rrf_k`, `rrf_dense_weight`.
- Candidate pool and reranker input depth: `pool_depth`, `rerank_depth` (5, 10, 15, 20).
- MMR diversity: `mmr_lambda` (0.5...1.0).
- Slot alignment: `slot_beta` (0...2.0).
- Deduplication: on/off.

Important: These candidate pools cannot recover gold documents that are absent from the `/retrieve` top 20. The 20-question visible benchmark is also a tiny set: directly maximizing it can overfit. Treat the winner as a candidate, not a guarantee on the hidden benchmark.
