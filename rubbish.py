"""RubbishRAG CLI — your evaluation and development toolkit.

Your boss said AI built this in 1 minute. Prove it wrong.

Commands:
  salam        — check key + quota
  bepar        — single probe:  rubbish bepar "سوال چی؟" --topk 5
  bench        — run visible bench (20 queries), save metrics.json
  bekesh       — generate diagnostic plots from your logs (alias: rapchik)
  bastesh      — stamp README report card + pack submission.zip
  chat         — interactive terminal chat with clarification loop
  ui           — launch the minimalistic web studio playground
  orchestrate  — run the end-to-end LangChain-style orchestrator
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from rubbish_rag.art import LOGO_CLEAN as LOGO
from rubbish_rag.identity import ensure_identity


def _check_sealed_compat():
    """Sealed candidate builds ship bytecode for one Python version only."""
    sealed = os.path.join(BASE, "server", "_hidden_retriever.pyc")
    src = os.path.join(BASE, "server", "_hidden_retriever.py")
    if os.path.exists(sealed) and not os.path.exists(src):
        info_p = os.path.join(BASE, "server", ".buildinfo.json")
        need = "3.11"
        try:
            need = json.load(open(info_p, encoding="utf-8"))["python"]
        except Exception:
            pass
        cur = f"{sys.version_info.major}.{sys.version_info.minor}"
        if cur != need:
            sys.exit(
                f"error: this sealed build needs Python {need} "
                f"(you have {cur}). Install Python {need} and retry.")


REPORT_START = "<!-- RUBBISH-REPORT:START -->"
REPORT_END = "<!-- RUBBISH-REPORT:END -->"


def _quota(key):
    p = os.path.join(BASE, "traces", "quota.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8")).get(key, 0)
    return 0


def cmd_salam(args):
    print(LOGO)
    print(f"Key: {args.key} | Quota used: {_quota(args.key)} / 1000")
    print("Corpus: corpus/test.csv (full copy, 330 distinct Persian credit QA rows)")
    print("Start: rubbish bepar \"رتبه C1 یعنی چی؟\"")


def cmd_bepar(args):
    print(LOGO)
    sys.path.insert(0, BASE)
    import rubbish_rag as R
    import rubbish_rag.pipeline  # noqa registers RubbishRAG stages
    # note: pipeline.retrieve ignores key; call remote directly for logging key
    hits = R.get("retriever")(args.query, topk=args.topk)
    hits = R.get("reranker")(args.query, hits)
    pred = R.get("resolver")(args.query, hits)
    print("--- hits (fused) ---")
    for h in hits[:args.topk]:
        print(f'doc={h["doc_id"]} fused={h["fused"]} bm25={h["bm25"]} '
              f'dense={h["dense"]} cat={h.get("category","")[:30]}')
        print("   " + h["text"][:160].replace("\n", " "))
    print("--- resolve ---")
    print(json.dumps(pred, ensure_ascii=False, indent=1)[:1500])


def _run_bench(key="demo-key"):
    from server.evaluator import score_one, aggregate
    vis = json.load(open(os.path.join(BASE, "server", "visible_bench.json"),
                         encoding="utf-8"))
    results, durs = [], []
    import rubbish_rag as R
    import rubbish_rag.pipeline  # noqa registers broken stages
    from server._hidden_retriever import remote_retrieve
    for item in vis:
        t0 = time.perf_counter()
        hits = R.get("retriever")(item["query"], topk=5, api_key=key)
        hits = R.get("reranker")(item["query"], hits)
        pred = R.get("resolver")(item["query"], hits)
        durs.append((time.perf_counter() - t0) * 1000.0)
        s = score_one(pred, item)
        results.append({**s, "id": item["id"]})
    agg = aggregate(results)
    import statistics
    agg["timing_ms"] = {
        "mean": round(statistics.mean(durs), 1),
        "p50": round(statistics.median(durs), 1),
        "p95": round(sorted(durs)[max(0, int(len(durs) * 0.95) - 1)], 1),
    }
    os.makedirs(os.path.join(BASE, "traces"), exist_ok=True)
    json.dump({"pipe": "naive", "agg": agg, "results": results},
              open(os.path.join(BASE, "traces", "metrics.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    return agg, results


def cmd_bench(args):
    print(LOGO)
    agg, _ = _run_bench(args.key)
    print("pipe=naive (RubbishRAG)")
    print(json.dumps(agg, ensure_ascii=False, indent=1))
    print("Saved: traces/metrics.json (every probe logged in traces/trace.jsonl)")


def cmd_bekesh(args):
    print(LOGO)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, BASE)
    import rubbish_rag as R
    import rubbish_rag.pipeline
    vis = json.load(open(os.path.join(BASE, "server", "visible_bench.json"),
                         encoding="utf-8"))
    gaps, lens, bms = [], [], []
    rep_bm, plain_bm = [], []

    def max_repeat(text):
        from collections import Counter
        toks = text.split()
        if not toks:
            return 0
        return max(Counter(toks).values())

    for item in vis[:20]:
        hits = R.get("retriever")(item['query'], topk=5)
        hits = R.get("reranker")(item['query'], hits)
        if hits:
            gaps.append(hits[0]["fused"] - hits[-1]["fused"])
        for h in hits[:2]:
            lens.append(len(h["text"]))
            bms.append(h["bm25"])
            if max_repeat(h["text"]) >= 8:
                rep_bm.append(h["bm25"])
            else:
                plain_bm.append(h["bm25"])
    os.makedirs(os.path.join(BASE, "plots"), exist_ok=True)
    plt.figure()
    plt.hist(gaps, bins=10)
    plt.title("Top1-Top5 fused gap per query")
    plt.xlabel("gap")
    plt.savefig(os.path.join(BASE, "plots", "score_spread.png"))
    plt.close()
    plt.figure()
    plt.scatter(lens, bms, s=12)
    plt.title("BM25 vs chunk length")
    plt.xlabel("chars")
    plt.ylabel("bm25")
    plt.savefig(os.path.join(BASE, "plots", "length_vs_bm25.png"))
    plt.close()
    plt.figure()
    plt.bar(["repetitive_hits", "other_hits"],
            [sum(rep_bm) / max(len(rep_bm), 1),
             sum(plain_bm) / max(len(plain_bm), 1)])
    plt.title("Mean BM25: repetitive vs other hits")
    plt.savefig(os.path.join(BASE, "plots", "repetition_vs_bm25.png"))
    plt.close()
    print("Generated: plots/score_spread.png plots/length_vs_bm25.png "
          "plots/repetition_vs_bm25.png")
    print("These are starting points — make your own diagrams and interpret them in PROOF.md.")
    _write_report_card(plt)


def _write_report_card(plt):
    """Performance report-card diagram from the fresh visible-bench metrics."""
    mp = os.path.join(BASE, "traces", "metrics.json")
    if not os.path.exists(mp):
        return
    m = json.load(open(mp, encoding="utf-8"))["agg"]
    tm = m.get("timing_ms", {})
    from rubbish_rag.identity import load_identity
    ident = load_identity() or {}
    who = ident.get("name", "?")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle(f"RubbishRAG report — {who} — visible bench "
                 f"acc={m.get('accuracy')}")
    ax[0].bar(["accuracy", "answer"], [m.get("accuracy", 0), m.get("answer_acc", 0)])
    ax[0].set_ylim(0, 1)
    ax[0].set_title("accuracy")
    ax[1].bar(["mean_ms", "p50_ms", "p95_ms"],
              [tm.get("mean", 0), tm.get("p50", 0), tm.get("p95", 0)])
    ax[1].set_title("latency per query (ms)")
    gaps = []
    try:
        from server._hidden_retriever import remote_retrieve
        vis = json.load(open(os.path.join(BASE, "server", "visible_bench.json"),
                             encoding="utf-8"))
        for item in vis[:20]:
            hits = remote_retrieve(item["query"], topk=5, api_key="report")
            if hits:
                gaps.append(hits[0]["fused"] - hits[-1]["fused"])
    except Exception:
        pass
    if gaps:
        ax[2].hist(gaps, bins=10)
    ax[2].set_title("top1-top5 fused gap")
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "plots", "report_card.png"))
    plt.close(fig)
    print("Generated: plots/report_card.png (your performance report card)")


def write_report_section():
    """(Re)write the README report card: identity + metrics + diagrams."""
    from rubbish_rag.identity import load_identity
    ident = load_identity() or {"name": "?", "email": "?", "github": "?"}
    mp = os.path.join(BASE, "traces", "metrics.json")
    metrics = json.load(open(mp, encoding="utf-8"))["agg"] if os.path.exists(mp) else {}
    tm = metrics.get("timing_ms", {})
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""{REPORT_START}
## 📊 My RubbishRAG Report

| card | |
|---|---|
| Candidate | {ident.get('name')} (@{ident.get('github')}) |
| Email | {ident.get('email')} |
| Date | {now} |

### Visible bench ({metrics.get('total', '?')} queries)

| metric | value |
|---|---|
| accuracy | {metrics.get('accuracy', '?')} |
| answer_acc | {metrics.get('answer_acc', '?')} |
| latency mean / p50 / p95 (ms) | {tm.get('mean', '?')} / {tm.get('p50', '?')} / {tm.get('p95', '?')} |

![report card](plots/report_card.png)
![score spread](plots/score_spread.png)
![length vs bm25](plots/length_vs_bm25.png)
![repetition vs bm25](plots/repetition_vs_bm25.png)

_Generated by `python3 rubbish.py bastesh`. Commit `plots/` + `README.md` so this renders on your fork._
{REPORT_END}"""
    rp = os.path.join(BASE, "README.md")
    text = open(rp, encoding="utf-8").read() if os.path.exists(rp) else ""
    pat = re.compile(re.escape(REPORT_START) + r".*?" + re.escape(REPORT_END),
                     re.S)
    if pat.search(text):
        text = pat.sub(body, text)
    else:
        text = text.rstrip() + "\n\n" + body + "\n"
    open(rp, "w", encoding="utf-8").write(text)
    print("README report card updated (commit plots/ + README.md).")


def cmd_bastesh(args):
    print(LOGO)
    write_report_section()
    out = os.path.join(BASE, "submissions", "submission.zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(os.path.join(BASE, "rubbish_rag")):
            for fn in files:
                if fn.endswith(".py"):
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.relpath(fp, BASE))
        for rel in ["traces/trace.jsonl", "traces/metrics.json",
                    ".candidate.json",
                    "plots/score_spread.png", "plots/length_vs_bm25.png",
                    "plots/repetition_vs_bm25.png", "plots/report_card.png",
                    "server/visible_bench.json"]:
            fp = os.path.join(BASE, rel)
            if os.path.exists(fp):
                z.write(fp, rel)
        proof = os.path.join(BASE, "docs", "PROOF.md")
        tmpl = os.path.join(BASE, "docs", "PROOF_template.md")
        if os.path.exists(proof):
            z.write(proof, "docs/PROOF.md")
        elif os.path.exists(tmpl):
            z.write(tmpl, "docs/PROOF.md")
            print("Warning: no docs/PROOF.md found — packed template instead. Write it!")
        mani = {"key": args.key, "quota_used": _quota(args.key)}
        z.writestr("manifest.json", json.dumps(mani, ensure_ascii=False))
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    print(f"Packed: {out}  sha={h}")
    print("Submit this file. Tampering is detectable.")


def cmd_orchestrate(args):
    print(LOGO)
    from rubbish_rag.orchestrator import get_orchestrator
    orch = get_orchestrator(api_key=args.key)
    res = orch.run(args.query, topk=args.topk)
    print("--- Orchestrator Result ---")
    print(json.dumps(res, ensure_ascii=False, indent=2))


def cmd_chat(args):
    print(LOGO)
    print("RubbishRAG Interactive Clarification Loop (Type 'exit' or 'q' to quit)")
    print("-" * 64)
    from rubbish_rag.orchestrator import get_orchestrator
    orch = get_orchestrator(api_key=args.key)
    while True:
        try:
            q = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not q or q.lower() in ("exit", "quit", "q"):
            break
        res = orch.run(q)
        if res.get("status") == "clarify":
            print(f"\n[⚠️  نیاز به شفاف‌سازی]: {res.get('question')}")
            opts = res.get("options", [])
            for idx, opt in enumerate(opts, 1):
                print(f"   {idx}) {opt}")
            try:
                choice = input("\nگزینه یا پاسخ شما > ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(opts):
                    selected = opts[int(choice) - 1]
                    res2 = orch.run(f"{q} ({selected})")
                else:
                    res2 = orch.run(f"{q} ({choice})")
                print(f"\n[✅ پاسخ نهایی]: {res2.get('brief')}")
                print(f"  ارجاعات (Cites): {res2.get('cites')}")
            except (EOFError, KeyboardInterrupt):
                break
        else:
            print(f"\n[✅ پاسخ]: {res.get('brief')}")
            print(f"  ارجاعات (Cites): {res.get('cites')}")


def cmd_ui(args):
    print(LOGO)
    print(f"Launching RubbishRAG Studio UI at http://localhost:{args.port} ...")
    try:
        import uvicorn
        uvicorn.run("server.app:app", host="0.0.0.0", port=args.port, reload=False)
    except ImportError:
        sys.exit("Error: uvicorn is required for web UI. Run `pip install -r requirements.txt`.")


def cmd_judge(args):
    print(LOGO)
    from rubbish_rag.judge import HeuristicFeatureJudge
    judge = HeuristicFeatureJudge()
    pre = judge.pre_retrieval(args.query)
    print("=== [System-1 Pre-Retrieval Decision] ===")
    print(json.dumps(pre.to_dict(), ensure_ascii=False, indent=2))
    if pre.needs_retrieval:
        from server._hidden_retriever import remote_retrieve
        hits = remote_retrieve(args.query, topk=pre.top_k, api_key=args.key)
        post = judge.post_retrieval(args.query, hits)
        print("\n=== [Post-Retrieval Evidence Assessment] ===")
        print(json.dumps(post.to_dict(), ensure_ascii=False, indent=2))


def cmd_judge_bench(args):
    print(LOGO)
    from rubbish_rag.judge import evaluate_judge
    print(f"Evaluating System-1 RAG Decision Judge (split={args.split or 'all'})...\n")
    metrics = evaluate_judge(split=args.split)
    print("=" * 55)
    print("      SYSTEM-1 RAG DECISION JUDGE BENCHMARK      ")
    print("=" * 55)
    print(f"Total Queries:         {metrics.get('total_queries')}")
    print(f"Routing Accuracy:      {metrics.get('routing_accuracy', 0) * 100:.1f}%")
    cla = metrics.get("clarification", {})
    print(f"Clarify Precision:     {cla.get('precision', 0) * 100:.1f}%")
    print(f"Clarify Recall:        {cla.get('recall', 0) * 100:.1f}%")
    print(f"Clarify F1:            {cla.get('f1', 0):.3f}")
    budget = metrics.get("context_budget", {})
    print(f"Mean Dynamic top-k:    {budget.get('mean_top_k')} (Baseline: 5)")
    print(f"Context Token Savings: {budget.get('token_savings_pct')}%")
    cal = metrics.get("calibration", {})
    print(f"Brier Score:           {cal.get('brier_score')}")
    print(f"Expected Calib Error:  {cal.get('expected_calibration_error')}")
    print("=" * 55)


def cmd_inspect_trace(args):
    print(LOGO)
    trace_file = os.path.join(BASE, "traces", "trace.jsonl")
    if not os.path.exists(trace_file):
        print("Error: traces/trace.jsonl not found.")
        return
    target = str(args.trace_id)
    found = []
    with open(trace_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if target in line or target == str(line_num):
                try:
                    found.append(json.loads(line))
                except Exception:
                    pass
    if not found:
        print(f"No trace found matching ID or query '{target}'.")
        return
    print(f"Found {len(found)} matching trace(s):")
    for t in found[:3]:
        print(json.dumps(t, ensure_ascii=False, indent=2))


def cmd_compare(args):
    print(LOGO)
    f1, f2 = args.file1, args.file2
    if not os.path.exists(f1) or not os.path.exists(f2):
        print(f"Error: One or both files not found: {f1}, {f2}")
        return
    try:
        m1 = json.load(open(f1, encoding="utf-8")).get("agg", {})
        m2 = json.load(open(f2, encoding="utf-8")).get("agg", {})
    except Exception as e:
        print(f"Error reading metrics JSON: {e}")
        return
    name1 = os.path.basename(f1)
    name2 = os.path.basename(f2)
    print(f"{'Metric':<25} | {name1:<20} | {name2:<20} | {'Delta':<10}")
    print("-" * 82)
    for k in ["accuracy", "answer_acc", "clarify_acc"]:
        v1 = float(m1.get(k, 0.0))
        v2 = float(m2.get(k, 0.0))
        delta = v2 - v1
        print(f"{k:<25} | {v1:<20.3f} | {v2:<20.3f} | {delta:+.3f}")
    tm1 = m1.get("timing_ms", {})
    tm2 = m2.get("timing_ms", {})
    print(f"{'latency_mean_ms':<25} | {str(tm1.get('mean', '?')):<20} | {str(tm2.get('mean', '?')):<20} | -")
    print(f"{'latency_p95_ms':<25} | {str(tm1.get('p95', '?')):<20} | {str(tm2.get('p95', '?')):<20} | -")


def main():
    ap = argparse.ArgumentParser(prog="rubbish")
    ap.add_argument("--key", default="demo-key")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("salam")
    p = sub.add_parser("bepar")
    p.add_argument("query")
    p.add_argument("--topk", type=int, default=5)
    sub.add_parser("bench")
    sub.add_parser("bekesh")
    sub.add_parser("rapchik")  # alias for bekesh
    sub.add_parser("bastesh")
    sub.add_parser("chat")

    p_orch = sub.add_parser("orchestrate")
    p_orch.add_argument("query")
    p_orch.add_argument("--topk", type=int, default=5)

    p_ui = sub.add_parser("ui")
    p_ui.add_argument("--port", type=int, default=8000)

    p_judge = sub.add_parser("judge", help="Run System-1 decision judge on query")
    p_judge.add_argument("query")

    p_jbench = sub.add_parser("judge-bench", help="Benchmark decision judge across dataset")
    p_jbench.add_argument("--split", default=None, choices=["train", "dev", "test"], help="Dataset split")

    p_trace = sub.add_parser("inspect-trace", help="Inspect trace entry by ID or query keyword")
    p_trace.add_argument("trace_id")

    p_comp = sub.add_parser("compare", help="Compare two metrics.json benchmark runs")
    p_comp.add_argument("file1")
    p_comp.add_argument("file2")

    args = ap.parse_args()
    _check_sealed_compat()
    ensure_identity()
    cmd_map = {
        "salam": cmd_salam,
        "bepar": cmd_bepar,
        "bench": cmd_bench,
        "bekesh": cmd_bekesh,
        "rapchik": cmd_bekesh,
        "bastesh": cmd_bastesh,
        "chat": cmd_chat,
        "orchestrate": cmd_orchestrate,
        "ui": cmd_ui,
        "judge": cmd_judge,
        "judge-bench": cmd_judge_bench,
        "inspect-trace": cmd_inspect_trace,
        "compare": cmd_compare,
    }
    cmd_map[args.cmd](args)


if __name__ == "__main__":
    main()

