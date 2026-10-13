# 🗑️ RubbishRAG — "AI made it in 1 minute!" …prove it wrong

```text
  ╔═════════════════════════════════════════════════════════════════╗
  ║    ____        _     _     _     _     ____      _    ____      ║
  ║   |  _ \ _   _| |__ | |__ (_)___| |__ |  _ \    / \  / ___|     ║
  ║   | |_) | | | | '_ \| '_ \| / __| '_ \| |_) |  / _ \| |  _      ║
  ║   |  _ <| |_| | |_) | |_) | \__ \ | | |  _ <  / ___ \ |_| |     ║
  ║   |_| \_\\__,_|_.__/|_.__/|_|___/_| |_|_| \_\/_/   \_\____|     ║
  ║                                                                 ║
  ║       "AI made it in 1 minute!" ... prove it wrong              ║
  ╚═════════════════════════════════════════════════════════════════╝
```

> **RubbishRAG** is a senior-level take-home engineering challenge centered on a retrieval-augmented generation (RAG) system for Persian credit-scoring regulations ([`corpus/test.csv`](corpus/test.csv), 330 verified QA pairs).
> 
> The story: an internal team claimed they built a production-ready Persian RAG in 1 minute using off-the-shelf AI. On the surface, the pipeline runs and returns answers. Under rigorous evaluation, **it achieves only ~25–30% accuracy on the visible benchmark**, suffers from rank distortions, falls for distractor documents, and hallucinates answers on ambiguous inputs.
>
> Your mission is to systematically probe the system, diagnose failure modes with empirical telemetry, implement principled engineering solutions in candidate-editable code, and defend your architectural choices with logged traces and ablation data.

📖 **Comprehensive Documentation:** Hosted on [GitHub Pages](https://alinikkhah2001.github.io/RubbishRAG_InterviewTakeHome/) (or browse locally in [`docs/index.html`](docs/index.html)).

---

## 1. System Architecture & Information Boundaries

```text
  User Query (Persian Credit Scoring)
                 │
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ CANDIDATE-EDITABLE PIPELINE (rubbish_rag/)                      │
  │                                                                 │
  │  [Query Normalizer]       Canonicalize script, digits, ZWNJ     │
  │  [Entity & Slot Guard]    Detect banks, rank codes (A1..E3)     │
  │  [Decision Judge]         Pre-retrieval route: Answer vs Clarify│
  │                                                                 │
  │  ┌──────────────┐         Calls sealed retrieval endpoint       │
  │  │ @retriever   │ ──────────────────────────────────────────┐   │
  │  └──────────────┘                                           │   │
  │  ┌──────────────┐                                           │   │
  │  │ @reranker    │ ◀─── Reranks candidates, filters traps    │   │
  │  └──────────────┘                                           │   │
  │  ┌──────────────┐                                           │   │
  │  │ @resolver    │ ───▶ Structured answer or clarification   │   │
  │  └──────────────┘                                           │   │
  └─────────────────────────────────────────────────────────────┼───┘
                                                                │
                                    Quota: 1000 calls/key       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ SEALED SERVER INTERFACE (server/)                               │
  │   POST /retrieve                                                │
  │   - Computes BM25, Dense similarity, and raw fused scores       │
  │   - Returns candidate hits with scores and document metadata    │
  │   - Governed by the Honor Code (Inspect via API probes only)    │
  └─────────────────────────────────────────────────────────────────┘
```

### Information Boundaries & Honor Code
- **Candidate-Editable (`rubbish_rag/`):** You have full ownership of `pipeline.py`, `normalize_fa.py`, `slots.py`, `judge.py`, and `orchestrator.py`. Write clean, typed, modular Python.
- **Sealed Evaluator (`server/`):** The server models a proprietary backend service. Do not inspect compiled bytecode or reverse-engineer private files. Probe the API via `python3 rubbish.py bepar` or HTTP requests, observe returned scores, and log empirical traces.
- **Ground Truth Corpus (`corpus/test.csv`):** 330 verified QA records from the Iranian Credit Scoring regulations.
- **Reproducibility:** The entire challenge runs **100% locally and offline on CPU** in `< 15ms` per query. No GPUs, external API keys, or paid cloud services are required.

---

## 2. 5-Minute Quickstart

### Prerequisites
- **Python 3.11.x** (Required for sealed server runtime compatibility)
- Standard developer environment (`macOS`, `Linux`, or `WSL`)

```bash
# 1) Clone your fork and set up virtual environment
git clone https://github.com/<YOUR-USERNAME>/RubbishRAG_InterviewTakeHome.git
cd RubbishRAG_InterviewTakeHome
python3 -m venv venv
source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Verify installation & run test suite
pytest tests/
python3 rubbish.py salam
```

### Core CLI Workflow
```bash
# Probe a single query and inspect raw retrieval sub-scores
python3 rubbish.py bepar "رتبه اعتباری B2 چیست؟" --topk 5

# Evaluate the visible benchmark (20 diverse queries)
python3 rubbish.py bench

# Compare baseline against your improved pipeline
python3 rubbish.py compare

# Inspect a specific execution trace by ID or keyword
python3 rubbish.py inspect-trace trace_20261007_120000_0

# Generate diagnostic evaluation plots
python3 rubbish.py bekesh

# (Staff Track) Benchmark the Adaptive System-1 Decision Judge
python3 rubbish.py judge-bench

# Package your submission (stamps README scorecard + builds submission.zip)
python3 rubbish.py bastesh
```

---

## 3. 3-Tier Task Architecture & Expected Scope

To respect your time and provide clear boundaries, the challenge is structured into three tiers. Choose your path based on your role focus and target level:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. REQUIRED CORE (~3–4 Hours) — Every Candidate Completes              │
│    Task 1: Scientific Diagnosis & Evidence (PROOF.md + Traces)         │
│    Task 2: Persian Normalization & Robust Retrieval                    │
│    Task 3: Faithful Answer Resolution & Clarification Guardrails       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. CHOOSE-2 ELECTIVES (~2 Hours) — Select Exactly Two                  │
│    Elective A: Entity-Aware Slot Reranking (Bank & Rank Sensitivity)   │
│    Elective B: Defensive Filtering Against Distractors & Poison Docs   │
│    Elective C: Observability & Diagnostics Tooling (CLI & Telemetry)   │
│    Elective D: LangChain LCEL Pipeline Orchestration                   │
│    Elective E: Interactive Web Studio & Latency Profiling              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. STAFF TRACK BONUS (+20% Bonus) — Optional Senior / Staff Extension │
│    Adaptive System-1 Decision Judge (Self-RAG / CRAG / Adaptive-RAG)   │
│    Pre-retrieval routing, dynamic-k stopping policy, calibration (ECE) │
└────────────────────────────────────────────────────────────────────────┘
```

See [`docs/tasks.html`](docs/tasks.html) for detailed task specifications and acceptance criteria.

---

## 4. Diagnostic Probing & Observable Symptoms

Rather than reading internal server source code, practice scientific AI engineering: formulate hypotheses and design targeted probes. Look for these observable symptoms:

1. **Persian Script Drift:** Queries containing Arabic characters (`ي`, `ك`), Persian/Arabic digits (`۱۲۳`), or missing Zero-Width Non-Joiners (ZWNJ) yield 0 BM25 term matches on relevant documents.
2. **Entity Inversion:** Inverting bank names (e.g. *بانک صادرات* vs *بانک ملی*) or rank grades (*A1* vs *E3*) fails to demote irrelevant hits due to unweighted bag-of-words scoring.
3. **Score Scale Distortion:** Outlier scores in sparse or dense retrievers distort naive linear combinations (`0.5 * bm25 + 0.5 * dense`), crowding out truly relevant hits.
4. **Length Bias:** Sparse scoring severely penalizes longer, informative credit scoring clauses in favor of short, terse fragments.
5. **Adversarial Distractors:** Keyword-stuffed or synthetic distractor documents outrank legitimate regulatory documents.
6. **Hallucination on Ambiguity:** Underspecified queries (e.g., *«بدهی دارم»* — loan debt vs tax lien vs bounced check) receive confident, incorrect answers instead of asking clarifying questions.

---

## 5. Staff Track: Adaptive System-1 Decision Judge

For candidates targeting Senior and Staff AI Engineer roles, RubbishRAG includes a major architecture track inspired by **Self-RAG**, **Adaptive-RAG**, and **Corrective RAG (CRAG)**:

- **Module:** [`rubbish_rag/judge.py`](rubbish_rag/judge.py)
- **Concept:** A fast, low-latency "System-1" decision model that evaluates queries and retrieved evidence before and after retrieval.
- **Capabilities:**
  - **Pre-retrieval routing:** Decides whether retrieval is necessary (`DIRECT_ANSWER`), whether clarification is mandatory (`CLARIFY`), or whether retrieval is required (`RETRIEVE`).
  - **Dynamic $k$ selection:** Adapts retrieval depth based on query complexity.
  - **Evidence sufficiency evaluation:** Decides whether retrieved chunks are sufficient to answer, require query reformulation, or warrant abstention.
  - **Probability Calibration:** Computes Brier Score and Expected Calibration Error (ECE) to ensure confidence scores are reliable.
  - **Evaluation Dataset:** 50 diverse Persian credit queries in [`data/judge_dataset.json`](data/judge_dataset.json).

```bash
# Run judge on a single query
python3 rubbish.py judge "رتبه C2 چه شرایطی دارد؟"

# Benchmark routing accuracy, calibration, and token savings
python3 rubbish.py judge-bench
```

Read the full architecture and state machine specifications in [`docs/adaptive-judge.html`](docs/adaptive-judge.html).

---

## 6. Deliverables & Evaluation Rubric

Your final submission must include:
1. **Pipeline Implementation:** Clean, modular overrides in `rubbish_rag/`.
2. **Evidence Report (`docs/PROOF.md`):** Complete report following [`docs/PROOF_template.md`](docs/PROOF_template.md), linking each change to exact trace IDs and ablation figures.
3. **Signed Archive (`submissions/submission.zip`):** Generated by `python3 rubbish.py bastesh` with automated SHA256 manifest.

### 7-Dimension Grading Rubric (100 Points + 20 Bonus)
- **1. Problem Diagnosis & Scientific Methodology (20%):** Hypothesis formulation, disciplined probing, trace-backed reasoning.
- **2. Retrieval Precision & IR Correctness (20%):** Persian normalization, entity sensitivity, score fusion, distractor resilience.
- **3. Answer Quality & Guardrails (15%):** Grounded brief answers, structured clarification on ambiguity, defensive abstention.
- **4. Software Architecture & Code Quality (15%):** Modular design, typing, testability, latency `< 100ms` on CPU.
- **5. Observability & Diagnostics (10%):** Trace structure, sub-score logging, inspection CLI tools.
- **6. Testing Rigor & Reproducibility (10%):** Unit tests in `tests/`, deterministic execution, clean CI.
- **7. Documentation & Communication (10%):** Crisp `docs/PROOF.md`, clear trade-off explanations.
- **Bonus Tracks (+20%):** Adaptive System-1 Judge, LangChain LCEL orchestrator, or Web Studio playground.

---

## 7. Repository Layout & Multi-Page Documentation

```text
corpus/test.csv               330 Persian Credit-Scoring QA Records
data/judge_dataset.json       50 Annotated Queries for Adaptive Decision Judge
rubbish_rag/                  Candidate Codebase
├── pipeline.py               Decorator-based pipeline stages (@chunker, @retriever, etc.)
├── normalize_fa.py           Persian text & digit canonicalization
├── slots.py                  Domain entity & slot extraction (banks, ranks, debts)
├── judge.py                  Adaptive System-1 Decision Judge & calibration
└── orchestrator.py           LangChain LCEL-compatible production pipeline
rubbish.py                    Candidate CLI toolkit (11 subcommands)
server/                       Sealed Evaluator & Retrieval Server (Honor Code)
tests/                        Automated test suite (pytest)
docs/                         Multi-Page Documentation Suite
├── index.html                Overview & Challenge Story
├── tasks.html                3-Tier Task Architecture & Specifications
├── architecture.html         Dual Pipeline Architecture & Scoring Math
├── adaptive-judge.html       Staff Track: Adaptive System-1 Decision Judge
├── guide.html                Engineering Guide: Scientific Probing & Traces
├── deliverables.html         Deliverables Checklist & 7-Dimension Rubric
├── glossary.html             Metrics Glossary (IR, Calibration, Latency)
├── ui.html                   Interactive Web Studio & Orchestrator Guide
├── CHALLENGE_AUDIT.md        Comprehensive Pre-Edit Challenge Audit
├── CHALLENGE_CHANGELOG.md    Challenge Evolution & Decision Log
└── PROOF_template.md         Hypothesis-Driven Evidence Report Template
```

---

<!-- RUBBISH-REPORT:START -->
## 📊 My RubbishRAG Report

| card | |
|---|---|
| Candidate | Dadbeh Tavanaei (@dadbeh123) |
| Email | dadbeh123@gmail.com |
| Date | 2026-10-13 12:05 UTC |

### Visible bench (20 queries)

| metric | value |
|---|---|
| accuracy | 0.75 |
| answer_acc | 0.75 |
| latency mean / p50 / p95 (ms) | 65.8 / 58.9 / 107.2 |

![report card](plots/report_card.png)
![score spread](plots/score_spread.png)
![length vs bm25](plots/length_vs_bm25.png)
![repetition vs bm25](plots/repetition_vs_bm25.png)

_Generated by `python3 rubbish.py bastesh`. Commit `plots/` + `README.md` so this renders on your fork._
<!-- RUBBISH-REPORT:END -->
