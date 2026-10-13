# Engineering Evidence Report (PROOF.md)

---

## Candidate Information

- **Candidate Name:** `Dadbeh Tavanaei` 
- **Submission Date:** `2026-10-13`  
- **Execution Target:** `WSL (CPU)`
- **Track Scope Completed:**
  - [x] Required Core (Tasks A-D)
  - [x] Elective 1: `Track E - Evidence Reranking & Diversity (dedup)`
  - [x] Elective 2: `Track F - Adversarial & Noise Document Defense (corpus grounding)`
  - [ ] Staff Track: Adaptive System-1 Decision Judge

---

## Executive Summary & Scorecard

Summarize your quantitative improvements over the naive baseline. Run `python3 rubbish.py compare` to extract these numbers.

| Metric | Naive Baseline | Your Final Pipeline | Delta ($\Delta$) | Target |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | ~28.6% | `75.0%` | `+45.0%` | $\ge 65\%$ |
| **Retrieval Recall@5** | ~42.0% | `__._%` | `+__.__%` | $\ge 75\%$ |
| **Clarification Accuracy** | 0.0% | `0.0%` | `+0.0%` | $\ge 50\%$ |
| **Mean Query Latency (CPU)**| ~12 ms | `69.0 ms` | `+57.0 ms` | $< 100$ ms |
| **Distractor / Poison Ingestion**| ~35.0% | `0.0%` | `-35.0%` | $0.0\%$ |

---

## Investigation & Architectural Changes

### Investigation 1: Credit rank code and repeated tokens test

#### 1. Observed Symptom

Short questions containing something like `C1` or `C3` rank document **9004** as the first document despite a specific document for rank definition being present lower in the list. Its returned text especially calls itself a fake answer and repeats rank codes (poisoned). With more constrained wording, the annotated C1 definition (`doc_id=106`) comes up to Top-1.

#### 2. Falsifiable Hypothesis

Term overlap or repetition works in favor of the code-stuffed document; adding a discriminative cue such as `بازه` or `معنی` should increase the relative retrieval rank of the true grade-specific definition.

#### 3. Diagnostic Probes & Exact Trace IDs

The `ts` + exact `q` pairs are listed below; example equivalent reproduction: `python3 rubbish.py bepar "رتبه C1 یعنی چی؟"`. The repeated-`رتبه` requests are controlled token-frequency variants.

#### 4. Quantitative Evidence & Sub-score Analysis

| Trace `ts` | Exact query `q` | Top-1 doc | BM25₁ | Dense₁ | Fused₁ | Fused₅ | Δ₁,₅ | Gold doc @ rank in top 5 |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| `1791535078.2078938` | `رتبه C1 یعنی چی؟` | `9004` | 8.9152 | 0.2500 | 1.0000 | 0.6593 | 0.3407 | `106` @ 2 |
| `1791535078.3566923` | `رتبه C3 یعنی چی؟` | `9004` | 9.3108 | 0.2500 | 1.0000 | 0.6579 | 0.3421 | `108` @ 2 |
| `1791535078.4171314` | `رتبه اعتباری C1 چیست؟` | `9004` | 8.9152 | 0.2500 | 0.8307 | 0.6411 | 0.1896 | `106` @ 2 |
| `1791535078.7181458` | `رتبه رتبه رتبه C1 یعنی چی؟` | `9004` | 12.7285 | 0.2500 | 1.0000 | 0.6973 | 0.3027 | `106` @ 2 |
| `1791535078.7368183` | `رتبه رتبه رتبه رتبه رتبه رتبه C1 یعنی چی؟` | `9004` | 18.4484 | 0.2500 | 1.0000 | 0.7248 | 0.2752 | `106` @ 2 |
| `1791535078.9436598` | `بازه رتبه اعتباری C1 از چند تا چند است؟` | `106` | 13.4375 | 0.3913 | 0.9861 | 0.7960 | 0.1901 | `106` @ 1 |
| `1791535078.9970095` | `معنی رتبه اعتباری C1 چیست؟` | `106` | 10.6550 | 0.2828 | 0.9183 | 0.6393 | 0.2790 | `106` @ 1 |

The identical C1 question at `ts=1791535078.2078938` places **9004** ahead of gold doc **106** (at rank 2). Repeating `رتبه` changes the Top-1 **BM25** component from **8.9152 → 12.7285 → 18.4484**, while Top-1 remains **9004** and its fused score remains **1.0000**. In contrast, the explicit range and meaning formulations put **106** at rank 1.

#### 5. Root Cause Analysis

The remote fusion formula is not exposed in the provided source, so the relative contribution of sparse vs dense ranking cannot be causally identified from five candidates. The *visible* local baseline in `rubbish_rag/pipeline.py` also has a shared word count `rerank()` implementation, but these traces measure the remote retriever and **do not test** the effect of that local reranker. Document 9004's synthetic repetition is verifiable from returned hit text.

#### 6. Architectural Modification

Candidate change for the subsequent retrieval task: repetition-aware score normalization and code-aware reranking that rewards explicit C1/C2/C3 range evidence. Assess performance against untouched queries and against other letter or number grades.

#### 7. Ablation Experiment

**Run after Task A - see [R4 slot-aware ablation](#r4-slot-aware-reranking-task-c-wiring--task-e) and [R2 poison defence](#r2-poison-defence-track-f).** On the 32 rank/bank-slot queries the slot bonus lifts Hit@3 50.0% -> 53.1% (one query; small, non-negative). The repetition hypothesis was falsified as a *defence*: a repetition filter alone still leaves poison in 5.5% of queries' top-3 and misses all ten fact-corrupted documents (9100-9109).

#### 8. Production Trade-offs

Rule-based grade sensitivity is cheap but risks overfitting surface forms and suppressing genuinely relevant overview documents. Prefer reusable token/slot features and evaluate generalization to unseen grades; no overhead numbers have been measured.

---

### Investigation 2: Persian/Arabic script, digits, and half-space drift

#### 1. Observed Symptom

Near identical inputs with Persian vs Arabic `ک/ك`, ASCII/Persian/Arabic digits, or a half-space vs plain space change the ranked results. The Arabic digit version of the 250–900 query loses the annotated range document from the top five, while ASCII and Persian-digit versions retrieve it at Top-1.

#### 2. Falsifiable Hypothesis

Character inconsistencies reach the remote retriever as different tokens. Comparing normalized query variants against a fixed index should improve agreement in the rankings for matched meaning.

#### 3. Diagnostic Probes & Exact Trace IDs

`ts` + `q` are exact trace locators. Example equivalent replay: `python3 rubbish.py bepar "محدوده امتیاز اعتباری ٢٥٠ تا ٩٠٠"`. The three digit strings are ASCII `250`, Persian `۲۵۰`, and Arabic `٢٥٠`; these are distinct Unicode sequences.

#### 4. Quantitative Evidence & Sub-score Analysis

| Trace `ts` | Exact query `q` | Top-1 doc | BM25₁ | Dense₁ | Fused₁ | Fused₅ | Δ₁,₅ | Gold doc @ rank in top 5 |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| `1791535079.0218961` | `کسب و کارهای ایرانی` | `31` | 9.6182 | 0.2372 | 0.8137 | 0.6562 | 0.1575 | not annotated |
| `1791535079.0484178` | `كسب و كارهاي ايراني` | `21` | 0.6524 | 0.2236 | 0.9815 | 0.8536 | 0.1279 | not annotated |
| `1791535079.0666535` | `محدوده امتیاز اعتباری 250 تا 900` | `1` | 11.6347 | 0.3501 | 0.9740 | 0.7543 | 0.2197 | `1` @ 1 |
| `1791535079.076967` | `محدوده امتیاز اعتباری ۲۵۰ تا ۹۰۰` | `1` | 9.9571 | 0.3501 | 0.9926 | 0.7768 | 0.2158 | `1` @ 1 |
| `1791535079.100755` | `محدوده امتیاز اعتباری ٢٥٠ تا ٩٠٠` | `235` | 3.7542 | 0.2462 | 0.8627 | 0.8435 | 0.0192 | `1` absent |
| `1791535079.1323123` | `چک‌های برگشتی چه تأثیری دارند؟` | `3` | 9.4353 | 0.2697 | 1.0000 | 0.7462 | 0.2538 | not annotated |
| `1791535079.155105` | `چک های برگشتی چه تأثیری دارند؟` | `61` | 8.3725 | 0.3062 | 0.8818 | 0.7602 | 0.1216 | not annotated |

For ASCII and Persian digits, the **same Top-1 doc 1** appears with BM25₁ **11.6347** vs **9.9571**; the Arabic version ranks **235** first (BM25₁ **3.7542**) and does not return annotated doc **1** in its top five. The `كسب` variation changes the top candidate **31 → 21**, but those BM25 scores belong to different documents. The `چک‌های` / `چک های` pair also changes Top-1 document suggestion.

#### 5. Root Cause Analysis

`rubbish_rag/normalize_fa.py` includes `normalize_fa()` with Arabic-to-Persian letter mapping, numeral mapping, and ZWNJ normalization. However, the baseline `rubbish_rag/pipeline.py retrieve()` directly calls `remote_retrieve(query, topk=topk)` and does not apply this normalizer.

#### 6. Architectural Modification

Proposed for a subsequent task: use one tested canonicalization policy at query and index boundaries where controllable; preserve original query for display/citation, and test script variants side-by-side. An intervening query-only normalization layer would require a separate before/after measurement.

#### 7. Ablation Experiment

**Run after Task A - see [R3 normalization A/B](#r3-normalization-task-c).** Three controlled perturbations of the 330 labelled questions, real retriever, raw vs old vs new normalizer. Script drift costs 11.9 pts of Hit@3 raw and the normalizer recovers all of it; ZWNJ-as-space is *not* neutral (only 5/148 queries keep an identical top-5) and the new restore-the-ZWNJ design beats the old ZWNJ->space design (MRR 0.404 vs 0.374).

#### 8. Production Trade-offs

Normalization is lightweight, but irreversible folding can lose meaningful distinctions in identifiers, named entities, or quoted text.
---

### Investigation 3: Query specificity, length, and entity-sensitive retrieval

#### 1. Observed Symptom

Single word and multiclause questions with overlapping topics can retrieve different documents. The general query `بازه امتیاز اعتباری` returns annotated doc 1 first, yet longer requests specifying natural and legal persons rank doc 2 first and remove annotated doc 1 from the top five. Substituting `بانک ملت` with `بانک صادرات` leaves the same Top-1 document 302.

#### 2. Falsifiable Hypothesis

Additional intent terms and named entities change sparse term frequency/semantic matching, potentially favoring broad related texts over a particular annotated answer source. Controlling sentence length while preserving the requested entity and answer type should test whether the shift is caused by query length, topic, or slot handling.

#### 3. Diagnostic Probes & Exact Trace IDs

These exact `ts` + `q` records cover a short keyword, a specific intent phrase, longer questions, and an entity substitution. Example equivalent replay: `python3 rubbish.py bepar "بانک ملت چه شرایط اعتباری دارد؟"`.

#### 4. Quantitative Evidence & Sub-score Analysis

| Trace `ts` | Exact query `q` | Top-1 doc | BM25₁ | Dense₁ | Fused₁ | Fused₅ | Δ₁,₅ | Gold doc @ rank in top 5 |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| `1791535078.5224473` | `وام` | `9000` | 3.6211 | 0.3536 | 0.9832 | 0.8441 | 0.1391 | not annotated |
| `1791535078.528374` | `شرایط دریافت وام` | `76` | 9.5404 | 0.3873 | 1.0000 | 0.5543 | 0.4457 | not annotated |
| `1791535078.5586638` | `آیا رتبه اعتباری برای دریافت تسهیلات بانکی اهمیت دارد؟` | `79` | 13.4411 | 0.3651 | 1.0000 | 0.7448 | 0.2552 | not annotated |
| `1791535078.635057` | `بازه امتیاز اعتباری` | `1` | 4.2678 | 0.2970 | 0.8039 | 0.7842 | 0.0197 | `1` @ 1 |
| `1791535078.6626906` | `بازه امتیاز اعتباری اشخاص حقیقی و حقوقی چند است؟` | `2` | 13.1713 | 0.4507 | 0.8509 | 0.7801 | 0.0708 | `1` absent |
| `1791535078.675036` | `من می‌خواهم بدانم در نظام اعتبارسنجی، حداقل و حداکثر امتیازی که برای اشخاص حقیقی و شرکت‌ها تعریف شده چه عددی است و آیا محدوده آن‌ها یکسان است؟` | `2` | 19.5810 | 0.2760 | 0.9781 | 0.7982 | 0.1799 | `1` absent |
| `1791535079.1717474` | `بانک ملت چه شرایط اعتباری دارد؟` | `302` | 8.1667 | 0.2333 | 0.9041 | 0.7498 | 0.1543 | not annotated |
| `1791535079.1898155` | `بانک صادرات چه شرایط اعتباری دارد؟` | `302` | 8.1667 | 0.2333 | 0.8749 | 0.8128 | 0.0621 | not annotated |

The annotated target for the score-range family is doc 1. Top-1 for the short range query, absent from top five for both person and company formulations. Doc 2 may still contain valid range information.  The bank pair holds Top-1 **doc 302** across different names, but no bank-specific gold judgments exist, so it only focuses on a slot sensitivity test. The single word `وام` yields Top-1 **9000**, while `شرایط دریافت وام` yields **76**, consistent with the value of added terms.

#### 5. Root Cause Analysis

The retriever is queried with unprocessed text. In `rubbish_rag/pipeline.py`, `rerank()` uses set intersection of surface tokens, and `resolve()` calls `extract_slots_broken(query)` but does not use the extracted result to constrain evidence. These source-level facts suggest places to investigate.

#### 6. Architectural Modification

Candidate future change: identify the requested answer type (e.g., score interval) and entity (person, company, bank), then apply a rerank or resolve constraint or clarification only if source coverage is inadequate.

#### 7. Ablation Experiment

**Partially run - see [R4](#r4-slot-aware-reranking-task-c-wiring--task-e).** Rank-slot on/off was run (31 rank queries). Bank-swap on/off could not be evaluated: only 1 of 330 corpus questions names a bank, so no statistical statement about bank-aware reranking is made.

#### 8. Production Trade-offs

Slots can improve precision but may misparse informal Persian and introduce false rejections. A clarification policy costs a turn and should not be triggered solely by a small fused score margin.

---

### Investigation 4: Fused-score saturation, off-topic queries, and baseline spread

#### 1. Observed Symptom

Some queries get a Top-1 fused score of **1.0000** with a Top-5 score of **0.0000**, even for an out-of-domain pizza question and for an ambiguous personal score question whose annotated target is absent. Another benchmark query has an extremely narrow Top-1–Top-5 spread despite missing its annotated target. Therefore a large or small gap alone does not prove correctness.

#### 2. Falsifiable Hypothesis

`fused` is a retrieval score subject to scale and normalization effects. If this is the case, `fused=1` and high Top-1–Top-5 margins will occur for both correct and incorrect items.

#### 3. Diagnostic Probes & Exact Trace IDs

The following `ts` + `q` pairs provide extremes and counterexamples. Example equivalent replay: `python3 rubbish.py bepar "طرز تهیه پیتزای مارگاریتا چیست؟"`.

#### 4. Quantitative Evidence & Sub-score Analysis

| Trace `ts` | Exact query `q` | Top-1 doc | BM25₁ | Dense₁ | Fused₁ | Fused₅ | Δ₁,₅ | Gold doc @ rank in top 5 |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| `1791535078.9194608` | `اعتبار` | `80` | 6.7348 | 0.1826 | 1.0000 | 0.0000 | 1.0000 | not annotated |
| `1791535079.2013032` | `طرز تهیه پیتزای مارگاریتا چیست؟` | `296` | 5.0506 | 0.0682 | 1.0000 | 0.0000 | 1.0000 | not annotated |
| `1791535079.319131` | `۴۸۰ شدم. چرا؟` | `110` | 7.0553 | 0.0891 | 1.0000 | 0.0000 | 1.0000 | `311` absent |
| `1791535079.3365996` | `برای اشخاص حقوقی، رخدادهای عملیاتی مانند «تعطیلی موقت» یا «توقف تولید» در گزارش اعتباری بازتاب دارد؟` | `137` | 40.8364 | 0.4500 | 1.0000 | 0.3981 | 0.6019 | `137` @ 1 |
| `1791535079.5950427` | `اگر شرکت بدهی به تامین‌کنندگان داشته باشد، آیا این بدهی در گزارش اعتباری ثبت می‌شود؟` | `65` | 14.1636 | 0.3546 | 0.8879 | 0.8507 | 0.0372 | `123` absent |
| `1791535079.6486773` | `تغییر محل قانونی و آدرس شرکت چه تأثیری بر امتیاز اعتباری دارد؟` | `15` | 15.4989 | 0.3466 | 0.9502 | 0.8087 | 0.1415 | `36` @ 5 |

The off-topic pizza case (`ts=1791535079.2013032`) has **Top-1 fused 1.0000 / Top-5 fused 0.0000** and no annotated target. The ambiguous `۴۸۰` case (`ts=1791535079.3191314`) similarly has gap **1.0000** while target doc **311** is absent. The supplier-debt question (`ts=1791535079.5950427`) has gap **0.0372** with target **123** absent; the production and off day question (`ts=1791535079.3365996`) has gap **0.6019** with gold **137** at Top-1.

#### 5. Root Cause Analysis

Only five fused hits per request are exposed, and some fallback hits have `fused=0`. A score of 1.0000 could be produced by per query scaling or other server behavior. The presence of an off-topic high score proves that treating `fused` as an answer-confidence probability would be unsafe.

#### 6. Architectural Modification

Candidate subsequent change: separate retrieval scoring from an evidence-sufficiency decision based on relevance validation, coverage, entity constraints, and calibrated held-out examples. A fixed threshold on fused Top-1 or Δ₁,₅ alone is not justified by these traces.

#### 7. Ablation Experiment

Clarification was not attempted. One *measured* fact from the 330-query set bears on the hypothesis: the Top1-Top5 server `fused` gap is rarely saturated (0.3% of queries >= 0.99; 1.8% have zero gap; median 0.19, p10 0.07, p90 0.39), so min-max saturation is a property of off-topic probes, not of in-domain queries.

#### 8. Production Trade-offs


---


## Empirical Validation (Tasks B-F)

### R1. Hybrid fusion (Task B)

Pipeline change (`rubbish_rag/pipeline.py`): `retrieve` now normalizes the query, **over-fetches 20** (the old code asked the server for 5 and re-ordered those same 5, so it could never recover a gold doc the server had already dropped), drops ungrounded chunks, and fuses. RRF (`rrf_fuse`) and length-compensated rescoring (`length_compensated_fuse`: `bm25' = bm25 * len(text)^gamma`) are both implemented; `FUSION` selects one. `fused` is kept on a 0-1 scale and the server value is preserved as `server_fused`.

| Score (grounded pool=20) | TEST Hit@3 | TEST MRR | ALL dMRR vs server (95% CI) | ALL dHit@3 vs server (95% CI) |
|---|---|---|---|---|
| server order | 37.7% | 0.338 | - | - |
| **RRF, equal weight, k=60** | 34.7% | 0.302 | **-0.035** (-0.056, -0.016) | **-3.9 pts** (-6.4, -1.5) |
| length-comp gamma=0.5, w_dense=0.25 | 41.3% | 0.357 | +0.013 (-0.007, +0.033) | +1.5 pts (-1.5, +4.6) |
| **length-comp gamma=0.75, w_dense=0 (selected on dev)** | 38.9% | 0.350 | +0.017 (-0.012, +0.044) | +1.5 pts (-2.4, +5.4) |

**Findings (plots: `plots/length_vs_bm25.png`):**

1. **Equal-weight RRF is significantly *worse*, on dev, test and all.** Per query it worsens gold rank for 42 queries and improves it for 18.
2. **Why:** dense is the weaker ranker here (Hit@3 26-29% vs BM25 37-41%) and RRF gives it an equal vote. Dense also adds nothing once candidates are pooled (best dev w_dense = 0 at every gamma). Weighted RRF with small w_dense just converges to BM25-only.
3. **The "RRF prevents outlier compression" hypothesis is falsified.** Outlier strength = the share of the pool's min-max BM25 range owned by the gap between the top hit and the runner up (exactly the quantity that squeezes everything else under min-max). On the top quartile (gap >= 34% of the range, n=83) RRF@5 *still loses* (Hit@3 **54.2% -> 47.0%**, MRR 0.489 -> 0.425; on the grounded pool 54.2% -> 49.4%); on the other 247 queries it also loses (31.6% -> 29.6%). Server Hit@3 *rises* with outlier strength (28.8% -> 54.5% across quintiles): a big BM25 outlier is an exact lexical match and rank fusion discards that confidence (`plots/outlier_hypothesis.png`, `eval/outlier_robust.py`).
4. **What was real:** BM25 favours short chunks (b=0.9): among grounded candidates the gold chunk's BM25 rises with length and compensation helps 49 queries and hurts 25 (mean +0.43 ranks). The Hit@3 gain is **not significant**. Length-compensation is delivered as default because it is positive on both splits and the mechanism is observed, I also kept `FUSION="rrf"` as selectable.
5. Removing poison is the largest single effect on ranking: poison docs sat in the top-3 of 14.8% of queries, and the original pipeline did nothing about them (15.5%).

### R2. Poison defence (Track F)

Two poison families exist in the pool (`plots/poison_and_redundancy.png`): **keyword-stuffed spam** and **fact-corrupted copies**.

Defence (`rubbish_rag/grounding.py`, used in `retrieve`): a chunk is kept iff its text is a **contiguous substring of the corpus record**. No ID threshold and no constants: an id that does not index a corpus row has nothing to be grounded in. If the corpus file is missing the filter fails to open with a warning.

| defence | poison in top-3 (330 q) | poison ids still leaking |
|---|---|---|
| none (server order) | 14.8% | all 18 |
| original pipeline | 15.5% | all 18 |
| repetition filter only (threshold = max over verified corpus records, 0.613 - derived, not chosen) | 5.5% | 9000, 9005, 9007 and **all of 9100-9109** |
| **corpus grounding** | **0.0%** | none (567/567 real chunks kept: zero false demotions) |

Honest limits: grounding is only as trustworthy as `corpus/test.csv`. A poisoned corpus record would pass (the repetition score is exposed as an independent diagnostic, `repetition_score`, but it is not used for filtering because on this corpus it adds nothing beyond grounding).

### R3. Normalization (Task C)

Old `normalize_fa` was **not idempotent** (`میشه` -> `می‌شود` -> `می شود`), had dead colloquial keys (`رتبه‌م`, `سابقه‌م` contained a ZWNJ that an earlier step had already stripped), turned `۱٬۰۰۰٬۰۰۰` into three numbers downstream, and converted ZWNJ to space without evidence. New design (`rubbish_rag/normalize_fa.py`):

* `ي ى ك ة أ إ ٱ ۀ` folded. `ئ ؤ` untouched. Digits of all scripts -> ASCII. `٬`/`٫` -> `,`/`.` **only between digits**. The Persian comma is left alone.
* **ZWNJ is canonical, not removed.** The corpus uses ZWNJ ~3x more than space and the index is ZWNJ-sensitive, so typed spaces are *restored*: rule-based (`می‌`/`نمی‌` prefix, `‌ها/‌های/‌هایی` suffix) plus a 149-pair lexicon **derived from the corpus by majority vote** (`scripts/build_zwnj_lexicon.py`; ZWNJ form must win and occur >= 2 times) with spaced and joined keys, applied to a fixpoint (so idempotence holds by construction).
* Colloquial -> formal at the **token** level (no `\b`), ZWNJ-insensitive lookup, punctuation-safe.
* Verified: **0 / 6,630** idempotence failures (whole corpus + 2 random perturbations each); 1,662 / 1,662 corpus texts give an identical canonical form under Arabic-letter + Arabic-digit drift; 77% of ZWNJ-bearing records are recovered exactly after ZWNJ->space.

Real-retriever (`eval/norm_ab.py`, `plots/normalization_ab.png`; grounded pool, MRR / Hit@3):

| perturbation (n) | clean | raw | OLD normalizer | NEW normalizer |
|---|---|---|---|---|
| Arabic keys + Arabic digits (329) | 0.345 / 38.6% | 0.211 / 26.7% | 0.339 / 38.0% | **0.349 / 38.9%** |
| ZWNJ typed as space (148) | 0.388 / 44.6% | 0.370 / 43.2% | 0.374 / 43.2% | **0.404 / 46.6%** |
| colloquial speech (155) | 0.384 / 42.6% | 0.394 / 45.8% | 0.394 / 45.8% | 0.393 / 43.9% |

Script drift is the big effect (-12 pts Hit@3 raw) and both normalizers absorb it. The ZWNJ result is where the redesign pays off. **Colloquial mapping shows no measurable retrieval effect** in this synthetic probe (these tokens behave like stop-words here); it is kept because it is harmless and is needed by the slot parser (`نمیشه` -> negation), not because retrieval improved.

Slots (`rubbish_rag/slots.py`): token and phrase matching instead of substring, longest-match, multi-valued (`BANK_ALL/DEBT_ALL/RANK_ALL`, legacy scalars = first mention), ambiguous bank names require `بانک`, ranks `A1..E3` incl. glued/spaced/Persian-digit forms with a context rule for lower-case spaced `a 1`, numbers keep semantics (`AMOUNT_VALUE`: `۳۵۰ هزار` -> 350000; the `1` of `C1` is not a quantity), persona phrases/idioms (`شخصیت حقوقی`, `منحصر به فرد`) handled, `بازه` removed from TIME. **Tests: passed** `tests/test_normalization.py` and `tests/test_slots.py`.

### R4. Slot-aware reranking (Task C wiring + Task E)

`rerank_hits` = slot-alignment bonus -> **document/content dedup** -> optional MMR. Dev-split grid (`eval/tune2.py`):

| config (DEV, n=163) | Hit@3 | MRR | dup doc@5 | dup text@5 |
|---|---|---|---|---|
| no rerank | 41.1% | 0.357 | 3.9% | 6.0% |
| MMR lambda=0.7 only | 41.7% | 0.356 | 2.8% | 3.4% |
| **dedup only** | 42.3% | 0.362 | **0.0%** | **0.0%** |
| dedup + MMR lambda=0.9 | 42.3% | 0.362 | 0.0% | 0.0% |
| **dedup + slot bonus 0.5 (final)** | **42.9%** | **0.364** | 0.0% | 0.0% |

* **Dedup works best** (same `doc_id` *and* identical text under different ids - the corpus contains duplicate rows such as 91/92). **MMR never beat plain dedup** (lambda=1.0 tied or won every row). Redundancy in top-5 drops 4.7%/5.6% -> 0%; the Hit@3 effect is small (dedup +1.2 pts on dev and +1.2 pts on test over fusion-only; with the slot bonus +1.8 on dev, unchanged on test).
* **Slot bonus** on the 32 queries that carry a rank/bank slot (31 rank, 1 bank): Hit@3 50.0% -> 53.1%, MRR 0.485 -> 0.493, no query worse. That is **one query**; treat it as weak evidence of a small effect.
* `retrieve` returns >= topk hits tagged with `topk`; `rerank` trims to it, so CLI callers (and `bekesh`'s Top1-Top5 gap) still see exactly `topk` hits.

**Ladder (ALL 330 / TEST 167)** - each row adds one change (`plots/length_vs_bm25.png`, panel B):

| | ALL Hit@3 | ALL MRR | ALL poison@3 | TEST Hit@3 | TEST MRR |
|---|---|---|---|---|---|
| 0 server order | 37.3% | 0.327 | 14.8% | 37.1% | 0.323 |
| 1 original pipeline (RRF@5 + MMR) | 34.5% | 0.288 | 15.5% | 33.5% | 0.276 |
| 2 + grounding | 38.5% | 0.337 | 0.0% | 37.7% | 0.338 |
| 3 + length-comp fusion | 40.0% | 0.354 | 0.0% | 38.9% | 0.350 |
| 4 + dedup | 41.2% | 0.357 | 0.0% | 40.1% | 0.352 |
| 5 + slot bonus = **final** | **41.5%** | **0.358** | **0.0%** | **40.1%** | **0.352** |

### R5. Chunker (Task D)

`chunk()` is boundary-aware (sentence > clause > word breaks, `[Category | Keyword]` header on every chunk, number+unit kept together, 450 char cap).

**Limit that must be stated:** no code path feeds `chunk()` output to the remote server (retrieved chunks carry the server's own ids such as `91:1` and are capped at 300 chars even after the new chunker existed), so `chunk()` cannot move any retrieval metric here and none is claimed. It is validated by unit tests.

### Reproduction

```bash
python -m pytest tests -q                        # 123 passed
python rubbish.py bench                          # visible bench; metrics.json
cd eval && python3 final_eval.py                     # ladder (replays eval/cache_clean.json offline)
python3 tune.py && python3 tune2.py                  # dev-split grids   | python3 confirm_fusion.py  # test + bootstrap
python3 norm_ab.py && python3 make_plots.py          # normalization A/B and all figures (offline from eval/qcache.json)
python3 fetch_clean.py                               # re-fetch raw hits (needs the server; 330 calls)
python3 ../scripts/build_zwnj_lexicon.py             # regenerate the ZWNJ lexicon from the corpus
```
Raw `/retrieve` calls made by the evaluation harness are kept apart in `eval/traces_eval_calls.jsonl`, so `traces/trace.jsonl` still contains only the Task A probes (plus later `bench`/`bepar` runs). The pre-fix visible-bench result is preserved in `traces/metrics_original_pipeline.json`.

### Known limitations (not hidden)

* The gold set is derived from the corpus and overrepresents questions whose wording already resembles their own answer.
* Hit@3 gains over the raw server order are small and their CI touches 0. The clearest wins are (i) poison 14.8% -> 0%, (ii) not regressing on RRF (+7.0 pts vs the original pipeline), (iii) script-drift/ZWNJ robustness, (iv) redundancy -> 0%.

---

## Deliberate Non-Interventions

1. **Rejected: RRF as the default fusion.** Measured significantly worse than the server order on test because the dense ranker is weaker and RRF discards BM25 magnitude. Kept as `FUSION="rrf"` for comparison.
2. **Rejected: MMR as the default diversifier.** Never beat plain dedup (`MMR_LAMBDA=1.0` (off)). Dedup alone removes all top-5 redundancy.
3. **Rejected: a repetition penalty as the poison defence.** Leaves 5.5% of queries poisoned and cannot see the ten corrupted documents. Corpus grounding is exact and had zero false positive instances.
4. **Rejected: ZWNJ -> space normalization** (the previous behaviour). Measured worse than restoring the ZWNJ (MRR 0.374 vs 0.404 on space-typed queries).

---

## Bonus Track: Adaptive System-1 Decision Judge *(Optional)*

*If completing the Staff Track, summarize your judge implementation here.*

- **Judge Architecture:** Heuristic Rule-Based / Open-Weight SLM (Llama-3.2 / Qwen)
- **Routing Accuracy on `data/judge_dataset.json`:** `__._%`
- **Clarification F1:** `_.___`
- **Token / Latency Savings:** `__._%`
- **Probability Calibration:**
  - Brier Score: `0.0___`
  - Expected Calibration Error (ECE): `0.0___`
- **Key Insight:** *Explain how your dynamic stopping policy decides when retrieved evidence is sufficient to answer.*

---

## Reproduction Checklist

Verify that the evaluation committee can reproduce your numbers cleanly:
- [x] `pytest tests/` passes cleanly (123 passed on Python 3.11; 117 on 3.12 with the 6 sealed-server CLI tests deselected).
- [x] `python3 rubbish.py bench` finishes in under 3 minutes on CPU (~0.5 s for 20 queries).
- [x] `python3 rubbish.py plot` generates all diagnostic figures in `plots/`.
- [x] `python3 rubbish.py bastesh` packages `submissions/submission.zip` with zero errors.