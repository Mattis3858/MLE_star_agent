# REFACTOR_PLAN.md

Master specification for refactoring `mle_star_agent.py` into an **autonomous MLE system** that reliably converges on the *best generalizing* model for the Rossmann forecasting task (and is structured to generalize to other tabular tasks later).

This supersedes `REFACTOR.md`. It keeps the original four phases (corrected) and adds the search-intelligence and autonomy pieces that the original plan was missing.

---

## 0. Framing: what "best model" actually requires

The original prototype has two structural weaknesses that no amount of prompt-tuning fixes:

1. **Dishonest metrics.** The Coder Agent writes the splitting *and* the scoring, so the reported MAPE can be leaked/overfit (the suspicious <1%). A great number that doesn't hold on unseen data is worse than useless — it poisons every downstream decision the Planner makes.
2. **Myopic search.** A single greedy `best_code` thread with one LLM-guessed strategy per iteration is hill-climbing on a noisy surface. It finds a local optimum and plateaus, and it never ensembles, never ablates, never tunes numerics properly.

So this plan is ordered: **Part A makes the number honest, Part B makes the search find better models, Part C/D make it run unattended and observable.** Do A before B — optimizing against a leaked metric is optimizing noise.

### Recommended implementation order
1. A1 — Leakage-safe harness contract
2. A2 — Held-out final test set (anti adaptive-overfitting)
3. A3 — Determinism + minimum-improvement threshold
4. A4 — Harness owns the metric (kill the regex)
5. B1 — Data-profiling / EDA agent
6. C1 — Dedicated debug/self-repair loop
7. B4 — Delegate numeric HPO to Optuna
8. B5 — Ensembling top-K
9. B2 / B3 — Solution-tree search + ablation-driven refinement
10. Everything else (C/D) as hardening

---

# PART A — Correctness & Honest Evaluation (foundation, do first)

## A1. Leakage-safe harness contract (corrected Phase 1)

### Why the original Phase 1 contract is still broken
The proposed contract —
```python
def engineering_and_train(train_df) -> xgb.XGBRegressor:
    ...
    return trained_model
```
— does **not** close the leak. The harness must score on a validation slice, which means it has to call `model.predict(val_features)`. But the feature engineering lives inside the LLM's function and only ran on `train_df`. The harness has no way to reproduce the same features on `val_df`, so:
- **train/val feature parity is unguaranteed**, and
- any *fittable* feature the LLM adds (target/mean encoding, store-level aggregates, embeddings) re-introduces leakage from inside the function.

Worse, **time-series lag/rolling features for the validation rows need to look back into the training history** — you physically cannot compute them by handing the function an isolated `val_df`.

### The fix: split responsibility by leakage risk, not by "one big function"
- **Harness owns all high-risk / temporal features.** Lag, rolling windows, expanding stats, target/mean encoding — anything that reads other rows — is computed *once*, correctly (row *t* only uses timestamps `< t`), by the harness, **before** the split. This is exactly where leakage normally happens, so take it out of the LLM's hands entirely.
- **LLM owns low-risk, deterministic transforms + model choice + hyperparams.** Date parts, promo interaction terms, `store.csv` joins, log-transform of the target, which estimator, which params.
- **Contract = a `Pipeline`, not a fitted model.** Have the LLM return an *unfitted* sklearn-compatible `Pipeline` (its FE transformers + estimator). The harness does `pipe.fit(train_slice)` then `pipe.predict(val_slice)` / `pipe.predict(test_slice)`. Any transformer that needs fitting is therefore fit on train only, by construction.

```python
# Agent contract (LLM fills the body; harness controls fit/predict/score)
def build_pipeline(feature_columns: list[str]) -> sklearn.pipeline.Pipeline:
    """Return an UNFITTED pipeline. No file IO, no splitting, no scoring."""
    ...
```

The harness, not the LLM, performs the chronological split and the scoring. This single change is what actually kills the <1% MAPE.

## A2. Held-out final test set — guard against adaptive overfitting (NEW, critical)

Even with A1, you run ~20 iterations where the Planner repeatedly reads the **same** validation MAPE to choose the next move. That is 20 rounds of model selection against one holdout = optimistic bias by multiple comparisons. The final reported `best_mape` will be too good.

**Fix:** carve three chronological slices, not two.
```
[ ........... train ........... | validation (selection) | test (report-only) ]
```
- The **validation** slice drives every Planner decision and `best_*` update.
- The **test** slice (e.g. final 6 weeks) is touched **exactly once**, after the whole agent run finishes, on the chosen `best_code`. The Planner never sees it.
- Report both numbers. A large val/test gap is itself a signal the search overfit the validation set.

## A3. Determinism + minimum-improvement threshold (NEW)

The whole loop assumes "MAPE went down ⇒ the strategy worked." But the code sets no seeds, so a 0.05% move is likely run-to-run noise, and the Planner will chase ghosts.

- Harness force-seeds everything: `random`, `numpy`, `PYTHONHASHSEED`, and the estimator (`random_state`, plus deterministic GPU flags where available).
- Establish a **noise floor**: run the same config 3–5 times once at startup, take the MAPE std.
- Only count an improvement as real if it beats `best_mape` by `> max(min_delta, k * noise_std)`. Otherwise treat as no-improvement (feeds `no_improve_rounds`). This stops the patience mechanism from being fooled by jitter.

## A4. Harness owns the metric — delete the regex (benefit of A1)

Once the harness computes MAPE itself, **remove** the fragile
`re.search(r"(FINAL_MAPE|...|MAPE)...", output)` stdout parsing. It can mis-match an intermediate printed MAPE and is a latent bug. The score becomes a return value from the harness, not text scraped from a subprocess.

## A5. Align with the real task (NEW, Rossmann-specific)

- The Rossmann competition's official metric is **RMSPE**, not MAPE, and the test period is a fixed **future ~6 weeks**. If the deliverable is meant to mean something, make the metric **pluggable** and default validation/test horizons to match the forecast horizon. Keeping MAPE is fine as long as it's a deliberate choice, not an accident.
- Keep the mandatory `Sales>0 & Open==1` filter and the `StateHoliday.astype(str)` fix from the original — but move them into the harness's data-loading step so the LLM can't drop them.

---

# PART B — Search Intelligence (this is what finds the *best* model)

## B1. Data-profiling / EDA agent (NEW — highest-leverage addition)

Right now the system **never looks at the data**. The design spec is generated from generic Rossmann lore and a paper. A truly autonomous system profiles the dataset first and conditions every later decision on real facts.

Add a node *before* design that produces a structured `data_profile`:
- per-column dtype, missingness, cardinality, min/max/skew;
- target distribution (justifies log-transform), zero-inflation;
- date coverage and gaps (informs the split + CV);
- candidate leakage columns (e.g. anything correlated with the target that wouldn't exist at prediction time);
- correlations / mutual information with the target.

Feed `data_profile` into the Research/Design and Planner prompts. This is cheap, deterministic (no LLM needed for the stats themselves — compute with pandas, then have the LLM *interpret*), and it lifts every downstream choice from "guess" to "informed."

## B2. From greedy hill-climb to a solution tree (NEW)

Replace the single `best_code` thread with a **solution tree / beam**:
- Persist every evaluated candidate as a node `{code, val_score, test_score?, parent_id, telemetry}`.
- Each iteration, the Planner may **branch from any promising past node**, not only the latest best — this escapes local minima that greedy single-threading gets stuck in.
- Keep a **beam of width K** (e.g. 3) as the active frontier; expand the most promising nodes (best score, or most uncertain).
- Add occasional **random restarts** (a fresh approach from scratch) to maintain exploration.
- **Prune-and-rollback + blacklist (from the 2nd-pass "MCTS" idea).** When a branch produces a run of consecutive non-improving steps, prune it: roll the active frontier back to the last global-best node, and record the failed *strategy* (not just the node) on a blacklist so the Planner doesn't re-propose the same dead end. Keep the blacklist in the experiment store (D3) and feed it into the Planner prompt as "already-tried, didn't work."

This is the structural change that most affects final model quality, because the search is no longer one-dimensional. (The second design pass called this "Tree-of-Thought / MCTS branching" — same intent; the node store in D3 plus the beam here *is* that mechanism, so don't build a separate parallel tree structure.)

## B3. Ablation-driven targeted refinement (NEW — the actual MLE-STAR idea)

MLE-STAR's core contribution is *targeted* refinement: it runs **ablation studies** to identify which pipeline component most affects the score, then concentrates edits there — rather than letting the Planner guess "the weakest component." Implement a lightweight version:
- Periodically, the harness ablates components of the current best pipeline (drop a feature group, swap the model, remove the log-transform) and measures the score delta.
- Feed the ablation deltas to the Planner as evidence: "removing lag features costs +3.1 MAPE; removing promo interactions costs +0.1." Now the Planner targets the high-impact block instead of guessing.

## B4. Delegate numeric HPO to Optuna; LLM owns structure (NEW)

LLMs are poor at numeric tuning (learning rate, `num_leaves`, regularization). Split the search:
- **LLM's job:** features, model family, pipeline structure, transforms — the *discrete/architectural* search.
- **Optuna's job (inside one evaluation):** the *continuous/numeric* search over the current architecture's hyperparameters, with a small trial budget and TPE/pruning.

A single "iteration" then = LLM proposes a structure → Optuna tunes it → harness scores the tuned best. This converges far faster and more reliably than asking the LLM to nudge numbers.

## B5. Ensembling top-K candidates (NEW — usually the single biggest win)

A best-model system should not return one pipeline. After the search, take the top-K diverse candidates from the solution tree and ensemble them (weighted average / stacking with a held-out blend). On Kaggle-style tabular tasks this is routinely the largest single improvement, and MLE-STAR has a dedicated ensembling stage. Make this a final node before the report, and report the ensemble's **test** score.

## B6. Proper time-series cross-validation (NEW)

A single holdout is high-variance, so the Planner's signal is noisy (compounds A3). Offer an **expanding-window time-series CV** (e.g. 3 folds, each validating on the next block) as the scoring mode. More compute, but a much more stable improvement signal — and it matches how the model will actually be used (forecasting forward). Keep single-holdout as a fast mode for early iterations, switch to CV for finalists.

## B7. Code-grounded retrieval / Hybrid RAG (NEW — adopted from 2nd-pass Phase 5)

The current research phase only ingests academic paper *text* (arxiv) and Tavily summaries. For a tabular task, the highest-value external knowledge is **actual feature-engineering code** — the pandas blocks Kaggle winners actually used (promo windows, holiday distances, store-level aggregates). Add a code-search retrieval step to the research/EDA stage:
- Query a GitHub/Kaggle code-search API for top Rossmann solutions and extract concrete FE snippets, not prose.
- Surface these to the Foundation Coder and Planner as candidate features to implement.

This pairs naturally with B1 (EDA): the profiler says *what the data looks like*, code-grounded RAG says *what worked on this exact dataset before*. **Security caveat (see C4):** retrieved code is untrusted — it must go through the same AST gate and sandbox as LLM-generated code, and must never be executed verbatim outside the harness. Treat it as *inspiration for features*, not a script to run.

---

# PART C — Robustness & Autonomy (run unattended)

## C1. Dedicated debug / self-repair sub-loop (NEW)

Currently a failure just reverts or asks the Planner to "fix it" — mixing *debugging* with *improving*. Separate them:
- On execution error, enter a **debug loop**: pass the parsed traceback (exception type + file + line, not the whole log) to a Debugger role, apply a fix, AST-check, re-run — up to `N` attempts.
- **Error-driven retrieval (adopted from 2nd-pass Phase 5).** Make Tavily/Arxiv (and the B7 code search) callable *inside* the debug loop, not just at startup. On a `Failed` status, inject: "execution failed with this traceback; you may search the error message before proposing a fix." This turns opaque library errors (version mismatches, CUDA/LightGBM `device='gpu'` build issues, dtype errors) into solvable ones. Gate it so it only fires on failure, to avoid burning tokens on every iteration.
- Only after a candidate *runs successfully* does control return to the Planner for *improvement*. This stops broken code from consuming improvement iterations and makes the system far more resilient unattended.

## C2. Dynamic termination & patience (corrected Phase 2)

Original Phase 2 is mostly right. Refinements:
- Keep **patience** (`no_improve_rounds >= P` ⇒ stop) and the **hard ceiling** (`MAX_ITERATIONS`) — these are the trustworthy stop conditions.
- Treat the LLM's self-reported `is_finished` as an **advisory signal only**. It's the least reliable (the model may always return `false`, or hallucinate completion). Never let it *extend* the run beyond the ceiling.
- "No improvement" must use the A3 min-delta threshold, not exact equality.

## C3. Checkpointing & resumability (NEW)

The entire run lives in memory; a crash at iteration 18 loses everything. Persist `AgentState` (or the solution tree + best pointers) to disk after each iteration and support `--resume`. Essential for long unattended runs.

## C4. Real sandboxing & resource enforcement (corrected guardrails)

**Important correction:** `ast.parse()` only catches syntax errors — it provides **zero** security isolation. LLM-generated code still runs via `subprocess` with full filesystem and network access. If "secure / industrial-grade" is a real goal:
- Enforce limits, don't just measure them: `resource.setrlimit` for memory/CPU, a hard `subprocess` timeout (already present — keep it).
- Disable network and restrict writable paths for the child process.
- For real isolation, run candidates in a container (Docker / nsjail / firejail).

Keep the AST check — but as a fast "fail before subprocess" gate, not as a security control.

**Extra surface from B7:** once you fetch code from GitHub/Kaggle (Hybrid RAG), you're introducing *untrusted external code* into the system. This makes the sandbox non-optional rather than nice-to-have — retrieved snippets must pass the same AST gate and run only inside the isolated child process.

## C5. Compute / token budget (NEW)

Since token bleeding is a stated concern, instrument it:
- Track cumulative tokens and per-call counts; enforce a budget ceiling that can trigger early termination.
- **Cache** the Research/EDA results so reruns don't re-pay for them.
- Wall-clock budget for the whole run, independent of iteration count.

---

# PART D — Code Editing, Reward & Observability

## D1. Patch-based editing — re-scoped (corrected Phase 3)

Note the tension with A1: once the LLM only writes a small `build_pipeline` function (~30–60 lines), full rewrites are *cheap* and the regression risk is low, so patch-based editing's payoff shrinks. **Recommendation: deprioritize Phase 3.** Under the narrow contract, regenerating the whole function per iteration is fine.

If you still want patching for larger surfaces:
- Use **SEARCH/REPLACE blocks** (aider-style) rather than a JSON diff — they're more robust in practice.
- Always have a fallback when the target block isn't found (re-request a full function), and AST-validate after patching.

## D2. Multi-dimensional telemetry & reward (corrected Phase 4)

Capturing `execution_time_seconds` and `peak_memory_mb` (via `time` / `psutil`) is good. But two corrections:

1. **Don't hand-weight a scalar reward.** `Reward = -MAPE - α·time` mixes incompatible units (MAPE ~5–50; seconds ~1–600), so α is brittle. And the *only consumer* of this reward is the **Planner LLM reading it as text** — it's not used for gradients/RL. So **pass the raw telemetry (MAPE, seconds, peak MB) separately** and let the Planner reason about the tradeoff ("last change cut MAPE 0.05% but tripled train time"). The LLM weighs this better than a fixed formula. *(Note: the 2nd-pass plan re-proposes this exact `-MAPE - α·time` formula — same unit-mismatch problem applies; prefer raw telemetry.)*
2. **For real limits, use hard constraints, not soft penalties.** Exceeding the time/memory ceiling ⇒ candidate fails. More controllable than a penalty term. This also fixes the `n_estimators=50000` OOM/runaway case the 2nd pass worried about — a memory/time ceiling kills it outright instead of merely down-weighting it.
3. **Smart log truncation (adopted from 2nd-pass Phase 4).** Replace the blind `output[-5000:]` slice fed to the Planner. Random `print`/progress output can push the real traceback out of the window. Instead, regex-extract the high-density region — from `Traceback (most recent call last):` to the final exception line — and pass *that*. On success, pass a compact metrics summary, not raw stdout. This both sharpens the Planner's signal and cuts tokens (supports C5). Also resolves the existing `output[-5000:]` vs "last 2000 chars" inconsistency in the prompts.

## D3. Structured experiment tracking (NEW)

`.log` files are not queryable. Persist each candidate to a structured store (SQLite or JSONL): `id, parent_id, config, val_score, test_score, telemetry, status, strategy, citation`. Benefits: powers the solution-tree search (B2), the ablation evidence (B3), resumability (C3), and a far better final report than parsing logs.

## D4. Cross-run memory — `learned_lessons.md` ledger (upgraded from 2nd-pass Phase 6)

Today "memory" is only `history` within one run. The second design pass proposes a concrete, cheap version worth adopting: a local **`learned_lessons.md`** ledger.
- At the end of `report_node`, have the Analyst append a short synthesis: top-3 high-impact features and top-3 failure modes from this run.
- At startup (`search_node`), ingest `learned_lessons.md` into context so the system accumulates wisdom across runs.

This is a good low-cost win, but add two guardrails the 2nd pass omits:
- **Keep entries structured and dated, and cap the file** (e.g. last N runs). Free-form append-forever ledgers rot and eventually blow the context budget.
- **Scope lessons to the dataset/task.** A lesson learned on Rossmann may actively mislead on a different dataset; tag each lesson with the task it came from and only inject matching ones. The experiment store (D3) is the more robust long-term backing for this; `learned_lessons.md` is the human-readable summary layer on top.

---

# PART E — Cross-cutting guardrails

- **Single source of truth for the metric:** only the harness computes scores; the LLM never reports its own number (kills D2's narrative-vs-reality gap and A4's regex).
- **Mandatory invariants enforced in the harness, not the prompt:** the `Sales>0 & Open==1` filter, `StateHoliday` cast, chronological split, seeds. Prompts are suggestions; the harness is law.
- **Pre-execution AST gate** before every `subprocess` (fast fail). Make it a *micro-repair loop*: if `ast.parse()` raises, do **not** spawn a subprocess — feed the syntax/indentation error straight back to the Coder for an in-memory fix and re-check. Only valid code ever reaches execution. (This is the useful half of the 2nd-pass Phase 3; the patch-based half is deprioritized per D1.)
- **Strict directory hygiene:** all per-iteration artifacts in `train_iter/`, `logs/`, `outputs/`, experiment DB in a known path; keep repo root clean.
- **Determinism end-to-end** so any reported improvement is reproducible.

---

## Reconciliation with the second (Gemini) design pass

The second `REFACTOR.md` has 6 phases. Four overlap with this plan (and re-introduce issues this plan already fixed); two contributed genuinely new material that has been folded in above.

| 2nd-pass phase | Status | Where it lands here |
|---|---|---|
| P1 Immutable harness, `engineering_and_train(train_df) -> model` | **Duplicate + reverts a fix** | Same contract this plan flagged as leaky. Use **A1** (return unfitted `Pipeline`; harness owns temporal features) instead. |
| P2 Termination + patience + `is_finished` | **Duplicate** | **C2.** Keep patience/ceiling; `is_finished` stays advisory (2nd pass terminates on it immediately — don't). |
| P3 Patch-based coding | **Partly redundant** | Patching → **D1** (deprioritized under narrow contract). AST pre-lint micro-loop → adopted in **Part E**. |
| P4 Telemetry + reward + log truncation | **Mixed** | Telemetry → **D2**. Reward `-MAPE-α·time` → rejected again (D2, unit mismatch). **Smart log truncation → newly adopted in D2 #3.** |
| P5 Error-driven search + Hybrid RAG | **New — adopted** | Error-driven retrieval → **C1**; code-grounded RAG → **B7**. |
| P6 MCTS branching + `learned_lessons.md` | **New — adopted (with guardrails)** | Branching/prune/blacklist → **B2**; cross-run ledger → **D4**. |

Net new from this pass: **B7** (code-grounded RAG), **D2 #3** (smart log truncation), **C1** (error-driven retrieval), **B2** (prune/rollback/blacklist), **D4** (concrete ledger), **Part E** (AST micro-repair loop). Everything else was already covered, and three items (P1 contract, P4 reward, P2 `is_finished`) repeat mistakes this plan deliberately corrects — don't regress to them.

---

## Summary: what changed vs. the original REFACTOR.md

| Original plan | Verdict | Change in this plan |
|---|---|---|
| P1 Harness owns split+MAPE | Right idea, **broken contract** | A1: split responsibility by leakage risk; return an unfitted `Pipeline` |
| P2 Termination + patience | Mostly correct | C2: keep patience+ceiling; demote `is_finished` to advisory |
| P3 Patch-based editing | **Low priority** under narrow contract | D1: deprioritize; if needed use SEARCH/REPLACE not JSON diff |
| P4 Scalar reward `-MAPE-α·time` | Unit-mismatched, wrong consumer | D2: pass raw telemetry to Planner; hard limits not soft penalty |
| `ast.parse` as security | **Misconception** | C4: AST = syntax gate only; use rlimit/sandbox for security |
| — | Missing | A2 test set, A3 seeds, B1 EDA, B2 tree search, B3 ablation, B4 Optuna, B5 ensembling, B6 CV, C1 debug loop, C3 resume, D3 experiment store |

**The two non-negotiables:** A1 (real leakage fix) and A2 (held-out test set). They decide whether the final reported number is honest. Everything in Part B decides whether that honest number is actually *good*.
