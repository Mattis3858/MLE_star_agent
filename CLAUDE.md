# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MLE-STAR v2 is an autonomous ML-engineering agent (LangGraph) for the Kaggle **Rossmann Store Sales** task. It runs a research → foundation → refine-loop → finalize → report pipeline where two Ollama Cloud LLMs (`deepseek-v3.1` reasoner, `qwen3-coder` coder) propose candidate models and a deterministic **harness** evaluates them honestly. The architecture is specified by `REFACTOR_PLAN.md` — read it to understand *why* things are shaped this way (item codes like A1/B2/C1 below refer to it).

## Commands

```bash
python -m mle_star                 # full run (canonical entry)
python -m mle_star --resume        # resume from checkpoint + experiment store (C3)
python -m mle_star --max-iter 5    # short run
python -m mle_star --force-data    # rebuild the prepared-data cache
python -m mle_star_agent           # legacy shim, same as python -m mle_star

python test/test_system.py        # no-LLM smoke suite (also works with pytest)
```

Tests need `train.csv`/`store.csv` in the repo root (gitignored) but **no LLM keys**. A full agent run additionally needs `.env` with `OLLAMA_API_KEY` (+ optional `TAVILY_API_KEY`). Tunables (`MAX_ITERATIONS`, metric, budgets) live in `mle_star/config.py`; some have env overrides (`MLE_STAR_METRIC`, `MLE_STAR_MAX_ITER`, `MLE_STAR_TOKEN_BUDGET`).

## Architecture — the one rule that matters

**The harness is law; prompts are suggestions.** The LLM never loads data, never splits, never computes a metric, never reports its own score. Anything correctness-critical is enforced in code, not in a prompt. If you find yourself adding a correctness rule to a prompt template, it belongs in `data.py`/`harness.py`/`runner.py` instead. This is the design's defense against the v1 failure mode (a leaked <1% MAPE produced by LLM-written scoring).

### The candidate contract (A1)
The LLM writes a small module defining `build_pipeline(feature_columns, categorical_columns) -> unfitted sklearn Pipeline` (final step named `"model"`), optionally `USE_LOG_TARGET = True` and `param_space(trial)` for Optuna. `mle_star/baseline_candidate.py` is the canonical example (also used for the noise floor). The harness fits/predicts/scores; fittable transforms are fit on train only by construction.

### Evaluation flow (one candidate)
`agents.refine_node` → `harness.evaluate()`: AST gate + static policy gate (no network/exec imports) → write to `train_iter/` → spawn `python -m mle_star.runner` subprocess (timeout + psutil memory watchdog kill) → runner seeds everything, fits, scores, writes a **JSON result file** (never parsed from stdout) → on failure, `agents._debug_loop` (C1) gets the extracted traceback, up to 3 fix attempts, before control returns to the Planner.

### Honesty machinery (don't weaken these)
- **3-way chronological split** (`data.py`): train | val (drives all decisions) | test (last 42 days, scored once in `finalize_node`). Val/test are each ≤ `HORIZON_DAYS`, and every lag/rolling feature is shifted ≥ `HORIZON_DAYS`, so no row sees its own slice.
- **Target encodings** fit only on data before `enc_cutoff` (val_start − 84d) so they stay honest in both holdout and expanding-window CV modes (B6).
- **`Customers`** exists in training data but not at prediction time — it is in `config.LEAKAGE_COLS` and must never become a feature.
- **Noise floor** (A3): baseline re-run across seeds at startup; an "improvement" must beat best by `min_delta = max(0.05, 2σ)`. Runner force-seeds `random`/numpy/estimator `random_state`; same seed ⇒ bit-identical score (asserted in tests).

### Search (B2/B3/B4/B5)
Every evaluated candidate is a node in `outputs/experiments.jsonl` (`store.ExperimentStore`) — this one store powers the beam (`frontier()`), prune/rollback (`path_failures()`), the strategy blacklist (strategies off the best lineage), resume, and the report. The Planner branches from any frontier node; `RESTART_AFTER` non-improving rounds force a from-scratch regeneration. Numeric HPO is Optuna's job inside the runner (LLM owns structure only). `finalize_node` CV-confirms top-K finalists, scores them on test once, and blends an ensemble with val-derived weights.

## Conventions

- All LLM calls go through `llm.call_reasoner`/`llm.call_coder` (tenacity retries + token ledger; `BudgetExceeded` ends the run gracefully). ChatOllama auth must use `client_kwargs={"headers": ...}` — a bare `headers=` kwarg is silently ignored by langchain-ollama 1.0.
- LLM-generated code passes through `llm.strip_fences` then `agents._ast_repair` (max 2 in-memory fixes) before any subprocess.
- Runtime artifacts: candidates → `train_iter/` (gitignored), prepared-data cache + per-candidate results → `outputs/cache/` (gitignored), deliverables (`final_best_model.py`, `analysis_report.md`, `final_results.json`, `experiments.jsonl`) → `outputs/`. Repo root stays clean.
- `search.py`/`ensemble.py` deliberately don't exist: tree *queries* live in `store.py`, search *policy* in `agents._plan_next`, ensembling in `agents.finalize_node`.
- Reference benchmark: the honest baseline scores ~8.2 MAPE / ~10.9 RMSPE on holdout (~16s fit). If a candidate reports MAPE < ~2, suspect leakage before celebrating — that's what the v1 system got wrong.
