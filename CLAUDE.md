# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MLE-STAR is an autonomous ML-engineering agent built on **LangGraph**. It runs a closed-loop "search → code → execute → evaluate → refine" cycle to produce an optimized model for the **Kaggle Rossmann Store Sales** forecasting task (metric: MAPE, lower is better). Nearly all logic lives in a single file: `mle_star_agent.py`.

## Running

```bash
pip install -r requirements.txt    # Python 3.13
python -m mle_star_agent           # full pipeline (~20 iterations, long-running)
```

Prerequisites (the script exits early if missing):
- `train.csv` and `store.csv` in the repo root (gitignored; not committed).
- A `.env` file with `OLLAMA_API_KEY` and `TAVILY_API_KEY`. Tavily is optional — if its key is absent, the research step skips web search but still runs.

There is no test suite or linter. `test/test_ollama.py` is a manual connectivity check for the Ollama Cloud endpoint, not a unit test.

## Architecture

A `StateGraph` over a single `AgentState` TypedDict (the shared blackboard) wires four nodes:

```
search → foundation → refine ⟲ (loop until MAX_ITERATIONS) → report → END
```

1. **`search_node`** (Research) — fetches the core MLE-STAR paper via Arxiv + Kaggle solutions via Tavily, then asks the *reasoner* LLM to emit `citations` and a `design_spec`. Always force-includes the MLE-STAR paper (arxiv:2506.15692) as a citation.
2. **`foundation_node`** (Foundation Coder) — *coder* LLM writes the first full training script from the spec. The prompt embeds **hard, non-negotiable rules** (see below).
3. **`refinement_node`** (the core loop) — for each iteration: writes the current `code` to `train_iter/train_iter_{n}.py`, runs it via `subprocess` (600s timeout, cwd = repo root so `train.csv`/`store.csv` resolve), regex-parses `FINAL_MAPE:` from stdout, then runs the **Planner** (reasoner, emits JSON strategy) and **Coder** (coder, applies the strategy to produce next `code`). Logs land in `logs/iter_{n}.log`.
4. **`report_node`** (Analyst) — reasoner LLM produces the report markdown from the `history`.

Output directories are defined as constants at the top of the file (`OUTPUT_DIR="outputs"`, `ITER_DIR="train_iter"`, `LOG_DIR="logs"`) and created at startup. On exit, `best_code` → `outputs/final_best_model.py` and `report` → `outputs/analysis_report.md`. The generated training scripts are prompted to save their own artifacts (plots, model dumps) under `outputs/` too.

### Two LLMs, two roles
- `llm_reasoner` = `deepseek-v3.1:671b-cloud` — research, planning, reporting.
- `llm_coder` = `qwen3-coder:480b-cloud` — writing/editing code only.

Both are **Ollama Cloud** (`base_url="https://ollama.com"` with a Bearer header), *not* a local Ollama install — despite what the README says about local/offline use. Both run at `temperature=0.0`. All LLM calls go through `call_llm_reasoner` / `call_llm_coder`, which are wrapped with `tenacity` retry (3 attempts, 5s wait) — add new calls through these wrappers, not by invoking the models directly.

### Self-correction & rollback logic (in `refinement_node`)
This is the trickiest part of the loop, keyed off MAPE:
- **Improved** (`mape < best_mape`): adopt as new `best_code`, reset `no_improve_rounds`.
- **Failed** (`mape == inf`): if no good baseline exists yet (`best_mape` still inf), keep the broken code and force the agent to fix it; otherwise **roll back** — feed `best_code` (not the broken code) to the Coder.
- **No improvement** (finite but not better): also feed `best_code` back, so refinement always branches from the best-known version.

`reward = -mape` (or `-1e9` on failure) is tracked in `history` for the report but does not currently gate control flow beyond the MAPE comparisons above.

## Conventions when editing

- **`MAX_ITERATIONS`** is set at the top of `mle_star_agent.py` (currently `20`; the README's "15" is stale). It's the only loop-termination control — `should_continue` checks nothing else.
- The pipeline's correctness depends on **prompt-embedded rules** that the generated code must obey. When changing prompts, preserve these or the loop breaks: filter `df = df[(df['Sales'] > 0) & (df['Open'] == 1)]` exactly once before any split; cast `StateHoliday` to `str`; numerically safe MAPE (no `inf`); print the result as exactly `FINAL_MAPE: {value}` (the regex `(FINAL_MAPE|Final MAPE|MAPE)\s*[:=]\s*...` depends on this); no `iterrows`/row loops; preserve GPU params.
- LLM code output is sanitized by stripping ` ```python ` / ` ``` ` fences — prompts instruct the model to emit raw Python, but the strip is the safety net.
- All run output is routed into subfolders via the `OUTPUT_DIR`/`ITER_DIR`/`LOG_DIR` constants — keep it that way so the repo root stays clean. `train.csv`, `store.csv`, `.env`, `venv/`, `__pycache__/`, `catboost_info/`, and `logs/` are gitignored.
