"""All prompt templates.

Note what is deliberately ABSENT compared to the v1 prompts: no FINAL_MAPE
print contract, no Sales>0 filter rule, no StateHoliday cast, no leakage
warnings. Those invariants moved into the harness (Part E: harness is law).
The LLM's job shrank to: design features, pick models, declare search spaces.
"""

CONTRACT = '''
# THE CANDIDATE CONTRACT (your entire output is ONE Python module)

Your module MUST define:

    def build_pipeline(feature_columns: list[str], categorical_columns: list[str]):
        """Return an UNFITTED sklearn-compatible Pipeline.
        The FINAL step MUST be named "model"."""

Optionally:

    USE_LOG_TARGET = True   # harness wraps the target in log1p/expm1

    def param_space(trial) -> dict:
        """Optuna search space. Keys are pipeline params, e.g.
        "model__n_estimators". The harness runs the tuning for you."""

Hard rules:
- NO file IO, NO reading CSVs, NO train/test splitting, NO metric computation,
  NO fitting at import time. The harness owns all of that.
- The pipeline receives a pandas DataFrame with EXACTLY feature_columns.
  Categorical (object-dtype) columns are listed in categorical_columns and the
  pipeline MUST encode them (e.g. OrdinalEncoder/OneHotEncoder inside a
  ColumnTransformer with remainder="passthrough").
- Some numeric columns contain NaN (early-history lags). Tree models like
  LightGBM/XGBoost/HistGradientBoosting handle NaN natively; if you use a
  model that does not, impute inside the pipeline.
- Allowed libraries: sklearn, xgboost, lightgbm, catboost, numpy, pandas.
- Output ONLY the Python module. No markdown fences, no commentary.
'''

EXAMPLE_NOTE = """
A known-good example of the contract (a HistGradientBoosting baseline) looks
like the module below. Improve on it; do not just echo it back.

{baseline_code}
"""


RESEARCH = """You are the Research Scientist of an autonomous ML agent for: {task}.

Dataset profile (computed from the real data by the harness):
{profile_md}

External search results (papers / Kaggle solutions / code snippets):
{search_results}

Tasks:
1. List 5 key references (papers, Kaggle winner solutions, method names) worth
   implementing for this task. One MUST be "MLE-STAR (arxiv:2506.15692)".
2. For each, give ONE concrete, implementable idea (a feature, a model choice,
   a transform) - not a vague theme.

Output each reference on its own line as: REFERENCE | concrete idea
"""

DESIGN = """You are the Lead ML Architect. Design a model-building strategy for: {task}.

Dataset profile:
{profile_md}

References and ideas from research:
{citations}

The execution harness ALREADY handles (do not redesign these):
- data loading, mandatory row filters, chronological train/val/test split
- temporal features: lags/rolling stats (horizon-safe), target encodings
- scoring ({metric}), Optuna hyperparameter tuning, ensembling

Your design space is the candidate module: which estimator family, how to
encode categoricals, which feature interactions to add as row-local pipeline
transforms, whether to log-transform the target, and what Optuna search space
to expose.

Write a concise design spec (<= 400 words) with: (1) first model to try and
why, (2) 3 candidate improvements ranked by expected impact, (3) what to
watch out for in this dataset based on the profile.
"""

FOUNDATION = """You are an expert ML engineer writing the FIRST candidate module
for: {task}.

Design spec:
{design_spec}

Available feature columns (with dtypes):
{feature_table}

{contract}

{example}
"""

PLANNER = """You are the Planner of an autonomous ML system (MLE-STAR style).
Decide the SINGLE next experiment. Evidence below; metric is {metric} (lower
is better). An improvement only counts if it beats best by > {min_delta:.3f}
(measured run-to-run noise; smaller deltas are jitter - do not chase them).

## Current frontier (top candidates; you may branch from ANY of them)
{frontier_md}

## Global best: node {best_id} with {metric} = {best_score:.4f}
## Consecutive non-improving rounds: {no_improve}

## Last evaluation
status: {last_status}
{last_detail}

## Ablation evidence (delta = how much {metric} WORSENS when the harness
removes a feature group from the best candidate; large positive = valuable,
near-zero/negative = dead weight). May be empty if not yet run.
{ablation_md}

## Telemetry of recent candidates (seconds / peak MB / n_features)
{telemetry_md}

## Strategy blacklist (already tried, did NOT help - do not re-propose)
{blacklist_md}

## Dataset profile reminder
{profile_md}

Rules:
- Propose ONE concrete change to the candidate module (model family, encoding,
  pipeline-level feature interaction, target transform, or Optuna space).
- The harness owns data/splits/lags/encodings - changes there are impossible.
- If status was a failure, the Debugger already handled it; you still propose
  an improvement, branching from a healthy node.
- If stuck >= {restart_after} rounds, consider "restart": a fresh approach
  from scratch (different model family / structure), not an edit.

Output STRICT JSON only:
{{
  "branch_from_node_id": <int - a node id from the frontier>,
  "restart": <true|false>,
  "component": "...",
  "strategy": "... one concrete change ...",
  "citation": "... reference motivating it, if any ...",
  "reasoning": "... why this is the highest-expected-value move ..."
}}
"""

CODER = """You are the Coder of an autonomous ML system.
Apply EXACTLY this plan to the candidate module below.

PLAN: {strategy}
COMPONENT: {component}
REASON: {reasoning}

BASE MODULE (modify this; keep everything else intact):
{base_code}

{contract}
"""

DEBUGGER = """You are the Debugger of an autonomous ML system. A candidate
module failed in the harness. Fix it with the SMALLEST change that makes it
run; do not redesign or "improve" anything else.

FAILURE ({status}):
{error}

{search_hint}

CANDIDATE MODULE:
{code}

{contract}
"""

SYNTAX_REPAIR = """The following Python module has a syntax error. Return the
corrected module ONLY (no fences, no commentary). Error:
{error}

MODULE:
{code}
"""

ANALYST = """You are a Senior Data Scientist writing the final Markdown report
for an autonomous ML run on: {task}.

## Honest evaluation protocol (explain this in the report)
- Chronological train/validation/test split; validation drove every decision;
  the test slice ({test_days} days) was scored EXACTLY ONCE at the end.
- All temporal features were horizon-safe (>= {horizon} day shifts); target
  encodings fit on old training data only. Metric computed by the harness,
  never self-reported by a model.
- Improvement threshold {min_delta:.3f} {metric} (measured noise floor).

## Results
- Best single model:  val {metric} = {best_val:.4f}, test {metric} = {best_test}
- Ensemble (top-{k}): test {metric} = {ensemble_test}
- A large val/test gap indicates the search overfit validation - comment on it.

## Experiment tree (every evaluated candidate)
{nodes_md}

## Ablation evidence on the final best
{ablation_md}

## Budget: {token_summary}; wall clock {wall_minutes:.0f} min; stop reason: {stop_reason}

Write the report with sections: Executive Summary; Methodology & Agent
Architecture (research/EDA -> foundation -> evaluate/debug/plan/code loop with
beam search -> ensemble -> report); Evaluation Protocol & Why the Number Is
Honest; Key Improvements & References; Versioned Experiments; Challenges &
Limitations; Business Insights. Output ONLY Markdown.
"""

LESSONS = """Distill the run below into AT MOST 3 reusable lessons for future
runs on the SAME task. Each lesson: one line, concrete, actionable
(features/models/transforms that worked or failed). Output as markdown bullets
only.

Run summary:
{summary}
"""
