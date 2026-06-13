"""Central configuration. The harness is law; prompts are suggestions (Part E)."""

import os
from dataclasses import dataclass, field

# --- Directories (strict hygiene: repo root stays clean) ---
OUTPUT_DIR = "outputs"
ITER_DIR = "train_iter"
LOG_DIR = "logs"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
STORE_PATH = os.path.join(OUTPUT_DIR, "experiments.jsonl")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.json")
LESSONS_PATH = "learned_lessons.md"  # D4 cross-run ledger (human-readable)

# --- Data ---
TRAIN_CSV = "train.csv"
STORE_CSV = "store.csv"
TARGET = "Sales"
DATE_COL = "Date"
GROUP_COL = "Store"
# Columns that exist in training data but NOT at prediction time -> never features.
LEAKAGE_COLS = ["Customers"]

# --- Splits (A2/A5): chronological, horizon-aligned (~6 weeks, like the competition) ---
HORIZON_DAYS = 42
VAL_DAYS = 42   # selection slice: drives every Planner decision
TEST_DAYS = 42  # report-only slice: touched exactly once at the very end

# --- Determinism (A3) ---
SEED = 42
NOISE_FLOOR_SEEDS = [42, 43, 44]  # baseline re-runs to estimate run-to-run noise
MIN_DELTA_ABS = 0.05              # improvement must beat best by max(this, K_NOISE*noise_std)
K_NOISE = 2.0

# --- Metric (A5): pluggable; harness computes both, METRIC drives selection ---
METRIC = os.getenv("MLE_STAR_METRIC", "mape")  # "mape" | "rmspe"

# --- Search (B2) ---
MAX_ITERATIONS = int(os.getenv("MLE_STAR_MAX_ITER", "20"))
BEAM_WIDTH = 3
PATIENCE = 6                # C2: stop after this many consecutive non-improvements
PRUNE_AFTER = 3             # B2: consecutive failures on a path before pruning back to best
RESTART_AFTER = 4           # B2: non-improving rounds before forcing a fresh-approach restart
BLACKLIST_CAP = 15
ABLATION_EVERY = 5          # B3: run ablation on current best every N iterations

# --- HPO (B4) ---
OPTUNA_TRIALS_FAST = 15     # per-candidate numeric tuning budget (holdout mode)
OPTUNA_TRIALS_FINAL = 40    # finalists / CV mode
OPTUNA_TIMEOUT_S = 900

# --- Ensembling (B5) ---
ENSEMBLE_TOP_K = 3

# --- Execution limits (C4/D2: hard constraints, not soft penalties) ---
EXEC_TIMEOUT_S = 900
MEMORY_LIMIT_MB = 8000
DEBUG_MAX_ATTEMPTS = 3      # C1 self-repair loop
AST_REPAIR_MAX = 2          # Part E micro-repair before any subprocess

# --- Budgets (C5) ---
TOKEN_BUDGET = int(os.getenv("MLE_STAR_TOKEN_BUDGET", "2000000"))
WALL_CLOCK_BUDGET_S = int(os.getenv("MLE_STAR_WALL_BUDGET_S", str(6 * 3600)))

# --- LLM ---
REASONER_MODEL = "qwen3-coder-next:cloud"
CODER_MODEL = "gemma4:31b-cloud"
OLLAMA_BASE_URL = "https://ollama.com"


@dataclass
class RunConfig:
    """Bundles tunables so tests can override without monkeypatching the module."""

    max_iterations: int = MAX_ITERATIONS
    beam_width: int = BEAM_WIDTH
    patience: int = PATIENCE
    metric: str = METRIC
    seed: int = SEED
    optuna_trials: int = OPTUNA_TRIALS_FAST
    exec_timeout_s: int = EXEC_TIMEOUT_S
    memory_limit_mb: int = MEMORY_LIMIT_MB
    noise_floor_seeds: list = field(default_factory=lambda: list(NOISE_FLOOR_SEEDS))
    resume: bool = False


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, ITER_DIR, LOG_DIR, CACHE_DIR):
        os.makedirs(d, exist_ok=True)
