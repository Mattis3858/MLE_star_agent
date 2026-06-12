"""Parent-side candidate evaluation harness.

Single source of truth for scores (Part E): only this module (via runner.py)
ever produces a metric. The LLM never reports its own number.

Hard constraints, not soft penalties (D2): wall-clock timeout and a memory
ceiling enforced by a psutil watchdog -> a runaway candidate FAILS, it is not
merely down-weighted.
"""

import ast
import json
import os
import subprocess
import sys
import threading
import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config

RUNNER = [sys.executable, "-m", "mle_star.runner"]
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_candidate.py")


# ---------------------------------------------------------------------------
# AST gate (Part E: fast fail before any subprocess; NOT a security control)
# ---------------------------------------------------------------------------

def ast_gate(code: str) -> Optional[str]:
    """Return None if the code parses, else the syntax error message."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno}): {e.text!r}"


FORBIDDEN_CALLS = {"eval", "exec", "__import__"}
FORBIDDEN_MODULES = {"socket", "urllib", "requests", "http", "ftplib", "shutil"}


def policy_gate(code: str) -> Optional[str]:
    """Cheap static policy check on candidate code (C4: defense in depth,
    not isolation). Network/exec primitives have no business in a pipeline."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return f"forbidden call: {node.func.id}()"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for n in names:
                if n.split(".")[0] in FORBIDDEN_MODULES:
                    return f"forbidden import: {n}"
    return None


# ---------------------------------------------------------------------------
# Smart log truncation (D2 #3)
# ---------------------------------------------------------------------------

def extract_error(output: str, max_chars: int = 3000) -> str:
    """Pull the traceback block out of child output instead of a blind tail
    slice. Falls back to the tail if no traceback marker is present."""
    idx = output.rfind("Traceback (most recent call last):")
    if idx >= 0:
        return output[idx:idx + max_chars]
    return output[-max_chars:]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    status: str                  # ok | syntax_error | policy_error | error | timeout | memory_limit
    candidate_path: str = ""
    scores: Dict[str, float] = field(default_factory=dict)       # val (or CV-mean) scores
    test_scores: Dict[str, float] = field(default_factory=dict)  # only in test mode
    telemetry: Dict = field(default_factory=dict)
    best_params: Dict = field(default_factory=dict)
    error: str = ""
    raw: Dict = field(default_factory=dict)

    def score(self, metric: str) -> float:
        src = self.scores or self.test_scores
        return float(src.get(metric, float("inf")))


def _watchdog(proc: subprocess.Popen, limit_mb: int, peak: dict, stop: threading.Event):
    try:
        import psutil

        p = psutil.Process(proc.pid)
        while not stop.is_set() and proc.poll() is None:
            try:
                rss = p.memory_info().rss / 1e6
                for c in p.children(recursive=True):
                    rss += c.memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                break
            peak["mb"] = max(peak.get("mb", 0.0), rss)
            if rss > limit_mb:
                peak["killed"] = True
                proc.kill()
                break
            time.sleep(0.5)
    except Exception:
        pass


def evaluate(
    code: str,
    name: str,
    mode: str = "holdout",
    seed: int = config.SEED,
    optuna_trials: int = 0,
    exclude_groups: Optional[List[str]] = None,
    best_params: Optional[Dict] = None,
    predictions_path: str = "",
    timeout_s: int = config.EXEC_TIMEOUT_S,
    memory_limit_mb: int = config.MEMORY_LIMIT_MB,
) -> EvalResult:
    """Gate -> write candidate -> run in subprocess -> parse JSON result."""
    config.ensure_dirs()

    err = ast_gate(code)
    if err:
        return EvalResult(status="syntax_error", error=err)
    err = policy_gate(code)
    if err:
        return EvalResult(status="policy_error", error=err)

    cand_path = os.path.join(config.ITER_DIR, f"{name}.py")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(code)

    result_path = os.path.join(config.CACHE_DIR, f"result_{name}.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    cmd = RUNNER + [
        "--candidate", cand_path,
        "--mode", mode,
        "--seed", str(seed),
        "--optuna-trials", str(optuna_trials),
        "--result", result_path,
        "--exclude-groups", ",".join(exclude_groups or []),
    ]
    if predictions_path:
        cmd += ["--predictions", predictions_path]

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(seed)
    if best_params:
        env["MLE_STAR_BEST_PARAMS"] = json.dumps(best_params)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    peak: dict = {"mb": 0.0}
    stop = threading.Event()
    wd = threading.Thread(
        target=_watchdog, args=(proc, memory_limit_mb, peak, stop), daemon=True
    )
    wd.start()

    try:
        output, _ = proc.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        timed_out = True
    finally:
        stop.set()

    telemetry = {"peak_mb_observed": round(peak.get("mb", 0.0), 1)}

    if peak.get("killed"):
        return EvalResult(
            status="memory_limit", candidate_path=cand_path, telemetry=telemetry,
            error=f"killed: exceeded {memory_limit_mb} MB (peak {peak['mb']:.0f} MB)",
        )
    if timed_out:
        return EvalResult(
            status="timeout", candidate_path=cand_path, telemetry=telemetry,
            error=f"killed: exceeded {timeout_s}s wall clock",
        )
    if proc.returncode != 0 or not os.path.exists(result_path):
        return EvalResult(
            status="error", candidate_path=cand_path, telemetry=telemetry,
            error=extract_error(output or ""),
        )

    with open(result_path, encoding="utf-8") as f:
        raw = json.load(f)

    telemetry.update(
        fit_seconds=raw.get("fit_seconds"),
        total_seconds=raw.get("total_seconds"),
        peak_mb=raw.get("peak_mb"),
        n_features=raw.get("n_features"),
        n_optuna_trials=raw.get("n_optuna_trials", 0),
        use_log_target=raw.get("use_log_target"),
    )
    return EvalResult(
        status="ok",
        candidate_path=cand_path,
        scores=raw.get("val_scores", {}),
        test_scores=raw.get("test_scores", {}),
        best_params=raw.get("best_params", {}),
        telemetry=telemetry,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# A3: noise floor + improvement threshold
# ---------------------------------------------------------------------------

def noise_floor(metric: str = config.METRIC, seeds: Optional[List[int]] = None) -> dict:
    """Run the harness baseline across seeds; the std is the run-to-run noise."""
    with open(BASELINE_PATH, encoding="utf-8") as f:
        code = f.read()
    scores = []
    for s in seeds or config.NOISE_FLOOR_SEEDS:
        r = evaluate(code, f"baseline_seed{s}", seed=s)
        if r.status == "ok":
            scores.append(r.score(metric))
    if not scores:
        return {"scores": [], "std": 0.0, "min_delta": config.MIN_DELTA_ABS}
    std = float(np.std(scores))
    return {
        "scores": scores,
        "std": std,
        "min_delta": max(config.MIN_DELTA_ABS, config.K_NOISE * std),
        "baseline_mean": float(np.mean(scores)),
    }


def is_real_improvement(new: float, best: float, min_delta: float) -> bool:
    return (best - new) > min_delta


# ---------------------------------------------------------------------------
# B3: ablation evidence
# ---------------------------------------------------------------------------

def ablation(code: str, base_score: float, feature_groups: List[str],
             metric: str = config.METRIC, seed: int = config.SEED) -> Dict[str, float]:
    """Score the candidate with each harness feature group removed.
    Positive delta = removing the group HURTS (the group is valuable)."""
    deltas = {}
    for g in feature_groups:
        r = evaluate(code, f"ablate_{g}", seed=seed, exclude_groups=[g])
        deltas[g] = (r.score(metric) - base_score) if r.status == "ok" else float("nan")
    return deltas
