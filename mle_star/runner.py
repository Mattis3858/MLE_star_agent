"""Child-process candidate evaluator.

The harness spawns this as a subprocess per candidate. It:
  1. seeds everything (A3),
  2. imports the candidate module (which only defines build_pipeline et al.),
  3. fits on the train slice, scores on val (or CV / final-test),
  4. writes a JSON result file (A4 - no stdout scraping).

The candidate NEVER sees the raw CSVs, never splits, never scores.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

from . import config, data


# ---------------------------------------------------------------------------
# Metrics (A4/A5: harness-owned, pluggable, numerically safe)
# ---------------------------------------------------------------------------

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100)


METRICS = {"mape": safe_mape, "rmspe": rmspe}


def score_all(y_true, y_pred) -> dict:
    return {name: fn(y_true, y_pred) for name, fn in METRICS.items()}


# ---------------------------------------------------------------------------
# Candidate loading + pipeline assembly
# ---------------------------------------------------------------------------

def load_candidate(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "build_pipeline"):
        raise AttributeError("candidate module must define build_pipeline(...)")
    return mod


def make_estimator(mod, feature_columns, categorical_columns, seed: int):
    from sklearn.compose import TransformedTargetRegressor

    pipe = mod.build_pipeline(feature_columns, categorical_columns)
    # Force determinism on the final estimator regardless of what the LLM set.
    final = pipe.steps[-1][1]
    if "random_state" in final.get_params():
        final.set_params(random_state=seed)

    if getattr(mod, "USE_LOG_TARGET", False):
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        ), pipe
    return pipe, pipe


def set_pipe_params(estimator, pipe, params: dict):
    """Apply Optuna params to the inner pipeline (works for both wrappings)."""
    pipe.set_params(**params)
    return estimator


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _fit_score(mod, X_tr, y_tr, X_ev, y_ev, feature_columns, categorical_columns, seed):
    est, _ = make_estimator(mod, feature_columns, categorical_columns, seed)
    t0 = time.time()
    est.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    y_pred = np.clip(est.predict(X_ev), 0, None)
    return score_all(y_ev, y_pred), y_pred, fit_s, est


def run_optuna(mod, X_tr, y_tr, X_val, y_val, feature_columns, categorical_columns,
               seed, n_trials, metric):
    """B4: numeric HPO inside one evaluation. Returns (best_params, n_done)."""
    if n_trials <= 0 or not hasattr(mod, "param_space"):
        return {}, 0
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        est, pipe = make_estimator(mod, feature_columns, categorical_columns, seed)
        set_pipe_params(est, pipe, mod.param_space(trial))
        est.fit(X_tr, y_tr)
        y_pred = np.clip(est.predict(X_val), 0, None)
        return METRICS[metric](y_val, y_pred)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=config.OPTUNA_TIMEOUT_S)
    return study.best_params, len(study.trials)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mode", choices=["holdout", "cv", "test"], default="holdout")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--optuna-trials", type=int, default=0)
    ap.add_argument("--result", required=True)
    ap.add_argument("--exclude-groups", default="")
    ap.add_argument("--predictions", default="")
    args = ap.parse_args()

    seed_everything(args.seed)

    df, meta = data.prepare()  # cache hit; parent prepared it
    metric = meta["metric"]

    feature_columns = list(meta["feature_columns"])
    excluded = [g for g in args.exclude_groups.split(",") if g]
    for g in excluded:
        feature_columns = [
            c for c in feature_columns if c not in set(meta["feature_groups"][g])
        ]
    categorical_columns = [
        c for c in feature_columns if meta["dtypes"][c] == "object"
    ]

    mod = load_candidate(args.candidate)
    y = df[meta["target"]]
    X = df[feature_columns]
    split = df["split"]
    dates = df[config.DATE_COL]

    t_start = time.time()
    result = {
        "mode": args.mode,
        "seed": args.seed,
        "excluded_groups": excluded,
        "n_features": len(feature_columns),
        "use_log_target": bool(getattr(mod, "USE_LOG_TARGET", False)),
    }

    if args.mode == "holdout":
        tr, ev = split == "train", split == "val"
        best_params, n_trials = run_optuna(
            mod, X[tr], y[tr], X[ev], y[ev],
            feature_columns, categorical_columns, args.seed, args.optuna_trials, metric,
        )
        est, pipe = make_estimator(mod, feature_columns, categorical_columns, args.seed)
        if best_params:
            set_pipe_params(est, pipe, best_params)
        t0 = time.time()
        est.fit(X[tr], y[tr])
        fit_s = time.time() - t0
        y_pred = np.clip(est.predict(X[ev]), 0, None)
        result.update(
            val_scores=score_all(y[ev], y_pred),
            best_params=best_params, n_optuna_trials=n_trials, fit_seconds=fit_s,
        )

    elif args.mode == "cv":
        # B6: expanding-window CV. 3 chronological folds ending at val_start.
        val_start = pd.Timestamp(meta["split_dates"]["val_start"])
        fold_scores = []
        for i in (2, 1, 0):
            cut = val_start - pd.Timedelta(days=i * config.VAL_DAYS)
            hi = cut + pd.Timedelta(days=config.VAL_DAYS)
            tr = (split != "test") & (dates < cut)
            ev = (split != "test") & (dates >= cut) & (dates < hi)
            scores, _, fit_s, _ = _fit_score(
                mod, X[tr], y[tr], X[ev], y[ev],
                feature_columns, categorical_columns, args.seed,
            )
            fold_scores.append(scores)
        result.update(
            fold_scores=fold_scores,
            val_scores={
                m: float(np.mean([f[m] for f in fold_scores])) for m in METRICS
            },
            val_scores_std={
                m: float(np.std([f[m] for f in fold_scores])) for m in METRICS
            },
        )

    else:  # test: refit on train+val, score the report-only slice ONCE (A2)
        tr, ev = split != "test", split == "test"
        est, pipe = make_estimator(mod, feature_columns, categorical_columns, args.seed)
        best_params = json.loads(os.environ.get("MLE_STAR_BEST_PARAMS", "{}"))
        if best_params:
            set_pipe_params(est, pipe, best_params)
        t0 = time.time()
        est.fit(X[tr], y[tr])
        fit_s = time.time() - t0
        y_pred = np.clip(est.predict(X[ev]), 0, None)
        result.update(test_scores=score_all(y[ev], y_pred), fit_seconds=fit_s)
        if args.predictions:
            np.savez(
                args.predictions,
                y_pred=y_pred, y_true=y[ev].to_numpy(),
                dates=dates[ev].astype("int64").to_numpy(),
                stores=df.loc[ev, config.GROUP_COL].to_numpy(),
            )

    result["total_seconds"] = time.time() - t_start
    try:
        import psutil

        info = psutil.Process().memory_info()
        result["peak_mb"] = round(getattr(info, "peak_wset", info.rss) / 1e6, 1)
    except Exception:
        result["peak_mb"] = None

    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
