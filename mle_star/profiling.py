"""B1: deterministic data profile. Pandas computes the facts; the LLM only
interprets them. Conditions every downstream design/planning decision on the
real dataset instead of generic Rossmann lore."""

import numpy as np
import pandas as pd

from . import config


def profile(df: pd.DataFrame, meta: dict) -> dict:
    """Structured profile of the PREPARED dataset (post-invariant rows)."""
    target = meta["target"]
    y = df[target]
    dates = df[config.DATE_COL]

    cols = {}
    for c in meta["feature_columns"]:
        s = df[c]
        info = {
            "dtype": str(s.dtype),
            "missing_frac": round(float(s.isna().mean()), 4),
            "nunique": int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            info.update(
                min=round(float(s.min()), 3),
                max=round(float(s.max()), 3),
                skew=round(float(s.skew()), 3),
            )
            # Correlation on a deterministic sample for speed.
            sample = df[[c, target]].dropna()
            if len(sample) > 50_000:
                sample = sample.iloc[:: len(sample) // 50_000]
            if sample[c].nunique() > 1:
                info["target_corr"] = round(float(sample[c].corr(sample[target])), 3)
        cols[c] = info

    day_counts = dates.dt.normalize().value_counts()
    expected_days = (dates.max() - dates.min()).days + 1

    high_corr = [
        c for c, i in cols.items() if abs(i.get("target_corr", 0)) > 0.9
    ]

    return {
        "n_rows": int(len(df)),
        "n_features": len(meta["feature_columns"]),
        "splits": meta["n_rows"],
        "split_dates": meta["split_dates"],
        "target": {
            "mean": round(float(y.mean()), 1),
            "median": round(float(y.median()), 1),
            "std": round(float(y.std()), 1),
            "skew": round(float(y.skew()), 3),
            "log_transform_recommended": bool(y.skew() > 1.0),
        },
        "date_coverage": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "distinct_days": int(day_counts.size),
            "expected_days": int(expected_days),
            "gap_days": int(expected_days - day_counts.size),
        },
        "columns": cols,
        "leakage": {
            "excluded_by_harness": config.LEAKAGE_COLS,
            "suspiciously_correlated": high_corr,
        },
    }


def render_markdown(p: dict, max_cols: int = 40) -> str:
    """Compact markdown for prompts (token-aware: facts only, no prose)."""
    lines = [
        f"Rows: {p['n_rows']:,} | features: {p['n_features']} | "
        f"splits train/val/test: {p['splits']['train']:,}/{p['splits']['val']:,}/{p['splits']['test']:,}",
        f"Dates {p['date_coverage']['start']} -> {p['date_coverage']['end']} "
        f"(gaps: {p['date_coverage']['gap_days']} days)",
        f"Target: mean={p['target']['mean']} median={p['target']['median']} "
        f"skew={p['target']['skew']} (log-transform recommended: {p['target']['log_transform_recommended']})",
        f"Harness-excluded leakage columns: {p['leakage']['excluded_by_harness']}",
    ]
    if p["leakage"]["suspiciously_correlated"]:
        lines.append(
            f"WARNING suspiciously target-correlated (>0.9): {p['leakage']['suspiciously_correlated']}"
        )
    lines.append("")
    lines.append("| column | dtype | miss | nuniq | corr(target) | skew |")
    lines.append("|---|---|---|---|---|---|")
    for c, i in list(p["columns"].items())[:max_cols]:
        lines.append(
            f"| {c} | {i['dtype']} | {i['missing_frac']} | {i['nunique']} "
            f"| {i.get('target_corr', '-')} | {i.get('skew', '-')} |"
        )
    return "\n".join(lines)
