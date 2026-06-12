"""Data loading, invariants, harness-owned features, chronological splits.

A1: every feature that reads *other rows* (lags, rolling stats, target encoding)
is computed here, by the harness, with explicit temporal discipline. The LLM
never touches them. LLM-side pipelines only see row-local raw columns plus
these precomputed ones.

Temporal discipline: every lag/rolling feature is shifted by >= HORIZON_DAYS.
Because the val and test slices are each <= HORIZON_DAYS long, no val/test row
can see sales from inside its own slice -- the features simulate a real
42-day-ahead forecast.

A5/E: the mandatory invariants (Sales>0 & Open==1 filter, StateHoliday cast)
live here, in code, not in a prompt.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config

PREPARED_PKL = os.path.join(config.CACHE_DIR, "prepared.pkl")
META_JSON = os.path.join(config.CACHE_DIR, "meta.json")

assert config.VAL_DAYS <= config.HORIZON_DAYS, "val slice must fit the forecast horizon"
assert config.TEST_DAYS <= config.HORIZON_DAYS, "test slice must fit the forecast horizon"


# ---------------------------------------------------------------------------
# Loading + invariants
# ---------------------------------------------------------------------------

def load_raw(train_csv: str = config.TRAIN_CSV, store_csv: str = config.STORE_CSV) -> pd.DataFrame:
    """Load, merge, and enforce the mandatory invariants (harness is law)."""
    df = pd.read_csv(train_csv, low_memory=False)
    store = pd.read_csv(store_csv)

    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    # Mixed int/str column in the raw data; cast before anything else.
    df["StateHoliday"] = df["StateHoliday"].astype(str)

    df = df.merge(store, on=config.GROUP_COL, how="left")
    df = df.sort_values([config.GROUP_COL, config.DATE_COL]).reset_index(drop=True)
    return df


def apply_row_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only open stores with positive sales. Applied AFTER temporal
    features are computed (the full calendar is needed for correct lags),
    and always BEFORE any split/scoring."""
    return df[(df[config.TARGET] > 0) & (df["Open"] == 1)].copy()


# ---------------------------------------------------------------------------
# Splits (A2): train | validation (selection) | test (report-only)
# ---------------------------------------------------------------------------

def split_dates(df: pd.DataFrame) -> Dict[str, str]:
    last = df[config.DATE_COL].max()
    test_start = last - pd.Timedelta(days=config.TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=config.VAL_DAYS)
    return {
        "val_start": str(val_start.date()),
        "test_start": str(test_start.date()),
        "last_date": str(last.date()),
    }


def assign_split(df: pd.DataFrame, dates: Dict[str, str]) -> pd.Series:
    d = df[config.DATE_COL]
    out = pd.Series("train", index=df.index)
    out[d >= pd.Timestamp(dates["val_start"])] = "val"
    out[d >= pd.Timestamp(dates["test_start"])] = "test"
    return out


# ---------------------------------------------------------------------------
# Harness-owned features (named groups -> enables B3 ablation)
# ---------------------------------------------------------------------------

def _date_features(df: pd.DataFrame) -> List[str]:
    d = df[config.DATE_COL].dt
    df["Year"] = d.year
    df["Month"] = d.month
    df["Day"] = d.day
    df["WeekOfYear"] = d.isocalendar().week.astype(int)
    df["DayOfYear"] = d.dayofyear
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
    return ["Year", "Month", "Day", "WeekOfYear", "DayOfYear", "DayOfWeek", "IsWeekend"]


def _store_static_features(df: pd.DataFrame) -> List[str]:
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    df["LogCompetitionDistance"] = np.log1p(df["CompetitionDistance"])

    # Months since competition opened (0 where unknown / not yet open).
    comp_open = pd.to_datetime(
        dict(
            year=df["CompetitionOpenSinceYear"].fillna(1900).astype(int),
            month=df["CompetitionOpenSinceMonth"].fillna(1).astype(int).clip(1, 12),
            day=1,
        ),
        errors="coerce",
    )
    months = (df[config.DATE_COL] - comp_open).dt.days / 30.44
    df["CompetitionOpenMonths"] = months.where(
        df["CompetitionOpenSinceYear"].notna(), 0
    ).clip(lower=0).fillna(0)

    # Promo2 participation: active months come as e.g. "Jan,Apr,Jul,Oct".
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)
    month_abbr = df[config.DATE_COL].dt.strftime("%b")
    interval = df["PromoInterval"].fillna("")
    in_interval = np.array([m in iv for m, iv in zip(month_abbr, interval)])
    df["IsPromo2Month"] = (df["Promo2"].eq(1) & in_interval).astype(int)

    p2_start = pd.to_datetime(
        df["Promo2SinceYear"].fillna(1900).astype(int).astype(str) + "-01-01"
    ) + pd.to_timedelta((df["Promo2SinceWeek"].fillna(1) - 1) * 7, unit="D")
    weeks = (df[config.DATE_COL] - p2_start).dt.days / 7.0
    df["Promo2Weeks"] = weeks.where(df["Promo2"].eq(1), 0).clip(lower=0).fillna(0)

    return [
        "StoreType", "Assortment", "CompetitionDistance", "LogCompetitionDistance",
        "CompetitionOpenMonths", "Promo2", "IsPromo2Month", "Promo2Weeks",
    ]


def _schedule_features(df: pd.DataFrame) -> List[str]:
    """Promo/holiday calendar features. These columns are known in advance at
    prediction time (they are schedules), so shift 0 is legitimate."""
    g = df.groupby(config.GROUP_COL, sort=False)

    # Promo run length: consecutive same-Promo days per store.
    promo = df["Promo"]
    run_id = (promo != g["Promo"].shift()).groupby(df[config.GROUP_COL], sort=False).cumsum()
    df["PromoStreak"] = (
        df.groupby([config.GROUP_COL, run_id], sort=False).cumcount() + 1
    ) * promo

    # Days since/until a state holiday, per store.
    is_holiday = df["StateHoliday"].ne("0")
    hol_date = df[config.DATE_COL].where(is_holiday)
    last_hol = hol_date.groupby(df[config.GROUP_COL], sort=False).ffill()
    next_hol = hol_date.groupby(df[config.GROUP_COL], sort=False).bfill()
    df["DaysSinceHoliday"] = (
        (df[config.DATE_COL] - last_hol).dt.days.fillna(365).clip(upper=365)
    )
    df["DaysUntilHoliday"] = (
        (next_hol - df[config.DATE_COL]).dt.days.fillna(365).clip(upper=365)
    )

    return [
        "Promo", "SchoolHoliday", "StateHoliday",
        "PromoStreak", "DaysSinceHoliday", "DaysUntilHoliday",
    ]


def _grid(df: pd.DataFrame) -> pd.DataFrame:
    """Complete daily Store x Date sales grid (NaN where no record), so that
    lag/rolling shifts are in *calendar days*, robust to closed-day gaps."""
    wide = df.pivot_table(
        index=config.DATE_COL, columns=config.GROUP_COL,
        values=config.TARGET, aggfunc="first",
    )
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    return wide.reindex(full_idx)


def _temporal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Horizon-safe lags and rolling stats. Every shift >= HORIZON_DAYS."""
    H = config.HORIZON_DAYS
    grid = _grid(df)

    # All wide frames share grid's shape, so stacking with future_stack=True
    # (which keeps NaNs) yields identical MultiIndexes -> one cheap concat.
    wides: Dict[str, pd.DataFrame] = {}
    for k in (H, H + 7, H + 14, H + 49, 364):
        wides[f"SalesLag{k}"] = grid.shift(k)
    lag_cols = list(wides)

    shifted = grid.shift(H)
    roll_cols = []
    for w in (7, 28, 91):
        wides[f"SalesRollMean{w}"] = shifted.rolling(w, min_periods=max(1, w // 2)).mean()
        wides[f"SalesRollStd{w}"] = shifted.rolling(w, min_periods=max(1, w // 2)).std()
        roll_cols += [f"SalesRollMean{w}", f"SalesRollStd{w}"]

    merged = pd.concat(
        {name: w.stack(future_stack=True) for name, w in wides.items()}, axis=1
    ).reset_index()
    merged = merged.rename(
        columns={merged.columns[0]: config.DATE_COL, merged.columns[1]: config.GROUP_COL}
    )

    df = df.merge(merged, on=[config.DATE_COL, config.GROUP_COL], how="left")
    return df, lag_cols, roll_cols


def enc_cutoff_date(dates: Dict[str, str]) -> pd.Timestamp:
    """Encodings are fit only on data older than the earliest expanding-window
    CV cutoff, so they stay honest in BOTH holdout and CV scoring modes."""
    return pd.Timestamp(dates["val_start"]) - pd.Timedelta(days=2 * config.VAL_DAYS)


def _target_encoding(df: pd.DataFrame, split: pd.Series, dates: Dict[str, str]) -> List[str]:
    """Fittable encodings, fit on OLD TRAIN ROWS ONLY (A1 + B6), applied everywhere."""
    train = df[(split == "train") & (df[config.DATE_COL] < enc_cutoff_date(dates))]
    gmean = train[config.TARGET].mean()

    enc_specs = {
        "StoreMeanSales": [config.GROUP_COL],
        "StoreMedianSales": [config.GROUP_COL],
        "StoreDowMeanSales": [config.GROUP_COL, "DayOfWeek"],
        "StorePromoMeanSales": [config.GROUP_COL, "Promo"],
    }
    cols = []
    for name, keys in enc_specs.items():
        agg = "median" if "Median" in name else "mean"
        table = train.groupby(keys)[config.TARGET].agg(agg).rename(name)
        df[name] = df[keys].merge(table, left_on=keys, right_index=True, how="left")[
            name
        ].fillna(gmean).values
        cols.append(name)
    return cols


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare(force: bool = False) -> Tuple[pd.DataFrame, dict]:
    """Build (or load from cache) the prepared dataset + metadata."""
    config.ensure_dirs()
    if not force and os.path.exists(PREPARED_PKL) and os.path.exists(META_JSON):
        with open(META_JSON, encoding="utf-8") as f:
            meta = json.load(f)
        return pd.read_pickle(PREPARED_PKL), meta

    df = load_raw()
    dates = split_dates(df)

    feature_groups: Dict[str, List[str]] = {}
    feature_groups["date"] = _date_features(df)
    feature_groups["store_static"] = _store_static_features(df)
    feature_groups["schedule"] = _schedule_features(df)
    df, lag_cols, roll_cols = _temporal_features(df)
    feature_groups["lag"] = lag_cols
    feature_groups["rolling"] = roll_cols

    # Filter AFTER temporal features (full calendar needed), BEFORE split use.
    df = apply_row_filter(df)
    df["split"] = assign_split(df, dates)
    feature_groups["target_enc"] = _target_encoding(df, df["split"], dates)

    feature_columns = [c for cols in feature_groups.values() for c in cols]
    assert not set(config.LEAKAGE_COLS) & set(feature_columns), "leakage column in features"

    keep = (
        [config.DATE_COL, config.GROUP_COL, config.TARGET, "split"] + feature_columns
    )
    df = df[keep].reset_index(drop=True)

    meta = {
        "target": config.TARGET,
        "metric": config.METRIC,
        "split_dates": dates,
        "enc_cutoff": str(enc_cutoff_date(dates).date()),
        "feature_groups": feature_groups,
        "feature_columns": feature_columns,
        "dtypes": {c: str(df[c].dtype) for c in feature_columns},
        "n_rows": {k: int((df["split"] == k).sum()) for k in ("train", "val", "test")},
    }

    df.to_pickle(PREPARED_PKL)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return df, meta
