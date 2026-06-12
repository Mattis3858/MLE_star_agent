"""No-LLM smoke tests for the refactored system.

Run:  python -m pytest test/test_system.py -v
(or:  python test/test_system.py  for a plain run)

Covers the honesty-critical machinery: split chronology, leakage exclusions,
temporal-feature horizon discipline, harness round-trip + failure paths,
metric correctness, store/tree queries. Requires train.csv/store.csv.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mle_star import config, data, harness, profiling, store
from mle_star.runner import rmspe, safe_mape


def test_metrics():
    y = np.array([100.0, 200.0, 0.0])
    p = np.array([110.0, 180.0, 10.0])
    m = safe_mape(y, p)
    assert np.isfinite(m), "safe_mape must never be inf"
    assert abs(safe_mape([100], [90]) - 10.0) < 1e-9
    assert abs(rmspe([100, 100], [90, 110]) - 10.0) < 1e-9


def test_prepared_data():
    df, meta = data.prepare()

    # Chronological, non-overlapping splits (A2).
    tr = df[df.split == "train"][config.DATE_COL]
    va = df[df.split == "val"][config.DATE_COL]
    te = df[df.split == "test"][config.DATE_COL]
    assert tr.max() < va.min() < va.max() < te.min()
    assert (te.max() - te.min()).days + 1 <= config.TEST_DAYS

    # Mandatory invariants applied in the harness (Part E).
    assert (df[config.TARGET] > 0).all()

    # Leakage columns excluded by construction (A1).
    for col in config.LEAKAGE_COLS:
        assert col not in meta["feature_columns"]

    # Horizon discipline: lag values for val/test rows must equal sales from
    # >= HORIZON_DAYS earlier (spot-check a store).
    s = df[(df[config.GROUP_COL] == 1) & df["SalesLag42"].notna()].tail(50)
    merged = s.merge(
        df[df[config.GROUP_COL] == 1][[config.DATE_COL, config.TARGET]],
        left_on=s[config.DATE_COL] - pd.Timedelta(days=42),
        right_on=config.DATE_COL, suffixes=("", "_past"),
    )
    matched = merged.dropna(subset=["Sales_past"])
    assert len(matched) > 0
    assert (matched["SalesLag42"] == matched["Sales_past"]).all()

    # Target encodings must be fit strictly before the CV-safe cutoff (B6).
    cutoff = pd.Timestamp(meta["enc_cutoff"])
    old = df[df[config.DATE_COL] < cutoff]
    expected = old[old[config.GROUP_COL] == 1][config.TARGET].mean()
    got = df[df[config.GROUP_COL] == 1]["StoreMeanSales"].iloc[0]
    assert abs(got - expected) < 1e-6, "encoding leaked newer data"


def test_profile():
    df, meta = data.prepare()
    p = profiling.profile(df, meta)
    assert p["target"]["log_transform_recommended"] in (True, False)
    assert "Customers" in p["leakage"]["excluded_by_harness"]
    md = profiling.render_markdown(p)
    assert "SalesLag42" in md


def test_harness_gates():
    assert harness.ast_gate("def f(:") is not None
    assert harness.ast_gate("def f(): pass") is None
    assert harness.policy_gate("import requests") is not None
    assert harness.policy_gate("import numpy") is None
    out = "noise\nTraceback (most recent call last):\n  boom\nValueError: x"
    assert harness.extract_error(out).startswith("Traceback")


def test_harness_roundtrip_and_determinism():
    with open(harness.BASELINE_PATH, encoding="utf-8") as f:
        code = f.read()
    r1 = harness.evaluate(code, "t_round1", seed=123)
    r2 = harness.evaluate(code, "t_round2", seed=123)
    assert r1.status == "ok", r1.error
    assert r1.scores["mape"] > 1.0, "suspiciously low MAPE - leakage?"
    # A3: same seed -> identical score.
    assert abs(r1.scores["mape"] - r2.scores["mape"]) < 1e-9


def test_harness_failure_paths():
    r = harness.evaluate("def build_pipeline(f, c):\n    raise RuntimeError('x')", "t_err")
    assert r.status == "error" and "Traceback" in r.error
    r = harness.evaluate("def build_pipeline(:", "t_syn")
    assert r.status == "syntax_error"


def test_store_tree():
    with tempfile.TemporaryDirectory() as d:
        st = store.ExperimentStore(os.path.join(d, "x.jsonl"))
        a = st.add(store.Node(id=1, parent_id=None, iteration=1, status="ok",
                              scores={"mape": 10.0}, strategy="s1"))
        b = st.add(store.Node(id=2, parent_id=1, iteration=2, status="ok",
                              scores={"mape": 8.0}, strategy="s2"))
        st.add(store.Node(id=3, parent_id=2, iteration=3, status="error", strategy="s3"))
        st.add(store.Node(id=4, parent_id=3, iteration=4, status="error", strategy="s4"))

        assert st.best("mape").id == 2
        assert [n.id for n in st.frontier("mape", 2)] == [2, 1]
        assert st.path_failures(2) == 2  # two consecutive failed descendants
        # Reload from disk (C3).
        st2 = store.ExperimentStore(st.path)
        assert len(st2.nodes) == 4 and st2.best("mape").id == 2


def test_improvement_threshold():
    assert harness.is_real_improvement(9.0, 10.0, 0.5)
    assert not harness.is_real_improvement(9.9, 10.0, 0.5)  # jitter, not progress


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"{fn.__name__} ...", end=" ", flush=True)
        fn()
        print("OK")
    print(f"\n{len(fns)} tests passed")
