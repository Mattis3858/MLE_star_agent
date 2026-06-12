"""Harness-owned baseline candidate. Used for the A3 noise-floor estimate and
as the canonical example of the build_pipeline contract shown to the LLM.

Contract (what every candidate module must provide):
    build_pipeline(feature_columns: list[str], categorical_columns: list[str])
        -> UNFITTED sklearn Pipeline whose FINAL step is named "model".
    USE_LOG_TARGET: bool  (optional, default False) - harness wraps the
        pipeline in TransformedTargetRegressor(log1p/expm1) when True.
    param_space(trial) -> dict  (optional) - Optuna search space mapping
        pipeline param names (e.g. "model__n_estimators") to suggestions.

No file IO, no splitting, no scoring, no fitting at import time.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

USE_LOG_TARGET = True


def build_pipeline(feature_columns, categorical_columns):
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.1,
        max_depth=None,
        random_state=0,  # harness overrides with the run seed
    )
    return Pipeline([("pre", pre), ("model", model)])


def param_space(trial):
    return {
        "model__max_iter": trial.suggest_int("model__max_iter", 150, 600),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.03, 0.3, log=True),
        "model__max_leaf_nodes": trial.suggest_int("model__max_leaf_nodes", 15, 255),
        "model__l2_regularization": trial.suggest_float("model__l2_regularization", 1e-3, 10.0, log=True),
    }
