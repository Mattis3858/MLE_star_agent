import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from lightgbm import LGBMRegressor

# Per Design Spec: Target is log1p(Sales)
USE_LOG_TARGET = True

def apply_rossmann_transforms(X):
    """
    Implements Design Spec (2C) and (3):
    - Fourier pairs for multi-scale seasonality.
    - Exponential decay for Promo/Holiday pulses.
    - Clipping of skewed time-delta features.
    """
    X = X.copy()
    
    # 1. Fourier Pairs (Periods: 365.25, 182.6, 91.3, 30.4, 7)
    # Use DayOfYear as the base for annual/semi-annual/quarterly/monthly
    # Use DayOfWeek for weekly
    if 'DayOfYear' in X.columns:
        day_of_year = X['DayOfYear']
        for period in [365.25, 182.6, 91.3, 30.4]:
            X[f'fourier_sin_{period}'] = np.sin(2 * np.pi * day_of_year / period)
            X[f'fourier_cos_{period}'] = np.cos(2 * np.pi * day_of_year / period)
    
    if 'DayOfWeek' in X.columns:
        dow = X['DayOfWeek']
        X['fourier_sin_7'] = np.sin(2 * np.pi * dow / 7)
        X['fourier_cos_7'] = np.cos(2 * np.pi * dow / 7)

    # 2. Exponential Decay Impulses (Design Spec 2C)
    if 'PromoStreak' in X.columns:
        X['exp_promo_decay'] = np.exp(-X['PromoStreak'] / 14)
    
    if 'DaysSinceHoliday' in X.columns:
        # Clip to 30 to avoid long-tail spurious splits (Design Spec 3)
        X['DaysSinceHoliday'] = X['DaysSinceHoliday'].clip(0, 30)
        X['exp_since_holiday_decay'] = np.exp(-X['DaysSinceHoliday'] / 30)
        
    if 'DaysUntilHoliday' in X.columns:
        # Clip to 14 (Design Spec 3)
        X['DaysUntilHoliday'] = X['DaysUntilHoliday'].clip(0, 14)
        X['exp_until_holiday_decay'] = np.exp(-X['DaysUntilHoliday'] / 7)

    # 3. CompetitionOpenMonths clip (Design Spec 3)
    if 'CompetitionOpenMonths' in X.columns:
        X['CompetitionOpenMonths'] = X['CompetitionOpenMonths'].clip(0, 24)

    return X

def build_pipeline(feature_columns: list[str], categorical_columns: list[str]):
    """
    Builds the Rossmann Store Sales pipeline.
    """
    
    # Custom transformer for the Rossmann-specific feature engineering
    custom_engineer = FunctionTransformer(apply_rossmann_transforms)
    
    # Categorical encoding
    pre_cat = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            ),
        ],
        remainder="passthrough",
    )
    
    # LightGBM GPU configuration (Design Spec 1)
    # Reduced n_estimators from 3000 to 1000 to prevent timeout/VRAM issues
    model = LGBMRegressor(
        device="gpu",
        max_bin=255,
        objective="regression_l1", 
        n_estimators=1000,
        importance_type="gain",
        min_child_samples=20, 
    )
    
    return Pipeline([
        ("engineer", custom_engineer),
        ("pre", pre_cat),
        ("model", model)
    ])

def param_space(trial):
    """
    Optuna search space defined in Design Spec.
    """
    return {
        "model__num_leaves": trial.suggest_int("model__num_leaves", 31, 255),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 1e-3, 5e-2, log=True),
        "model__min_child_samples": trial.suggest_int("model__min_child_samples", 20, 200),
        "model__feature_fraction": trial.suggest_float("model__feature_fraction", 0.5, 0.9),
        "model__bagging_fraction": trial.suggest_float("model__bagging_fraction", 0.5, 0.9),
        "model__bagging_freq": 5,
        "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 1.0, log=True),
        "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 1.0, log=True),
    }