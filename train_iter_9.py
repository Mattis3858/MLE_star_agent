import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Detect GPU availability
def get_tree_method():
    try:
        from numba import cuda
        if cuda.is_available():
            return 'gpu_hist'
    except ImportError:
        pass
    return 'hist'

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'gpu'
    except ImportError:
        pass
    return 'cpu'

# Safe MAPE implementation
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero:
    y_true_safe = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

# Data preprocessing pipeline
def automated_preprocessing(train_path, store_path):
    # Read datasets
    train_df = pd.read_csv(train_path, parse_dates=['Date'])
    store_df = pd.read_csv(store_path)
    
    # Merge datasets on 'Store' key
    df = pd.merge(train_df, store_df, on='Store', how='left')
    
    # Agentic decision: Filter based on business rules
    # Remove closed stores (Sales = 0 when Store is closed)
    df = df[(df['Sales'] > 0) & (df['Open'] == 1)]
    
    # Automated missing value handling
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Competition distance: fill large values for stores without competition
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].max() * 2)
    
    # Competition open date: create flag for missing competition
    df['HasCompetition'] = ~df['CompetitionOpenSinceYear'].isna()
    
    return df

# Calendar/date features
def generate_calendar_features(df):
    # Basic temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    
    # Advanced temporal features (agentic search)
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsMonthStart'] = (df['Date'].dt.day == 1).astype(int)
    df['IsMonthEnd'] = (df['Date'].dt.day == df['Date'].dt.days_in_month).astype(int)
    
    # Seasonal features
    df['Season'] = (df['Month'] % 12 + 3) // 3
    df['IsHolidayPeriod'] = ((df['Month'] == 12) | (df['Month'] == 1)).astype(int)
    
    return df

# Store-level features
def generate_store_features(df):
    # Store characteristics
    store_encoders = {
        'StoreType': LabelEncoder(),
        'Assortment': LabelEncoder(),
        'StateHoliday': LabelEncoder()
    }
    
    for col, encoder in store_encoders.items():
        # Fix data type consistency: ensure all values are strings before encoding
        df[col] = df[col].astype(str)
        df[col + '_Encoded'] = encoder.fit_transform(df[col].fillna('Unknown'))
    
    # Competition features (agentic refinement)
    df['CompetitionMonths'] = ((df['Year'] - df['CompetitionOpenSinceYear']) * 12 + 
                              (df['Month'] - df['CompetitionOpenSinceMonth'])).clip(lower=0)
    
    # Promo2 duration
    df['Promo2Weeks'] = ((df['Year'] - df['Promo2SinceYear']) * 52 + 
                        (df['Week'] - df['Promo2SinceWeek'])).clip(lower=0)
    
    # Distance-based features
    df['CompetitionDistanceLog'] = np.log1p(df['CompetitionDistance'])
    
    return df

# Promotion/holiday features
def generate_promotion_features(df):
    # Promo interaction features
    df['PromoWithHoliday'] = (df['Promo'] & (df['StateHoliday'] != '0')).astype(int)
    df['PromoWithSchoolHoliday'] = (df['Promo'] & df['SchoolHoliday']).astype(int)
    
    # Promo2 extended features
    promo2_months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    
    for month_num, month_name in promo2_months.items():
        df[f'Promo2_{month_name}'] = df['PromoInterval'].str.contains(month_name).fillna(False).astype(int)
    
    return df

# Feature engineering pipeline
def feature_engineering_pipeline(df):
    df = generate_calendar_features(df)
    df = generate_store_features(df)
    df = generate_promotion_features(df)
    
    # Select final features
    feature_columns = [
        'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday',
        'Year', 'Month', 'Week', 'DayOfYear', 'Quarter',
        'IsWeekend', 'IsMonthStart', 'IsMonthEnd', 'Season', 'IsHolidayPeriod',
        'StoreType_Encoded', 'Assortment_Encoded', 'StateHoliday_Encoded',
        'CompetitionDistance', 'CompetitionMonths', 'Promo2Weeks',
        'CompetitionDistanceLog', 'HasCompetition',
        'PromoWithHoliday', 'PromoWithSchoolHoliday'
    ]
    
    # Add Promo2 monthly features
    promo2_features = [col for col in df.columns if col.startswith('Promo2_')]
    feature_columns.extend(promo2_features)
    
    return df[feature_columns], df['Sales']

# Model creation
def create_models():
    tree_method = get_tree_method()
    device = get_device()
    
    models = {
        'xgb': xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method=tree_method,
            random_state=42
        ),
        'lgbm': lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            device=device,
            random_state=42
        )
    }
    return models

# Stacking ensemble with meta-learner
def create_stacking_ensemble(models, X_train, y_train, X_val):
    # Get predictions from base models on training set
    base_predictions_train = []
    base_predictions_val = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        base_predictions_train.append(train_pred)
        base_predictions_val.append(val_pred)
    
    # Stack predictions as new features
    X_meta_train = np.column_stack(base_predictions_train)
    X_meta_val = np.column_stack(base_predictions_val)
    
    # Train meta-learner
    meta_learner = LinearRegression()
    meta_learner.fit(X_meta_train, y_train)
    
    # Final predictions
    final_predictions = meta_learner.predict(X_meta_val)
    
    return final_predictions, meta_learner

# Weighted averaging ensemble
def create_weighted_ensemble(models, X_train, y_train, X_val, y_val):
    predictions = {}
    weights = {}
    
    # Train models and get individual predictions
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        predictions[name] = pred
        mape = safe_mape(y_val, pred)
        # Convert MAPE to weight (inverse relationship)
        weights[name] = 1 / (mape + 1e-8)  # Small epsilon to avoid division by zero
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {name: weight/total_weight for name, weight in weights.items()}
    
    # Compute weighted average
    final_predictions = np.zeros_like(list(predictions.values())[0])
    for name, pred in predictions.items():
        final_predictions += weights[name] * pred
    
    return final_predictions, weights

# Main training function
def train_model():
    # Load and preprocess data
    df = automated_preprocessing('train.csv', 'store.csv')
    
    # Feature engineering
    X, y = feature_engineering_pipeline(df)
    
    # Time series split (last 6 weeks for validation)
    split_date = df['Date'].max() - pd.Timedelta(weeks=6)
    train_mask = df['Date'] <= split_date
    val_mask = df['Date'] > split_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    # Handle any remaining missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Convert back to DataFrame to preserve column names
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_val.columns)
    
    # Train models
    models = create_models()
    
    # Implement stacking ensemble
    print("Training stacking ensemble...")
    stacked_predictions, meta_learner = create_stacking_ensemble(
        models, X_train_imputed, y_train, X_val_imputed
    )
    stacked_mape = safe_mape(y_val, stacked_predictions)
    print(f"Stacking ensemble MAPE: {stacked_mape:.4f}")
    
    # Implement weighted averaging ensemble
    print("Training weighted averaging ensemble...")
    weighted_predictions, weights = create_weighted_ensemble(
        models, X_train_imputed, y_train, X_val_imputed, y_val
    )
    weighted_mape = safe_mape(y_val, weighted_predictions)
    print(f"Weighted averaging ensemble MAPE: {weighted_mape:.4f}")
    print(f"Model weights: {weights}")
    
    # Select best ensemble method
    if stacked_mape <= weighted_mape:
        final_mape = stacked_mape
        print("Selected stacking ensemble as best method")
    else:
        final_mape = weighted_mape
        print("Selected weighted averaging ensemble as best method")
    
    print(f"FINAL_MAPE: {final_mape}")

if __name__ == "__main__":
    train_model()