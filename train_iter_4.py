import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Set matplotlib backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Detect hardware acceleration
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'gpu'
    except ImportError:
        pass
    return 'cpu'

DEVICE = get_device()

# Safe MAPE implementation
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    y_true_safe = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

# Load and merge data
train_df = pd.read_csv('train.csv', parse_dates=['Date'])
store_df = pd.read_csv('store.csv')

# Merge datasets
df = pd.merge(train_df, store_df, on='Store', how='left')

# Data filtering (CRITICAL)
df = df[(df['Sales'] > 0) & (df['Open'] == 1)].copy()

# Fix data types (CRITICAL)
df['StateHoliday'] = df['StateHoliday'].astype(str)

# Feature engineering
def create_features(df):
    # Calendar features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.day_of_year
    df['quarter'] = df['Date'].dt.quarter
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['is_year_end'] = (df['Date'].dt.month == 12).astype(int)
    
    # Store-level features
    le_store_type = LabelEncoder()
    le_assortment = LabelEncoder()
    df['store_type_encoded'] = le_store_type.fit_transform(df['StoreType'])
    df['assortment_encoded'] = le_assortment.fit_transform(df['Assortment'])
    df['competition_distance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    df['competition_missing'] = df['CompetitionDistance'].isna().astype(int)
    
    # Competition timing features
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(method='ffill')
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(method='ffill')
    df['competition_open_months'] = (
        (df['year'] - df['CompetitionOpenSinceYear']) * 12 + 
        (df['month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0)
    
    # Promotional features
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    promo2_start = pd.to_datetime(
        df['Promo2SinceYear'].astype(str) + '-' + 
        df['Promo2SinceWeek'].astype(str) + '-1', 
        format='%Y-%W-%w', errors='coerce'
    )
    df['promo2_since_days'] = (df['Date'] - promo2_start).dt.days.fillna(0)
    
    # Holiday features
    state_holidays = ['DE_BW', 'DE_BY', 'DE_BE', 'DE_BB', 'DE_HB', 'DE_HH', 
                      'DE_HE', 'DE_MV', 'DE_NI', 'DE_NW', 'DE_RP', 'DE_SL', 
                      'DE_SN', 'DE_ST', 'DE_SH', 'DE_TH']
    for state in state_holidays:
        df[f'holiday_{state}'] = (df['StateHoliday'] == state).astype(int)
    df['school_holiday'] = df['SchoolHoliday']
    
    return df

# Apply feature engineering
df = create_features(df)

# Define features for modeling
feature_columns = [
    'Store', 'DayOfWeek', 'Promo', 'year', 'month', 'week', 'day_of_week',
    'day_of_year', 'quarter', 'is_month_start', 'is_month_end',
    'is_quarter_start', 'is_year_end', 'store_type_encoded',
    'assortment_encoded', 'competition_distance', 'competition_missing',
    'competition_open_months', 'promo2_since_days', 'school_holiday'
] + [f'holiday_{state}' for state in ['DE_BW', 'DE_BY', 'DE_BE', 'DE_BB', 'DE_HB', 'DE_HH', 
                                      'DE_HE', 'DE_MV', 'DE_NI', 'DE_NW', 'DE_RP', 'DE_SL', 
                                      'DE_SN', 'DE_ST', 'DE_SH', 'DE_TH']]

# Time-based split (last 6 weeks for validation)
split_date = df['Date'].max() - pd.Timedelta(weeks=6)
train_data = df[df['Date'] < split_date]
val_data = df[df['Date'] >= split_date]

# Prepare features and targets
X_train = train_data[feature_columns]
y_train = train_data['Sales']
X_val = val_data[feature_columns]
y_val = val_data['Sales']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Model training with hardware acceleration
models = {}

# LightGBM model
lgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 8,
    'num_leaves': 32,
    'random_state': 42,
    'device': DEVICE,
    'verbosity': -1
}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)])
models['lightgbm'] = lgb_model

# XGBoost model
xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 8,
    'random_state': 42,
    'tree_method': 'hist'
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train_scaled, y_train, 
              eval_set=[(X_val_scaled, y_val)], 
              verbose=False)
models['xgboost'] = xgb_model

# Model evaluation
results = {}
for name, model in models.items():
    y_pred = model.predict(X_val_scaled)
    mape = safe_mape(y_val, y_pred)
    results[name] = mape
    print(f"{name} MAPE: {mape:.4f}")

# Select best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
final_mape = results[best_model_name]

# Feature importance plot for best model
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Top 20 Feature Importances ({best_model_name})")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')

# Print final result (CRITICAL)
print(f"FINAL_MAPE: {final_mape}")