import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import warnings
warnings.filterwarnings("ignore")

# Check for GPU availability
def check_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

GPU_AVAILABLE = check_gpu()

# Safe MAPE implementation
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero:
    y_true_safe = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

# Load and merge data
def load_data():
    train_df = pd.read_csv('train.csv', low_memory=False)
    store_df = pd.read_csv('store.csv', low_memory=False)
    
    # Merge datasets
    df = pd.merge(train_df, store_df, on='Store', how='left')
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index and ensure it's in datetime format
    df.set_index('Date', inplace=True)
    # Explicitly convert index to datetime format for time series operations
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Validate that the index is in correct datetime format
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index is not in datetime format. Time series operations cannot be performed.")
    
    # Filter data as per requirements
    df = df[(df['Sales'] > 0) & (df['Open'] == 1)]
    
    # Convert StateHoliday to string
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    return df

# Feature engineering
def engineer_features(df):
    # Optimize data types before feature engineering
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Apply log transform to the target variable
    df['Sales_log'] = np.log1p(df['Sales'])
    
    # Temporal features
    df['Year'] = df.index.year.astype('int16')
    df['Month'] = df.index.month.astype('int8')
    df['Day'] = df.index.day.astype('int8')
    df['DayOfWeek'] = df.index.dayofweek.astype('int8')
    df['WeekOfYear'] = df.index.isocalendar().week.astype('int8')
    df['Quarter'] = df.index.quarter.astype('int8')
    
    # Cyclical encoding
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7).astype('float32')
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7).astype('float32')
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12).astype('float32')
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12).astype('float32')
    
    # Competition features
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    
    # Promo2 features
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)
    
    # Days since competition opened
    df['CompetitionOpenSince'] = pd.to_datetime(
        df['CompetitionOpenSinceYear'].astype(str) + '-' + 
        df['CompetitionOpenSinceMonth'].astype(str) + '-01',
        errors='coerce'
    )
    df['DaysSinceCompetitionOpen'] = (df.index - df['CompetitionOpenSince']).dt.days
    df['DaysSinceCompetitionOpen'] = df['DaysSinceCompetitionOpen'].fillna(0)
    df['DaysSinceCompetitionOpen'] = df['DaysSinceCompetitionOpen'].apply(lambda x: 0 if x < 0 else x)
    
    # Promo2 related features - Fixed implementation
    # Create a boolean flag for Promo2 active status based on year and week comparison
    df['Promo2Active'] = (
        (df['Promo2'] == 1) & 
        (df['Promo2SinceYear'] > 0) & 
        (
            (df['Year'] > df['Promo2SinceYear']) | 
            ((df['Year'] == df['Promo2SinceYear']) & (df['WeekOfYear'] >= df['Promo2SinceWeek']))
        )
    ).astype('int8')
    
    # For date calculation, use the first day of the Promo2SinceYear as a safe approximation
    # when Promo2SinceWeek is missing or zero
    df['Promo2Start'] = pd.to_datetime(
        df['Promo2SinceYear'].astype(int).astype(str) + '-01-01',
        errors='coerce'
    )
    
    # Adjust for weeks where data is available
    has_week_info = (df['Promo2SinceWeek'] > 0) & (df['Promo2SinceYear'] > 0)
    df.loc[has_week_info, 'Promo2Start'] = pd.to_datetime(
        df.loc[has_week_info, 'Promo2SinceYear'].astype(int).astype(str) + '-01-01'
    ) + pd.to_timedelta((df.loc[has_week_info, 'Promo2SinceWeek'] - 1) * 7, unit='days')
    
    df['DaysSincePromo2Start'] = (df.index - df['Promo2Start']).dt.days
    df['DaysSincePromo2Start'] = df['DaysSincePromo2Start'].fillna(0)
    df['DaysSincePromo2Start'] = df['DaysSincePromo2Start'].apply(lambda x: 0 if x < 0 else x)
    
    # Is promo interval month
    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['MonthStr'] = df['Month'].map(month_map)
    df['IsPromoMonth'] = df.apply(lambda row: 1 if row['PromoInterval'] != 'None' and row['MonthStr'] in row['PromoInterval'] else 0, axis=1).astype('int8')
    
    # Sort by Store and Date for lag features
    df.sort_values(['Store', 'Date'], inplace=True)
    
    # Extended lag features for sales
    lag_days = [1, 2, 3, 5, 7, 14, 21, 28]
    for lag in lag_days:
        df[f'Sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag).astype('float32')
        df[f'Customers_lag_{lag}'] = df.groupby('Store')['Customers'].shift(lag).astype('float32')
    
    # Rolling features with multiple windows
    windows = [3, 7, 14, 30, 60, 90]
    for window in windows:
        df[f'Sales_rolling_mean_{window}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        ).astype('float32')
        df[f'Sales_rolling_std_{window}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).astype('float32')
        df[f'Customers_rolling_mean_{window}'] = df.groupby('Store')['Customers'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        ).astype('float32')
    
    # Exponential weighted moving averages
    ewm_spans = [7, 14, 30]
    for span in ewm_spans:
        df[f'Sales_ewm_mean_{span}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.ewm(span=span, min_periods=1).mean()
        ).astype('float32')
    
    # Mean encoding
    store_mean_sales = df.groupby('Store')['Sales'].mean()
    df['Store_mean_sales'] = df['Store'].map(store_mean_sales).astype('float32')
    
    dayofweek_mean_sales = df.groupby('DayOfWeek')['Sales'].mean()
    df['DayOfWeek_mean_sales'] = df['DayOfWeek'].map(dayofweek_mean_sales).astype('float32')
    
    month_mean_sales = df.groupby('Month')['Sales'].mean()
    df['Month_mean_sales'] = df['Month'].map(month_mean_sales).astype('float32')
    
    # Fill NaN values created by lag features
    df.fillna(0, inplace=True)
    
    return df

# Prepare features and target
def prepare_data(df):
    # Categorical columns
    categorical_cols = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 
                        'SchoolHoliday', 'StoreType', 'Assortment', 'Promo2', 
                        'PromoInterval']
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Select features for modeling
    feature_cols = [
        'Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
        'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter',
        'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos',
        'DaysSinceCompetitionOpen', 'DaysSincePromo2Start', 'IsPromoMonth',
        'Sales_lag_1', 'Sales_lag_2', 'Sales_lag_3', 'Sales_lag_5', 
        'Sales_lag_7', 'Sales_lag_14', 'Sales_lag_21', 'Sales_lag_28',
        'Customers_lag_1', 'Customers_lag_2', 'Customers_lag_3', 'Customers_lag_5',
        'Customers_lag_7', 'Customers_lag_14', 'Customers_lag_21', 'Customers_lag_28',
        'Sales_rolling_mean_3', 'Sales_rolling_std_3',
        'Sales_rolling_mean_7', 'Sales_rolling_std_7',
        'Sales_rolling_mean_14', 'Sales_rolling_std_14',
        'Sales_rolling_mean_30', 'Sales_rolling_std_30',
        'Sales_rolling_mean_60', 'Sales_rolling_std_60',
        'Sales_rolling_mean_90', 'Sales_rolling_std_90',
        'Customers_rolling_mean_3', 'Customers_rolling_mean_7',
        'Customers_rolling_mean_14', 'Customers_rolling_mean_30',
        'Customers_rolling_mean_60', 'Customers_rolling_mean_90',
        'Sales_ewm_mean_7', 'Sales_ewm_mean_14', 'Sales_ewm_mean_30',
        'Store_mean_sales', 'DayOfWeek_mean_sales', 'Month_mean_sales'
    ]
    
    # Remove any columns not in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Sales_log']  # Use log-transformed target
    
    return X, y, feature_cols

# Stacking ensemble with non-linear meta-learner and weighted ensemble
def train_stacking_ensemble(X, y):
    # Time series split (last 6 weeks for validation)
    split_date = X.index.max() - pd.Timedelta(weeks=6)
    
    # Use integer positions instead of boolean indexing to reduce memory usage
    train_mask = X.index <= split_date
    val_mask = X.index > split_date
    
    # Get integer positions
    train_positions = np.where(train_mask)[0]
    val_positions = np.where(val_mask)[0]
    
    # Use iloc for memory-efficient splitting
    X_train = X.iloc[train_positions]
    X_val = X.iloc[val_positions]
    y_train = y.iloc[train_positions]
    y_val = y.iloc[val_positions]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    # Base models
    models = {
        'LightGBM': lgb.LGBMRegressor(
            device='gpu' if GPU_AVAILABLE else 'cpu',
            random_state=42,
            n_estimators=1000
        ),
        'XGBoost': xgb.XGBRegressor(
            tree_method='hist',
            random_state=42,
            n_estimators=1000
        ),
        'CatBoost': cb.CatBoostRegressor(
            verbose=False,
            random_state=42,
            iterations=1000,
            task_type='GPU' if GPU_AVAILABLE else 'CPU'
        )
    }
    
    # Out-of-fold predictions for stacking
    oof_predictions = pd.DataFrame(index=X_train.index)
    val_predictions = pd.DataFrame(index=X_val.index)
    
    # Train base models and get predictions
    model_weights = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Get predictions on training set (for stacking)
        train_preds = model.predict(X_train_scaled)
        oof_predictions[name] = train_preds
        
        # Get predictions on validation set
        val_preds = model.predict(X_val_scaled)
        val_predictions[name] = val_preds
        
        # Calculate individual model MAPE for weighting
        val_preds_original = np.expm1(val_preds)
        y_val_original = np.expm1(y_val)
        model_mape = safe_mape(y_val_original, val_preds_original)
        model_weights[name] = 1.0 / (model_mape + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # Weighted ensemble
    weighted_val_preds = np.zeros(len(X_val))
    for name, weight in model_weights.items():
        weighted_val_preds += weight * val_predictions[name]
    
    # Transform predictions back to original scale
    weighted_val_preds_original = np.expm1(weighted_val_preds)
    y_val_original = np.expm1(y_val)
    
    # Calculate MAPE for weighted ensemble
    weighted_mape = safe_mape(y_val_original, weighted_val_preds_original)
    print(f"Weighted Ensemble MAPE: {weighted_mape:.4f}")
    
    # Print model weights
    print("Model weights:")
    for name, weight in model_weights.items():
        print(f"  {name}: {weight:.4f}")
    
    # Non-linear meta-learner (LightGBM)
    print("Training non-linear meta-learner (LightGBM)...")
    meta_learner = lgb.LGBMRegressor(
        device='gpu' if GPU_AVAILABLE else 'cpu',
        random_state=42,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3
    )
    
    meta_learner.fit(oof_predictions, y_train)
    stacked_val_preds = meta_learner.predict(val_predictions)
    
    # Transform predictions back to original scale
    stacked_val_preds_original = np.expm1(stacked_val_preds)
    
    # Calculate MAPE for non-linear stacking
    stacked_mape = safe_mape(y_val_original, stacked_val_preds_original)
    print(f"Non-linear Stacking MAPE: {stacked_mape:.4f}")
    
    # Use the better of the two approaches
    if stacked_mape < weighted_mape:
        final_mape = stacked_mape
        print("Using non-linear stacking result")
    else:
        final_mape = weighted_mape
        print("Using weighted ensemble result")
    
    return final_mape

# Main execution
def main():
    print("Loading data...")
    df = load_data()
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Preparing data...")
    X, y, feature_cols = prepare_data(df)
    
    print("Training stacking ensemble...")
    final_mape = train_stacking_ensemble(X, y)
    
    print(f"FINAL_MAPE: {final_mape}")

if __name__ == "__main__":
    main()