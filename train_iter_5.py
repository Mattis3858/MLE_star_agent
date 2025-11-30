import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
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

# Weight optimization for ensemble
def optimize_weights(y_true, pred1, pred2):
    best_mape = float('inf')
    best_weight = 0.0
    
    # Try weights from 0 to 1 in steps of 0.05
    for w in np.arange(0.0, 1.05, 0.05):
        ensemble_pred = w * pred1 + (1 - w) * pred2
        mape = safe_mape(y_true, ensemble_pred)
        if mape < best_mape:
            best_mape = mape
            best_weight = w
            
    return best_weight, best_mape

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
    predictions = {}
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_imputed, y_train)
        pred = model.predict(X_val_imputed)
        predictions[name] = pred
        
        # Calculate MAPE
        mape = safe_mape(y_val, pred)
        results[name] = mape
        print(f"{name} MAPE: {mape:.4f}")
        
        # Plot feature importance for XGBoost
        if name == 'xgb':
            feature_importance = model.feature_importances_
            features = X_train_imputed.columns
            importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
            importance_df = importance_df.sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances (XGBoost)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
    
    # Optimize ensemble weights
    xgb_pred = predictions['xgb']
    lgbm_pred = predictions['lgbm']
    
    print("\nOptimizing ensemble weights...")
    best_weight, ensemble_mape = optimize_weights(y_val, xgb_pred, lgbm_pred)
    
    print(f"Optimal XGBoost weight: {best_weight:.2f}")
    print(f"Optimal LightGBM weight: {1 - best_weight:.2f}")
    print(f"Ensemble MAPE: {ensemble_mape:.4f}")
    
    # Determine final MAPE (ensemble if better than individual models)
    individual_mapes = list(results.values())
    best_individual_mape = min(individual_mapes)
    
    if ensemble_mape < best_individual_mape:
        final_mape = ensemble_mape
        print("Ensemble performs better than individual models.")
    else:
        final_mape = best_individual_mape
        print("Individual model performs better than ensemble.")
    
    print(f"FINAL_MAPE: {final_mape}")

if __name__ == "__main__":
    train_model()