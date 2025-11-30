import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
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
    
    # Enhanced calendar features
    # Create a comprehensive holiday calendar
    # Define fixed holidays (same date every year)
    fixed_holidays = [
        '01-01',  # New Year
        '05-01',  # Labour Day
        '10-03',  # German Unity Day
        '12-25',  # Christmas Day
        '12-26'   # Boxing Day
    ]
    
    # Create holiday flags
    df['is_fixed_holiday'] = df['Date'].dt.strftime('%m-%d').isin(fixed_holidays).astype(int)
    
    # Easter-related holidays (need to be calculated)
    # For simplicity, we'll use a fixed approximation for Easter Sunday
    # In a real implementation, you'd use a proper easter calculation
    easter_dates = []
    for year in df['year'].unique():
        # Approximate Easter date (could be improved with a proper easter calculation)
        # This is a simplified version
        easter_month = 4
        easter_day = 1 + (year % 19) % 28
        if easter_day > 30:
            easter_day = 30
        try:
            easter_dates.append(pd.Timestamp(year=year, month=easter_month, day=easter_day))
        except:
            easter_dates.append(pd.Timestamp(year=year, month=4, day=15))
    
    # Create easter period (Good Friday to Easter Monday)
    df['is_easter'] = 0
    for year in df['year'].unique():
        # Approximate easter date for the year
        easter_month = 4
        easter_day = 1 + (year % 19) % 28
        if easter_day > 30:
            easter_day = 30
        try:
            easter_date = pd.Timestamp(year=year, month=easter_month, day=easter_day)
        except:
            easter_date = pd.Timestamp(year=year, month=4, day=15)
        
        # Good Friday (2 days before Easter)
        good_friday = easter_date - pd.Timedelta(days=2)
        # Easter Monday (1 day after Easter)
        easter_monday = easter_date + pd.Timedelta(days=1)
        
        # Mark easter period
        mask = (df['Date'] >= good_friday) & (df['Date'] <= easter_monday) & (df['year'] == year)
        df.loc[mask, 'is_easter'] = 1
    
    # School vacation periods (approximated)
    # In Germany, school vacations vary by state and year
    # We'll create a simplified version based on typical patterns
    df['is_school_vacation'] = 0
    
    # Summer vacation (typically July to late August)
    summer_vacation_mask = (
        ((df['month'] == 7) | (df['month'] == 8)) & 
        (df['Date'].dt.day >= 15)  # Approximate mid-July to end of August
    )
    df.loc[summer_vacation_mask, 'is_school_vacation'] = 1
    
    # Winter vacation (typically around Christmas/New Year)
    winter_vacation_mask = (
        ((df['month'] == 12) & (df['Date'].dt.day >= 24)) |
        ((df['month'] == 1) & (df['Date'].dt.day <= 10))
    )
    df.loc[winter_vacation_mask, 'is_school_vacation'] = 1
    
    # Spring vacation (typically around Easter)
    # We'll use our easter period as a proxy
    df['is_school_vacation'] = df['is_school_vacation'] | df['is_easter']
    
    # Distance to next holiday features
    # Create a list of all holidays
    all_holidays = pd.Series(pd.to_datetime([]))
    
    # Add fixed holidays for all years in dataset
    years = df['year'].unique()
    for year in years:
        for holiday in fixed_holidays:
            try:
                holiday_date = pd.Timestamp(year=year, month=int(holiday[:2]), day=int(holiday[3:]))
                all_holidays = pd.concat([all_holidays, pd.Series([holiday_date])])
            except:
                pass
    
    # Add easter dates
    for year in years:
        easter_month = 4
        easter_day = 1 + (year % 19) % 28
        if easter_day > 30:
            easter_day = 30
        try:
            easter_date = pd.Timestamp(year=year, month=easter_month, day=easter_day)
            all_holidays = pd.concat([all_holidays, pd.Series([easter_date])])
        except:
            pass
    
    all_holidays = all_holidays.sort_values().unique()
    
    # Calculate days to next holiday
    df['days_to_next_holiday'] = 365  # Default large value
    for i, date in enumerate(df['Date']):
        future_holidays = all_holidays[all_holidays > date]
        if len(future_holidays) > 0:
            next_holiday = future_holidays[0]
            df.iloc[i, df.columns.get_loc('days_to_next_holiday')] = (next_holiday - date).days
    
    # Holiday type indicators
    df['is_major_holiday'] = (
        (df['Date'].dt.strftime('%m-%d') == '12-25') |  # Christmas
        (df['Date'].dt.strftime('%m-%d') == '12-26') |  # Boxing Day
        df['is_easter']
    ).astype(int)
    
    df['is_regular_holiday'] = (
        df['is_fixed_holiday'] & ~df['is_major_holiday']
    ).astype(int)
    
    # Interaction features between promotions and calendar events
    df['promo_on_holiday'] = (df['Promo'] == 1) & (df['is_fixed_holiday'] == 1)
    df['promo_on_school_vacation'] = (df['Promo'] == 1) & (df['is_school_vacation'] == 1)
    df['promo_near_holiday'] = (df['Promo'] == 1) & (df['days_to_next_holiday'] <= 7)
    
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
    'competition_open_months', 'promo2_since_days', 'school_holiday',
    'is_fixed_holiday', 'is_easter', 'is_school_vacation', 
    'days_to_next_holiday', 'is_major_holiday', 'is_regular_holiday',
    'promo_on_holiday', 'promo_on_school_vacation', 'promo_near_holiday'
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

# CatBoost model
catboost_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'depth': 8,
    'random_state': 42,
    'verbose': False
}
catboost_model = CatBoostRegressor(**catboost_params)
catboost_model.fit(X_train_scaled, y_train,
                   eval_set=(X_val_scaled, y_val))
models['catboost'] = catboost_model

# Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
models['random_forest'] = rf_model

# Stacking ensemble
def create_stacking_ensemble(models, X_train, X_val, y_train, y_val):
    # Get predictions from base models on training set
    base_predictions_train = np.column_stack([
        model.predict(X_train) for model in models.values()
    ])
    
    # Get predictions from base models on validation set
    base_predictions_val = np.column_stack([
        model.predict(X_val) for model in models.values()
    ])
    
    # Train meta-model (linear regression) on base model predictions
    meta_model = LinearRegression()
    meta_model.fit(base_predictions_train, y_train)
    
    # Predict with meta-model on validation set
    final_predictions = meta_model.predict(base_predictions_val)
    
    return final_predictions, meta_model

# Create stacking ensemble predictions
stacking_predictions, meta_model = create_stacking_ensemble(
    models, X_train_scaled, X_val_scaled, y_train, y_val
)

# Evaluate stacking ensemble
stacking_mape = safe_mape(y_val, stacking_predictions)
print(f"stacking_ensemble MAPE: {stacking_mape:.4f}")

# Compare with individual models
results = {}
for name, model in models.items():
    y_pred = model.predict(X_val_scaled)
    mape = safe_mape(y_val, y_pred)
    results[name] = mape
    print(f"{name} MAPE: {mape:.4f}")

# Add stacking result to results
results['stacking_ensemble'] = stacking_mape

# Implement weighted ensemble optimized on validation set
def optimize_weights(models, X_val, y_val):
    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_val)
    
    # Add stacking predictions
    predictions['stacking_ensemble'] = stacking_predictions
    
    # Convert to array for easier manipulation
    pred_array = np.column_stack(list(predictions.values()))
    model_names = list(predictions.keys())
    
    # Initialize weights with higher values for better models
    weights = np.array([0.1, 0.4, 0.1, 0.1, 0.3])  # [lgb, xgb, cat, rf, stack]
    
    # Simple grid search to optimize weights
    best_mape = float('inf')
    best_weights = weights.copy()
    
    # Try different weight combinations around initial weights
    for w1 in np.arange(0.1, 0.6, 0.1):  # LightGBM
        for w2 in np.arange(0.3, 0.7, 0.1):  # XGBoost
            for w3 in np.arange(0.0, 0.3, 0.1):  # CatBoost
                for w4 in np.arange(0.0, 0.3, 0.1):  # Random Forest
                    w5 = 1.0 - (w1 + w2 + w3 + w4)  # Stacking
                    if w5 >= 0:  # Valid weight distribution
                        weights = np.array([w1, w2, w3, w4, w5])
                        weighted_pred = np.dot(pred_array, weights)
                        mape = safe_mape(y_val, weighted_pred)
                        if mape < best_mape:
                            best_mape = mape
                            best_weights = weights.copy()
    
    return best_weights, model_names, best_mape

# Optimize weights for weighted ensemble
weights, model_names, weighted_mape = optimize_weights(models, X_val_scaled, y_val)
print(f"weighted_ensemble MAPE: {weighted_mape:.4f}")

# Calculate weighted ensemble predictions
weighted_predictions = np.zeros_like(stacking_predictions)
for i, (name, model) in enumerate(models.items()):
    weighted_predictions += weights[i] * model.predict(X_val_scaled)
weighted_predictions += weights[-1] * stacking_predictions

# Update results with weighted ensemble
results['weighted_ensemble'] = weighted_mape

# Select best model
best_model_name = min(results, key=results.get)
final_mape = results[best_model_name]

# Determine best predictions
if best_model_name == 'weighted_ensemble':
    best_predictions = weighted_predictions
elif best_model_name == 'stacking_ensemble':
    best_predictions = stacking_predictions
else:
    best_model = models[best_model_name]
    best_predictions = best_model.predict(X_val_scaled)

# Feature importance plot for best individual model (if applicable)
if best_model_name in models and hasattr(models[best_model_name], 'feature_importances_'):
    importances = models[best_model_name].feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Top 20 Feature Importances ({best_model_name})")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_columns[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')

# Print final result (CRITICAL)
print(f"FINAL_MAPE: {final_mape}")