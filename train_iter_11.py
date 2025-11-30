import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

def safe_mape(y_true, y_pred):
    """MAPE that handles zero sales without infinite errors"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return np.mean(np.abs(y_pred)) * 100
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_and_merge_data():
    """Load and merge train and store data"""
    train_df = pd.read_csv('train.csv', parse_dates=['Date'])
    store_df = pd.read_csv('store.csv')
    df = pd.merge(train_df, store_df, on='Store', how='left')
    return df

def preprocess_data(df):
    """Preprocess the merged data according to specifications"""
    # Convert StateHoliday to string
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    # Filter out records with Sales <= 0 or Open == 0
    df = df[(df['Sales'] > 0) & (df['Open'] == 1)].copy()
    
    # Handle missing values
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
    df['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    df['Promo2SinceWeek'].fillna(0, inplace=True)
    df['Promo2SinceYear'].fillna(0, inplace=True)
    df['PromoInterval'].fillna('no_promo_interval', inplace=True)
    
    return df

def engineer_features(df):
    """Engineer features as per specification"""
    # Calendar features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
    
    # Fourier features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Competition features
    df['competition_open_months'] = (
        (df['year'] - df['CompetitionOpenSinceYear']) * 12 + 
        (df['month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0)
    
    # Promo2 features
    df['promo2_weeks'] = (
        (df['year'] - df['Promo2SinceYear']) * 52 + 
        (df['week'] - df['Promo2SinceWeek'])
    ).clip(lower=0)
    
    # Store type encodings
    df['is_assortment_b'] = (df['Assortment'] == 'b').astype(int)
    df['is_assortment_c'] = (df['Assortment'] == 'c').astype(int)
    
    # Mean encoding
    store_sales_mean = df.groupby('Store')['Sales'].mean()
    df['store_sales_mean_encoded'] = df['Store'].map(store_sales_mean)
    
    # Day-of-week encoding by store
    dow_encoding = df.groupby(['Store', 'day_of_week'])['Sales'].mean()
    df['dow_store_encoded'] = df.apply(
        lambda x: dow_encoding.get((x['Store'], x['day_of_week']), store_sales_mean[x['Store']]), 
        axis=1
    )
    
    # Enhanced calendar features
    # Holiday effects
    df['is_state_holiday'] = (df['StateHoliday'] != '0').astype(int)
    df['is_public_holiday'] = (df['StateHoliday'] == 'a').astype(int)
    df['is_easter_holiday'] = (df['StateHoliday'] == 'b').astype(int)
    df['is_christmas_holiday'] = (df['StateHoliday'] == 'c').astype(int)
    
    # School vacation patterns
    df['is_school_vacation'] = df['SchoolHoliday'].astype(int)
    
    # Enhanced temporal features
    df['days_from_start'] = (df['Date'] - df['Date'].min()).dt.days
    
    # Distance to next holiday (simplified approach)
    # Create a holiday flag series
    holiday_dates = df[df['is_state_holiday'] == 1]['Date'].unique()
    holiday_dates = pd.to_datetime(holiday_dates)
    holiday_dates = pd.Series(holiday_dates).sort_values()
    
    # For each date, find the next holiday
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    df_sorted['next_holiday_in_days'] = np.nan
    
    for i, date in enumerate(df_sorted['Date'].values):
        future_holidays = holiday_dates[holiday_dates >= pd.to_datetime(date)]
        if len(future_holidays) > 0:
            next_holiday = future_holidays.min()
            df_sorted.loc[i, 'next_holiday_in_days'] = (next_holiday - pd.to_datetime(date)).days
        else:
            # If no future holiday, use a large number
            df_sorted.loc[i, 'next_holiday_in_days'] = 365
    
    # Merge back to original df
    df = df.merge(df_sorted[['Date', 'Store', 'next_holiday_in_days']], on=['Date', 'Store'], how='left')
    
    # Promotion calendar features
    df['promo_running'] = df['Promo'].astype(int)
    df['promo2_running'] = ((df['Promo2'] == 1) & 
                           (df['year'] > df['Promo2SinceYear']) | 
                           ((df['year'] == df['Promo2SinceYear']) & 
                            (df['week'] >= df['Promo2SinceWeek']))).astype(int)
    
    # Promo interval handling
    promo_intervals = {
        'Jan,Apr,Jul,Oct': 1,
        'Feb,May,Aug,Nov': 2,
        'Mar,Jun,Sept,Dec': 3,
        'no_promo_interval': 0
    }
    df['promo_interval_encoded'] = df['PromoInterval'].map(promo_intervals).fillna(0)
    
    # Month-specific promo flag
    df['is_promo_month'] = 0
    for interval, months in [('Jan,Apr,Jul,Oct', [1,4,7,10]), 
                            ('Feb,May,Aug,Nov', [2,5,8,11]), 
                            ('Mar,Jun,Sept,Dec', [3,6,9,12])]:
        mask = (df['PromoInterval'] == interval) & (df['month'].isin(months))
        df.loc[mask, 'is_promo_month'] = 1
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'].isin([5, 6])).astype(int)  # Saturday=5, Sunday=6
    
    # Beginning and end of month effects
    df['is_month_beginning'] = (df['Date'].dt.day <= 7).astype(int)
    df['is_month_ending'] = (df['Date'].dt.day >= 24).astype(int)
    
    # Seasonal indicators
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    
    return df

def prepare_features(df):
    """Prepare final feature set"""
    # Categorical encoding
    le = LabelEncoder()
    categorical_features = ['Store', 'DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']
    for col in categorical_features:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Select features for modeling
    feature_columns = [
        'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear',
        'PromoInterval', 'year', 'month', 'week', 'day_of_week', 'day_of_year',
        'quarter', 'is_month_start', 'is_month_end', 'is_quarter_start',
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'competition_open_months', 'promo2_weeks', 'is_assortment_b',
        'is_assortment_c', 'store_sales_mean_encoded', 'dow_store_encoded',
        # New features
        'is_state_holiday', 'is_public_holiday', 'is_easter_holiday', 'is_christmas_holiday',
        'is_school_vacation', 'days_from_start', 'next_holiday_in_days',
        'promo_running', 'promo2_running', 'promo_interval_encoded', 'is_promo_month',
        'is_weekend', 'is_month_beginning', 'is_month_ending',
        'is_spring', 'is_summer', 'is_autumn', 'is_winter'
    ]
    
    return df[feature_columns], df['Sales']

def time_series_split(df, validation_months=6):
    """Split data for time series validation"""
    # Sort by date
    df = df.sort_values('Date')
    
    # Determine split date
    max_date = df['Date'].max()
    split_date = max_date - pd.DateOffset(months=validation_months)
    
    # Split data
    train_df = df[df['Date'] <= split_date]
    val_df = df[df['Date'] > split_date]
    
    return train_df, val_df

def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with hardware acceleration"""
    # Detect GPU availability using a more reliable method
    try:
        # Try to use lightgbm's built-in GPU support detection
        import lightgbm as lgb
        device = 'gpu' if lgb.get_device_name(0) is not None else 'cpu'
    except (AttributeError, Exception):
        # Fallback to default CPU if GPU detection fails
        device = 'cpu'
    
    # Prepare categorical feature indices
    categorical_feature_indices = [X_train.columns.get_loc(c) for c in X_train.columns if c in [
        'Store', 'DayOfWeek', 'StateHoliday', 'StoreType', 'Assortment', 'PromoInterval'
    ]]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        categorical_feature=categorical_feature_indices
    )
    
    val_data = lgb.Dataset(
        X_val, 
        label=y_val,
        categorical_feature=categorical_feature_indices,
        reference=train_data
    )
    
    # Define parameters - removed verbose and using verbosity in params
    params = {
        'objective': 'regression',
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,  # Correct way to control verbosity in LightGBM
        'device': device,
        'random_state': 42
    }
    
    # Train model with early_stopping callback - removed verbose parameter
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
        # Removed verbose parameter as it's not accepted by lgb.train()
    )
    
    return model

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    mape = safe_mape(y_val, y_pred)
    return mape, y_pred

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    """Main training pipeline"""
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_merge_data()
    df = preprocess_data(df)
    df = engineer_features(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Time series split
    train_df, val_df = time_series_split(df)
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    
    # Remove Date column as it's not a feature
    X_train = X_train.drop('Date', axis=1)
    X_val = X_val.drop('Date', axis=1)
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    mape, y_pred = evaluate_model(model, X_val, y_val)
    final_mape = mape
    
    # Print final MAPE
    print(f"FINAL_MAPE: {final_mape}")
    
    # Plot feature importance
    plot_feature_importance(model, X_train.columns)
    
    # Save model
    model.save_model('rossmann_model.txt')

if __name__ == "__main__":
    main()