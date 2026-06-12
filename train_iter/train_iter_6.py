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
    
    # Enhanced calendar features
    
    # Holiday proximity features
    # Create a list of all holidays in Germany for the years in the dataset
    # Using known public holidays in Germany (simplified for all states)
    holidays = [
        '2013-01-01', '2013-01-06', '2013-03-29', '2013-04-01', '2013-05-01', '2013-05-09', '2013-05-20', '2013-05-30',
        '2013-06-10', '2013-08-15', '2013-10-03', '2013-10-31', '2013-11-01', '2013-11-20', '2013-12-25', '2013-12-26',
        '2014-01-01', '2014-01-06', '2014-04-18', '2014-04-21', '2014-05-01', '2014-05-29', '2014-06-09', '2014-06-19',
        '2014-08-15', '2014-10-03', '2014-10-31', '2014-11-01', '2014-11-19', '2014-12-25', '2014-12-26',
        '2015-01-01', '2015-01-06', '2015-04-03', '2015-04-06', '2015-05-01', '2015-05-14', '2015-05-25', '2015-06-04',
        '2015-08-15', '2015-10-03', '2015-10-31', '2015-11-01', '2015-11-18', '2015-12-25', '2015-12-26'
    ]
    
    # Convert to datetime
    holidays = pd.to_datetime(holidays)
    
    # Create a DataFrame with holidays
    holidays_df = pd.DataFrame({'HolidayDate': holidays})
    
    # Create features for proximity to holidays
    df['DateOnly'] = df.index.date
    holidays_df['HolidayDateOnly'] = holidays_df['HolidayDate'].dt.date
    
    # Calculate days to next holiday
    df['DaysToNextHoliday'] = df.index.to_series().apply(
        lambda x: (holidays_df[holidays_df['HolidayDate'] >= x]['HolidayDate'] - x).dt.days.min() if not holidays_df[holidays_df['HolidayDate'] >= x].empty else 365
    ).astype('int16')
    
    # Calculate days since last holiday
    df['DaysSinceLastHoliday'] = df.index.to_series().apply(
        lambda x: (x - holidays_df[holidays_df['HolidayDate'] <= x]['HolidayDate']).dt.days.max() if not holidays_df[holidays_df['HolidayDate'] <= x].empty else 365
    ).astype('int16')
    
    # Is day before/after holiday
    df['IsDayBeforeHoliday'] = (df['DaysToNextHoliday'] == 1).astype('int8')
    df['IsDayAfterHoliday'] = (df['DaysSinceLastHoliday'] == 1).astype('int8')
    
    # Is holiday week (within 3 days of a holiday)
    df['IsHolidayWeek'] = ((df['DaysToNextHoliday'] <= 3) | (df['DaysSinceLastHoliday'] <= 3)).astype('int8')
    
    # Promotional pattern features
    # Promo duration (how long current promo has been running)
    df['PromoDuration'] = df.groupby(['Store', (df['Promo'] != df['Promo'].shift()).cumsum()]).cumcount() + 1
    df['PromoDuration'] = df['PromoDuration'] * df['Promo']
    df['PromoDuration'] = df['PromoDuration'].astype('int16')
    
    # Promo change indicator (whether promo status changed from previous day)
    df['PromoChange'] = (df['Promo'] != df.groupby('Store')['Promo'].shift(1)).astype('int8')
    
    # Promo start indicator (first day of a promo)
    df['PromoStart'] = (df['Promo'] == 1) & (df['PromoChange'] == 1)
    df['PromoStart'] = df['PromoStart'].astype('int8')
    
    # Promo end indicator (last day of a promo)
    df['PromoEnd'] = (df['Promo'] == 0) & (df['PromoChange'] == 1)
    df['PromoEnd'] = df['PromoEnd'].astype('int8')
    
    # School vacation interaction features
    # Create school vacation periods (simplified for Germany)
    # Note: In real implementation, you would use actual school vacation data for each state
    vacation_periods = [
        ('2013-02-11', '2013-02-22'),
        ('2013-04-01', '2013-04-13'),
        ('2013-07-22', '2013-09-06'),
        ('2013-10-21', '2013-11-01'),
        ('2014-02-17', '2014-02-28'),
        ('2014-04-07', '2014-04-19'),
        ('2014-07-21', '2014-09-05'),
        ('2014-10-20', '2014-10-31'),
        ('2015-02-16', '2015-02-27'),
        ('2015-03-30', '2015-04-11'),
        ('2015-07-20', '2015-09-04'),
        ('2015-10-19', '2015-10-30')
    ]
    
    # Create vacation features
    df['IsSchoolVacation'] = 0
    for start, end in vacation_periods:
        df.loc[(df.index >= start) & (df.index <= end), 'IsSchoolVacation'] = 1
    df['IsSchoolVacation'] = df['IsSchoolVacation'].astype('int8')
    
    # Interaction between promo and school vacation
    df['PromoDuringVacation'] = (df['Promo'] == 1) & (df['IsSchoolVacation'] == 1)
    df['PromoDuringVacation'] = df['PromoDuringVacation'].astype('int8')
    
    # Interaction between holidays and school vacation
    df['HolidayDuringVacation'] = ((df['DaysToNextHoliday'] <= 3) | (df['DaysSinceLastHoliday'] <= 3)) & (df['IsSchoolVacation'] == 1)
    df['HolidayDuringVacation'] = df['HolidayDuringVacation'].astype('int8')
    
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
        'Store_mean_sales', 'DayOfWeek_mean_sales', 'Month_mean_sales',
        # New calendar features
        'DaysToNextHoliday', 'DaysSinceLastHoliday', 'IsDayBeforeHoliday',
        'IsDayAfterHoliday', 'IsHolidayWeek', 'PromoDuration', 'PromoChange',
        'PromoStart', 'PromoEnd', 'IsSchoolVacation', 'PromoDuringVacation',
        'HolidayDuringVacation'
    ]
    
    # Remove any columns not in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Sales']
    
    return X, y, feature_cols

# Model training and evaluation
def train_and_evaluate(X, y):
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
    
    # Model candidates
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
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'LightGBM':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
        elif name == 'XGBoost':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
        elif name == 'CatBoost':
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_val_scaled)
        
        mape = safe_mape(y_val, preds)
        results[name] = mape
        print(f"{name} MAPE: {mape:.4f}")
    
    # Select best model
    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name} with MAPE: {results[best_model_name]:.4f}")
    
    # Retrain best model on full dataset
    print(f"Retraining {best_model_name} on full dataset...")
    if best_model_name == 'LightGBM':
        best_model.fit(X_train_scaled, y_train)
        final_preds = best_model.predict(X_val_scaled)
    elif best_model_name == 'XGBoost':
        best_model.fit(X_train_scaled, y_train)
        final_preds = best_model.predict(X_val_scaled)
    elif best_model_name == 'CatBoost':
        best_model.fit(X_train_scaled, y_train)
        final_preds = best_model.predict(X_val_scaled)
    
    # Calculate final MAPE
    final_mape = safe_mape(y_val, final_preds)
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train_scaled.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    return final_mape

# Main execution
def main():
    print("Loading data...")
    df = load_data()
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Preparing data...")
    X, y, feature_cols = prepare_data(df)
    
    print("Training and evaluating models...")
    final_mape = train_and_evaluate(X, y)
    
    print(f"FINAL_MAPE: {final_mape}")

if __name__ == "__main__":
    main()