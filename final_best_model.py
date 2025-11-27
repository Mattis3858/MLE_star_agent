import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# For hardware acceleration detection
import torch

plt_backend = 'Agg'
import matplotlib
matplotlib.use(plt_backend)
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self):
        self.store_encoders = {}
        
    def load_data(self, train_path, store_path):
        """Load and merge train and store data"""
        train_df = pd.read_csv(train_path)
        store_df = pd.read_csv(store_path)
        
        # Merge datasets
        merged_df = pd.merge(train_df, store_df, on='Store', how='left')
        return merged_df
    
    def handle_missing_values(self, df):
        """Handle missing values based on data characteristics"""
        # Store-related missing values
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(1900)
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(1900)
        df['PromoInterval'] = df['PromoInterval'].fillna('')
        
        return df
    
    def encode_categoricals(self, df):
        """Encode categorical variables"""
        categorical_cols = ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']
        
        for col in categorical_cols:
            if col not in self.store_encoders:
                self.store_encoders[col] = LabelEncoder()
                df[col] = self.store_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.store_encoders[col].transform(df[col].astype(str))
        
        return df

class FeatureEngineer:
    def __init__(self):
        self.store_type_embeddings = {}
        
    def create_date_features(self, df):
        """Extract comprehensive date-based features"""
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic date components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Seasonal features
        df['Season'] = df['Month'] % 12 // 3 + 1
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        
        return df
    
    def create_competitor_features(self, df):
        """Create competitor-related features with proper datetime handling"""
        # Fill missing values first
        df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(1900)
        df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
        
        # Data validation: Replace invalid year/month values
        # Invalid years: less than 1900 or greater than current year + 10
        current_year = pd.Timestamp.now().year
        invalid_year_mask = (df['CompetitionOpenSinceYear'] < 1900) | (df['CompetitionOpenSinceYear'] > current_year + 10)
        df.loc[invalid_year_mask, 'CompetitionOpenSinceYear'] = 1900
        
        # Invalid months: not in range 1-12
        invalid_month_mask = (df['CompetitionOpenSinceMonth'] < 1) | (df['CompetitionOpenSinceMonth'] > 12)
        df.loc[invalid_month_mask, 'CompetitionOpenSinceMonth'] = 1  # Default to January
        
        # Create a safe CompetitionOpenDate with error handling
        # Create year and month as integers
        year = df['CompetitionOpenSinceYear'].astype(int)
        month = df['CompetitionOpenSinceMonth'].astype(int)
        
        # Create date strings in a safe way
        date_str = year.astype(str) + '-' + month.astype(str) + '-15'
        
        # Convert to datetime with error handling
        df['CompetitionOpenDate'] = pd.to_datetime(date_str, errors='coerce')
        
        # Calculate days open
        df['CompetitionDaysOpen'] = df['Date'].sub(df['CompetitionOpenDate']).dt.days
        df['CompetitionDaysOpen'] = df['CompetitionDaysOpen'].fillna(0)
        df['CompetitionDaysOpen'] = df['CompetitionDaysOpen'].apply(lambda x: 0 if x < 0 else x)
        
        # Competition distance features
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(0)
        df['HasCompetitor'] = (df['CompetitionDistance'] > 0).astype(int)
        
        return df
    
    def create_promo_features(self, df):
        """Create promotional campaign features with proper handling of missing/invalid Promo2 dates"""
        # Promo2 duration - handle missing values
        df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
        df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
        
        # Handle invalid values: Year 1900 indicates missing data, Week 0 is invalid
        invalid_promo2_mask = (df['Promo2SinceYear'] == 1900) | (df['Promo2SinceWeek'] == 0)
        
        # Create safe year and week columns
        safe_year = df['Promo2SinceYear'].copy()
        safe_week = df['Promo2SinceWeek'].copy()
        
        # Replace invalid values with NaN for safe conversion
        safe_year[invalid_promo2_mask] = np.nan
        safe_week[invalid_promo2_mask] = np.nan
        
        # Create Promo2 start date with error handling
        try:
            # Create date strings in format YYYY-WW-1 (Monday of the week)
            date_str = safe_year.astype(str) + '-' + safe_week.astype(str) + '-1'
            df['Promo2StartDate'] = pd.to_datetime(date_str, format='%Y-%U-%w', errors='coerce')
        except:
            # Fallback method if format parsing fails
            df['Promo2StartDate'] = pd.NaT
            
        # Calculate Promo2 days with proper handling of NaT values
        df['Promo2Days'] = df['Date'].sub(df['Promo2StartDate']).dt.days
        df['Promo2Days'] = df['Promo2Days'].fillna(0)
        df['Promo2Days'] = df['Promo2Days'].apply(lambda x: 0 if x < 0 else x)
        
        # Promo interval features - Add type checking and handling
        df['IsPromo2Month'] = 0
        
        # Ensure PromoInterval is string type and handle NaN/missing values
        df['PromoInterval'] = df['PromoInterval'].fillna('')
        df['PromoInterval'] = df['PromoInterval'].astype(str)
        
        for interval in df['PromoInterval'].unique():
            if interval and interval != 'nan':  # Check for non-empty and not 'nan' strings
                try:
                    months = interval.split(',')
                    for month in months:
                        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 
                                   'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                                   'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                        if month.strip() in month_map:
                            df.loc[(df['PromoInterval'] == interval) & 
                                  (df['Month'] == month_map[month.strip()]), 
                                  'IsPromo2Month'] = 1
                except AttributeError:
                    # Skip non-string values that might have slipped through
                    continue
        
        return df
    
    def create_store_type_embeddings(self, df):
        """Create store type embeddings to capture store-specific patterns"""
        # Create mean sales per store type for embedding
        if not self.store_type_embeddings:
            # Only calculate during training
            store_type_sales = df.groupby('StoreType')['Sales'].mean()
            self.store_type_embeddings = store_type_sales.to_dict()
        
        # Map store type to average sales embedding
        df['StoreTypeEmbedding'] = df['StoreType'].map(self.store_type_embeddings)
        
        return df

    def create_lag_features(self, df, lag_days=[1, 2, 3, 7, 14]):
        """Create lag features for sales"""
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        for lag in lag_days:
            df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
            
        return df
    
    def create_rolling_features(self, df, windows=[7, 14, 30]):
        """Create rolling window statistics for sales"""
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        for window in windows:
            # Rolling mean
            df[f'Sales_Rolling_Mean_{window}'] = df.groupby('Store')['Sales'].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
            # Rolling std
            df[f'Sales_Rolling_Std_{window}'] = df.groupby('Store')['Sales'].rolling(window, min_periods=1).std().reset_index(0, drop=True)
            # Rolling min
            df[f'Sales_Rolling_Min_{window}'] = df.groupby('Store')['Sales'].rolling(window, min_periods=1).min().reset_index(0, drop=True)
            # Rolling max
            df[f'Sales_Rolling_Max_{window}'] = df.groupby('Store')['Sales'].rolling(window, min_periods=1).max().reset_index(0, drop=True)
            
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between temporal variables and store characteristics"""
        # Promo interaction with day of week
        df['Promo_DayOfWeek'] = df['Promo'] * df['DayOfWeek']
        df['Promo_Weekend'] = df['Promo'] * df['IsWeekend']
        
        # Store type interactions with seasonal patterns
        df['StoreType_Season'] = df['StoreType'] * df['Season']
        df['StoreType_Month'] = df['StoreType'] * df['Month']
        
        # Promo2 interactions
        df['Promo2_DayOfWeek'] = df['Promo2'] * df['DayOfWeek']
        df['Promo2_Weekend'] = df['Promo2'] * df['IsWeekend']
        df['Promo2_Season'] = df['Promo2'] * df['Season']
        
        # Store type interactions with promotions
        df['StoreType_Promo'] = df['StoreType'] * df['Promo']
        df['StoreType_Promo2'] = df['StoreType'] * df['Promo2']
        
        # Assortment interactions
        df['Assortment_DayOfWeek'] = df['Assortment'] * df['DayOfWeek']
        df['Assortment_Season'] = df['Assortment'] * df['Season']
        
        # Competition interactions with temporal features
        df['HasCompetitor_DayOfWeek'] = df['HasCompetitor'] * df['DayOfWeek']
        df['HasCompetitor_Weekend'] = df['HasCompetitor'] * df['IsWeekend']
        
        return df

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Exclude non-feature columns
        exclude_cols = ['Date', 'Sales', 'Customers']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = np.log1p(df['Sales'])  # Log transform sales
        
        self.feature_cols = feature_cols
        return X, y
    
    def detect_device(self):
        """Detect if GPU is available for training"""
        if torch.cuda.is_available():
            return 'gpu'
        else:
            return 'cpu'
    
    def train_model(self, X_train, y_train):
        """Train the best performing model (LightGBM) with hardware acceleration"""
        device = self.detect_device()
        
        if device == 'gpu':
            params = {
                'objective': 'regression',
                'metric': 'mape',
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        else:
            params = {
                'objective': 'regression',
                'metric': 'mape',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(params, train_data, num_boost_round=1000)
        return self.model

class RossmannPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self, train_path, store_path):
        """Execute complete pipeline"""
        print("Step 1: Loading and preprocessing data...")
        df = self.preprocessor.load_data(train_path, store_path)
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.encode_categoricals(df)
        
        print("Step 2: Feature engineering...")
        df = self.feature_engineer.create_date_features(df)
        df = self.feature_engineer.create_competitor_features(df)
        df = self.feature_engineer.create_promo_features(df)
        df = self.feature_engineer.create_store_type_embeddings(df)
        
        # Add lag features and rolling window statistics
        df = self.feature_engineer.create_lag_features(df)
        df = self.feature_engineer.create_rolling_features(df)
        
        # Add interaction features
        df = self.feature_engineer.create_interaction_features(df)
        
        # CRITICAL: Filter only open stores and non-zero sales BEFORE any split
        df = df[(df['Open'] == 1) & (df['Sales'] > 0)]
        
        print("Step 3: Preparing features for modeling...")
        X, y = self.model_trainer.prepare_features(df)
        
        # Convert datetime columns to numerical features
        datetime_cols = ['Date', 'CompetitionOpenDate', 'Promo2StartDate']
        for col in datetime_cols:
            if col in X.columns:
                # Convert to timestamp integers
                X[col] = pd.to_datetime(X[col]).astype('int64') // 10**9
        
        # Encode categorical PromoInterval as integer codes
        if 'PromoInterval' in X.columns:
            X['PromoInterval'] = pd.Categorical(X['PromoInterval']).codes
        
        # Time-based split (using date ordering)
        df_sorted = df.sort_values('Date')
        split_idx = int(len(df_sorted) * 0.8)
        
        train_indices = df_sorted.index[:split_idx]
        test_indices = df_sorted.index[split_idx:]
        
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        
        print("Step 4: Model training...")
        model = self.model_trainer.train_model(X_train, y_train)
        
        print("Step 5: Model evaluation...")
        y_pred = model.predict(X_test, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
        
        # Inverse log transform predictions for MAPE calculation
        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred)
        
        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
        
        print(f"FINAL_MAPE: {mape:.6f}")
        
        return model, mape

# Usage
if __name__ == "__main__":
    pipeline = RossmannPipeline()
    model, mape = pipeline.run_pipeline('train.csv', 'store.csv')