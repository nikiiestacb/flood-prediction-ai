"""
Feature Engineering for Flood Prediction
Creates temporal and spatial features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class FloodFeatureEngineer:
    """Create features for flood prediction"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            df: DataFrame with timestamp and raw features
        """
        self.df = df.copy()
        self.feature_names = []
        
    def create_temporal_features(self) -> pd.DataFrame:
        """Create time-based features"""
        
        # Cyclical encoding for hour, day, month
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['timestamp'].dt.hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['timestamp'].dt.hour / 24)
        
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['timestamp'].dt.dayofweek / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['timestamp'].dt.dayofweek / 7)
        
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['timestamp'].dt.month / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['timestamp'].dt.month / 12)
        
        # Season indicator
        self.df['season'] = self.df['timestamp'].dt.month % 12 // 3
        self.df['is_monsoon'] = self.df['timestamp'].dt.month.isin([6, 7, 8, 9]).astype(int)
        
        return self.df
    
    def create_lag_features(self, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            columns: List of column names to create lags for
            lags: List of lag periods
        """
        for col in columns:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        return self.df
    
    def create_rolling_features(
        self, 
        columns: List[str], 
        windows: List[int]
    ) -> pd.DataFrame:
        """
        Create rolling statistics
        
        Args:
            columns: List of column names
            windows: List of window sizes
        """
        for col in columns:
            for window in windows:
                # Rolling mean
                self.df[f'{col}_rolling_mean_{window}'] = (
                    self.df[col].rolling(window=window).mean()
                )
                
                # Rolling std
                self.df[f'{col}_rolling_std_{window}'] = (
                    self.df[col].rolling(window=window).std()
                )
                
                # Rolling max
                self.df[f'{col}_rolling_max_{window}'] = (
                    self.df[col].rolling(window=window).max()
                )
                
                # Rolling min
                self.df[f'{col}_rolling_min_{window}'] = (
                    self.df[col].rolling(window=window).min()
                )
        
        return self.df
    
    def create_derived_features(self) -> pd.DataFrame:
        """Create domain-specific derived features"""
        
        # Antecedent Precipitation Index (API)
        # Weighted sum of previous rainfall
        k = 0.85  # Recession coefficient
        api = np.zeros(len(self.df))
        
        for i in range(1, len(self.df)):
            if pd.notna(self.df.loc[i-1, 'rainfall']):
                api[i] = k * api[i-1] + self.df.loc[i, 'rainfall']
        
        self.df['api'] = api
        
        # Cumulative rainfall
        for days in [3, 7, 14, 30]:
            self.df[f'cumulative_rainfall_{days}d'] = (
                self.df['rainfall'].rolling(window=days).sum()
            )
        
        # Rate of change
        self.df['discharge_rate_change'] = self.df['river_discharge'].diff()
        self.df['rainfall_intensity_change'] = self.df['rainfall'].diff()
        
        # Soil saturation indicator
        if 'soil_moisture' in self.df.columns:
            # Normalize to 0-1
            max_moisture = self.df['soil_moisture'].quantile(0.99)
            self.df['soil_saturation_index'] = (
                self.df['soil_moisture'] / max_moisture
            ).clip(0, 1)
        
        return self.df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features"""
        
        # Rainfall × Soil Moisture
        if 'soil_moisture' in self.df.columns and 'rainfall' in self.df.columns:
            self.df['rainfall_soil_interaction'] = (
                self.df['rainfall'] * self.df['soil_moisture']
            )
        
        # River discharge × Rainfall
        if 'river_discharge' in self.df.columns and 'rainfall' in self.df.columns:
            self.df['discharge_rainfall_interaction'] = (
                self.df['river_discharge'] * self.df['rainfall']
            )
        
        return self.df
    
    def get_feature_importance_ready_data(
        self, 
        target_column: str = 'flood_event'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training
        
        Args:
            target_column: Name of target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Drop non-feature columns
        drop_cols = ['timestamp', target_column]
        drop_cols = [col for col in drop_cols if col in self.df.columns]
        
        X = self.df.drop(columns=drop_cols)
        y = self.df[target_column] if target_column in self.df.columns else None
        
        # Drop rows with NaN (from rolling/lag operations)
        if y is not None:
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]
        else:
            X = X.dropna()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all features
    
    Args:
        df: Input DataFrame with basic columns
        
    Returns:
        DataFrame with all engineered features
    """
    engineer = FloodFeatureEngineer(df)
    
    # Create all feature types
    engineer.create_temporal_features()
    
    # Lag features for key columns
    lag_columns = ['rainfall', 'river_discharge', 'soil_moisture']
    lag_columns = [col for col in lag_columns if col in df.columns]
    engineer.create_lag_features(lag_columns, lags=[1, 3, 7, 14])
    
    # Rolling features
    engineer.create_rolling_features(lag_columns, windows=[3, 7, 14])
    
    # Derived features
    engineer.create_derived_features()
    
    # Interaction features
    engineer.create_interaction_features()
    
    print(f"Created {len(engineer.df.columns)} total features")
    print(f"Original features: {len(df.columns)}")
    print(f"New features: {len(engineer.df.columns) - len(df.columns)}")
    
    return engineer.df


if __name__ == "__main__":
    # Example usage
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'rainfall': np.random.exponential(5, 1000),
        'river_discharge': np.random.normal(1000, 200, 1000),
        'soil_moisture': np.random.uniform(0.3, 0.9, 1000),
        'flood_event': np.random.binomial(1, 0.1, 1000)
    })
    
    # Create features
    featured_df = create_all_features(sample_df)
    print("\nSample of created features:")
    print(featured_df.head())
    print(f"\nShape: {featured_df.shape}")
