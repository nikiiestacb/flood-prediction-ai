"""
Data processing module for flood prediction
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodDataProcessor:
    """Process raw hydrological data for flood prediction"""
    
    def __init__(self, data_path: str):
        """
        Initialize data processor
        
        Args:
            data_path: Path to raw data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load raw data from CSV"""
        logger.info(f"Loading data from {self.data_path}")
        try:
            self.data = pd.read_csv(self.data_path, parse_dates=['timestamp'])
            logger.info(f"Loaded {len(self.data)} records")
            return self.data
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
    
    def clean_data(self):
        """Clean and validate data"""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        logger.info(f"Removed {before - after} duplicate records")
        
        # Handle missing values
        missing = self.data.isnull().sum()
        logger.info(f"Missing values:\n{missing[missing > 0]}")
        
        # Fill missing with forward fill
        self.data = self.data.fillna(method='ffill')
        
        return self.data
    
    def add_features(self):
        """Add derived features"""
        logger.info("Creating features...")
        
        # Temporal features
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        self.data['month'] = self.data['timestamp'].dt.month
        
        # Lagged features
        for lag in [1, 3, 7, 14]:
            self.data[f'rainfall_lag_{lag}'] = self.data['rainfall'].shift(lag)
        
        # Rolling statistics
        self.data['rainfall_7d_mean'] = self.data['rainfall'].rolling(7).mean()
        self.data['discharge_3d_max'] = self.data['river_discharge'].rolling(3).max()
        
        logger.info(f"Created {self.data.shape[1]} total features")
        return self.data
    
    def save_processed_data(self, output_path: str):
        """Save processed data"""
        logger.info(f"Saving to {output_path}")
        self.data.to_csv(output_path, index=False)
        logger.info("Data saved successfully")


if __name__ == "__main__":
    # Example usage
    processor = FloodDataProcessor("data/raw/raw_flood_data.csv")
    processor.load_data()
    processor.clean_data()
    processor.add_features()
    processor.save_processed_data("data/processed/flood_data.csv")
