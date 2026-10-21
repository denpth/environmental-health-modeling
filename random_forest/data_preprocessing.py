"""
Data Preprocessing Pipeline for EPA Air Quality and Health Risk Prediction
Author: AI Optimization for Computer Scientists - D682
Description: Loads, cleans, and prepares the air quality dataset for Random Forest modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AirQualityDataProcessor:
    """
    A comprehensive data preprocessing class for air quality and health risk prediction.

    This class handles:
    - Data loading and initial exploration
    - Missing value analysis and treatment
    - Feature engineering for temporal patterns
    - Data quality validation
    - Train/test splitting for model development
    """

    def __init__(self, data_path='dataset.csv'):
        """
        Initialize the data processor.

        Args:
            data_path (str): Path to the CSV dataset file
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_column = 'healthRiskScore'

    def load_data(self):
        """
        Load the dataset and perform initial data exploration.

        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("Loading air quality dataset...")
        self.df = pd.read_csv(self.data_path)

        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

        return self.df

    def explore_data(self):
        """
        Perform comprehensive data exploration and quality assessment.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*60)
        print("DATA EXPLORATION SUMMARY")
        print("="*60)

        # Basic info
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes.value_counts())

        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        }).sort_values('Missing_Count', ascending=False)

        print(f"\nMissing Values Summary:")
        print(missing_summary[missing_summary['Missing_Count'] > 0])

        # Target variable analysis
        if self.target_column in self.df.columns:
            print(f"\nTarget Variable ({self.target_column}) Statistics:")
            print(self.df[self.target_column].describe())

        # Numerical features summary
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\nNumerical Features: {len(numerical_cols)}")
        print(f"Categorical Features: {len(self.df.columns) - len(numerical_cols)}")

        return missing_summary

    def clean_data(self):
        """
        Clean the dataset by handling missing values and data quality issues.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)

        initial_shape = self.df.shape

        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows...")
            self.df = self.df.drop_duplicates()

        # Handle missing values
        missing_summary = self.df.isnull().sum()
        if missing_summary.sum() > 0:
            print(f"Handling missing values...")

            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  - Filled {col} missing values with median: {median_val:.2f}")

            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"  - Filled {col} missing values with mode: {mode_val}")

        # Remove outliers using IQR method for key pollution variables
        pollution_vars = ['pm2.5', 'no2', 'co2', 'healthRiskScore']
        for var in pollution_vars:
            if var in self.df.columns:
                Q1 = self.df[var].quantile(0.25)
                Q3 = self.df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((self.df[var] < lower_bound) | (self.df[var] > upper_bound)).sum()
                if outliers > 0:
                    print(f"  - Removing {outliers} outliers from {var}")
                    self.df = self.df[(self.df[var] >= lower_bound) & (self.df[var] <= upper_bound)]

        print(f"\nData cleaning completed!")
        print(f"Shape change: {initial_shape} -> {self.df.shape}")

        return self.df

    def engineer_features(self):
        """
        Create additional features to improve model performance.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        # Convert datetime if it exists
        if 'datetimeEpoch' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetimeEpoch'], unit='s')
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
            print("  - Created datetime features: hour, day_of_year")

        # Create pollution index (composite measure)
        if all(col in self.df.columns for col in ['pm2.5', 'no2', 'co2']):
            # Normalize each pollutant to 0-1 scale then average
            pm25_norm = (self.df['pm2.5'] - self.df['pm2.5'].min()) / (self.df['pm2.5'].max() - self.df['pm2.5'].min())
            no2_norm = (self.df['no2'] - self.df['no2'].min()) / (self.df['no2'].max() - self.df['no2'].min())
            co2_norm = (self.df['co2'] - self.df['co2'].min()) / (self.df['co2'].max() - self.df['co2'].min())

            self.df['pollution_index'] = (pm25_norm + no2_norm + co2_norm) / 3
            print("  - Created pollution_index (composite measure)")

        # Create temperature comfort index
        if all(col in self.df.columns for col in ['temp', 'humidity']):
            # Simple comfort index based on temperature and humidity
            self.df['comfort_index'] = 100 - abs(self.df['temp'] - 72) - abs(self.df['humidity'] - 50) * 0.5
            print("  - Created comfort_index")

        # Create wind chill factor
        if all(col in self.df.columns for col in ['temp', 'windspeed']):
            # Simplified wind chill calculation
            self.df['wind_chill'] = self.df['temp'] - (self.df['windspeed'] * 0.7)
            print("  - Created wind_chill factor")

        # Encode categorical variables
        categorical_cols = ['isWeekend']
        for col in categorical_cols:
            if col in self.df.columns:
                if self.df[col].dtype == 'object' or self.df[col].dtype == 'bool':
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                    print(f"  - Encoded {col} as {col}_encoded")

        print(f"\nFeature engineering completed!")
        print(f"New shape: {self.df.shape}")

        return self.df

    def prepare_features(self):
        """
        Prepare final feature set for modeling by selecting relevant columns.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*60)
        print("FEATURE PREPARATION")
        print("="*60)

        # Define features to exclude (non-predictive or target-related)
        exclude_cols = [
            'datetimeEpoch', 'datetime', 'sunriseEpoch', 'sunsetEpoch',
            'healthRiskScore',  # This is our target
            'isWeekend'  # We'll use the encoded version
        ]

        # Select feature columns
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]

        # Ensure we have the target column
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        self.feature_names = feature_cols
        print(f"Selected {len(feature_cols)} features for modeling:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")

        return feature_cols

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.feature_names is None:
            self.prepare_features()

        print("\n" + "="*60)
        print("DATA SPLITTING")
        print("="*60)

        # Prepare features and target
        X = self.df[self.feature_names]
        y = self.df[self.target_column]

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]}")

        # Display target distribution
        print(f"\nTarget variable distribution:")
        print(f"Training - Mean: {self.y_train.mean():.3f}, Std: {self.y_train.std():.3f}")
        print(f"Testing  - Mean: {self.y_test.mean():.3f}, Std: {self.y_test.std():.3f}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_data_summary(self):
        """
        Generate a comprehensive summary of the processed data.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        summary = {
            'total_samples': len(self.df),
            'total_features': len(self.df.columns),
            'target_column': self.target_column,
            'feature_columns': self.feature_names,
            'train_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'target_stats': {
                'mean': self.df[self.target_column].mean(),
                'std': self.df[self.target_column].std(),
                'min': self.df[self.target_column].min(),
                'max': self.df[self.target_column].max()
            }
        }

        return summary

def main():
    """
    Main execution function demonstrating the data preprocessing pipeline.
    """
    print("EPA Air Quality Data Preprocessing Pipeline")
    print("=" * 60)

    # Initialize processor
    processor = AirQualityDataProcessor('dataset.csv')

    # Execute preprocessing pipeline
    try:
        # Load and explore data
        processor.load_data()
        processor.explore_data()

        # Clean and engineer features
        processor.clean_data()
        processor.engineer_features()

        # Prepare features and split data
        processor.prepare_features()
        processor.split_data()

        # Generate summary
        summary = processor.get_data_summary()

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Total samples: {summary['total_samples']}")
        print(f"Training samples: {summary['train_samples']}")
        print(f"Testing samples: {summary['test_samples']}")
        print(f"Features: {len(summary['feature_columns'])}")
        print(f"Target range: {summary['target_stats']['min']:.2f} - {summary['target_stats']['max']:.2f}")

        return processor

    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    processor = main()