"""
Data preprocessing module.
Handles data loading, cleaning, imputation, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings
from pathlib import Path

from src.config import STARS_CSV, NUMERIC_COLUMNS, RANDOM_STATE, TEST_SIZE, OUTPUTS_DIR, DATA_PROCESSED

warnings.filterwarnings('ignore')


def load_data(file_path=None):
    """
    Load the stars dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file. If None, uses default from config.
    
    Returns:
        DataFrame with the loaded data
    """
    if file_path is None:
        file_path = STARS_CSV
    
    df = pd.read_csv(file_path)
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    return df


def explore_data(df):
    """
    Perform basic exploratory data analysis.
    
    Args:
        df: DataFrame to explore
    
    Returns:
        Dictionary with basic statistics
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nNumber of duplicates:", df.duplicated().sum())
    
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing)
    print("\nPercentage of missing values:")
    print((missing / len(df) * 100).round(2))
    
    print("\nStatistical description:")
    print(df.describe().round(2))
    
    print("\nDistribution of Star Types:")
    print(df['Star type'].value_counts())
    
    print("\nDistribution of Star Colors:")
    print(df['Star color'].value_counts())
    
    print("\nDistribution of Spectral Classes:")
    print(df['Spectral Class'].value_counts())
    
    return {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'n_duplicates': df.duplicated().sum(),
        'missing_values': missing.to_dict()
    }


def clean_data(df):
    """
    Clean the dataset by removing duplicates.
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    
    print(f"Removed {n_before - n_after} duplicate rows")
    print(f"Number of samples after cleaning: {n_after}")
    
    return df


def impute_missing_values(df, numeric_cols=None):
    """
    Impute missing values using KNN imputation.
    
    Args:
        df: DataFrame with missing values
        numeric_cols: List of numeric column names. If None, uses config default.
    
    Returns:
        DataFrame with imputed values
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS
    
    print("\n" + "="*50)
    print("MISSING VALUE IMPUTATION")
    print("="*50)
    
    # Check if there are any missing values in numeric columns
    missing = df[numeric_cols].isnull().sum()
    if missing.sum() == 0:
        print("No missing values in numeric columns. Skipping imputation.")
        return df
    
    print(f"Missing values before imputation:\n{missing[missing > 0]}")
    
    # Use KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    print("Missing values imputed using KNN (k=5)")
    
    return df_imputed


def prepare_features(df, numeric_cols=None):
    """
    Prepare features (X) and target (y) for modeling.
    
    Args:
        df: DataFrame with features and target
        numeric_cols: List of numeric column names. If None, uses config default.
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS
    
    X = df[numeric_cols].copy()
    y = df['Star type'].copy()
    
    return X, y


def split_data(X, y, test_size=None, random_state=None):
    """
    Split data into training and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set. If None, uses config default.
        random_state: Random state for reproducibility. If None, uses config default.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nData split: {len(X_train)} training samples, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n" + "="*50)
    print("FEATURE SCALING")
    print("="*50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler (mean=0, std=1)")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(file_path=None, save_processed=False):
    """
    Complete preprocessing pipeline.
    
    Args:
        file_path: Path to CSV file. If None, uses config default.
        save_processed: Whether to save processed data to disk
    
    Returns:
        Dictionary with all preprocessed data and objects
    """
    # Load data
    df = load_data(file_path)
    
    # Explore
    explore_data(df)
    
    # Clean
    df = clean_data(df)
    
    # Impute missing values
    df = impute_missing_values(df)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Save processed data if requested
    if save_processed:
        df.to_csv(DATA_PROCESSED / "processed_data.csv", index=False)
        print(f"\nProcessed data saved to {DATA_PROCESSED / 'processed_data.csv'}")
    
    return {
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler
    }

