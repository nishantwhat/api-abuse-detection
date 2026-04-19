"""
preprocessing.py
----------------
Handles all data cleaning tasks before features are used for training.

Key responsibilities:
  1. Separate features (X) from the target label (y)
  2. Handle missing values in a meaningful way (not just drop them)
  3. Remove or cap extreme outliers that could distort model training
  4. Report a brief data quality summary for transparency
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Map numeric labels to human-readable class names for display purposes
LABEL_MAP = {0: "Normal", 1: "Suspicious", 2: "Abuse"}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning pipeline. Applies all cleaning steps in order and
    returns a clean DataFrame ready for feature engineering.

    Args:
        df: Raw DataFrame from data_loader.

    Returns:
        Cleaned DataFrame.
    """
    print("\n[PREPROCESSING] Starting data cleaning...")
    _report_data_quality(df)

    df = df.copy()

    # Step 1: Drop rows where the label is missing — we can't train on unlabeled data
    before = len(df)
    df = df.dropna(subset=["label"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[PREPROCESSING] Dropped {dropped} rows with missing labels.")

    # Step 2: Fill missing numeric values with the column median
    # Median is more robust than mean when data has outliers
    feature_cols = [c for c in df.columns if c != "label"]
    missing_counts = df[feature_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    if not cols_with_missing.empty:
        print(f"[PREPROCESSING] Imputing missing values using column median:")
        for col, count in cols_with_missing.items():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   - {col}: {count} values filled with median={median_val:.4f}")

    # Step 3: Cap extreme outliers using the IQR method
    # Values beyond 3×IQR from the 25th/75th percentile are capped
    df = _cap_outliers(df, feature_cols)

    print(f"[PREPROCESSING] Cleaning complete. Final shape: {df.shape}")
    return df


def split_features_labels(df: pd.DataFrame):
    """
    Separate the dataset into feature matrix X and target vector y.

    Args:
        df: Cleaned DataFrame.

    Returns:
        X (DataFrame of features), y (Series of labels)
    """
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)
    return X, y


def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets with stratification.
    Stratification ensures class proportions are preserved in both splits.

    Args:
        X: Feature matrix.
        y: Target labels.
        test_size: Fraction of data to use for testing (default 20%).
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,          # Preserve class balance in both splits
        random_state=random_state
    )

    print(f"\n[PREPROCESSING] Train/Test Split:")
    print(f"   Training samples : {len(X_train)}")
    print(f"   Testing samples  : {len(X_test)}")
    print(f"   Test size        : {test_size * 100:.0f}%")

    return X_train, X_test, y_train, y_test


def _cap_outliers(df: pd.DataFrame, feature_cols: list, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Cap outliers beyond multiplier×IQR at the fence values.
    This preserves the row (unlike dropping) while reducing distortion.

    Args:
        df: DataFrame with features.
        feature_cols: List of column names to process.
        multiplier: IQR multiplier to define fence boundaries.

    Returns:
        DataFrame with capped values.
    """
    capped_count = 0
    for col in feature_cols:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            capped_count += outliers

    print(f"[PREPROCESSING] Outlier capping: {capped_count} values capped across all features.")
    return df


def _report_data_quality(df: pd.DataFrame) -> None:
    """
    Print a brief data quality summary to help understand the raw dataset.
    """
    print(f"   Shape           : {df.shape}")
    print(f"   Total NaN cells : {df.isnull().sum().sum()}")
    print(f"   Duplicate rows  : {df.duplicated().sum()}")
    if "label" in df.columns:
        dist = df["label"].value_counts().sort_index()
        print("   Class distribution:")
        for label_id, count in dist.items():
            name = LABEL_MAP.get(int(label_id), "Unknown")
            pct  = 100 * count / len(df)
            print(f"     {label_id} ({name}): {count} ({pct:.1f}%)")
