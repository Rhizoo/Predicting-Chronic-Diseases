"""
Data Preprocessing Module
Handles missing values, outlier detection, encoding, and scaling.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODELS_DIR, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, TEST_SIZE, RANDOM_STATE,
)


def load_dataset(disease: str) -> pd.DataFrame:
    """Load a disease dataset from CSV."""
    path = os.path.join(DATA_DIR, f"{disease}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"    Removed {removed} duplicate rows")
    return df


def detect_and_cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using IQR method for numeric features."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower, upper)
    return df


def encode_features(df: pd.DataFrame, disease: str, fit: bool = True) -> pd.DataFrame:
    """Encode categorical features using LabelEncoder."""
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        if fit:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            # Load saved encoder
            encoder_path = os.path.join(MODELS_DIR, f"{disease}_encoder_{col}.joblib")
            le = joblib.load(encoder_path)
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    if fit and encoders:
        for col, le in encoders.items():
            encoder_path = os.path.join(MODELS_DIR, f"{disease}_encoder_{col}.joblib")
            joblib.dump(le, encoder_path)

    return df


def scale_features(df: pd.DataFrame, disease: str, fit: bool = True) -> pd.DataFrame:
    """Scale numeric features using StandardScaler."""
    df = df.copy()
    cols_to_scale = [c for c in NUMERIC_FEATURES if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        scaler_path = os.path.join(MODELS_DIR, f"{disease}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
    else:
        scaler_path = os.path.join(MODELS_DIR, f"{disease}_scaler.joblib")
        scaler = joblib.load(scaler_path)
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def preprocess_dataset(disease: str):
    """
    Full preprocessing pipeline for a disease dataset.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print(f"  Preprocessing {disease} dataset...")

    # Load
    df = load_dataset(disease)
    print(f"    Loaded {len(df)} samples")

    # Clean
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = detect_and_cap_outliers(df)

    # Split features and target
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["target"].values

    # Encode and scale
    X = encode_features(X, disease, fit=True)
    X = scale_features(X, disease, fit=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"    Train: {len(X_train)} | Test: {len(X_test)} | Positive rate: {y.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, feature_cols


def preprocess_single_input(data: dict, disease: str) -> pd.DataFrame:
    """Preprocess a single patient input for prediction."""
    df = pd.DataFrame([data])

    # Ensure correct types
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = handle_missing_values(df)
    df = encode_features(df, disease, fit=False)
    df = scale_features(df, disease, fit=False)

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    return df[feature_cols]
