"""
Feature Engineering Module
Handles derived features, feature selection, and class imbalance via SMOTE.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-meaningful derived features."""
    X = X.copy()

    # BMI category (encoded as ordinal after scaling, so we use raw if available)
    # These work on scaled data too — they capture relative thresholds
    if "bmi" in X.columns and "age" in X.columns:
        X["bmi_age_interaction"] = X["bmi"] * X["age"]

    if "systolic_bp" in X.columns and "diastolic_bp" in X.columns:
        X["pulse_pressure"] = X["systolic_bp"] - X["diastolic_bp"]
        X["mean_arterial_pressure"] = X["diastolic_bp"] + (X["systolic_bp"] - X["diastolic_bp"]) / 3

    if "glucose" in X.columns and "hba1c" in X.columns:
        X["glucose_hba1c_ratio"] = X["glucose"] / (X["hba1c"] + 1e-6)

    if "cholesterol" in X.columns and "bmi" in X.columns:
        X["cholesterol_bmi_interaction"] = X["cholesterol"] * X["bmi"]

    return X


def select_features(X_train: pd.DataFrame, y_train: np.ndarray,
                     X_test: pd.DataFrame, top_k: int = None):
    """
    Feature selection using mutual information.
    If top_k is None, keeps all features with MI > 0.01.
    """
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    feature_importance = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

    if top_k is None:
        selected = feature_importance[feature_importance > 0.01].index.tolist()
    else:
        selected = feature_importance.head(top_k).index.tolist()

    if len(selected) < 3:
        selected = feature_importance.head(5).index.tolist()

    return X_train[selected], X_test[selected], selected


def apply_smote(X_train: pd.DataFrame, y_train: np.ndarray):
    """Apply SMOTE to handle class imbalance."""
    positive_rate = y_train.mean()

    # Only apply SMOTE if there's meaningful imbalance
    if 0.35 <= positive_rate <= 0.65:
        print(f"    Class balance OK ({positive_rate:.1%} positive) — skipping SMOTE")
        return X_train, y_train

    print(f"    Applying SMOTE (positive rate: {positive_rate:.1%})...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"    -> Resampled: {len(X_resampled)} samples (was {len(X_train)})")
    return pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled


def engineer_features(X_train, X_test, y_train):
    """
    Full feature engineering pipeline:
    1. Add derived features
    2. Apply SMOTE for class imbalance
    3. Select top features via mutual information
    """
    # Add derived features
    X_train = add_derived_features(X_train)
    X_test = add_derived_features(X_test)

    # Fill any NaN introduced by derived features
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Handle class imbalance
    X_train, y_train = apply_smote(X_train, y_train)

    # Feature selection
    X_train, X_test, selected_features = select_features(X_train, y_train, X_test)

    print(f"    Selected {len(selected_features)} features: {selected_features[:5]}...")

    return X_train, X_test, y_train, selected_features
