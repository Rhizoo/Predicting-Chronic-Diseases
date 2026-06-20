"""
Model Training Module
Trains 6 ML algorithms per disease, selects the best model, and saves it.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIGS, MODELS_DIR, RANDOM_STATE


MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "SVC": SVC,
    "XGBClassifier": XGBClassifier,
    "LGBMClassifier": LGBMClassifier,
}


def train_single_model(name: str, config: dict, X_train, y_train, X_test, y_test):
    """Train a single model and return it with basic metrics."""
    model_class = MODEL_CLASSES[config["model"]]
    model = model_class(**config["params"])

    model.fit(X_train, y_train)

    # Quick evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = 0.0

    return model, accuracy, auc


def train_all_models(X_train, X_test, y_train, y_test, disease: str, feature_names: list):
    """
    Train all 6 models for a disease, compare them, and save the best.
    Returns: dict of {model_name: (model, accuracy, auc)}
    """
    print(f"\n  Training models for {disease}...")
    results = {}

    for name, config in MODEL_CONFIGS.items():
        try:
            model, acc, auc = train_single_model(
                name, config, X_train, y_train, X_test, y_test
            )
            results[name] = {"model": model, "accuracy": acc, "auc": auc}
            print(f"    {name:25s} -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        except Exception as e:
            print(f"    {name:25s} -> FAILED: {e}")

    # Select best model by AUC (or accuracy if AUC unavailable)
    best_name = max(results, key=lambda k: results[k]["auc"] if results[k]["auc"] > 0 else results[k]["accuracy"])
    best_model = results[best_name]["model"]

    # Save best model
    model_path = os.path.join(MODELS_DIR, f"{disease}_best_model.joblib")
    joblib.dump(best_model, model_path)

    # Save metadata
    meta = {
        "disease": disease,
        "best_model": best_name,
        "accuracy": results[best_name]["accuracy"],
        "auc": results[best_name]["auc"],
        "feature_names": feature_names,
        "all_results": {
            k: {"accuracy": v["accuracy"], "auc": v["auc"]}
            for k, v in results.items()
        },
    }
    meta_path = os.path.join(MODELS_DIR, f"{disease}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK] Best model: {best_name} (AUC: {results[best_name]['auc']:.4f})")
    print(f"    Saved to {model_path}\n")

    return results, best_name
