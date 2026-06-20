"""
Model Explainability Module
Generates SHAP-based explanations for model predictions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHARTS_DIR, MODELS_DIR, DISEASE_DISPLAY_NAMES


def generate_global_explanations(model, X_test, feature_names: list, disease: str):
    """
    Generate and save global SHAP summary plots.
    Uses TreeExplainer for tree models, KernelExplainer otherwise.
    """
    print(f"  Generating SHAP explanations for {disease}...")

    try:
        # Try TreeExplainer first (much faster for tree-based models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fall back to KernelExplainer with a subsample
        print("    Using KernelExplainer (slower)...")
        background = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test.iloc[:200])

    # Handle multi-output shap_values (take positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease)
    plt.title(f"Feature Importance (SHAP) — {display_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(CHARTS_DIR, f"{disease}_shap_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Summary beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary — {display_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    bee_path = os.path.join(CHARTS_DIR, f"{disease}_shap_summary.png")
    plt.savefig(bee_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Save mean SHAP values as JSON for the web app
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, [float(v) for v in mean_shap]))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    imp_path = os.path.join(CHARTS_DIR, f"{disease}_feature_importance.json")
    with open(imp_path, "w") as f:
        json.dump(importance, f, indent=2)

    print(f"  [OK] SHAP explanations saved for {disease}\n")


def explain_single_prediction(model, input_data: pd.DataFrame, feature_names: list, disease: str):
    """
    Generate SHAP explanation for a single patient prediction.
    Returns dict of {feature: shap_value}.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
    except Exception:
        try:
            background = shap.sample(input_data, min(1, len(input_data)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(input_data)
        except Exception:
            # If SHAP fails, return feature importance from model if available
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return dict(zip(feature_names, [float(v) for v in importances]))
            return {}

    # Handle multi-output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    explanation = dict(zip(feature_names, [float(v) for v in values]))
    explanation = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True))

    return explanation
