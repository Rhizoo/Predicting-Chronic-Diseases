"""
Model Evaluation Module
Computes detailed metrics, generates confusion matrices, ROC curves,
and model comparison charts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHARTS_DIR, DISEASE_DISPLAY_NAMES


def evaluate_model(model, X_test, y_test, model_name: str):
    """Compute all metrics for a single model."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["y_proba"] = y_proba
    else:
        metrics["roc_auc"] = 0.0
        metrics["y_proba"] = None

    metrics["y_pred"] = y_pred
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    return metrics


def evaluate_all_models(results: dict, X_test, y_test, disease: str):
    """Evaluate all trained models and generate comparison charts."""
    print(f"\n  Evaluating models for {disease}...")
    all_metrics = {}

    for name, result in results.items():
        metrics = evaluate_model(result["model"], X_test, y_test, name)
        all_metrics[name] = metrics
        print(f"    {name:25s} -> Acc: {metrics['accuracy']:.3f} | "
              f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1']:.3f} | AUC: {metrics['roc_auc']:.3f}")

    # Generate charts
    _plot_model_comparison(all_metrics, disease)
    _plot_roc_curves(all_metrics, y_test, disease)
    _plot_confusion_matrices(all_metrics, y_test, disease)

    # Save detailed metrics as JSON
    serializable = {}
    for name, m in all_metrics.items():
        serializable[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in m.items()
            if k not in ("y_pred", "y_proba", "confusion_matrix")
        }
        serializable[name]["confusion_matrix"] = m["confusion_matrix"].tolist()

    metrics_path = os.path.join(CHARTS_DIR, f"{disease}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  [OK] Evaluation complete for {disease}\n")
    return all_metrics


def _plot_model_comparison(all_metrics: dict, disease: str):
    """Generate a grouped bar chart comparing all model metrics."""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(all_metrics.keys())

    data = {m: [all_metrics[model][m] for model in model_names] for m in metric_names}
    df = pd.DataFrame(data, index=model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis", edgecolor="white")

    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease)
    ax.set_title(f"Model Comparison — {display_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f"{disease}_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc_curves(all_metrics: dict, y_test, disease: str):
    """Plot ROC curves for all models on one chart."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set2(np.linspace(0, 1, len(all_metrics)))
    for (name, metrics), color in zip(all_metrics.items(), colors):
        if metrics["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})",
                    color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)

    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease)
    ax.set_title(f"ROC Curves — {display_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f"{disease}_roc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrices(all_metrics: dict, y_test, disease: str):
    """Plot confusion matrices for all models in a grid."""
    n_models = len(all_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes_flat = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, metrics) in enumerate(all_metrics.items()):
        ax = axes_flat[idx]
        cm = metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    # Hide empty subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease)
    fig.suptitle(f"Confusion Matrices — {display_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f"{disease}_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
