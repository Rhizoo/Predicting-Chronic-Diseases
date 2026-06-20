"""
End-to-End Pipeline Runner
Executes the full ML pipeline: data generation -> preprocessing ->
feature engineering -> model training -> evaluation -> SHAP explanations.

Usage: python run_pipeline.py
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DISEASES, DISEASE_DISPLAY_NAMES
from data.generate_data import generate_all_datasets
from preprocessing.preprocess import preprocess_dataset
from preprocessing.feature_engineering import engineer_features
from models.train import train_all_models
from models.evaluate import evaluate_all_models
from models.explain import generate_global_explanations


def run_pipeline():
    """Execute the complete ML pipeline."""
    start = time.time()

    print("=" * 70)
    print("  CHRONIC DISEASE PREDICTION - ML PIPELINE")
    print("=" * 70)

    # ── Step 1: Generate Data ──────────────────────────────────────────────
    print("\n[1/5] Generating synthetic datasets...")
    generate_all_datasets()

    # ── Step 2-5: Per-disease pipeline ─────────────────────────────────────
    for disease in DISEASES:
        display = DISEASE_DISPLAY_NAMES.get(disease, disease)
        print("-" * 70)
        print(f"  DISEASE: {display}")
        print("-" * 70)

        # Step 2: Preprocess
        print("\n[2/5] Preprocessing...")
        X_train, X_test, y_train, y_test, feature_names = preprocess_dataset(disease)

        # Step 3: Feature Engineering
        print("\n[3/5] Feature Engineering...")
        X_train, X_test, y_train, selected_features = engineer_features(
            X_train, X_test, y_train
        )

        # Step 4: Train Models
        print("\n[4/5] Training Models...")
        results, best_name = train_all_models(
            X_train, X_test, y_train, y_test, disease, selected_features
        )

        # Step 5: Evaluate
        print("[5/5] Evaluating Models...")
        all_metrics = evaluate_all_models(results, X_test, y_test, disease)

        # Step 6: SHAP Explanations (on best model)
        print("  Generating SHAP explanations...")
        best_model = results[best_name]["model"]
        try:
            generate_global_explanations(best_model, X_test, selected_features, disease)
        except Exception as e:
            print(f"    [!] SHAP generation failed: {e}")
            print("    Continuing without SHAP plots...")

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"  [OK] PIPELINE COMPLETE -- {elapsed:.1f}s")
    print("=" * 70)
    print("\n  Next steps:")
    print("    1. Review charts in static/charts/")
    print("    2. Launch web app: python app.py")
    print("    3. Open http://localhost:5000 in your browser\n")


if __name__ == "__main__":
    run_pipeline()
