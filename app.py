"""
Flask Web Application — Chronic Disease Prediction Dashboard
Routes:
  GET  /          → Main prediction dashboard
  POST /predict   → Run prediction and return results
  GET  /models    → Model comparison data (JSON)
  GET  /api/predict → JSON API endpoint
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DISEASES, DISEASE_DISPLAY_NAMES, DISEASE_ICONS,
    MODELS_DIR, CHARTS_DIR, ALL_FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    FEATURE_DISPLAY_NAMES, FEATURE_RANGES, CATEGORICAL_OPTIONS,
)
from preprocessing.preprocess import preprocess_single_input
from models.explain import explain_single_prediction

app = Flask(__name__)

# ─── Load Models on Startup ────────────────────────────────────────────────────
loaded_models = {}
model_metadata = {}


def load_all_models():
    """Load best trained model and metadata for each disease."""
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f"{disease}_best_model.joblib")
        meta_path = os.path.join(MODELS_DIR, f"{disease}_metadata.json")

        if os.path.exists(model_path):
            loaded_models[disease] = joblib.load(model_path)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    model_metadata[disease] = json.load(f)
            print(f"  [OK] Loaded model for {disease}")
        else:
            print(f"  [!] No trained model found for {disease}")


# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template(
        "index.html",
        diseases=DISEASES,
        disease_names=DISEASE_DISPLAY_NAMES,
        disease_icons=DISEASE_ICONS,
        features=ALL_FEATURES,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        feature_display_names=FEATURE_DISPLAY_NAMES,
        feature_ranges=FEATURE_RANGES,
        categorical_options=CATEGORICAL_OPTIONS,
        models_available=len(loaded_models) > 0,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Run disease prediction for a patient."""
    try:
        data = request.get_json()
        disease = data.get("disease", "diabetes")

        if disease not in loaded_models:
            return jsonify({"error": f"No trained model for {disease}. Run the pipeline first."}), 400

        # Extract patient features
        patient = {}
        for feat in ALL_FEATURES:
            val = data.get(feat)
            if val is not None and val != "":
                if feat in NUMERIC_FEATURES:
                    patient[feat] = float(val)
                else:
                    patient[feat] = str(val)

        # Preprocess
        model = loaded_models[disease]
        meta = model_metadata.get(disease, {})
        feature_names = meta.get("feature_names", ALL_FEATURES)

        input_df = preprocess_single_input(patient, disease)

        # Ensure columns match training features
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        # Predict
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            risk_score = float(proba[1]) * 100
            prediction = int(proba[1] > 0.5)
        else:
            prediction = int(model.predict(input_df)[0])
            risk_score = 100.0 if prediction == 1 else 0.0

        # SHAP explanation
        explanation = explain_single_prediction(model, input_df, feature_names, disease)

        # Determine risk level
        if risk_score >= 70:
            risk_level = "High Risk"
            risk_color = "#ef4444"
        elif risk_score >= 40:
            risk_level = "Moderate Risk"
            risk_color = "#f59e0b"
        else:
            risk_level = "Low Risk"
            risk_color = "#10b981"

        # Top contributing factors
        top_factors = list(explanation.items())[:5]
        factors = [
            {
                "name": FEATURE_DISPLAY_NAMES.get(k, k),
                "impact": round(v, 4),
                "direction": "increases" if v > 0 else "decreases",
            }
            for k, v in top_factors
        ]

        return jsonify({
            "success": True,
            "disease": DISEASE_DISPLAY_NAMES.get(disease, disease),
            "risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "prediction": prediction,
            "factors": factors,
            "model_used": meta.get("best_model", "Unknown"),
            "model_accuracy": round(meta.get("accuracy", 0) * 100, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models")
def models_info():
    """Return model comparison data."""
    result = {}
    for disease in DISEASES:
        # Load metrics
        metrics_path = os.path.join(CHARTS_DIR, f"{disease}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                result[disease] = {
                    "display_name": DISEASE_DISPLAY_NAMES.get(disease, disease),
                    "metrics": json.load(f),
                    "best_model": model_metadata.get(disease, {}).get("best_model", "N/A"),
                }

        # Load feature importance
        imp_path = os.path.join(CHARTS_DIR, f"{disease}_feature_importance.json")
        if os.path.exists(imp_path) and disease in result:
            with open(imp_path) as f:
                result[disease]["feature_importance"] = json.load(f)

    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint — same as /predict but explicitly for programmatic use."""
    return predict()


# ─── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Loading trained models...")
    load_all_models()
    print(f"\n  Starting Flask server...")
    print(f"  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
