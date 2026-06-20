"""
Configuration for Chronic Disease Prediction System
Central configuration for paths, features, model hyperparameters, and disease definitions.
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset Configuration ─────────────────────────────────────────────────────
NUM_SAMPLES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Disease Definitions ───────────────────────────────────────────────────────
DISEASES = ["diabetes", "heart_disease", "kidney_disease", "hypertension"]

DISEASE_DISPLAY_NAMES = {
    "diabetes": "Type 2 Diabetes",
    "heart_disease": "Heart Disease",
    "kidney_disease": "Chronic Kidney Disease",
    "hypertension": "Hypertension",
}

DISEASE_ICONS = {
    "diabetes": "🩸",
    "heart_disease": "❤️",
    "kidney_disease": "🫘",
    "hypertension": "💉",
}

# ─── Feature Definitions ───────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp",
    "glucose", "cholesterol", "hba1c", "creatinine",
    "physical_activity_hours",
]

CATEGORICAL_FEATURES = [
    "gender", "smoking_status", "alcohol_consumption", "family_history",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DISPLAY_NAMES = {
    "age": "Age (years)",
    "gender": "Gender",
    "bmi": "BMI",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "glucose": "Fasting Glucose (mg/dL)",
    "cholesterol": "Total Cholesterol (mg/dL)",
    "hba1c": "HbA1c (%)",
    "creatinine": "Serum Creatinine (mg/dL)",
    "smoking_status": "Smoking Status",
    "alcohol_consumption": "Alcohol Consumption",
    "physical_activity_hours": "Physical Activity (hrs/week)",
    "family_history": "Family History",
}

FEATURE_RANGES = {
    "age": (18, 90),
    "bmi": (15.0, 50.0),
    "systolic_bp": (80, 200),
    "diastolic_bp": (50, 130),
    "glucose": (60, 300),
    "cholesterol": (100, 400),
    "hba1c": (4.0, 14.0),
    "creatinine": (0.4, 6.0),
    "physical_activity_hours": (0, 20),
}

CATEGORICAL_OPTIONS = {
    "gender": ["Male", "Female"],
    "smoking_status": ["Never", "Former", "Current"],
    "alcohol_consumption": ["None", "Light", "Moderate", "Heavy"],
    "family_history": ["No", "Yes"],
}

# ─── Model Hyperparameters ──────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Logistic Regression": {
        "model": "LogisticRegression",
        "params": {"max_iter": 1000, "random_state": RANDOM_STATE, "class_weight": "balanced"},
    },
    "Decision Tree": {
        "model": "DecisionTreeClassifier",
        "params": {"max_depth": 10, "random_state": RANDOM_STATE, "class_weight": "balanced"},
    },
    "Random Forest": {
        "model": "RandomForestClassifier",
        "params": {"n_estimators": 200, "max_depth": 15, "random_state": RANDOM_STATE, "class_weight": "balanced", "n_jobs": -1},
    },
    "SVM": {
        "model": "SVC",
        "params": {"kernel": "rbf", "probability": True, "random_state": RANDOM_STATE, "class_weight": "balanced"},
    },
    "XGBoost": {
        "model": "XGBClassifier",
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "random_state": RANDOM_STATE, "use_label_encoder": False, "eval_metric": "logloss"},
    },
    "LightGBM": {
        "model": "LGBMClassifier",
        "params": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "random_state": RANDOM_STATE, "verbose": -1},
    },
}
