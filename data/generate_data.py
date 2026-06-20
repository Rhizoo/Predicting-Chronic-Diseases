"""
Synthetic Healthcare Data Generator
Generates medically-plausible patient records for 4 chronic diseases:
  - Type 2 Diabetes
  - Heart Disease
  - Chronic Kidney Disease
  - Hypertension

Each disease dataset has ~5000 samples with realistic feature distributions
and target labels based on medical risk factor thresholds with added noise
for realism.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, NUM_SAMPLES, RANDOM_STATE


def _base_population(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a base population with demographic and clinical features."""
    age = rng.normal(52, 15, n).clip(18, 90).astype(int)
    gender = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    bmi = rng.normal(27, 6, n).clip(15, 50).round(1)
    systolic_bp = rng.normal(125, 18, n).clip(80, 200).astype(int)
    diastolic_bp = rng.normal(80, 12, n).clip(50, 130).astype(int)
    glucose = rng.normal(110, 35, n).clip(60, 300).astype(int)
    cholesterol = rng.normal(210, 45, n).clip(100, 400).astype(int)
    hba1c = rng.normal(5.8, 1.2, n).clip(4.0, 14.0).round(1)
    creatinine = rng.exponential(0.9, n).clip(0.4, 6.0).round(2)
    smoking = rng.choice(["Never", "Former", "Current"], n, p=[0.50, 0.30, 0.20])
    alcohol = rng.choice(["None", "Light", "Moderate", "Heavy"], n, p=[0.30, 0.35, 0.25, 0.10])
    activity = rng.normal(4, 3, n).clip(0, 20).round(1)
    family_history = rng.choice(["No", "Yes"], n, p=[0.65, 0.35])

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "glucose": glucose,
        "cholesterol": cholesterol,
        "hba1c": hba1c,
        "creatinine": creatinine,
        "smoking_status": smoking,
        "alcohol_consumption": alcohol,
        "physical_activity_hours": activity,
        "family_history": family_history,
    })


def _diabetes_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign diabetes label based on medical risk factors."""
    score = np.zeros(len(df))
    score += (df["glucose"] > 126).astype(float) * 3.0
    score += (df["hba1c"] > 6.5).astype(float) * 3.0
    score += (df["bmi"] > 30).astype(float) * 2.0
    score += (df["age"] > 45).astype(float) * 1.5
    score += (df["family_history"] == "Yes").astype(float) * 2.0
    score += (df["physical_activity_hours"] < 2).astype(float) * 1.0
    score += (df["cholesterol"] > 240).astype(float) * 0.5
    # Normalize to probability and add noise
    prob = 1 / (1 + np.exp(-(score - 6) / 2))
    noise = rng.normal(0, 0.08, len(df))
    prob = np.clip(prob + noise, 0, 1)
    return (prob > 0.5).astype(int)


def _heart_disease_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign heart disease label based on cardiac risk factors."""
    score = np.zeros(len(df))
    score += (df["age"] > 55).astype(float) * 2.5
    score += (df["cholesterol"] > 240).astype(float) * 3.0
    score += (df["systolic_bp"] > 140).astype(float) * 2.5
    score += (df["smoking_status"] == "Current").astype(float) * 2.5
    score += (df["bmi"] > 30).astype(float) * 1.5
    score += (df["gender"] == "Male").astype(float) * 1.0
    score += (df["physical_activity_hours"] < 2).astype(float) * 1.5
    score += (df["family_history"] == "Yes").astype(float) * 2.0
    score += (df["glucose"] > 126).astype(float) * 1.0
    prob = 1 / (1 + np.exp(-(score - 7) / 2.5))
    noise = rng.normal(0, 0.08, len(df))
    prob = np.clip(prob + noise, 0, 1)
    return (prob > 0.5).astype(int)


def _kidney_disease_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign CKD label based on renal risk factors."""
    score = np.zeros(len(df))
    score += (df["creatinine"] > 1.3).astype(float) * 3.5
    score += (df["systolic_bp"] > 140).astype(float) * 2.5
    score += (df["glucose"] > 126).astype(float) * 2.0
    score += (df["age"] > 60).astype(float) * 2.0
    score += (df["bmi"] > 30).astype(float) * 1.0
    score += (df["smoking_status"] == "Current").astype(float) * 1.5
    score += (df["family_history"] == "Yes").astype(float) * 1.5
    prob = 1 / (1 + np.exp(-(score - 6) / 2))
    noise = rng.normal(0, 0.08, len(df))
    prob = np.clip(prob + noise, 0, 1)
    return (prob > 0.5).astype(int)


def _hypertension_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign hypertension label based on BP and related risk factors."""
    score = np.zeros(len(df))
    score += (df["systolic_bp"] > 130).astype(float) * 3.5
    score += (df["diastolic_bp"] > 85).astype(float) * 3.0
    score += (df["bmi"] > 30).astype(float) * 2.0
    score += (df["age"] > 50).astype(float) * 1.5
    score += (df["smoking_status"] == "Current").astype(float) * 1.5
    score += (df["alcohol_consumption"] == "Heavy").astype(float) * 2.0
    score += (df["physical_activity_hours"] < 2).astype(float) * 1.5
    score += (df["family_history"] == "Yes").astype(float) * 1.5
    score += (df["cholesterol"] > 240).astype(float) * 1.0
    prob = 1 / (1 + np.exp(-(score - 7) / 2.5))
    noise = rng.normal(0, 0.08, len(df))
    prob = np.clip(prob + noise, 0, 1)
    return (prob > 0.5).astype(int)


LABEL_GENERATORS = {
    "diabetes": _diabetes_label,
    "heart_disease": _heart_disease_label,
    "kidney_disease": _kidney_disease_label,
    "hypertension": _hypertension_label,
}


def generate_all_datasets():
    """Generate synthetic datasets for all four chronic diseases."""
    rng = np.random.default_rng(RANDOM_STATE)
    os.makedirs(DATA_DIR, exist_ok=True)

    for disease, label_fn in LABEL_GENERATORS.items():
        print(f"  Generating {disease} dataset ({NUM_SAMPLES} samples)...")
        df = _base_population(NUM_SAMPLES, rng)

        # Inject some missing values for realism (2% random NaN in numeric cols)
        numeric_cols = ["bmi", "glucose", "cholesterol", "hba1c", "creatinine"]
        for col in numeric_cols:
            mask = rng.random(NUM_SAMPLES) < 0.02
            df.loc[mask, col] = np.nan

        df["target"] = label_fn(df, rng)

        path = os.path.join(DATA_DIR, f"{disease}.csv")
        df.to_csv(path, index=False)

        pos_rate = df["target"].mean() * 100
        print(f"    -> Saved to {path}  |  Positive rate: {pos_rate:.1f}%")

    print("  [OK] All datasets generated.\n")


if __name__ == "__main__":
    generate_all_datasets()
