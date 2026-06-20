# ChronoPredict — AI Chronic Disease Risk Predictor

A Flask-based web application that predicts the risk of chronic diseases (Diabetes, Heart Disease, Hypertension, Kidney Disease) using machine learning models with SHAP-based explainability.

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** installed ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/Predicting-Chronic-Diseases.git
   cd Predicting-Chronic-Diseases
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**
   ```bash
   python app.py
   ```

6. **Open in browser**
   Navigate to [http://localhost:5000](http://localhost:5000)

## 📁 Project Structure

```
├── app.py                 # Flask web server & routes
├── config.py              # Feature definitions & configuration
├── run_pipeline.py        # ML training pipeline
├── requirements.txt       # Python dependencies
├── data/                  # Training datasets (CSV)
├── models/                # Model training & SHAP explanation modules
├── preprocessing/         # Data preprocessing pipeline
├── saved_models/          # Trained model files (.joblib)
├── static/                # CSS, JS, and frontend assets
└── templates/             # Jinja2 HTML templates
```

## 🔬 Retraining Models

To retrain the models from scratch:
```bash
python run_pipeline.py
```

## 🛠️ Tech Stack

- **Backend:** Flask, scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **Data:** pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
