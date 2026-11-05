from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import ks_2samp
import pandas as pd
import json
import os

app = FastAPI(title="EconoCausal API", description="REST API for Causal ML Dynamic Pricing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_PATH = "data/processed/optimization_metrics.json"
BASELINE_DATA_PATH = "data/mock/mock_customers.csv"

@app.get("/")
def read_root():
    return {"status": "EconoCausal Engine is running."}

@app.get("/api/metrics")
def get_optimization_metrics():
    """Returns the business KPIs for the React Natural Language Summary."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics file not found. Run optimization script first.")
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    return metrics

@app.post("/api/detect_drift")
async def detect_drift(file: UploadFile = File(...)):
    """
    Analyzes uploaded campaign data for covariate shift against the baseline model data.
    Uses the Kolmogorov-Smirnov test for continuous variables.
    """
    if not os.path.exists(BASELINE_DATA_PATH):
        raise HTTPException(status_code=500, detail="Baseline data missing. Cannot calculate drift.")

    # 1. Load the Baseline Data (the data the model was trained on)
    df_baseline = pd.read_csv(BASELINE_DATA_PATH)
    
    # 2. Load the New Data from the uploaded file
    try:
        new_data = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file uploaded")

    # The confounders our causal model relies on
    confounders = ['income', 'prior_orders', 'tenure_days']
    drift_report = {}
    is_drifting = False

    # 3. Mathematically compare the distributions
    for feature in confounders:
        if feature not in new_data.columns:
            drift_report[feature] = {"status": "Missing in new data", "drift_detected": True}
            is_drifting = True
            continue
            
        # The KS test compares two distributions. A p-value < 0.05 means they are significantly different.
        stat, p_value = ks_2samp(df_baseline[feature].dropna(), new_data[feature].dropna())
        
        # FIX: Wrap in bool() and float() to strip away NumPy data types
        drift = bool(p_value < 0.05)
        if drift:
            is_drifting = True
            
        drift_report[feature] = {
            "p_value": float(round(p_value, 4)),
            "drift_detected": drift
        }

    return {
        "overall_drift_detected": is_drifting,
        "warning_message": "WARNING: Customer behavior has shifted. Re-train the Causal Model!" if is_drifting else "Data distributions are stable.",
        "feature_breakdown": drift_report
    }
@app.get("/api/scores")
def get_scores():
    """Reads the Causal AI output and serves it to the frontend."""
    file_path = "data/processed/ite_scores_crossfit.csv"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Scores file not found. Run ML pipeline first.")
    
    # Read the file and convert it to a JSON-friendly format
    df = pd.read_csv(file_path)
    
    # We only need specific columns for the dashboard
    cols = ['customer_id', 'ite_10', 'ite_20', 'discount_amount', 'purchase_made']
    if 'ite_diff' in df.columns:
        cols.append('ite_diff')
        
    df = df[[c for c in cols if c in df.columns]]
    
    return df.to_dict(orient="records")