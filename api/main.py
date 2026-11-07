from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import ks_2samp
import pandas as pd
import json
import os
import numpy as np
from scipy.optimize import linprog

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
@app.get("/api/optimize")
def optimize_allocation(budget: float = 5000.0):
    """
    Week 3 SciPy Optimization Solver:
    Maximizes causal lift subject to a strict budget constraint.
    """
    file_path = "data/processed/ite_scores_crossfit.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Scores file not found.")
        
    df = pd.read_csv(file_path).fillna(0)
    
    # 1. Pre-filter: Only target customers where a discount actually causes a positive lift
    candidates = df[(df['ite_10'] > 0) | (df['ite_20'] > 0)].copy()
    
    # Limit to top 1000 prospects to ensure the solver runs instantly on any hardware
    candidates['max_ite'] = candidates[['ite_10', 'ite_20']].max(axis=1)
    candidates = candidates.sort_values('max_ite', ascending=False).head(1000)
    
    n = len(candidates)
    if n == 0:
        return {"summary": {"budget_utilized": 0, "projected_revenue_lift": 0}, "prescriptions": []}

    # 2. Formulate Linear Programming Problem (Knapsack)
    # Objective: Maximize total ITE (SciPy minimizes, so we make ITE negative)
    # Variables: [customer1_$10, customer1_$20, customer2_$10, customer2_$20...]
    c = np.zeros(2 * n)
    c[0::2] = -candidates['ite_10'].values
    c[1::2] = -candidates['ite_20'].values
    
    # Constraint A: Total Cost Cannot Exceed Budget
    A_budget = np.zeros((1, 2 * n))
    A_budget[0, 0::2] = 10
    A_budget[0, 1::2] = 20
    b_budget = np.array([budget])
    
    # Constraint B: Mutually Exclusive (Cannot give a user BOTH $10 and $20)
    A_eq = np.zeros((n, 2 * n))
    for i in range(n):
        A_eq[i, 2*i] = 1
        A_eq[i, 2*i + 1] = 1
    b_eq = np.ones(n)
    
    A_ub = np.vstack([A_budget, A_eq])
    b_ub = np.concatenate([b_budget, b_eq])
    
    # 3. Solve using SciPy
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')
    
    x = res.x
    candidates['offer_10'] = np.round(x[0::2])
    candidates['offer_20'] = np.round(x[1::2])
    
    # 4. Format Output for React
    candidates['prescription'] = 0
    candidates.loc[candidates['offer_10'] == 1, 'prescription'] = 10
    candidates.loc[candidates['offer_20'] == 1, 'prescription'] = 20
    
    final_targets = candidates[candidates['prescription'] > 0]
    
    total_spent = float((final_targets['prescription']).sum())
    projected_lift = float((final_targets['offer_10'] * final_targets['ite_10'] + 
                            final_targets['offer_20'] * final_targets['ite_20']).sum())
    
    results = final_targets[['customer_id', 'prescription', 'ite_10', 'ite_20']].to_dict(orient="records")
    
    return {
        "summary": {
            "budget_utilized": total_spent,
            "projected_revenue_lift": projected_lift,
            "customers_targeted": len(final_targets)
        },
        "prescriptions": results
    }