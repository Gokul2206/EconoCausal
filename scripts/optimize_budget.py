import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix, vstack

SCORES_PATH = 'data/processed/ite_scores_crossfit.csv'
OUT_PATH = 'data/processed/allocation.csv'
LEVELS = [10, 20]
COLS = {10: 'ite_10', 20: 'ite_20'}

# Add this near the top with your constants
AOV = 100  # Assume an Average Order Value of $100 per repeat purchase


def solve_milp(scores, budget):
    n = len(scores)
    n_vars = 2 * n

    c = np.zeros(n_vars)
    cost = np.zeros(n_vars)
    for k, level in enumerate(LEVELS):
        # Multiply the predicted extra repeats by AOV to optimize for Revenue
        c[k::2] = -(scores[COLS[level]].values * AOV)  
        cost[k::2] = level

    # Constraint 1: at most one level per customer (sparse, two entries per row)
    rows = np.repeat(np.arange(n), 2)
    cols = np.arange(n_vars)
    A_one = csr_matrix((np.ones(n_vars), (rows, cols)), shape=(n, n_vars))

    # Constraint 2: total spend within budget (one dense-looking row, stored sparse)
    A_budget = csr_matrix(cost.reshape(1, -1))

    A = vstack([A_one, A_budget]).tocsr()
    lb = np.concatenate([np.zeros(n), [0]])
    ub = np.concatenate([np.ones(n), [budget]])
    constraints = LinearConstraint(A, lb, ub)

    res = milp(c=c, constraints=constraints,
               integrality=np.ones(n_vars), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"Solver failed: {res.message}")
    return np.round(res.x).astype(int)

def build_allocation(scores, x):
    rows = []
    for i, cid in enumerate(scores['customer_id'].values):
        for k, level in enumerate(LEVELS):
            if x[2 * i + k] == 1:
                rows.append({
                    'customer_id': cid,
                    'discount': level,
                    'predicted_effect': scores[COLS[level]].iloc[i],
                    'cost': level,
                })
    return pd.DataFrame(rows)


def baseline_first_ids(scores, budget):
    spend, gain = 0, 0.0
    for eff in scores['ite_10']:
        if spend + 10 > budget:
            break
        spend += 10
        gain += eff
    return spend, gain


def baseline_random(scores, budget, seed=42):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(scores))
    spend, gain = 0, 0.0
    for i in order:
        if spend + 10 > budget:
            break
        spend += 10
        gain += scores['ite_10'].iloc[i]
    return spend, gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=float, default=50000,
                        help='Marketing budget in dollars')
    args = parser.parse_args()
    if args.budget <= 0:
        raise SystemExit('Budget must be greater than 0')

    scores = pd.read_csv(SCORES_PATH)
    x = solve_milp(scores, args.budget)
    alloc = build_allocation(scores, x)

    spend = alloc['cost'].sum()
    gain = alloc['predicted_effect'].sum()

    os.makedirs('data/processed', exist_ok=True)
    alloc.to_csv(OUT_PATH, index=False)

    base_spend, base_gain = baseline_first_ids(scores, args.budget)
    rand_spend, rand_gain = baseline_random(scores, args.budget)

    print(f"Budget: ${args.budget:,.0f}")
    print(f"Optimized (SciPy milp): spend ${spend:,.0f}, customers {len(alloc):,}")
    print(f"  -> Predicted Revenue: ${gain * AOV:,.2f} (ROI: {((gain * AOV) - spend) / spend * 100:.1f}%)")
    
    print(f"First-IDs baseline: spend ${base_spend:,.0f}")
    print(f"  -> Predicted Revenue: ${base_gain * AOV:,.2f} (ROI: {((base_gain * AOV) - base_spend) / base_spend * 100:.1f}%)")
    
    print(f"Random baseline: spend ${rand_spend:,.0f}")
    print(f"  -> Predicted Revenue: ${rand_gain * AOV:,.2f} (ROI: {((rand_gain * AOV) - rand_spend) / rand_spend * 100:.1f}%)")
    
    print(f"Saved {OUT_PATH}")


if __name__ == '__main__':
    main()