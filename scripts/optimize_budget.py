import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, vstack

SCORES_PATH = 'data/processed/ite_scores_crossfit.csv'
ALLOC_PATH = 'data/processed/allocation.csv'
METRICS_PATH = 'data/processed/optimization_metrics.json'

LEVELS = [10, 20]                     # discount options; $0 is "no offer"
COLS = {10: 'ite_10', 20: 'ite_20'}

# ASSUMPTION, not in the data: average order value per repeat purchase.
# All revenue figures scale linearly with this value.
AOV = 100
N_RANDOM_SEEDS = 20


def solve_milp(scores, budget):
    """Choose one of {$0, $10, $20} per customer to maximise assumed revenue
    (predicted extra repeats x AOV) subject to total discount spend <= budget.
    Variables: x[i, k] = 1 if customer i gets LEVELS[k]. The $0 option is
    implicit: a customer with no variable set to 1 gets no offer."""
    n = len(scores)
    n_levels = len(LEVELS)
    n_vars = n_levels * n

    gain = np.zeros(n_vars)
    cost = np.zeros(n_vars)
    for k, level in enumerate(LEVELS):
        gain[k::n_levels] = scores[COLS[level]].values * AOV   # assumed revenue
        cost[k::n_levels] = level                              # discount paid

    # Each customer receives at most one level
    rows = np.repeat(np.arange(n), n_levels)
    cols = np.arange(n_vars)
    A_one = csr_matrix((np.ones(n_vars), (rows, cols)), shape=(n, n_vars))

    # Total discount spend within budget
    A_budget = csr_matrix(cost.reshape(1, -1))

    A = vstack([A_one, A_budget]).tocsr()
    lb = np.concatenate([np.zeros(n), [0]])
    ub = np.concatenate([np.ones(n), [budget]])

    res = milp(
        c=-gain,                              # milp minimises, so negate
        constraints=LinearConstraint(A, lb, ub),
        integrality=np.ones(n_vars),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"Solver failed: {res.message}")
    return np.round(res.x).astype(int)


def build_allocation(scores, x):
    """One row per customer. Customers with no level chosen are listed as $0."""
    n_levels = len(LEVELS)
    rows = []
    for i, cid in enumerate(scores['customer_id'].values):
        chosen = 0
        for k, level in enumerate(LEVELS):
            if x[n_levels * i + k] == 1:
                chosen = level
        rows.append({
            'customer_id': cid,
            'discount': chosen,
            'cost': chosen,                       # discount paid = cost
            'predicted_effect': scores[COLS[chosen]].iloc[i] if chosen else 0.0,
        })
    return pd.DataFrame(rows)


def baseline_first_ids(scores, budget):
    """$10 to customers in ID order until the budget runs out."""
    spend, gain = 0, 0.0
    for eff in scores['ite_10']:
        if spend + 10 > budget:
            break
        spend += 10
        gain += eff
    return spend, gain


def baseline_random(scores, budget, seeds=N_RANDOM_SEEDS):
    """$10 to random customers until the budget runs out, over several seeds."""
    results = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(scores))
        spend, gain = 0, 0.0
        for i in order:
            if spend + 10 > budget:
                break
            spend += 10
            gain += scores['ite_10'].iloc[i]
        results.append((spend, gain))
    spends = [s for s, _ in results]
    gains = [g for _, g in results]
    return {
        'spend_mean': float(np.mean(spends)),
        'repeats_mean': float(np.mean(gains)),
        'repeats_min': float(np.min(gains)),
        'repeats_max': float(np.max(gains)),
    }


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

    spend = float(alloc['cost'].sum())
    repeats = float(alloc['predicted_effect'].sum())
    funded = int((alloc['discount'] > 0).sum())
    assert spend <= args.budget + 1e-6, "Budget constraint violated"

    base_spend, base_repeats = baseline_first_ids(scores, args.budget)
    rand = baseline_random(scores, args.budget)

    os.makedirs('data/processed', exist_ok=True)
    alloc.to_csv(ALLOC_PATH, index=False)

    metrics = {
        'budget': args.budget,
        'aov_assumed': AOV,
        'optimized': {
            'spend': spend,
            'customers_funded': funded,
            'predicted_repeats': repeats,
            'assumed_revenue': repeats * AOV,
        },
        'first_ids_baseline': {
            'spend': float(base_spend),
            'predicted_repeats': float(base_repeats),
        },
        'random_baseline': rand,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Budget: ${args.budget:,.0f}   (AOV is an assumption: ${AOV})")
    print(f"Optimized: spend ${spend:,.0f}, customers funded {funded:,}, "
          f"predicted extra repeats {repeats:,.1f}")
    print(f"First-IDs baseline: spend ${base_spend:,.0f}, "
          f"predicted extra repeats {base_repeats:,.1f}")
    print(f"Random baseline ({N_RANDOM_SEEDS} seeds): "
          f"repeats mean {rand['repeats_mean']:,.1f} "
          f"(range {rand['repeats_min']:,.1f} to {rand['repeats_max']:,.1f})")
    print(f"Saved {ALLOC_PATH} and {METRICS_PATH}")


if __name__ == '__main__':
    main()