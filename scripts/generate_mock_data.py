import numpy as np
import pandas as pd
import os

# Design fixed before any model is run. Do not change after seeing results.
RNG_SEED = 42
N = 100_000
rng = np.random.default_rng(RNG_SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrate_intercept(logit, target_mean):
    # Shift the intercept so the mean probability equals target_mean.
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() > target_mean:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---- Confounders (measured before any offer) ----
income = rng.normal(0, 1, N)
prior_orders = rng.poisson(0.5, N).astype(float)
tenure_days = rng.exponential(200, N)
season = rng.integers(0, 4, N)              # 0 Winter, 1 Spring, 2 Summer, 3 Autumn

tenure_z = (tenure_days - tenure_days.mean()) / tenure_days.std()
prior_sat = np.minimum(prior_orders, 2.0)   # saturates after 2 prior orders
income_above_1 = (income > 1).astype(float)
tenure_above_300 = (tenure_days > 300).astype(float)
season_summer = (season == 2).astype(float)

# ---- Offer: targeting with mixed functional forms ----
offer_logit = (0.6 * income
               + 0.8 * income_above_1
               + 0.5 * tenure_above_300
               + 0.3 * prior_sat
               + 0.3 * season_summer)
offer_logit = offer_logit + calibrate_intercept(offer_logit, 0.40)
p_offer = sigmoid(offer_logit)
offered = rng.binomial(1, p_offer)

# Each offered customer gets $10 or $20 with equal probability
level = rng.choice([10, 20], N)
discount_amount = np.where(offered == 1, level, 0)

# ---- Planted individual effects (design values) ----
# $10: threshold in income (responds only below income 0)
design_ite_10 = np.where(income < 0, 0.08, 0.02) - 0.01 * income
# $20: depends on prior orders and tenure, different shape from $10
design_ite_20 = (0.04 + 0.03 * prior_sat
                 - 0.04 * np.maximum(tenure_z, 0)
                 - 0.02 * income)

# ---- Outcome: repeat purchase within 90 days ----
base_logit = (0.5 * income
              + 0.6 * prior_sat
              + 0.4 * (tenure_days > 300)
              + 0.2 * (season == 1))
base_logit = base_logit + calibrate_intercept(base_logit, 0.08)
p_base = sigmoid(base_logit)

design_effect = np.where(discount_amount == 20, design_ite_20,
                np.where(discount_amount == 10, design_ite_10, 0.0))

# Keep the outcome probability valid, then store the REALIZED effect as the truth
p_outcome = np.clip(p_base + design_effect, 0.001, 0.999)
true_ite = p_outcome - p_base                 # realized effect, used for validation
purchase_made = rng.binomial(1, p_outcome)

# Per-level realized truths (for validation)
true_ite_10 = np.where(discount_amount == 10, true_ite, design_ite_10)
true_ite_20 = np.where(discount_amount == 20, true_ite, design_ite_20)

df = pd.DataFrame({
    'customer_id': np.arange(N),
    'income': income,
    'prior_orders': prior_orders,
    'tenure_days': tenure_days,
    'season': season,
    'discount_amount': discount_amount,
    'discount_given': (discount_amount > 0).astype(int),
    'purchase_made': purchase_made,
    'true_ite_10': true_ite_10,   # validation only
    'true_ite_20': true_ite_20,   # validation only
    'true_ite': true_ite,         # validation only
})

os.makedirs('data/mock', exist_ok=True)
df.to_csv('data/mock/mock_customers.csv', index=False)