import numpy as np
import pandas as pd
import os

# Fixed design. Generated once. Do not change after seeing results.
SEED = 2026
N = 100_000
rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrate_intercept(logit, target_mean):
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() > target_mean:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---- Unobserved driver (NOT saved) ----
engagement = rng.normal(0, 1, N)
engagement_proxy = 0.7 * engagement + rng.normal(0, 0.7, N)

# ---- Observed variables ----
income_true = rng.normal(0, 1, N)
prior_orders = rng.poisson(0.5, N).astype(float)
tenure_days = rng.exponential(200, N)
season = rng.integers(0, 4, N)                 # 0 Winter 1 Spring 2 Summer 3 Autumn
device_mobile = rng.binomial(1, 0.6, N)

# Measurement noise on income: the analysis sees a noisy version
income = income_true + rng.normal(0, 0.3, N)

tenure_z = (tenure_days - tenure_days.mean()) / tenure_days.std()
prior_sat = np.minimum(prior_orders, 2.0)
income_above_1 = (income_true > 1).astype(float)
tenure_above_300 = (tenure_days > 300).astype(float)
season_summer = (season == 2).astype(float)

# ---- Offer: observed and unobserved drivers ----
offer_logit = (0.6 * income_true + 0.8 * income_above_1 + 0.5 * tenure_above_300
               + 0.3 * prior_sat + 0.3 * season_summer
               + 0.5 * engagement)
offer_logit = offer_logit + calibrate_intercept(offer_logit, 0.40)
offered = rng.binomial(1, sigmoid(offer_logit))
level = rng.choice([10, 20], N)
discount_amount = np.where(offered == 1, level, 0)

# ---- Planted effects (validation only) ----
design_ite_10 = np.where(income_true < 0, 0.08, 0.02) - 0.01 * income_true
design_ite_20 = (0.04 + 0.03 * prior_sat
                 - 0.04 * np.maximum(tenure_z, 0) - 0.02 * income_true)
design_effect = np.where(discount_amount == 20, design_ite_20,
                np.where(discount_amount == 10, design_ite_10, 0.0))

# ---- Outcome: observed, unobserved, and outcome-only driver ----
base_logit = (0.5 * income_true + 0.6 * prior_sat + 0.4 * tenure_above_300
              + 0.2 * (season == 1) + 0.5 * engagement
              + 0.3 * device_mobile)
base_logit = base_logit + calibrate_intercept(base_logit, 0.08)
p_base = sigmoid(base_logit)
p_outcome = np.clip(p_base + design_effect, 0.001, 0.999)
true_ite = p_outcome - p_base
purchase_made = rng.binomial(1, p_outcome)

true_ite_10 = np.where(discount_amount == 10, true_ite, design_ite_10)
true_ite_20 = np.where(discount_amount == 20, true_ite, design_ite_20)

df = pd.DataFrame({
    'customer_id': np.arange(N),
    'income': income,
    'prior_orders': prior_orders,
    'tenure_days': tenure_days,
    'season': season,
    'device_mobile': device_mobile,
    'discount_amount': discount_amount,
    'discount_given': (discount_amount > 0).astype(int),
    'purchase_made': purchase_made,
    'engagement_proxy': engagement_proxy,
    'true_ite_10': true_ite_10,   # validation only
    'true_ite_20': true_ite_20,   # validation only
    'true_ite': true_ite,         # validation only
})

os.makedirs('data/mock', exist_ok=True)
df.to_csv('data/mock/mock_customers.csv', index=False)
print(f"Customers: {N:,}  Repeat rate: {purchase_made.mean():.3f}")
print(f"Share by amount: {df['discount_amount'].value_counts(normalize=True).sort_index().round(3).to_dict()}")