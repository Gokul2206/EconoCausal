import numpy as np
import pandas as pd
import os
from scipy.optimize import brentq

rng = np.random.default_rng(42)

raw = pd.read_csv('data/processed/order_level_data.csv')
df = (raw.sort_values(['customer_id', 'prior_orders'])
         .drop_duplicates('customer_id', keep='last')
         .reset_index(drop=True))
n = len(df)

def z(x):
    x = x.to_numpy(float)
    return (x - x.mean()) / (x.std() + 1e-9)

z_income = z(df['income_proxy'])
z_prior = np.clip(z(df['prior_orders']), -2, 2)
z_recency = np.clip(z(df['days_since_prev']), -2, 2)
z_spend = np.clip(z(df['prior_spend']), -2, 2)
z_items = np.clip(z(df['n_items']), -2, 2)
z_freight = np.clip(z(df['freight']), -2, 2)
z_weight = np.clip(z(df['avg_weight']), -2, 2)
is_summer = (df['order_season'].values == 'Summer').astype(float)
is_spring = (df['order_season'].values == 'Spring').astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calibrate(logit, target):
    return brentq(lambda b: sigmoid(logit + b).mean() - target, -10, 10)

# Offer: same targeting as the frozen version
logit_offer = 0.4 * (0.5 * z_income + 0.3 * z_prior + 0.2 * is_summer)
p_offer = sigmoid(logit_offer + calibrate(logit_offer, 0.32))
got_offer = rng.binomial(1, p_offer)
amount = np.where(got_offer == 1, rng.choice([10, 20], n), 0)

# Planted individual effects (never model inputs)
true_ite_10 = np.clip(0.05 - 0.02 * z_income, -0.02, 0.15)
true_ite_20 = np.clip(0.10 - 0.04 * z_income, -0.02, 0.25)
true_ite = np.where(amount == 20, true_ite_20,
           np.where(amount == 10, true_ite_10, 0.0))

# Outcome: real pre-offer features, fixed coefficients, calibrated to 3%
logit_y = (0.5 * z_income + 0.3 * z_prior + 0.2 * z_recency + 0.2 * z_spend
           + 0.1 * is_spring + 0.3 * z_items + 0.3 * z_weight + 0.2 * z_freight)
p0 = sigmoid(logit_y + calibrate(logit_y, 0.03))
p_y = np.clip(p0 + true_ite, 0.001, 0.999)
repeat = rng.binomial(1, p_y)

out = pd.DataFrame({
    'customer_id': df['customer_id'].values,
    'income_proxy': df['income_proxy'].values,
    'prior_orders': df['prior_orders'].values,
    'days_since_prev': df['days_since_prev'].values,
    'prior_spend': df['prior_spend'].values,
    'order_season': df['order_season'].values,
    'n_items': df['n_items'].values,
    'freight': df['freight'].values,
    'avg_weight': df['avg_weight'].values,
    'discount_amount': amount,
    'discount_given': (amount > 0).astype(int),
    'purchase_made': repeat,
    'true_ite': true_ite,        # validation only
    'true_ite_10': true_ite_10,  # validation only
    'true_ite_20': true_ite_20,  # validation only
})

os.makedirs('data/mock', exist_ok=True)
out.to_csv('data/mock/mock_customers.csv', index=False)

print(f"Customers: {n:,}")
print(f"Share by amount: {out['discount_amount'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
print(f"Repeat rate: {out['purchase_made'].mean():.3f}")
print(f"Naive gap (any discount vs none): "
      f"{out.loc[out.discount_given==1,'purchase_made'].mean() - out.loc[out.discount_given==0,'purchase_made'].mean():.4f}")
print(f"Planted ATE, any discount: {out.loc[out.discount_given==1,'true_ite'].mean():.4f}")