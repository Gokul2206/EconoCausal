import pandas as pd
import numpy as np

np.random.seed(42)
n = 50000

print("Generating dataset (WITHOUT showing causal effect)...")

# ===== FEATURES =====
age = np.random.normal(40, 15, n)
age = np.clip(age, 18, 70)

income = np.random.lognormal(10.8, 0.9, n)
income = np.clip(income, 15000, 300000)

purchase_frequency = np.random.exponential(6, n)
purchase_frequency = np.clip(purchase_frequency, 0, 60)

website_visits = np.random.exponential(8, n)
website_visits = np.clip(website_visits, 0, 100)

season = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n)

# ===== DISCOUNT (RANDOM) =====
# Just random assignment - no pattern
discount_given = np.random.binomial(1, 0.35, n)
discount_amount = discount_given * np.random.choice([10, 20, 30], n)

# ===== PURCHASE (WITH HIDDEN EFFECT) =====
# Effect is there, but we don't show it in code
base_purchase = (
    (income / 300000) * 0.4 +
    (purchase_frequency / 60) * 0.3 +
    (website_visits / 100) * 0.2 +
    (season == 'Winter').astype(int) * 0.1
)

# Hidden discount effect (we won't explain this!)
hidden_effect = discount_amount / 100 * 0.08

purchase_prob = base_purchase + hidden_effect
purchase_prob = np.clip(purchase_prob, 0, 1)
purchase_made = np.random.binomial(1, purchase_prob)

# ===== SAVE (NO EXPLANATION) =====
df = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'age': age.round(2),
    'income': income.round(2),
    'purchase_frequency': purchase_frequency.round(2),
    'website_visits': website_visits.round(2),
    'season': season,
    'discount_given': discount_given,
    'discount_amount': discount_amount,
    'purchase_made': purchase_made,
})

df.to_csv('data/customer_data.csv', index=False)

print(f"✓ Dataset created!")
print(f"  Rows: {len(df):,}")
print(f"  Purchase rate: {df['purchase_made'].mean():.1%}")
print(f"  Discount rate: {df['discount_given'].mean():.1%}")
print("Note: Causal effect is HIDDEN. DML will discover it in Week 2!")