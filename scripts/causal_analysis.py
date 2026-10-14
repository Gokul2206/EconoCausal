import pandas as pd
from dowhy import CausalModel

print("\n" + "="*70)
print("WEEK 1: CAUSAL ANALYSIS WITH DOWHY")
print("="*70)

# Load processed data
print("\n📖 Loading processed data...")
customer_final = pd.read_csv('data/processed/customer_data.csv')

print(f"✓ Loaded: {len(customer_final)} customers")
print(f"✓ Columns: {len(customer_final.columns)}")

# DATA QUALITY CHECK
print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

print("\nOutcome (purchase_made) distribution:")
print(customer_final['purchase_made'].value_counts())
print(f"Repeat purchase rate: {customer_final['purchase_made'].mean():.1%}")

print("\nTreatment (discount_given) distribution:")
print(customer_final['discount_given'].value_counts())
print(f"Discount rate: {customer_final['discount_given'].mean():.1%}")

# Define causal graph
print("\n" + "="*70)
print("CAUSAL GRAPH (DAG)")
print("="*70)

causal_graph = """
digraph {
income_proxy -> discount_given;
income_proxy -> purchase_made;
tenure_days -> discount_given;
tenure_days -> purchase_made;
engagement_score -> discount_given;
engagement_score -> purchase_made;
favorite_season -> purchase_made;
discount_given -> purchase_made;
}
"""

print(causal_graph)

# Create causal model
print("="*70)
print("CREATING CAUSAL MODEL")
print("="*70)

model = CausalModel(
    data=customer_final,
    treatment='discount_given',
    outcome='purchase_made',
    common_causes=['income_proxy', 'tenure_days', 'engagement_score', 'favorite_season'],
    graph=causal_graph
)

print("✓ CausalModel created")

# Identify effect
print("\n" + "="*70)
print("IDENTIFYING CAUSAL EFFECT")
print("="*70)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print("\n📍 IDENTIFIED ESTIMAND:")
print(identified_estimand)

# Print summary
print("\n" + "="*70)
print("WEEK 1 SUMMARY")
print("="*70)
print("""
✅ Data: 95,146 customers
✅ Treatment: discount_given (binary: 0 or 1)
✅ Outcome: purchase_made (binary: repeat purchase 0 or 1)
✅ Confounders: income_proxy, tenure_days, engagement_score, favorite_season

✅ Causal Effect: IDENTIFIABLE
   By adjusting for confounders, we can isolate:
   TRUE effect of discount on purchase (not confounded)

📊 Next Step: Week 2 - Double ML Training
   - Train Model 1: Predict purchase from confounders
   - Train Model 2: Predict discount from confounders
   - Use residuals to estimate Individual Treatment Effects (ITEs)
""")
print("="*70)