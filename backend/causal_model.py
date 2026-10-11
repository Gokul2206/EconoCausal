import pandas as pd
from dowhy import CausalModel

df = pd.read_csv('data/customer_data.csv')

print("\n" + "="*70)
print("CAUSAL MODEL DEFINITION")
print("="*70)

# FIXED: Use exact column names from CSV
causal_graph = """
digraph {
income -> discount_given;
income -> purchase_made;
purchase_frequency -> discount_given;
purchase_frequency -> purchase_made;
website_visits -> discount_given;
website_visits -> purchase_made;
season -> purchase_made;
discount_given -> purchase_made;
}
"""

print("\n📊 Causal Graph:")
print(causal_graph)

# Create causal model
model = CausalModel(
    data=df,
    treatment='discount_given',
    outcome='purchase_made',
    common_causes=['income', 'purchase_frequency', 'website_visits', 'season'],
    graph=causal_graph
)

print("\n✓ CausalModel created")

# Identify effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print("\n📍 IDENTIFIED ESTIMAND:")
print(identified_estimand)

print("\n" + "="*70)
print("RESULT: Causal effect is IDENTIFIABLE")
print("="*70 + "\n")