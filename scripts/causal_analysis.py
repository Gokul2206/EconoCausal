import pandas as pd
from dowhy import CausalModel

print("=" * 70)
print("CAUSAL ANALYSIS: DAG ON MOCK DATA")
print("=" * 70)

df = pd.read_csv('data/mock/mock_customers.csv')

# Planted columns are for validation only. They must never be used here.
PLANTED = ['true_ite', 'true_ite_10', 'true_ite_20']
assert not any(c in df.columns and c in ['income_proxy', 'prior_orders', 'days_since_prev',
                                          'prior_spend', 'order_season']
               for c in PLANTED)

# Encode season as numbers so the model can use it
df['order_season'] = df['order_season'].astype('category').cat.codes

CONFOUNDERS = ['income_proxy', 'prior_orders', 'days_since_prev',
               'prior_spend', 'order_season']

causal_graph = """
digraph {
income_proxy -> discount_given;
income_proxy -> purchase_made;
prior_orders -> discount_given;
prior_orders -> purchase_made;
days_since_prev -> discount_given;
days_since_prev -> purchase_made;
prior_spend -> discount_given;
prior_spend -> purchase_made;
order_season -> discount_given;
order_season -> purchase_made;
discount_given -> purchase_made;
}
"""

print(causal_graph)

model = CausalModel(
    data=df,
    treatment='discount_given',
    outcome='purchase_made',
    common_causes=CONFOUNDERS,
    graph=causal_graph,
)

print("=" * 70)
print("IDENTIFIED ESTIMAND (backdoor adjustment set)")
print("=" * 70)

estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(estimand)