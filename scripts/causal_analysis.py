import pandas as pd
from dowhy import CausalModel

df = pd.read_csv('data/mock/mock_customers.csv')

graph = """
digraph {
income -> discount_amount;
income -> purchase_made;
prior_orders -> discount_amount;
prior_orders -> purchase_made;
tenure_days -> discount_amount;
tenure_days -> purchase_made;
season -> discount_amount;
season -> purchase_made;
discount_amount -> purchase_made;
}
"""

model = CausalModel(
    data=df,
    treatment='discount_amount',
    outcome='purchase_made',
    common_causes=['income', 'prior_orders', 'tenure_days', 'season'],
    graph=graph,
)

estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(estimand)