import pandas as pd
import os

# Load delivered orders, items, and customers
orders = pd.read_csv('data/raw/olist_orders_dataset.csv',
                     parse_dates=['order_purchase_timestamp'])
orders = orders[orders['order_status'] == 'delivered']
items = pd.read_csv('data/raw/olist_order_items_dataset.csv')
customers = pd.read_csv('data/raw/olist_customers_dataset.csv')

# One row per item, with the real person ID
df = (items.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp']], on='order_id')
           .merge(customers[['customer_id', 'customer_unique_id', 'customer_state']], on='customer_id'))

# One row per order
order = (df.groupby(['customer_unique_id', 'order_id'])
           .agg(order_time=('order_purchase_timestamp', 'first'),
                order_value=('price', 'sum'),
                customer_state=('customer_state', 'first'))
           .reset_index()
           .rename(columns={'customer_unique_id': 'customer_id'})
           .sort_values(['customer_id', 'order_time']))

# Pre-order history: uses only earlier orders of the same customer
grp = order.groupby('customer_id')
order['prior_orders'] = grp.cumcount()
order['days_since_prev'] = (order['order_time'] - grp['order_time'].shift()).dt.days.fillna(0)
order['prior_spend'] = grp['order_value'].cumsum() - order['order_value']

# Confounders
state_income = {
    'SP': 0.85, 'RJ': 0.80, 'MG': 0.65, 'RS': 0.78, 'BA': 0.48,
    'PE': 0.42, 'PR': 0.70, 'SC': 0.75, 'ES': 0.70, 'DF': 0.90,
    'PA': 0.45, 'CE': 0.45, 'PB': 0.40, 'GO': 0.65, 'AM': 0.50
}
order['income_proxy'] = order['customer_state'].map(state_income).fillna(0.55)
order['order_season'] = order['order_time'].dt.month.map(
    lambda m: 'Summer' if m in [12, 1, 2] else
              'Autumn' if m in [3, 4, 5] else
              'Winter' if m in [6, 7, 8] else 'Spring')

# Save only what the mock generator uses
os.makedirs('data/processed', exist_ok=True)
cols = ['customer_id', 'prior_orders', 'days_since_prev', 'prior_spend',
        'income_proxy', 'order_season']
order[cols].to_csv('data/processed/order_level_data.csv', index=False)

print(f"Rows: {len(order):,}")
print(f"Customers: {order['customer_id'].nunique():,}")
print("Saved data/processed/order_level_data.csv")