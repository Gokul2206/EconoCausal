import pandas as pd
import os

orders = pd.read_csv('data/raw/olist_orders_dataset.csv',
                     parse_dates=['order_purchase_timestamp'])
orders = orders[orders['order_status'] == 'delivered']
items = pd.read_csv('data/raw/olist_order_items_dataset.csv')
customers = pd.read_csv('data/raw/olist_customers_dataset.csv')
products = pd.read_csv('data/raw/olist_products_dataset.csv')

df = (items.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp']], on='order_id')
           .merge(customers[['customer_id', 'customer_unique_id', 'customer_state']], on='customer_id')
           .merge(products[['product_id', 'product_weight_g']], on='product_id', how='left'))
df = df.drop(columns='customer_id').rename(columns={'customer_unique_id': 'customer_id'})
df['product_weight_g'] = df['product_weight_g'].fillna(df['product_weight_g'].median())

order = (df.groupby(['customer_id', 'order_id'])
           .agg(order_time=('order_purchase_timestamp', 'first'),
                order_value=('price', 'sum'),
                n_items=('order_item_id', 'count'),
                freight=('freight_value', 'sum'),
                avg_weight=('product_weight_g', 'mean'),
                customer_state=('customer_state', 'first'))
           .reset_index()
           .sort_values(['customer_id', 'order_time']))

grp = order.groupby('customer_id')
order['prior_orders'] = grp.cumcount()
order['days_since_prev'] = (order['order_time'] - grp['order_time'].shift()).dt.days.fillna(0)
order['prior_spend'] = grp['order_value'].cumsum() - order['order_value']

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

os.makedirs('data/processed', exist_ok=True)
cols = ['customer_id', 'prior_orders', 'days_since_prev', 'prior_spend',
        'income_proxy', 'order_season', 'n_items', 'freight', 'avg_weight']
order[cols].to_csv('data/processed/order_level_data.csv', index=False)

print(f"Rows: {len(order):,}   Customers: {order['customer_id'].nunique():,}")
print("Saved data/processed/order_level_data.csv")