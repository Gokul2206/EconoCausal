import pandas as pd
import numpy as np
import os

print("="*70)
print("STEP 1: LOAD RAW DATA")
print("="*70)

# Load datasets
customers = pd.read_csv('data/raw/olist_customers_dataset.csv')
orders = pd.read_csv('data/raw/olist_orders_dataset.csv')
order_items = pd.read_csv('data/raw/olist_order_items_dataset.csv')
products = pd.read_csv('data/raw/olist_products_dataset.csv')

print(f"✓ Customers: {customers.shape}")
print(f"✓ Orders: {orders.shape}")
print(f"✓ Order items: {order_items.shape}")
print(f"✓ Products: {products.shape}")

# Merge datasets
print("\n" + "="*70)
print("STEP 2: MERGE DATASETS")
print("="*70)

df = orders.merge(customers, on='customer_id', how='left')
df = df.merge(order_items, on='order_id', how='left')
df = df.merge(products, on='product_id', how='left')

print(f"✓ Merged shape: {df.shape}")

# Clean data
print("\n" + "="*70)
print("STEP 3: CLEAN DATA")
print("="*70)

df = df.dropna(subset=['price', 'product_category_name'])
df = df[df['order_status'] == 'delivered'].copy()

print(f"✓ After cleaning: {len(df)} rows")

# Keep only needed columns
needed_cols = [
    'customer_id',
    'customer_state', 
    'order_purchase_timestamp',
    'price',
    'product_category_name',
    'order_status'
]
df = df[needed_cols].copy()

print("\n" + "="*70)
print("STEP 4: ENGINEER FEATURES")
print("="*70)

# Convert timestamp
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Season
df['month'] = df['order_purchase_timestamp'].dt.month
df['season'] = df['month'].apply(lambda x:
    'Summer' if x in [12, 1, 2] else
    'Autumn' if x in [3, 4, 5] else
    'Winter' if x in [6, 7, 8] else 'Spring'
)

# Income proxy from state
state_income_map = {
    'SP': 0.85, 'RJ': 0.80, 'MG': 0.65, 'RS': 0.78, 'BA': 0.48, 
    'PE': 0.42, 'PR': 0.70, 'SC': 0.75, 'ES': 0.70, 'DF': 0.90,
    'PA': 0.45, 'CE': 0.45, 'PB': 0.40, 'GO': 0.65, 'AM': 0.50
}
df['income_proxy'] = df['customer_state'].map(state_income_map).fillna(0.55)

# Discount
category_median = df.groupby('product_category_name')['price'].transform('median')
df['discount_given'] = (df['price'] < category_median * 0.85).astype(int)
df['discount_amount'] = (category_median - df['price']).clip(0, None)

print("✓ Features engineered")

# Tenure and engagement
print("\n" + "="*70)
print("STEP 5: AGGREGATE TO CUSTOMER LEVEL")
print("="*70)

customer_tenure = df.groupby('customer_id')['order_purchase_timestamp'].agg(['min', 'max'])
customer_tenure.columns = ['first_order_date', 'last_order_date']
max_date = df['order_purchase_timestamp'].max()
customer_tenure['tenure_days'] = (max_date - customer_tenure['first_order_date']).dt.days
customer_tenure['purchase_frequency'] = df.groupby('customer_id').size()
customer_tenure['engagement_score'] = customer_tenure['purchase_frequency'] / customer_tenure['purchase_frequency'].max()

# Final customer dataset
customer_final = df.groupby('customer_id').agg({
    'customer_state': 'first',
    'income_proxy': 'first',
    'season': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Spring',
    'discount_given': 'max',  # FIXED: was 'mean', now 'max'
    'discount_amount': 'mean',
    'price': 'mean'
}).reset_index()

customer_final = customer_final.merge(customer_tenure.reset_index(), on='customer_id')

# FIXED: Engineer outcome properly - repeat purchase (bought 2+ times)
customer_final['purchase_made'] = (customer_final['purchase_frequency'] > 1).astype(int)

# Rename columns
customer_final.columns = ['customer_id', 'customer_state', 'income_proxy', 'favorite_season',
                          'discount_given', 'discount_amount', 'avg_price',
                          'first_order_date', 'last_order_date', 'tenure_days',
                          'purchase_frequency', 'engagement_score', 'purchase_made']

print(f"✓ Aggregated to {len(customer_final)} customers")

# Rename columns
customer_final.columns = ['customer_id', 'customer_state', 'income_proxy', 'favorite_season',
                          'discount_given', 'discount_amount', 'avg_price',
                          'first_order_date', 'last_order_date', 'tenure_days',
                          'purchase_frequency', 'engagement_score', 'purchase_made']

print(f"✓ Aggregated to {len(customer_final)} customers")

# Save
print("\n" + "="*70)
print("STEP 6: SAVE DATA")
print("="*70)

os.makedirs('data/processed', exist_ok=True)
customer_final.to_csv('data/processed/customer_data.csv', index=False)

print(f"✓ Saved to: data/processed/customer_data.csv")
print(f"✓ Rows: {len(customer_final)}")
print(f"✓ Columns: {len(customer_final.columns)}")

# Summary
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Income proxy: {customer_final['income_proxy'].min():.2f} to {customer_final['income_proxy'].max():.2f}")
print(f"Discount rate: {customer_final['discount_given'].mean():.1%}")
print(f"Discount amount: ${customer_final['discount_amount'].mean():.2f}")
print(f"Tenure: {customer_final['tenure_days'].min()} to {customer_final['tenure_days'].max()} days")
print(f"Purchase frequency: {customer_final['purchase_frequency'].min()} to {customer_final['purchase_frequency'].max()}")
print(f"Engagement: {customer_final['engagement_score'].min():.3f} to {customer_final['engagement_score'].max():.3f}")
print("="*70)
print("\n✓ PROCESS_DATA.PY COMPLETED SUCCESSFULLY")