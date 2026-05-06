import pandas as pd
# =========================
# 1. LOAD DATA
# =========================
orders = pd.read_csv("datasets/olist_orders_dataset.csv")
order_items = pd.read_csv("datasets/olist_order_items_dataset.csv")
products = pd.read_csv("datasets/olist_products_dataset.csv")
customers = pd.read_csv("datasets/olist_customers_dataset.csv")
payments = pd.read_csv("datasets/olist_order_payments_dataset.csv")
# =========================
# 2. FILTER ORDER STATUS (DELIVERED)
# =========================
orders = orders[orders["order_status"] == "delivered"]
# =========================
# 3. KONVERSI TIMESTAMP
# =========================
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)
# =========================
# 4. FEATURE ENGINEERING TANGGAL
# =========================
orders["order_year"] = orders["order_purchase_timestamp"].dt.year
orders["order_month"] = orders["order_purchase_timestamp"].dt.month
orders["order_day"] = orders["order_purchase_timestamp"].dt.day
orders["order_dayofweek"] = orders["order_purchase_timestamp"].dt.dayofweek
orders["order_hour"] = orders["order_purchase_timestamp"].dt.hour
# =========================
# 5. JOIN ORDERS + CUSTOMERS
# =========================
df = orders.merge(customers, on="customer_id", how="left")
# =========================
# 6. JOIN ORDER ITEMS
# =========================
df = df.merge(order_items, on="order_id", how="left")
# =========================
# 7. JOIN PRODUCTS
# =========================
df = df.merge(products, on="product_id", how="left")
# =========================
# 8. JOIN PAYMENTS
# =========================
df = df.merge(payments, on="order_id", how="left")
# =========================
# 9. SELECT VARIABLE + TANGGAL
# =========================
df_final = df[[
    "order_id",
    "order_year",
    "order_month",
    "order_day",
    "order_dayofweek",
    "order_hour",
    "payment_value",
    "price",
    "freight_value",
    "product_weight_g",
    "product_category_name",
    "customer_state",
    "payment_type"
]]
# =========================
# 10. AGREGASI KE LEVEL ORDER
# =========================
df_final = df_final.groupby("order_id").agg({
    "payment_value": "sum",
    "price": "mean",
    "freight_value": "mean",
    "product_weight_g": "mean",
    "product_category_name": "first",
    "customer_state": "first",
    "payment_type": "first",
    "order_year": "first",
    "order_month": "first",
    "order_day": "first",
    "order_dayofweek": "first",
    "order_hour": "first"
}).reset_index()
# =========================
# 11. SAVE DATASET
# =========================
df_final.to_csv("datasets/olist_orders_success.csv", index=False)
# =========================
print("Success")
print("Shape:", df_final.shape)