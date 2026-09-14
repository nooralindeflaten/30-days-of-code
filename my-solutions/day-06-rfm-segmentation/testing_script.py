# %% [markdown]
# # Day 6 - 10: Customer Intelligence
# 
# 1. Business Problem
# Who are our customers and where should we focus retention?
# 
# 2. Dataset
# What does each row represent?
# What are the limitations?
# 
# 3. Data Quality
# Missing values
# Duplicates
# Cancellations
# Negative quantities
# Outliers
# 
# 4. Exploratory Analysis
# Revenue over time
# Customers
# Countries
# Orders
# Products
# 
# 5. Customer-Level Feature Engineering
# RFM
# AOV
# Purchase frequency
# Product diversity
# Cancellation behavior
# 
# 6. Customer Segmentation
# RFM analysis
# K-Means
# Cluster interpretation
# 
# 7. Churn Definition
# What does "inactive" mean?
# Why did I choose this threshold?
# 
# 8. Churn Prediction
# Baseline
# Model
# Cross-validation
# Metrics
# 
# 9. Business Analysis
# Who should we target?
# How many customers?
# Expected impact?
# 
# 10. Recommendation
# What should the company do?
# 
# 11. Limitations
# What can't this analysis tell us?
# 
# 12. Next Step
# What experiment/data would I want next?

# %%
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
x = online_retail.data.features 
y = online_retail.data.targets 
data = online_retail.data.original
# metadata 
print(online_retail.metadata) 
  
# variable information 
print(online_retail.variables) 

# %% [markdown]
# # 1. Explore the data
# 
# ### Task
# Customer
# |
# 
# ├── Recency
# 
# ├── Frequency
# 
# ├── Monetary value
# 
# ├── Average order value
# 
# ├── Number of products
# 
# ├── Cancellation rate
# 
# └── Country
# 
# 
# RFM Segmentation
# 
# Champions
# Loyal customers
# Potential loyalists
# At risk
# Lost customers
# New customers
# 
# 
# 
# ### Business question
# 
# "Who are our customers, which customers are most valuable, and where should we focus retention efforts?"
# 
# 
# 
# ### Some objectives from research reports
# Which items / products ’web pages has a
# customer visited? 
# 
# How long has a customer
# stayed with each web page, and in which
# sequence has a customer visited a set of
# products ’web pages?
# 
# 
# Who are the most / least valuable customers
# to the business? 
# 
# 
# What are the distinct
# characteristics of them?
# 
# 
# Who are the most / least loyal customers,
# and how are they characterized?
# 
# 
# What are customers ’purchase behaviour
# patterns? 
# 
# 
# Which products / items have
# customers purchased together often? 
# 
# In
# which sequence the products have been
# purchased?
# 
# 
# Which types of customers are more likely
# to respond to a certain promotion mailing?
# 
# 
# and
# What are the sales patterns in terms of
# various perspectives such as products /
# items, regions and time (weekly, monthly,
# quarterly, yearly and seasonally), and
# so on? 

# %%
import pandas as pd
print("Missing values in features:\n", x.isnull().sum())
print("stats:\n", x.describe())
#print(online_retail.data)
df = data.copy()


print("Number of unique customers:", df['CustomerID'].nunique())
print("Number of unique products:", df['StockCode'].nunique())
print("Number of unique countries:", df['Country'].nunique())
print("Number of unique invoices:", df['InvoiceNo'].nunique())
print("Duplicate rows:", df.duplicated().sum())

# %%
df['IsCancellation'] = df['InvoiceNo'].astype(str).str.upper().str.startswith('C')

print(df["IsCancellation"].value_counts())
print(
    pd.crosstab(
        df["IsCancellation"],
        df["Quantity"] < 0
    )
)

# %% [markdown]
# # Figuring out revenue factors
# - Total amount spent by a customer on a product
# - Customer data
# - Analyze products and sales

# %%
import numpy as np
import matplotlib.pyplot as plt

# Drop rows with missing CustomerID and Description
df = df.dropna(subset=['CustomerID', 'Description'])
df = df.drop_duplicates().copy()


purchases = df[~df['IsCancellation'] & (df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
purchases['Revenue'] = purchases['Quantity'] * purchases['UnitPrice']

print("Purchase rows:", len(purchases))
print("Customers:", purchases["CustomerID"].nunique())
print("Revenue:", purchases["Revenue"].sum())

# %%
total_revenue = purchases["Revenue"].sum()
n_orders = purchases["InvoiceNo"].nunique()
n_customers = purchases["CustomerID"].nunique()

avg_order_value = total_revenue / n_orders

print(f"Revenue: £{total_revenue:,.2f}")
print(f"Orders: {n_orders:,}")
print(f"Customers: {n_customers:,}")
print(f"Average order value: £{avg_order_value:,.2f}")

# %%
# top 10 products by quantity sold and revenue
most_sold_products = purchases.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
highest_revenue_products = purchases.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.barh(most_sold_products.index, most_sold_products.values)
plt.title('Top 10 Products by Quantity Sold')
plt.xlabel('Quantity Sold')
plt.subplot(1, 2, 2)
plt.barh(highest_revenue_products.index, highest_revenue_products.values)
plt.title('Top 10 Products by Revenue')
plt.xlabel('Revenue (£)')
plt.tight_layout()
plt.show()

# %%
country_revenue = purchases.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
plt.barh(country_revenue.index, country_revenue.values)
plt.title('Top 10 Countries by Revenue')
plt.xlabel('Revenue (£)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Time-series data

# %%
purchases['InvoiceDate'] = pd.to_datetime(purchases['InvoiceDate'])
purchases['InvoiceMonth'] = purchases['InvoiceDate'].dt.to_period('M')

monthly_revenue = purchases.groupby('InvoiceMonth')['Revenue'].sum()
plt.figure(figsize=(12, 5))
plt.plot(monthly_revenue.index.astype(str), monthly_revenue.values, marker='o')
plt.title('Monthly Revenue Over Time')
plt.xlabel('Month')
plt.ylabel('Revenue (£)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Preprocessing
# 
# - Make customer data tables 
# - calculate returns, return amount and total amount 

# %%
import pandas as pd
import datetime as dt

# Orders: Unique invoices, items, unique items
orders = purchases.groupby(['CustomerID', 'InvoiceNo']).agg(
    OrderValue=('Revenue', 'sum'),
    Items=('Quantity', 'sum'),
    UniqueItems=('StockCode', 'nunique')
    )

cancelled_orders = df[df['IsCancellation']].groupby('CustomerID').agg(
    CancelledInvoices=('InvoiceNo', 'nunique'),
    CancelledOrders=('IsCancellation', 'sum')
    )

cancelled_orders['TotalInvoices'] = orders.groupby('CustomerID').size()
cancelled_orders['CancelledPercentage'] = (cancelled_orders['CancelledInvoices'] / cancelled_orders['TotalInvoices']) * 100


# %% [markdown]
# # RFM

# %%
snapshot_date = purchases['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = (
    purchases.groupby("CustomerID")
    .agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    )
)

rfm["Recency"] = (
    snapshot_date - rfm["LastPurchase"]
).dt.days

rfm = rfm.drop(columns="LastPurchase")


# %% [markdown]
# # Features

# %%
customers = rfm.copy()
customer_ids = customers.index.tolist()
customer_metrics = pd.DataFrame(index=customer_ids, columns=['AverageOrderValue', 'AverageItemsPerOrder', 'AverageUniqueItemsPerOrder'])
for customer_id in customer_ids:
    customer_orders = orders.loc[customer_id]
    customer_metrics.at[customer_id, 'AverageOrderValue'] = customer_orders['OrderValue'].mean()
    customer_metrics.at[customer_id, 'AverageItemsPerOrder'] = customer_orders['Items'].mean()
    customer_metrics.at[customer_id, 'AverageUniqueItemsPerOrder'] = customer_orders['UniqueItems'].mean()

# Combine RFM and customer metrics
customers = customers.join(customer_metrics)
# Add cancellation metrics

customers = customers.join(cancelled_orders[['CancelledInvoices', 'CancelledOrders', 'CancelledPercentage']])
customers[['CancelledInvoices', 'CancelledOrders', 'CancelledPercentage']] = customers[['CancelledInvoices', 'CancelledOrders', 'CancelledPercentage']].fillna(0)
print(customers.head())

# %%

customers["R_Score"] = pd.qcut(
    customers["Recency"].rank(method="first"),
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)
customers["F_Score"] = pd.qcut(
    customers["Frequency"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)
customers["M_Score"] = pd.qcut(
    customers["Monetary"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

customers["RFM_Score"] = (
    customers["R_Score"]
    + customers["F_Score"]
    + customers["M_Score"]
)

customers["RFM_Segment"] = customers["R_Score"].astype(str)+ customers["F_Score"].astype(str)+ customers["M_Score"].astype(str)

# %% [markdown]
# # Scoring Metrics

# %%
top_customers = customers.sort_values('RFM_Score', ascending=False).head(10)
print(top_customers)

# Segment count (for plotting)
print(customers['RFM_Segment'].value_counts())

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(customers[['Recency', 'Frequency', 'Monetary']])
rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=customers.index)


# %%
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(rfm_scaled_df)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# %%
kmeans = KMeans(n_clusters=4, random_state=42)
customers['Cluster'] = kmeans.fit_predict(rfm_scaled_df)

# %%
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customers, x='Recency', y='Frequency', hue='Cluster', palette='Set1')
plt.title('RFM Segments (Recency vs Frequency)')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customers, x='Monetary', y='Frequency', hue='Cluster', palette='Set2')
plt.title('RFM Segments (Monetary vs Frequency)')
plt.show()

# %%
import plotly.express as px

# Compute cluster summary separately
cluster_summary = customers.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(1)

# Add count of members in each cluster
cluster_summary['Count'] = customers.groupby('Cluster').size()

print(cluster_summary)

# Create the scatter plot
fig = px.scatter_3d(
    customers,
    x='Recency',
    y='Frequency',
    z='Monetary',
    color=customers['Cluster'].astype(str),
    title='3D RFM Cluster Visualization',
    labels={'color': 'Cluster'}
)

fig.show()

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(rfm_scaled_df)

labels = kmeans.labels_
score = silhouette_score(rfm_scaled_df, labels)

print(f"Silhouette Score: {score:.3f}")

# %%
sil_scores = []
K = range(2, 11)  # Test k from 2 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled_df)
    labels = kmeans.labels_
    score = silhouette_score(rfm_scaled_df, labels)
    sil_scores.append(score)

# Plot the scores
plt.figure(figsize=(8,4))
plt.plot(K, sil_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Print best k
best_k = K[sil_scores.index(max(sil_scores))]
print(f"Optimal k (based on silhouette score): {best_k}")


