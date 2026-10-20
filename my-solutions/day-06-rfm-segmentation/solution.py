# %% [markdown]
# # Challenge 01 — E-Commerce Business Diagnosis
# UK-based online retailer, ~1 year of transactions (UCI Online Retail dataset)

# %%
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# %% [markdown]
# ## 1. Audit the raw data
# We look before we touch anything: dimensions, dtypes, missingness, duplicates,
# and the specific data-quality flags the brief calls out. Nothing gets dropped
# in this section.

# %%
online_retail = fetch_ucirepo(id=352)
data = online_retail.data.original
df = data.copy()

df["IsCancellation"] = df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

audit = {
    "dataset_dimensions": df.shape,
    "dtypes": df.dtypes.to_dict(),
    "missing_values": df.isnull().sum().to_dict(),
    "duplicate_rows": int(df.duplicated().sum()),
    "missing_customer_id": int(df["CustomerID"].isnull().sum()),
    "missing_description": int(df["Description"].isnull().sum()),
    "zero_or_negative_qty": int((df["Quantity"] <= 0).sum()),
    "zero_or_negative_price": int((df["UnitPrice"] <= 0).sum()),
    "cancellation_rows": int(df["IsCancellation"].sum()),
}

print("=== DATA QUALITY AUDIT ===")
for k, v in audit.items():
    if k not in ("dtypes", "missing_values"):
        print(f"{k}: {v}")
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())

# Are missing descriptions and missing CustomerIDs the same rows, or different problems?
missing_desc_no_custid = df[df["Description"].isnull() & df["CustomerID"].isnull()].shape[0]
missing_desc_total = df["Description"].isnull().sum()
print(f"\nRows missing BOTH Description and CustomerID: {missing_desc_no_custid} / {missing_desc_total} missing-description rows")

# StockCodes tied to missing descriptions — are they real products or adjustments (POST, D, M, BANK CHARGES, etc.)?
missing_description_stockcodes = df[df["Description"].isnull()]["StockCode"].unique()
print(f"\nStockCodes with missing descriptions: {missing_description_stockcodes[:20]}")

# Zero/negative quantity and price rows — inspect what they actually are before deciding
zero_quantity = df[df["Quantity"] <= 0]
zero_price = df[df["UnitPrice"] <= 0]
print(f"\nZero/negative Quantity rows: {zero_quantity.shape[0]} "
      f"({(zero_quantity['IsCancellation']).sum()} are cancellations)")
print(f"Zero/negative UnitPrice rows: {zero_price.shape[0]} "
      f"(sample StockCodes: {zero_price['StockCode'].unique()[:10]})")

cancellation_example = df[df["IsCancellation"]].head(5)
print("\nSample cancellation rows:\n", cancellation_example)

# %% [markdown]
# **Audit takeaways (documented, not yet acted on):**
# - Missing `CustomerID` rows cannot be used for any *customer-level* analysis
#   (RFM, concentration, cohort work) but CAN still count toward gross revenue
#   if we're only answering "how much did the business sell."
# - Missing `Description` rows are almost always non-product adjustment codes
#   (postage, bank charges, manual entries) rather than real inventory — these
#   should not be treated as ordinary products in product-performance analysis.
# - Negative/zero `Quantity` overlaps heavily with cancellations (expected: a
#   cancellation is a negative-quantity entry against a prior order) — but not
#   perfectly, so we check both instead of assuming full overlap.
# - Zero `UnitPrice` rows are typically non-sale adjustments (samples, damages,
#   free items) — they contribute zero revenue by construction, so they don't
#   need to be dropped, they just naturally fall out of revenue sums.

# %% [markdown]
# ## 2. Define valid purchases
# **Definition used in this analysis:**
# - **Completed purchase** = not a cancellation AND `Quantity > 0` AND `UnitPrice > 0`
#   AND has a non-null `CustomerID` (for customer-level metrics) — for
#   business-wide KPIs that don't require a customer, we relax the CustomerID
#   requirement.
# - **Cancellation / return** = InvoiceNo starts with `C` (regardless of
#   quantity/price sign) — kept as its own group, not discarded, because it
#   feeds cancellation-rate metrics later.
# - **Unusable / ambiguous** = missing `Description` with non-numeric
#   `StockCode` (adjustment/admin codes like POST, D, M, BANK CHARGES) — these
#   are excluded from *product* performance analysis but may still count in
#   revenue totals since money still changed hands.

# %%
df_clean = df.dropna(subset=["CustomerID", "Description"]).copy()
df_clean.drop_duplicates(inplace=True)
print("Cleaned dataset dimensions:", df_clean.shape)

customer_transactions = df_clean.copy()

purchases = customer_transactions.loc[
    (~customer_transactions["IsCancellation"])
    & (customer_transactions["Quantity"] > 0)
    & (customer_transactions["UnitPrice"] > 0)
].copy()

cancellations_raw = customer_transactions.loc[customer_transactions["IsCancellation"]].copy()
# Numeric-stockcode cancellations = real product returns (vs. admin/adjustment cancellations)
cancellations = cancellations_raw[cancellations_raw["StockCode"].str.isnumeric()].copy()

print(f"Completed purchases: {purchases.shape[0]:,} rows")
print(f"Cancellations (all): {cancellations_raw.shape[0]:,} rows")
print(f"Cancellations (real product returns, numeric StockCode): {cancellations.shape[0]:,} rows")

# %% [markdown]
# ## 3. Business KPIs
# Gross vs. net is the key distinction the brief asks for: gross = all sales
# activity before subtracting returns; net = what the business actually keeps.

# %%
gross_revenue = customer_transactions.loc[customer_transactions["Quantity"] > 0, "Revenue"].sum()
cancellation_value = cancellations["Revenue"].abs().sum()  # cancellations store negative revenue
net_revenue = purchases["Revenue"].sum()  # equivalent to gross - cancellations restricted to real returns

num_orders = purchases["InvoiceNo"].nunique()
num_customers = purchases["CustomerID"].nunique()
num_products_sold = purchases["StockCode"].nunique()
units_sold = purchases["Quantity"].sum()
avg_order_value = net_revenue / num_orders
items_per_order = units_sold / num_orders

total_invoices_all = customer_transactions["InvoiceNo"].nunique()
cancelled_invoices = cancellations_raw["InvoiceNo"].nunique()
cancellation_rate = cancelled_invoices / total_invoices_all

kpi_summary = pd.Series({
    "Gross Revenue (£)": gross_revenue,
    "Cancellation Value (£)": cancellation_value,
    "Net Revenue (£)": net_revenue,
    "Number of Orders (completed)": num_orders,
    "Number of Customers": num_customers,
    "Number of Distinct Products Sold": num_products_sold,
    "Units Sold": units_sold,
    "Average Order Value (£)": avg_order_value,
    "Items per Order": items_per_order,
    "Total Invoices (incl. cancellations)": total_invoices_all,
    "Cancelled Invoices": cancelled_invoices,
    "Cancellation Rate": cancellation_rate,
})

print("=== KPI SUMMARY ===")
print(kpi_summary.to_string())

# %% [markdown]
# ## 4. Revenue over time
# Monthly granularity (dataset spans ~1 year, so monthly is the natural
# resolution for spotting trend/seasonality without over-fitting to noise).

# %%
purchases["Month"] = purchases["InvoiceDate"].dt.to_period("M")

monthly = (
    purchases.groupby("Month")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
        UnitsSold=("Quantity", "sum"),
        ProductsSold=("StockCode", "nunique"),
    )
)
monthly.index = monthly.index.to_timestamp()

monthly["AverageOrderValue"] = monthly["Revenue"] / monthly["Orders"]
monthly["OrdersPerCustomer"] = monthly["Orders"] / monthly["Customers"]
monthly["ItemsPerOrder"] = monthly["UnitsSold"] / monthly["Orders"]
monthly["RevenuePerItem"] = monthly["Revenue"] / monthly["UnitsSold"]
monthly["RevenueGrowthPct"] = monthly["Revenue"].pct_change() * 100

print(monthly)

# Monthly cancellation rate, to see if returns spike in specific months
monthly_cancel = (
    customer_transactions.assign(Month=customer_transactions["InvoiceDate"].dt.to_period("M"))
    .groupby("Month")
    .agg(TotalInvoices=("InvoiceNo", "nunique"))
)
monthly_cancel_only = (
    cancellations_raw.assign(Month=cancellations_raw["InvoiceDate"].dt.to_period("M"))
    .groupby("Month")["InvoiceNo"].nunique()
    .rename("CancelledInvoices")
)
monthly_cancel = monthly_cancel.join(monthly_cancel_only).fillna(0)
monthly_cancel["CancellationRate"] = monthly_cancel["CancelledInvoices"] / monthly_cancel["TotalInvoices"]
monthly_cancel.index = monthly_cancel.index.to_timestamp()
print("\nMonthly cancellation rate:\n", monthly_cancel[["CancellationRate"]])

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(monthly.index, monthly["Revenue"], marker="o")
axes[0, 0].set_title("Monthly Revenue")
axes[0, 0].tick_params(axis="x", rotation=45)

axes[0, 1].bar(monthly.index, monthly["RevenueGrowthPct"])
axes[0, 1].axhline(0, linestyle="--", color="grey")
axes[0, 1].set_title("Month-over-Month Revenue Growth (%)")
axes[0, 1].tick_params(axis="x", rotation=45)

axes[1, 0].plot(monthly.index, monthly["Orders"], marker="o", label="Orders")
axes[1, 0].plot(monthly.index, monthly["Customers"], marker="o", label="Customers")
axes[1, 0].set_title("Monthly Orders vs Active Customers")
axes[1, 0].legend()
axes[1, 0].tick_params(axis="x", rotation=45)

axes[1, 1].plot(monthly.index, monthly["AverageOrderValue"], marker="o")
axes[1, 1].set_title("Average Order Value Over Time")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Revenue driver decomposition
# Revenue = Customers × Orders-per-Customer × Average-Order-Value. This
# decomposes each month's revenue *change* into how much came from each factor,
# rather than just reporting raw deltas side by side.

# %%
def decompose_revenue_drivers(monthly_df):
    d = monthly_df.copy()
    d["Revenue_prev"] = d["Revenue"].shift(1)
    d["Customers_prev"] = d["Customers"].shift(1)
    d["OrdersPerCustomer_prev"] = d["OrdersPerCustomer"].shift(1)
    d["AOV_prev"] = d["AverageOrderValue"].shift(1)

    # % change in each driver
    d["CustomerGrowthPct"] = (d["Customers"] / d["Customers_prev"] - 1) * 100
    d["OrdersPerCustomerGrowthPct"] = (d["OrdersPerCustomer"] / d["OrdersPerCustomer_prev"] - 1) * 100
    d["AOVGrowthPct"] = (d["AverageOrderValue"] / d["AOV_prev"] - 1) * 100

    # Which single driver moved the most this month (crude but interpretable
    # "primary driver" flag)
    driver_cols = ["CustomerGrowthPct", "OrdersPerCustomerGrowthPct", "AOVGrowthPct"]
    d["PrimaryDriver"] = d[driver_cols].abs().idxmax(axis=1)
    d["PrimaryDriver"] = d["PrimaryDriver"].replace({
        "CustomerGrowthPct": "Customer count",
        "OrdersPerCustomerGrowthPct": "Orders per customer",
        "AOVGrowthPct": "Average order value",
    })
    return d[["Revenue", "RevenueGrowthPct", "CustomerGrowthPct",
              "OrdersPerCustomerGrowthPct", "AOVGrowthPct", "PrimaryDriver"]]

driver_decomposition = decompose_revenue_drivers(monthly)
print("\n=== Revenue Driver Decomposition ===")
print(driver_decomposition)

# %% [markdown]
# ## 5. Geographic analysis

# %%
country_summary = (
    purchases.groupby("Country")
    .agg(
        Revenue=("Revenue", "sum"),
        Customers=("CustomerID", "nunique"),
        Orders=("InvoiceNo", "nunique"),
    )
)
country_summary["AvgOrderValue"] = country_summary["Revenue"] / country_summary["Orders"]
country_summary["RevenueSharePct"] = country_summary["Revenue"] / country_summary["Revenue"].sum() * 100
country_summary = country_summary.sort_values("Revenue", ascending=False)

print("=== Country Summary (all) ===")
print(country_summary)

# Flag thin-data countries so small-sample noise isn't over-interpreted
country_summary["LowSampleWarning"] = country_summary["Customers"] < 10
print("\nCountries with fewer than 10 customers (interpret cautiously):")
print(country_summary[country_summary["LowSampleWarning"]])

uk_share = country_summary.loc["United Kingdom", "RevenueSharePct"] if "United Kingdom" in country_summary.index else None
print(f"\nUK share of total revenue: {uk_share:.1f}%" if uk_share else "UK not found in data")

# %%
top_countries_ex_uk = country_summary.drop(index="United Kingdom", errors="ignore").head(10)
plt.figure(figsize=(8, 5))
plt.barh(top_countries_ex_uk.index, top_countries_ex_uk["Revenue"])
plt.title("Top 10 Countries by Revenue (excl. UK)")
plt.xlabel("Revenue (£)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Product performance

# %%
product_summary = (
    purchases.groupby("Description")
    .agg(
        Revenue=("Revenue", "sum"),
        UnitsSold=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
    )
    .sort_values("Revenue", ascending=False)
)

most_sold_by_qty = product_summary.sort_values("UnitsSold", ascending=False).head(10)
highest_revenue_products = product_summary.head(10)
most_frequently_purchased = product_summary.sort_values("Orders", ascending=False).head(10)

# Overlap check: are top-revenue and top-volume products the same?
top_rev_set = set(highest_revenue_products.index)
top_qty_set = set(most_sold_by_qty.index)
overlap = top_rev_set & top_qty_set
print(f"Overlap between top-10 revenue and top-10 volume products: {len(overlap)}/10")
print("Products in both lists:", overlap)

# Cancellation activity by product (using real product-return cancellations)
product_cancellations = (
    cancellations.groupby("Description")
    .agg(CancelledUnits=("Quantity", "sum"), CancelledOrders=("InvoiceNo", "nunique"))
)
product_cancellations["CancelledUnits"] = product_cancellations["CancelledUnits"].abs()
product_sold_units = purchases.groupby("Description")["Quantity"].sum().rename("SoldUnits")

product_cancel_rate = product_cancellations.join(product_sold_units, how="left").fillna(0)
product_cancel_rate["CancellationRate"] = (
    product_cancel_rate["CancelledUnits"]
    / (product_cancel_rate["CancelledUnits"] + product_cancel_rate["SoldUnits"])
)
# Only meaningful for products with enough volume to not be noise
top_cancelled_products = product_cancel_rate[
    product_cancel_rate["SoldUnits"] + product_cancel_rate["CancelledUnits"] >= 20
].sort_values("CancellationRate", ascending=False).head(10)

print("\n=== Highest-Revenue Products ===\n", highest_revenue_products)
print("\n=== Highest-Volume Products ===\n", most_sold_by_qty)
print("\n=== Most Frequently Purchased (by order count) ===\n", most_frequently_purchased)
print("\n=== Products with Highest Cancellation Rate (min. 20 combined units) ===\n", top_cancelled_products)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(most_sold_by_qty.index, most_sold_by_qty["UnitsSold"])
axes[0].set_title("Top 10 Products by Units Sold")
axes[0].invert_yaxis()

axes[1].barh(highest_revenue_products.index, highest_revenue_products["Revenue"])
axes[1].set_title("Top 10 Products by Revenue")
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Customer concentration (Pareto analysis)

# %%
revenue_per_customer = (
    purchases.groupby("CustomerID")["Revenue"]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
    .reset_index(drop=True)
)

revenue_per_customer["cum_revenue_pct"] = (
    revenue_per_customer["Revenue"].cumsum() / revenue_per_customer["Revenue"].sum() * 100
)
revenue_per_customer["customer_rank_pct"] = (
    (revenue_per_customer.index + 1) / len(revenue_per_customer) * 100
)

# Concrete Pareto readout: revenue share held by top 10% / 20% of customers
def revenue_share_of_top_pct(df, pct):
    n = int(len(df) * pct / 100)
    return df.iloc[:n]["Revenue"].sum() / df["Revenue"].sum() * 100

top10_share = revenue_share_of_top_pct(revenue_per_customer, 10)
top20_share = revenue_share_of_top_pct(revenue_per_customer, 20)

print(f"Top 10% of customers generate {top10_share:.1f}% of revenue")
print(f"Top 20% of customers generate {top20_share:.1f}% of revenue")

# %%
plt.figure(figsize=(8, 5))
plt.plot(revenue_per_customer["customer_rank_pct"], revenue_per_customer["cum_revenue_pct"])
plt.axhline(80, linestyle="--", color="grey")
plt.axvline(20, linestyle="--", color="grey")
plt.title("Cumulative Revenue Contribution by Customer (Pareto Curve)")
plt.xlabel("% of Customers (ranked by revenue, descending)")
plt.ylabel("Cumulative % of Revenue")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(revenue_per_customer["Revenue"], bins=30)
plt.title("Customer Revenue Distribution")
plt.xlabel("Revenue per Customer (£)")
plt.tight_layout()
plt.show()