from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -------------------------
# Paths / Config
# -------------------------
OUT_DIR = Path("data/rfm/reports/day07")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
}

CLUSTER_FEATURES = [
    "Recency",
    "Frequency",
    "MonetaryValue",
    "AvgOrderValue",
    "ItemsPerOrder",
    "ProductDiversity",
    "CustomerTenure",
    "AvgDaysBetweenOrders",
    "CancellationRate",
]


# -------------------------
# Data loading and cleaning
# -------------------------
def load_transactions():
    online_retail = fetch_ucirepo(id=352)
    df = online_retail.data.original.copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["NonProduct"] = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)

    # Customer-level analysis requires CustomerID.
    customer_transactions = df.dropna(subset=["CustomerID"]).copy()
    customer_transactions["CustomerID"] = customer_transactions["CustomerID"].astype(int)

    return customer_transactions


def split_purchase_cancel(df):
    df = df.copy()
    df.dropna(subset=["CustomerID"], inplace=True)
    purchases = df.loc[
        (~df["IsCancellation"])
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["NonProduct"])
    ].copy()

    cancellations = df.loc[
        (df["IsCancellation"])
        & (df["Quantity"] < 0)
        & (df["UnitPrice"] > 0)
        & (~df["NonProduct"])
    ].copy()

    # Revenue is already negative for cancellation rows because Quantity < 0.
    return purchases, cancellations


# -------------------------
# Feature engineering
# -------------------------
def avg_days_between_orders(dates: pd.Series):
    unique_sorted = np.sort(dates.dt.floor("D").unique())
    if len(unique_sorted) < 2:
        return np.nan
    diffs = np.diff(unique_sorted) / np.timedelta64(1, "D")
    return float(np.mean(diffs))


def build_customer_dataset():
    tx = load_transactions()
    purchases, cancellations = split_purchase_cancel(tx)

    snapshot_date = purchases["InvoiceDate"].max() + dt.timedelta(days=1)

    recency = (
        purchases.groupby("CustomerID")["InvoiceDate"]
        .max()
        .apply(lambda x: (snapshot_date - x).days)
        .rename("Recency")
    )

    frequency = (
        purchases.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .rename("Frequency")
    )

    gross_monetary = (
        purchases.groupby("CustomerID")["Revenue"]
        .sum()
        .rename("GrossMonetaryValue")
    )

    cancellation_value = (
        cancellations.groupby("CustomerID")["Revenue"]
        .sum()
        .rename("CancellationValue")
    )

    monetary_block = pd.concat([gross_monetary, cancellation_value], axis=1).fillna(0)
    monetary_block["MonetaryValue"] = monetary_block["GrossMonetaryValue"] + monetary_block["CancellationValue"]

    avg_order_value = (monetary_block["MonetaryValue"] / frequency).rename("AvgOrderValue")

    qty_per_invoice = purchases.groupby(["CustomerID", "InvoiceNo"])["Quantity"].sum()
    items_per_order = qty_per_invoice.groupby("CustomerID").mean().rename("ItemsPerOrder")

    product_diversity = purchases.groupby("CustomerID")["StockCode"].nunique().rename("ProductDiversity")

    first_purchase = purchases.groupby("CustomerID")["InvoiceDate"].min()
    last_purchase = purchases.groupby("CustomerID")["InvoiceDate"].max()
    customer_tenure = (last_purchase - first_purchase).dt.days.rename("CustomerTenure")

    days_since_first_purchase = (
        first_purchase.apply(lambda x: (snapshot_date - x).days).rename("DaysSinceFirstPurchase")
    )

    purchase_intervals = (
        purchases.groupby("CustomerID")["InvoiceDate"]
        .apply(avg_days_between_orders)
        .rename("AvgDaysBetweenOrders")
    )

    invoice_status = (
        tx[["CustomerID", "InvoiceNo", "IsCancellation"]]
        .drop_duplicates()
    )
    cancellation_metrics = (
        invoice_status.groupby("CustomerID")
        .agg(
            TotalInvoices=("InvoiceNo", "nunique"),
            CancelledInvoices=("IsCancellation", "sum")
        )
    )
    cancellation_metrics["CancellationRate"] = (
        cancellation_metrics["CancelledInvoices"] / cancellation_metrics["TotalInvoices"]
    )

    customer_df = pd.concat(
        [
            recency,
            frequency,
            monetary_block[["GrossMonetaryValue", "CancellationValue", "MonetaryValue"]],
            avg_order_value,
            items_per_order,
            product_diversity,
            customer_tenure,
            days_since_first_purchase,
            purchase_intervals,
            cancellation_metrics[["TotalInvoices", "CancelledInvoices", "CancellationRate"]],
        ],
        axis=1,
    ).reset_index()

    customer_df.to_csv(OUT_DIR / "customer_features.csv", index=False)

    feature_dict = pd.DataFrame({
        "feature": [
            "Recency", "Frequency", "GrossMonetaryValue", "CancellationValue", "MonetaryValue",
            "AvgOrderValue", "ItemsPerOrder", "ProductDiversity", "CustomerTenure",
            "DaysSinceFirstPurchase", "AvgDaysBetweenOrders", "TotalInvoices",
            "CancelledInvoices", "CancellationRate"
        ],
        "definition": [
            "Days since most recent successful purchase invoice.",
            "Count of successful purchase invoices (cancellations excluded).",
            "Sum of successful purchase revenue per customer.",
            "Sum of cancellation revenue per customer (usually negative).",
            "GrossMonetaryValue + CancellationValue (net economic value).",
            "MonetaryValue / Frequency.",
            "Average quantity per purchase invoice.",
            "Unique products (StockCode) purchased.",
            "Days between first and last successful purchase.",
            "Days since first successful purchase to snapshot date.",
            "Mean days between successful purchase dates.",
            "Unique invoices observed for this customer (purchase + cancellation).",
            "Count of cancellation invoices for this customer.",
            "CancelledInvoices / TotalInvoices."
        ]
    })
    feature_dict.to_csv(OUT_DIR / "feature_dictionary.csv", index=False)

    return customer_df


# -------------------------
# RFM scoring and rule segments
# -------------------------
def score_rfm(customer_df):
    out = customer_df.copy()

    out["RecencyScore"] = pd.qcut(
        out["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    out["FrequencyScore"] = pd.qcut(
        out["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    out["MonetaryScore"] = pd.qcut(
        out["MonetaryValue"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    out["RFM_Score"] = (
        out["RecencyScore"].astype(str)
        + out["FrequencyScore"].astype(str)
        + out["MonetaryScore"].astype(str)
    )
    return out


def assign_business_segments(rfm_df):
    customers = rfm_df.copy()

    conditions = [
        (
            (customers["RecencyScore"] >= 4)
            & (customers["FrequencyScore"] >= 4)
            & (customers["MonetaryScore"] >= 4)
        ),
        (
            (customers["RecencyScore"] >= 3)
            & (customers["FrequencyScore"] >= 4)
            & (customers["MonetaryScore"] >= 3)
        ),
        (
            (customers["RecencyScore"] >= 4)
            & (customers["FrequencyScore"].between(2, 3))
        ),
        (
            (customers["RecencyScore"] >= 4)
            & (customers["FrequencyScore"] == 1)
        ),
        (
            (customers["RecencyScore"] <= 2)
            & (customers["FrequencyScore"] >= 3)
            & (customers["MonetaryScore"] >= 3)
        ),
        (
            (customers["RecencyScore"] <= 2)
            & (customers["FrequencyScore"] <= 2)
        )
    ]

    segments = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "New Customers",
        "At Risk",
        "Inactive / Low Engagement"
    ]

    customers["BusinessSegment"] = np.select(conditions, segments, default="Other")
    customers.to_csv(OUT_DIR / "rfm_scored_customers.csv", index=False)
    return customers


def build_segment_profile(seg_df):
    profile = (
        seg_df.groupby("BusinessSegment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetaryValue=("MonetaryValue", "mean"),
            TotalMonetaryValue=("MonetaryValue", "sum"),
            AvgOrderValue=("AvgOrderValue", "mean"),
            AvgItemsPerOrder=("ItemsPerOrder", "mean"),
            AvgProductDiversity=("ProductDiversity", "mean"),
            AvgCustomerTenure=("CustomerTenure", "mean"),
            AvgDaysBetweenOrders=("AvgDaysBetweenOrders", "mean"),
            AvgCancelledInvoices=("CancelledInvoices", "mean"),
            AvgCancellationRate=("CancellationRate", "mean"),
        )
        .reset_index()
    )

    profile["CustomerShare"] = profile["Customers"] / profile["Customers"].sum()
    profile["RevenueShare"] = profile["TotalMonetaryValue"] / profile["TotalMonetaryValue"].sum()
    profile = profile.sort_values("RevenueShare", ascending=False)

    profile.to_csv(OUT_DIR / "segment_profile.csv", index=False)
    return profile


# -------------------------
# Clustering
# -------------------------
def prepare_clustering_features(customer_df):
    X = customer_df[CLUSTER_FEATURES].copy()

    X["AvgDaysBetweenOrders"] = X["AvgDaysBetweenOrders"].fillna(
        X["AvgDaysBetweenOrders"].median()
    )

    log_cols = [
        "Frequency",
        "MonetaryValue",
        "AvgOrderValue",
        "ItemsPerOrder",
        "ProductDiversity",
        "CustomerTenure",
        "AvgDaysBetweenOrders",
    ]
    for col in log_cols:
        X[col] = np.log1p(X[col].clip(lower=0))

    for col in X.columns:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lower, upper=upper)

    return X


def evaluate_kmeans(X_scaled, k_range=range(2, 11)):
    rows = []
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        rows.append({"k": k, "inertia": model.inertia_, "silhouette_score": silhouettes[-1]})

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(OUT_DIR / "cluster_evaluation.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(list(k_range), inertias, marker="o")
    ax1.set_title("KMeans Elbow (Inertia)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kmeans_elbow.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), silhouettes, marker="o")
    plt.title("KMeans Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "kmeans_silhouette.png", dpi=160)
    plt.close()

    return eval_df


def cluster_customers(customer_df, n_clusters=5):
    X = prepare_clustering_features(customer_df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    evaluate_kmeans(X_scaled, k_range=range(2, 11))

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_df = customer_df.copy()
    cluster_df["Cluster"] = model.fit_predict(X_scaled)

    cluster_df.to_csv(OUT_DIR / "clustered_customers.csv", index=False)
    return cluster_df


def build_cluster_profile(cluster_df):
    profile = (
        cluster_df.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetaryValue=("MonetaryValue", "mean"),
            TotalMonetaryValue=("MonetaryValue", "sum"),
            AvgOrderValue=("AvgOrderValue", "mean"),
            AvgItemsPerOrder=("ItemsPerOrder", "mean"),
            AvgProductDiversity=("ProductDiversity", "mean"),
            AvgCustomerTenure=("CustomerTenure", "mean"),
            AvgDaysBetweenOrders=("AvgDaysBetweenOrders", "mean"),
            AvgCancelledInvoices=("CancelledInvoices", "mean"),
            AvgCancellationRate=("CancellationRate", "mean"),
        )
        .reset_index()
    )

    profile["CustomerShare"] = profile["Customers"] / profile["Customers"].sum()
    profile["RevenueShare"] = profile["TotalMonetaryValue"] / profile["TotalMonetaryValue"].sum()
    profile = profile.sort_values("RevenueShare", ascending=False)

    profile.to_csv(OUT_DIR / "cluster_profile.csv", index=False)
    return profile


# -------------------------
# Compare rule vs clustering
# -------------------------
def compare_segments_and_clusters(cluster_df, segment_profile, cluster_profile):
    crosstab_counts = pd.crosstab(cluster_df["BusinessSegment"], cluster_df["Cluster"])
    crosstab_pct = pd.crosstab(
        cluster_df["BusinessSegment"], cluster_df["Cluster"], normalize="index"
    ).round(3)

    crosstab_counts.to_csv(OUT_DIR / "segment_cluster_crosstab_counts.csv")
    crosstab_pct.to_csv(OUT_DIR / "segment_cluster_crosstab_pct.csv")

    top_segment = segment_profile.sort_values("RevenueShare", ascending=False)["BusinessSegment"].iloc[0]
    top_cluster = cluster_profile.sort_values("RevenueShare", ascending=False)["Cluster"].iloc[0]

    seg_customers = set(cluster_df.loc[cluster_df["BusinessSegment"] == top_segment, "CustomerID"])
    clu_customers = set(cluster_df.loc[cluster_df["Cluster"] == top_cluster, "CustomerID"])
    overlap = seg_customers & clu_customers

    summary = {
        "top_business_segment": top_segment,
        "top_cluster": int(top_cluster),
        "top_segment_size": len(seg_customers),
        "top_cluster_size": len(clu_customers),
        "overlap_customers": len(overlap),
        "overlap_pct_of_segment": round(100 * len(overlap) / len(seg_customers), 1) if seg_customers else 0.0,
        "overlap_pct_of_cluster": round(100 * len(overlap) / len(clu_customers), 1) if clu_customers else 0.0,
    }

    pd.Series(summary).to_csv(OUT_DIR / "segment_cluster_overlap_summary.csv", header=["value"])
    return crosstab_counts, crosstab_pct, summary


# -------------------------
# Recommendations
# -------------------------
def write_recommendations(segment_profile, cluster_profile, overlap_summary):
    top_seg = segment_profile.iloc[0]["BusinessSegment"]
    top_seg_share = segment_profile.iloc[0]["RevenueShare"]

    top_cluster = int(cluster_profile.iloc[0]["Cluster"])
    top_cluster_share = cluster_profile.iloc[0]["RevenueShare"]

    text = f"""# Commercial Recommendations (Day 07)

## Key findings
- Highest-value business segment: **{top_seg}** ({top_seg_share:.1%} of net customer value).
- Highest-value cluster: **Cluster {top_cluster}** ({top_cluster_share:.1%} of net customer value).
- Overlap between highest-value segment and cluster:
  - {overlap_summary['overlap_customers']} customers
  - {overlap_summary['overlap_pct_of_segment']:.1f}% of top segment
  - {overlap_summary['overlap_pct_of_cluster']:.1f}% of top cluster

## Recommended actions by group
1. **Champions / highest-value cluster**
   - VIP retention: early access, loyalty rewards, priority support.
   - Protect margin with targeted offers (not blanket discounting).

2. **Loyal + Potential Loyalists**
   - Cross-sell bundles and category expansion nudges.
   - Time-based reactivation before expected next-order window.

3. **At Risk**
   - Win-back journey based on recency + historical value.
   - Test incentive levels and monitor margin impact.

4. **Inactive / Low Engagement**
   - Low-cost channels first.
   - Suppress expensive channels unless uplift is proven.

## RFM vs clustering
- **RFM** is more interpretable and easy for stakeholder activation.
- **Clustering** captures richer multivariate behavior.
- Recommended operating model: use RFM for campaign rules and clustering for strategy refinement.
"""
    (OUT_DIR / "recommendations.md").write_text(text, encoding="utf-8")


def main():
    customer_df = build_customer_dataset()
    rfm_df = score_rfm(customer_df)
    segmented_df = assign_business_segments(rfm_df)

    segment_profile = build_segment_profile(segmented_df)
    cluster_df = cluster_customers(segmented_df, n_clusters=5)
    cluster_profile = build_cluster_profile(cluster_df)

    _, _, overlap_summary = compare_segments_and_clusters(
        cluster_df, segment_profile, cluster_profile
    )

    write_recommendations(segment_profile, cluster_profile, overlap_summary)

    print("Saved outputs to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()