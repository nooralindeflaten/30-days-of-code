from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = '/Users/nooralindeflaten/30-days-of-code/data/rfm/customer_dataset_test.csv'


def build_customer_dataset():
    online_retail = fetch_ucirepo(id=352)

    data = online_retail.data.original
    df = data.copy()
    df['IsCancellation'] = df['InvoiceNo'].astype(str).str.upper().str.startswith('C')
    df['InvoiceDate'] = pd.to_datetime(df["InvoiceDate"])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df_clean = df.dropna(subset=['CustomerID', 'Description'])
    df_clean.drop_duplicates(inplace=True)

    customer_transactions = df_clean.copy()
    purchases = customer_transactions.loc[
        (~customer_transactions["IsCancellation"])
        & (customer_transactions["Quantity"] > 0)
        & (customer_transactions["UnitPrice"] > 0)
    ].copy()

    snapshot_date = purchases['InvoiceDate'].max() + dt.timedelta(days=1)

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

    monetary = (
        purchases.groupby("CustomerID")["Revenue"]
        .sum()
        .rename("MonetaryValue")
    )

    avg_order_value = (monetary / frequency).rename("AvgOrderValue")

    first_purchase = purchases.groupby("CustomerID")["InvoiceDate"].min()
    last_purchase = purchases.groupby("CustomerID")["InvoiceDate"].max()
    tenure = (last_purchase - first_purchase).dt.days.rename("CustomerTenure")

    days_since_first = (
        first_purchase.apply(lambda x: (snapshot_date - x).days)
        .rename("DaysSinceFirstPurchase")
    )

    def avg_days_between_orders(dates):
        dates = dates.sort_values().unique()
        if len(dates) < 2:
            return np.nan
        diffs = np.diff(dates) / np.timedelta64(1, "D")
        return diffs.mean()

    purchase_intervals = (
        purchases.groupby("CustomerID")["InvoiceDate"]
        .apply(avg_days_between_orders)
        .rename("AvgDaysBetweenOrders")
    )

    # Cancellation behaviour is based on ALL invoices (purchases + cancellations),
    # not just the numeric-stockcode subset, so the rate reflects the customer's
    # real cancel-vs-order ratio.
    invoice_status = (
        customer_transactions[["CustomerID", "InvoiceNo", "IsCancellation"]]
        .drop_duplicates()
    )

    cancellation_metrics = (
        invoice_status
        .groupby("CustomerID")
        .agg(
            TotalInvoices=("InvoiceNo", "nunique"),
            CancelledInvoices=("IsCancellation", "sum")
        )
    )
    cancellation_metrics["CancellationRate"] = (
        cancellation_metrics["CancelledInvoices"]
        / cancellation_metrics["TotalInvoices"]
    )

    qty_per_invoice = purchases.groupby(["CustomerID", "InvoiceNo"])["Quantity"].sum()
    items_per_order = (
        qty_per_invoice.groupby("CustomerID")
        .mean()
        .rename("ItemsPerOrder")
    )

    product_diversity = (
        purchases.groupby("CustomerID")["StockCode"]
        .nunique()
        .rename("ProductDiversity")
    )

    customer_df = pd.concat(
        [
            recency,
            frequency,
            monetary,
            avg_order_value,
            items_per_order,
            product_diversity,
            tenure,
            days_since_first,
            purchase_intervals,
            cancellation_metrics[["CancelledInvoices", "CancellationRate"]],
        ],
        axis=1,
    )

    customer_df = customer_df.reset_index().rename(columns={"index": "CustomerID"})

    print(customer_df.head(10).to_string(index=False))
    print(f"\nShape: {customer_df.shape}")
    print(f"\nMissing values per column:\n{customer_df.isna().sum()}")

    customer_df.to_csv(OUT, index=False)
    return customer_df


def score_customers():
    customer_df = pd.read_csv(OUT)

    # FIX: Recency and MonetaryValue commonly have duplicate/tied values in this
    # dataset, which makes plain pd.qcut throw "Bin edges must be unique". Using
    # .rank(method='first') first (as you already did for Frequency) guarantees
    # unique bin edges for all three scores.
    customer_df['RecencyScore'] = pd.qcut(
        customer_df['Recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]
    )
    customer_df['FrequencyScore'] = pd.qcut(
        customer_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]
    )
    customer_df['MonetaryScore'] = pd.qcut(
        customer_df['MonetaryValue'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]
    )

    customer_df['RFM_Score'] = (
        customer_df['RecencyScore'].astype(str) +
        customer_df['FrequencyScore'].astype(str) +
        customer_df['MonetaryScore'].astype(str)
    )

    customers = customer_df.copy()
    conditions = [
        (
            (customers["RecencyScore"] >= 3)
            & (customers["FrequencyScore"] >= 3)
            & (customers["MonetaryScore"] >= 3)
        ),
        (
            (customers["FrequencyScore"] >= 3)
            & (customers["MonetaryScore"] >= 3)
        ),
        (
            (customers["RecencyScore"] >= 3)
            & (customers["FrequencyScore"] >= 2)
        ),
        (
            (customers["RecencyScore"] == 4)
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

    return customers


def build_segment_profile(customer_df):
    segment_profile = (
        customer_df.groupby("BusinessSegment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetaryValue=("MonetaryValue", "mean"),
            AvgOrderValue=("AvgOrderValue", "mean"),
            AvgItemsPerOrder=("ItemsPerOrder", "mean"),
            AvgProductDiversity=("ProductDiversity", "mean"),
            AvgCustomerTenure=("CustomerTenure", "mean"),
            AvgDaysSinceFirstPurchase=("DaysSinceFirstPurchase", "mean"),
            AvgDaysBetweenOrders=("AvgDaysBetweenOrders", "mean"),
            AvgCancelledInvoices=("CancelledInvoices", "mean"),
            AvgCancellationRate=("CancellationRate", "mean")
        )
        .reset_index()
    )
    segment_profile["CustomerShare"] = (
        segment_profile["Customers"] / segment_profile["Customers"].sum()
    )
    segment_profile["RevenueShare"] = (
        segment_profile["AvgMonetaryValue"] * segment_profile["Customers"]
    )
    segment_profile["RevenueShare"] = (
        segment_profile["RevenueShare"] / segment_profile["RevenueShare"].sum()
    )
    segment_profile = segment_profile.sort_values("RevenueShare", ascending=False)
    return segment_profile


def cluster_customers(customer_df, n_clusters=5, n_clusters_range=range(2, 11), show_elbow=True):
    cluster_df = customer_df.copy()
    features = [
        "Recency",
        "Frequency",
        "MonetaryValue",
        "AvgOrderValue",
        "ItemsPerOrder",
        "ProductDiversity",
        "CustomerTenure",
        "DaysSinceFirstPurchase",
        "AvgDaysBetweenOrders",
        "CancelledInvoices",
        "CancellationRate"
    ]

    X = cluster_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if show_elbow:
        silhouette_scores = []
        # FIX: loop variable renamed from n_clusters -> k so it no longer
        # shadows/overwrites the function's n_clusters parameter.
        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))

        plt.figure(figsize=(10, 6))
        plt.plot(list(n_clusters_range), silhouette_scores, marker='o')
        plt.title('Silhouette Analysis for KMeans Clustering')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.xticks(list(n_clusters_range))
        plt.grid()
        plt.show()

    # FIX: fit on the requested n_clusters (now safe from shadowing), assign
    # to cluster_df (not the caller's original customer_df), and return that.
    cluster_df['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)
    return cluster_df


def build_cluster_profile(customer_df):
    cluster_profiles = (
        customer_df.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetaryValue=("MonetaryValue", "mean"),
            AvgOrderValue=("AvgOrderValue", "mean"),
            AvgItemsPerOrder=("ItemsPerOrder", "mean"),
            AvgProductDiversity=("ProductDiversity", "mean"),
            AvgCustomerTenure=("CustomerTenure", "mean"),
            AvgDaysSinceFirstPurchase=("DaysSinceFirstPurchase", "mean"),
            AvgDaysBetweenOrders=("AvgDaysBetweenOrders", "mean"),
            AvgCancelledInvoices=("CancelledInvoices", "mean"),
            AvgCancellationRate=("CancellationRate", "mean")
        )
        .reset_index()
    )

    cluster_profiles["CustomerShare"] = (
        cluster_profiles["Customers"] / cluster_profiles["Customers"].sum()
    )
    cluster_profiles["RevenueShare"] = (
        cluster_profiles["AvgMonetaryValue"] * cluster_profiles["Customers"]
    )
    cluster_profiles["RevenueShare"] = (
        cluster_profiles["RevenueShare"] / cluster_profiles["RevenueShare"].sum()
    )
    cluster_profiles = cluster_profiles.sort_values("RevenueShare", ascending=False)
    return cluster_profiles


def compare_segments_and_clusters(customer_df, segment_profile, cluster_profiles):
    """
    Answers: do the business-rule segments and the ML clusters agree on who
    the high-value customers are?

    FIX: the original merge joined segment_profile (keyed by string labels
    like "Champions") to cluster_profiles (keyed by integer cluster IDs like
    0, 1, 2) on those mismatched key columns -- there's no reason cluster 3
    corresponds to "Champions", so that merge produced a meaningless result.

    Instead: build a customer-level crosstab of BusinessSegment x Cluster to
    see how segment members actually distribute across clusters, then check
    whether the top-value segment and top-value cluster overlap on the same
    customers.
    """
    # Raw counts: how many customers in each business segment fall into each cluster
    crosstab_counts = pd.crosstab(customer_df["BusinessSegment"], customer_df["Cluster"])

    # Row-normalized: within each business segment, % of customers in each cluster
    crosstab_pct = pd.crosstab(
        customer_df["BusinessSegment"], customer_df["Cluster"], normalize="index"
    ).round(3)

    # Identify the "high-value" group on each side by revenue share
    top_segment = segment_profile.sort_values("RevenueShare", ascending=False)["BusinessSegment"].iloc[0]
    top_cluster = cluster_profiles.sort_values("RevenueShare", ascending=False)["Cluster"].iloc[0]

    top_segment_customers = set(customer_df.loc[customer_df["BusinessSegment"] == top_segment, "CustomerID"])
    top_cluster_customers = set(customer_df.loc[customer_df["Cluster"] == top_cluster, "CustomerID"])

    overlap = top_segment_customers & top_cluster_customers
    overlap_pct_of_segment = len(overlap) / len(top_segment_customers) if top_segment_customers else 0
    overlap_pct_of_cluster = len(overlap) / len(top_cluster_customers) if top_cluster_customers else 0

    summary = {
        "top_business_segment": top_segment,
        "top_cluster": int(top_cluster),
        "top_segment_size": len(top_segment_customers),
        "top_cluster_size": len(top_cluster_customers),
        "overlap_customers": len(overlap),
        "overlap_pct_of_segment": round(overlap_pct_of_segment * 100, 1),
        "overlap_pct_of_cluster": round(overlap_pct_of_cluster * 100, 1),
    }

    return crosstab_counts, crosstab_pct, summary


def main():
    customer_df = score_customers()
    segment_profile = build_segment_profile(customer_df)

    cluster_df = cluster_customers(customer_df, n_clusters=5, show_elbow=True)
    cluster_profiles = build_cluster_profile(cluster_df)

    crosstab_counts, crosstab_pct, overlap_summary = compare_segments_and_clusters(
        cluster_df, segment_profile, cluster_profiles
    )

    print("Segment Profile:\n", segment_profile)
    print("\nCluster Profiles:\n", cluster_profiles)
    print("\nBusinessSegment x Cluster (counts):\n", crosstab_counts)
    print("\nBusinessSegment x Cluster (% within segment):\n", crosstab_pct)
    print("\nHigh-value overlap summary:\n", overlap_summary)
    
    crosstab_counts.to_csv('/Users/nooralindeflaten/30-days-of-code/data/rfm/crosstab_counts.csv')
    segment_profile.to_csv('/Users/nooralindeflaten/30-days-of-code/data/rfm/segment_profile.csv', index=False)
    cluster_profiles.to_csv('/Users/nooralindeflaten/30-days-of-code/data/rfm/cluster_profiles.csv', index=False)
    


if __name__ == "__main__":
    #build_customer_dataset()
    main()