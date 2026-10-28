from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise ImportError(
        "Please install ucimlrepo first: pip install ucimlrepo"
    ) from e


OUT = Path("data/rfm/reports/day6")
OUT.mkdir(parents=True, exist_ok=True)

NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
}
MIN_COUNTRY_CUSTOMERS = 5


def load_data():
    online_retail = pd.read_parquet('data/rfm/online_retail.parquet')
    df = online_retail.copy()

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.upper()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["NonProduct"] = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)
    return df


def run_audit(df):
    lines = ["# Data Quality Audit", ""]

    lines.append(f"- Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append(f"- Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    lines.append("")
    lines.append("## Missing values")
    lines.append("```")
    lines.append(df.isna().sum().to_string())
    lines.append("```")
    lines.append("")

    lines.append(f"- Duplicate rows: {df.duplicated().sum():,}")
    lines.append(f"- Missing CustomerID rows: {df['CustomerID'].isna().sum():,}")
    lines.append(f"- Missing Description rows: {df['Description'].isna().sum():,}")
    lines.append(f"- Quantity <= 0 rows: {(df['Quantity'] <= 0).sum():,}")
    lines.append(f"- UnitPrice <= 0 rows: {(df['UnitPrice'] <= 0).sum():,}")
    lines.append(f"- Cancellation rows: {df['IsCancellation'].sum():,}")
    lines.append(f"- Non-product code rows: {df['NonProduct'].sum():,}")
    lines.append("")

    lines.append("## Analysis impact")
    lines.append("- Missing CustomerID affects only customer-level analysis.")
    lines.append("- Cancellation rows are preserved for net revenue and cancellation analysis.")
    lines.append("- Non-positive quantity/price and non-product codes are excluded from valid purchases.")
    lines.append("- Duplicates are reported; not removed blindly.")

    (OUT / "audit.md").write_text("\n".join(lines), encoding="utf-8")


def classify_records(df):
    is_purchase = (
        (~df["IsCancellation"])
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["NonProduct"])
    )
    is_cancellation = (
        (df["IsCancellation"])
        & (df["Quantity"] < 0)
        & (df["UnitPrice"] > 0)
        & (~df["NonProduct"])
    )

    df = df.copy()
    df["RecordType"] = np.select(
        [is_purchase, is_cancellation],
        ["purchase", "cancellation"],
        default="ambiguous"
    )
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    summary = (
        df.groupby("RecordType")
        .agg(rows=("InvoiceNo", "size"), value=("Revenue", "sum"))
        .reset_index()
    )
    summary.to_csv(OUT / "record_classification.csv", index=False)

    purchases = df[df["RecordType"] == "purchase"].copy()
    cancellations = df[df["RecordType"] == "cancellation"].copy()
    cancellations["CancelValue"] = -cancellations["Revenue"]
    cancellations["CancelUnits"] = -cancellations["Quantity"]

    return df, purchases, cancellations


def compute_kpis(purchases, cancellations):
    gross_revenue = purchases["Revenue"].sum()
    cancelled_value = cancellations["CancelValue"].sum()
    net_revenue = gross_revenue - cancelled_value

    orders = purchases["InvoiceNo"].nunique()
    cancellation_invoices = cancellations["InvoiceNo"].nunique()

    kpis = {
        "gross_revenue": gross_revenue,
        "cancelled_value": cancelled_value,
        "net_revenue": net_revenue,
        "orders": orders,
        "customers": purchases["CustomerID"].nunique(),
        "products_sold": purchases["StockCode"].nunique(),
        "units_sold": purchases["Quantity"].sum(),
        "units_cancelled": cancellations["CancelUnits"].sum(),
        "avg_order_value": gross_revenue / orders if orders else np.nan,
        "items_per_order_units": purchases["Quantity"].sum() / orders if orders else np.nan,
        "cancellation_rate_value": cancelled_value / gross_revenue if gross_revenue else np.nan,
        "cancellation_rate_orders": cancellation_invoices / orders if orders else np.nan,
        "revenue_share_missing_customer": (
            purchases.loc[purchases["CustomerID"].isna(), "Revenue"].sum() / gross_revenue
            if gross_revenue else np.nan
        ),
    }

    pd.Series(kpis).to_csv(OUT / "kpis.csv", header=["value"])
    return kpis


def revenue_over_time(purchases, cancellations):
    monthly = purchases.groupby("Month").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
        UnitsSold=("Quantity", "sum"),
        ProductsSold=("StockCode", "nunique"),
    )

    # FIX 1: OrdersPerCustomer must use orders from known customers only
    known_orders = (
        purchases.dropna(subset=["CustomerID"])
        .groupby("Month")["InvoiceNo"].nunique()
    )

    monthly["AverageOrderValue"] = monthly["Revenue"] / monthly["Orders"]
    monthly["OrdersPerCustomer"] = known_orders / monthly["Customers"]
    monthly["ItemsPerOrder"] = monthly["UnitsSold"] / monthly["Orders"]
    monthly["RevenuePerItem"] = monthly["Revenue"] / monthly["UnitsSold"]

    monthly["CancelledValue"] = cancellations.groupby("Month")["CancelValue"].sum()
    monthly["CancelledValue"] = monthly["CancelledValue"].fillna(0)
    monthly["NetRevenue"] = monthly["Revenue"] - monthly["CancelledValue"]
    monthly["RevenueGrowthMoM"] = monthly["Revenue"].pct_change()

    monthly["IsPartialMonth"] = monthly.index == monthly.index.max()
    monthly.to_csv(OUT / "monthly_metrics.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    monthly["Revenue"].plot(ax=axes[0, 0], marker="o", label="Gross Revenue")
    monthly["NetRevenue"].plot(ax=axes[0, 0], marker="o", label="Net Revenue")
    axes[0, 0].set_title("Revenue over time")
    axes[0, 0].legend()

    monthly["Orders"].plot(ax=axes[0, 1], marker="o", title="Orders over time")
    monthly["Customers"].plot(ax=axes[1, 0], marker="o", title="Customers over time")
    monthly["AverageOrderValue"].plot(ax=axes[1, 1], marker="o", title="AOV over time")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT / "revenue_over_time.png", dpi=160)
    plt.close()

    known = purchases.dropna(subset=["CustomerID"]).copy()
    monthly_known = known.groupby("Month").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    )
    monthly_known["OrdersPerCustomer"] = monthly_known["Orders"] / monthly_known["Customers"]
    monthly_known["AOV"] = monthly_known["Revenue"] / monthly_known["Orders"]

    contrib = np.log(monthly_known[["Customers", "OrdersPerCustomer", "AOV"]]).diff()

    # FIX 2: drop first NaN row and partial last month
    contrib = contrib.iloc[1:-1]

    # FIX 3: convert datetime index to strings to avoid pandas bar/period converter bug
    contrib.index = contrib.index.strftime("%Y-%m")

    contrib.to_csv(OUT / "revenue_driver_contributions.csv")

    contrib.plot(kind="bar", stacked=True, figsize=(12, 5))
    plt.title("Revenue change decomposition (log contribution)")
    plt.xlabel("Month")
    plt.ylabel("Log-point contribution")
    plt.tight_layout()
    plt.savefig(OUT / "revenue_driver_contributions.png", dpi=160)
    plt.close()

    return monthly


def geography_analysis(purchases, cancellations):
    country = purchases.groupby("Country").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    )
    country["AOV"] = country["Revenue"] / country["Orders"]
    country["RevenueShare"] = country["Revenue"] / country["Revenue"].sum()
    country["CancelledValue"] = cancellations.groupby("Country")["CancelValue"].sum()
    country["CancelledValue"] = country["CancelledValue"].fillna(0)
    country["LowSampleFlag"] = country["Customers"] < MIN_COUNTRY_CUSTOMERS
    country = country.sort_values("Revenue", ascending=False)
    country.to_csv(OUT / "country_metrics.csv")

    top10 = country.head(10)
    top10["Revenue"].plot(kind="bar", figsize=(9, 5), title="Top 10 countries by revenue")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUT / "country_revenue_top10.png", dpi=160)
    plt.close()

    non_uk = country.drop(index="United Kingdom", errors="ignore").head(10)
    non_uk["Revenue"].plot(kind="bar", figsize=(9, 5), title="Top 10 non-UK countries by revenue")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUT / "country_revenue_non_uk_top10.png", dpi=160)
    plt.close()

    return country


def product_analysis(purchases, cancellations):
    product = purchases.groupby("StockCode").agg(
        Revenue=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
    )

    desc = purchases.groupby("StockCode")["Description"].agg(
        lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""
    )
    product["Description"] = desc

    canc = cancellations.groupby("StockCode").agg(
        CancelUnits=("CancelUnits", "sum"),
        CancelValue=("CancelValue", "sum"),
    )
    product = product.join(canc).fillna({"CancelUnits": 0, "CancelValue": 0})
    product["CancelUnitRate"] = product["CancelUnits"] / product["Units"]

    product.sort_values("Revenue", ascending=False).head(20).to_csv(OUT / "top_products_revenue.csv")
    product.sort_values("Units", ascending=False).head(20).to_csv(OUT / "top_products_units.csv")
    product.sort_values("Orders", ascending=False).head(20).to_csv(OUT / "top_products_orders.csv")
    product[product["Units"] >= 100].sort_values("CancelUnitRate", ascending=False).head(20).to_csv(
        OUT / "top_products_cancel_rate.csv"
    )
    product.sort_values("CancelValue", ascending=False).head(20).to_csv(
        OUT / "top_products_cancel_value.csv"
    )

    top20_units = set(product.nlargest(20, "Units").index)
    top20_revenue = set(product.nlargest(20, "Revenue").index)
    overlap = len(top20_units & top20_revenue)

    product_stats = {
        "top20_units_revenue_overlap_count": overlap,
        "top20_revenue_share": (
            product.nlargest(20, "Revenue")["Revenue"].sum() / product["Revenue"].sum()
            if product["Revenue"].sum() else np.nan
        ),
    }
    pd.Series(product_stats).to_csv(OUT / "product_stats.csv", header=["value"])
    return product, product_stats


def customer_concentration(purchases, cancellations):
    known_p = purchases.dropna(subset=["CustomerID"])
    customer_rev = known_p.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)

    known_c = cancellations.dropna(subset=["CustomerID"])
    customer_cancel = known_c.groupby("CustomerID")["CancelValue"].sum()

    customer = pd.DataFrame({"GrossRevenue": customer_rev})
    customer["CancelledValue"] = customer_cancel
    customer["CancelledValue"] = customer["CancelledValue"].fillna(0)
    customer["NetRevenue"] = customer["GrossRevenue"] - customer["CancelledValue"]
    customer = customer.sort_values("GrossRevenue", ascending=False)
    customer.to_csv(OUT / "customer_revenue.csv")

    rev = customer["GrossRevenue"].values
    cum = np.cumsum(rev) / rev.sum()
    x = np.arange(1, len(rev) + 1) / len(rev)

    def share_top(q):
        idx = int(np.ceil(len(cum) * q)) - 1
        return float(cum[max(idx, 0)])

    metrics = {
        "top_1pct_share": share_top(0.01),
        "top_5pct_share": share_top(0.05),
        "top_10pct_share": share_top(0.10),
        "top_20pct_share": share_top(0.20),
        "customers_for_50pct_revenue": int(np.searchsorted(cum, 0.5) + 1),
        "customers_for_80pct_revenue": int(np.searchsorted(cum, 0.8) + 1),
        "note": "concentration computed on identified customers only",
    }

    asc = np.sort(rev)
    n = len(asc)
    gini = (2 * np.sum(np.arange(1, n + 1) * asc) / (n * asc.sum())) - (n + 1) / n
    metrics["gini"] = float(gini)

    pd.Series(metrics).to_csv(OUT / "customer_concentration.csv", header=["value"])

    plt.figure(figsize=(7, 5))
    plt.plot(x, cum, label="Cumulative revenue")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect equality")
    plt.title("Customer revenue concentration")
    plt.xlabel("Share of customers")
    plt.ylabel("Share of revenue")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "customer_concentration_curve.png", dpi=160)
    plt.close()

    return metrics


def write_executive_summary(kpis, monthly, country, product_stats, concentration):
    complete = monthly[~monthly["IsPartialMonth"]]
    best_month = complete["Revenue"].idxmax() if len(complete) else monthly["Revenue"].idxmax()
    worst_month = complete["Revenue"].idxmin() if len(complete) else monthly["Revenue"].idxmin()

    uk_share = country.loc["United Kingdom", "RevenueShare"] if "United Kingdom" in country.index else np.nan

    text = f"""# Executive Summary

## Main findings
- Gross revenue: {kpis['gross_revenue']:,.0f}
- Cancelled value: {kpis['cancelled_value']:,.0f}
- Net revenue: {kpis['net_revenue']:,.0f}
- Orders: {kpis['orders']:,}
- Customers (identified): {kpis['customers']:,}
- Average order value: {kpis['avg_order_value']:.2f}
- Cancellation rate (value): {kpis['cancellation_rate_value']:.1%}
- Cancellation rate (orders): {kpis['cancellation_rate_orders']:.1%}
- UK revenue share: {uk_share:.1%}
- Top 20 product revenue share: {product_stats['top20_revenue_share']:.1%}
- Top 20% customers revenue share: {concentration['top_20pct_share']:.1%}
- Gini (identified customers): {concentration['gini']:.2f}

## Stability over time
- Strongest complete month: {best_month:%Y-%m}
- Weakest complete month: {worst_month:%Y-%m}
- Note: final month in dataset is partial and should be interpreted cautiously.

## Caveats
- Customer concentration uses identified customers only.
- Cancellations are timestamped on cancellation date, not matched back to original purchase date.
- Non-product stock codes and ambiguous rows excluded from valid purchase KPIs.
"""
    (OUT / "executive_summary.md").write_text(text, encoding="utf-8")


def main():
    df = load_data()
    run_audit(df)

    _, purchases, cancellations = classify_records(df)

    kpis = compute_kpis(purchases, cancellations)
    monthly = revenue_over_time(purchases, cancellations)
    country = geography_analysis(purchases, cancellations)
    _, product_stats = product_analysis(purchases, cancellations)
    concentration = customer_concentration(purchases, cancellations)

    write_executive_summary(kpis, monthly, country, product_stats, concentration)

    print("Saved outputs to:", OUT.resolve())


if __name__ == "__main__":
    main()