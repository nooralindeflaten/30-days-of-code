from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise ImportError("Please install ucimlrepo first: pip install ucimlrepo") from e

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
except ImportError as e:
    raise ImportError(
        "Please install mlxtend first: pip install mlxtend"
    ) from e


# -------------------------
# Paths / Config
# -------------------------
OUT_DIR = Path("data/rfm/reports/day08")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY07_SEGMENTS_PATH = Path("reports/day07/rfm_scored_customers.csv")

NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
}

MIN_PRODUCT_ORDERS = 50
SUPPORT_GRID = [0.02, 0.01, 0.005]
MIN_CONFIDENCE = 0.20
MIN_LIFT = 1.20


# -------------------------
# Data loading / cleaning
# -------------------------
def load_data():
    online_retail = pd.read_parquet('data/rfm/online_retail.parquet')
    df = online_retail.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["NonProduct"] = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)

    return df


def split_valid_purchase_rows(df):
    purchases = df.loc[
        (~df["IsCancellation"])
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (~df["NonProduct"])
    ].copy()

    return purchases


# -------------------------
# 1) Product performance
# -------------------------
def product_performance(purchases):
    perf = purchases.groupby("StockCode").agg(
        Description=("Description", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else ""),
        UnitsSold=("Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    )
    perf["AvgQtyPerOrder"] = perf["UnitsSold"] / perf["Orders"]
    perf["AvgRevenuePerOrder"] = perf["Revenue"] / perf["Orders"]
    perf = perf.sort_values("Revenue", ascending=False)

    perf.to_csv(OUT_DIR / "product_performance.csv")

    perf.sort_values("Revenue", ascending=False).head(20).to_csv(OUT_DIR / "top_products_revenue.csv")
    perf.sort_values("UnitsSold", ascending=False).head(20).to_csv(OUT_DIR / "top_products_units.csv")
    perf.sort_values("Orders", ascending=False).head(20).to_csv(OUT_DIR / "top_products_orders.csv")

    # quick visual
    top_rev = perf.head(15).sort_values("Revenue")
    plt.figure(figsize=(9, 6))
    plt.barh(top_rev["Description"].replace("", top_rev.index), top_rev["Revenue"])
    plt.title("Top 15 Products by Revenue")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_products_revenue.png", dpi=160)
    plt.close()

    return perf


# -------------------------
# 2) Basket construction
# -------------------------
def build_baskets(purchases):
    product_order_counts = purchases.groupby("StockCode")["InvoiceNo"].nunique()
    keep_codes = product_order_counts[product_order_counts >= MIN_PRODUCT_ORDERS].index

    filtered = purchases[purchases["StockCode"].isin(keep_codes)].copy()

    basket_lines = (
        filtered.groupby(["InvoiceNo", "StockCode"])["Quantity"]
        .sum()
        .unstack(fill_value=0)
    )

    basket_binary = (basket_lines > 0).astype(int)

    basket_binary.to_csv(OUT_DIR / "basket_matrix_binary.csv")

    basket_summary = (
        filtered.groupby("InvoiceNo")
        .agg(
            BasketUnits=("Quantity", "sum"),
            BasketUniqueProducts=("StockCode", "nunique"),
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "first"),
            Country=("Country", "first"),
        )
        .reset_index()
    )
    basket_summary.to_csv(OUT_DIR / "basket_summary.csv", index=False)

    pd.Series({
        "purchase_rows": len(purchases),
        "purchase_orders": purchases["InvoiceNo"].nunique(),
        "kept_products_for_rules": len(keep_codes),
        "orders_after_product_filter": basket_binary.shape[0],
    }).to_csv(OUT_DIR / "basket_build_stats.csv", header=["value"])

    return basket_binary, basket_summary, filtered


# -------------------------
# 3) Basket characteristics
# -------------------------
def basket_characteristics(basket_summary):
    s = basket_summary["BasketUniqueProducts"]

    stats = {
        "orders": int(len(basket_summary)),
        "mean_unique_products_per_order": float(s.mean()),
        "median_unique_products_per_order": float(s.median()),
        "p90_unique_products_per_order": float(s.quantile(0.90)),
        "single_product_order_share": float((s == 1).mean()),
        "multi_product_order_share": float((s >= 2).mean()),
    }
    pd.Series(stats).to_csv(OUT_DIR / "basket_characteristics.csv", header=["value"])

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=30)
    plt.title("Distribution of Unique Products per Order")
    plt.xlabel("Unique products in order")
    plt.ylabel("Number of orders")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "basket_size_distribution.png", dpi=160)
    plt.close()

    return stats


# -------------------------
# 4-5) Frequent itemsets + rules
# -------------------------
def mine_itemsets_and_rules(basket_binary):
    eval_rows = []
    saved_itemsets = None
    saved_rules = None
    selected_support = None

    for min_support in SUPPORT_GRID:
        itemsets = fpgrowth(
            basket_binary,
            min_support=min_support,
            use_colnames=True
        )
        if itemsets.empty:
            eval_rows.append({
                "min_support": min_support,
                "n_itemsets": 0,
                "n_rules": 0
            })
            continue

        rules = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
        rules = rules[rules["lift"] >= MIN_LIFT].copy()

        eval_rows.append({
            "min_support": min_support,
            "n_itemsets": len(itemsets),
            "n_rules": len(rules),
        })

        if saved_rules is None and len(rules) > 0:
            saved_itemsets = itemsets.copy()
            saved_rules = rules.copy()
            selected_support = min_support

    eval_df = pd.DataFrame(eval_rows).sort_values("min_support", ascending=False)
    eval_df.to_csv(OUT_DIR / "association_threshold_grid.csv", index=False)

    if saved_itemsets is None or saved_rules is None:
        pd.DataFrame().to_csv(OUT_DIR / "frequent_itemsets.csv", index=False)
        pd.DataFrame().to_csv(OUT_DIR / "association_rules.csv", index=False)
        pd.Series({"selected_support": np.nan, "note": "No rules found with current thresholds"}).to_csv(
            OUT_DIR / "association_selection.csv",
            header=["value"]
        )
        return None, None

    # make readable text columns
    saved_itemsets = saved_itemsets.copy()
    saved_itemsets["itemset_size"] = saved_itemsets["itemsets"].apply(len)
    saved_itemsets["itemsets_text"] = saved_itemsets["itemsets"].apply(lambda x: ", ".join(sorted(list(x))))
    saved_itemsets.sort_values(["itemset_size", "support"], ascending=[True, False], inplace=True)

    rules = saved_rules.copy()
    rules["antecedents_text"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents_text"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)

    # sort for practicality
    rules.sort_values(["lift", "confidence", "support"], ascending=[False, False, False], inplace=True)

    saved_itemsets.to_csv(OUT_DIR / "frequent_itemsets.csv", index=False)
    rules.to_csv(OUT_DIR / "association_rules.csv", index=False)

    pd.Series({
        "selected_support": selected_support,
        "min_confidence": MIN_CONFIDENCE,
        "min_lift": MIN_LIFT,
        "n_itemsets_selected": len(saved_itemsets),
        "n_rules_selected": len(rules),
    }).to_csv(OUT_DIR / "association_selection.csv", header=["value"])

    return saved_itemsets, rules


# -------------------------
# 6) Commercial shortlist
# -------------------------
def shortlist_rules(rules, product_perf):
    if rules is None or rules.empty:
        pd.DataFrame().to_csv(OUT_DIR / "commercial_rule_shortlist.csv", index=False)
        return pd.DataFrame()

    # product revenue lookup for consequence/antecedent relevance
    rev_map = product_perf["Revenue"].to_dict()

    shortlist = rules.copy()
    shortlist = shortlist[
        (shortlist["confidence"] >= 0.30) |
        (shortlist["lift"] >= 2.0) |
        (shortlist["support"] >= 0.02)
    ].copy()

    def rule_revenue_score(row):
        ants = list(row["antecedents"])
        cons = list(row["consequents"])
        vals = [rev_map.get(x, 0) for x in ants + cons]
        return float(np.mean(vals)) if vals else 0.0

    shortlist["avg_product_revenue_in_rule"] = shortlist.apply(rule_revenue_score, axis=1)
    shortlist.sort_values(
        ["lift", "confidence", "support", "avg_product_revenue_in_rule"],
        ascending=[False, False, False, False],
        inplace=True
    )

    keep_cols = [
        "antecedents_text", "consequents_text",
        "support", "confidence", "lift", "leverage", "conviction",
        "antecedent_support", "consequent_support",
        "avg_product_revenue_in_rule"
    ]
    shortlist[keep_cols].head(100).to_csv(OUT_DIR / "commercial_rule_shortlist.csv", index=False)

    return shortlist


# -------------------------
# 7) Segment-specific basket insights
# -------------------------
def segment_specific_insights(filtered_purchases):
    if not DAY07_SEGMENTS_PATH.exists():
        pd.Series({
            "status": "skipped",
            "reason": f"missing {DAY07_SEGMENTS_PATH}"
        }).to_csv(OUT_DIR / "segment_join_status.csv", header=["value"])
        return None

    segments = pd.read_csv(DAY07_SEGMENTS_PATH)
    if "CustomerID" not in segments.columns or "BusinessSegment" not in segments.columns:
        pd.Series({
            "status": "skipped",
            "reason": "day07 file exists but required columns missing"
        }).to_csv(OUT_DIR / "segment_join_status.csv", header=["value"])
        return None

    # make IDs comparable
    segments["CustomerID"] = pd.to_numeric(segments["CustomerID"], errors="coerce")
    fp = filtered_purchases.copy()
    fp["CustomerID"] = pd.to_numeric(fp["CustomerID"], errors="coerce")

    merged = fp.merge(
        segments[["CustomerID", "BusinessSegment"]],
        on="CustomerID",
        how="inner"
    )

    if merged.empty:
        pd.Series({
            "status": "skipped",
            "reason": "join produced no rows"
        }).to_csv(OUT_DIR / "segment_join_status.csv", header=["value"])
        return None

    seg_product = (
        merged.groupby(["BusinessSegment", "StockCode"])
        .agg(
            Orders=("InvoiceNo", "nunique"),
            Units=("Quantity", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    # Within each segment, compute product share by orders.
    seg_totals = seg_product.groupby("BusinessSegment")["Orders"].sum().rename("SegmentOrders")
    seg_product = seg_product.merge(seg_totals, on="BusinessSegment", how="left")
    seg_product["OrderShareWithinSegment"] = seg_product["Orders"] / seg_product["SegmentOrders"]

    seg_product.sort_values(["BusinessSegment", "OrderShareWithinSegment"], ascending=[True, False], inplace=True)
    seg_product.to_csv(OUT_DIR / "segment_product_preferences.csv", index=False)

    # Top 15 per segment to keep readable
    top15 = seg_product.groupby("BusinessSegment").head(15).copy()
    top15.to_csv(OUT_DIR / "segment_top_products.csv", index=False)

    pd.Series({
        "status": "ok",
        "joined_rows": len(merged),
        "segments_found": merged["BusinessSegment"].nunique(),
    }).to_csv(OUT_DIR / "segment_join_status.csv", header=["value"])

    return seg_product


# -------------------------
# 8) Optional product network (lightweight)
# -------------------------
def build_product_network_edges(rules):
    if rules is None or rules.empty:
        pd.DataFrame().to_csv(OUT_DIR / "product_network_edges.csv", index=False)
        return

    edges = rules.copy()
    # Keep only simple 1->1 rules for interpretable graph edges
    edges = edges[(edges["antecedents"].apply(len) == 1) & (edges["consequents"].apply(len) == 1)].copy()

    edges["source"] = edges["antecedents"].apply(lambda x: next(iter(x)))
    edges["target"] = edges["consequents"].apply(lambda x: next(iter(x)))

    edges = edges.sort_values(["lift", "confidence", "support"], ascending=False).head(300)

    edge_cols = ["source", "target", "support", "confidence", "lift", "leverage", "conviction"]
    edges[edge_cols].to_csv(OUT_DIR / "product_network_edges.csv", index=False)


# -------------------------
# Recommendations
# -------------------------
def write_recommendations(basket_stats, shortlist):
    if shortlist is None or shortlist.empty:
        text = """# Day 08 Recommendations

No association rules met the current thresholds.
Suggested next steps:
1. Lower min_support grid (e.g., 0.003).
2. Reduce product-frequency filter from 50 to 25 orders.
3. Re-run and validate top rules with lift > 1.
"""
        (OUT_DIR / "recommendations.md").write_text(text, encoding="utf-8")
        return

    top_lift = shortlist.head(10)
    top_conf = shortlist.sort_values(["confidence", "support"], ascending=False).head(10)
    top_support = shortlist.sort_values(["support", "confidence"], ascending=False).head(10)

    lines = [
        "# Day 08 Recommendations",
        "",
        "## Basket behavior relevance",
        f"- Single-product order share: {basket_stats['single_product_order_share']:.1%}",
        f"- Multi-product order share: {basket_stats['multi_product_order_share']:.1%}",
        "- Cross-sell analysis is most useful when multi-product share is substantial.",
        "",
        "## Rule shortlist strategy",
        "- Use high-lift rules for discovery of non-obvious pairings.",
        "- Use high-confidence rules for operational add-on recommendations.",
        "- Use high-support rules for broad merchandising/bundles.",
        "",
        "## Top high-lift opportunities (sample)",
    ]

    for _, r in top_lift.iterrows():
        lines.append(
            f"- {r['antecedents_text']} -> {r['consequents_text']} "
            f"(support={r['support']:.3f}, confidence={r['confidence']:.2f}, lift={r['lift']:.2f})"
        )

    lines += [
        "",
        "## Top high-confidence cross-sell opportunities (sample)",
    ]

    for _, r in top_conf.iterrows():
        lines.append(
            f"- {r['antecedents_text']} -> {r['consequents_text']} "
            f"(confidence={r['confidence']:.2f}, support={r['support']:.3f}, lift={r['lift']:.2f})"
        )

    lines += [
        "",
        "## Top high-support combinations (sample)",
    ]
    for _, r in top_support.iterrows():
        lines.append(
            f"- {r['antecedents_text']} -> {r['consequents_text']} "
            f"(support={r['support']:.3f}, confidence={r['confidence']:.2f}, lift={r['lift']:.2f})"
        )

    lines += [
        "",
        "## Operational recommendations",
        "1. Add rule-based add-on suggestions on product/cart pages for high-confidence pairs.",
        "2. Test bundle pricing for high-support, moderate-lift combinations.",
        "3. Prioritize high-revenue products that also appear in strong rules.",
        "4. If segment joins are available, tailor top bundles per segment instead of one global rule set.",
    ]

    (OUT_DIR / "recommendations.md").write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    df = load_data()
    purchases = split_valid_purchase_rows(df)

    perf = product_performance(purchases)
    basket_binary, basket_summary, filtered = build_baskets(purchases)
    basket_stats = basket_characteristics(basket_summary)

    _, rules = mine_itemsets_and_rules(basket_binary)
    shortlist = shortlist_rules(rules, perf)

    segment_specific_insights(filtered)
    build_product_network_edges(rules)
    write_recommendations(basket_stats, shortlist)

    print("Saved outputs to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()