from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise ImportError("Please install ucimlrepo first: pip install ucimlrepo") from e


# -------------------------
# Config
# -------------------------
OUT_DIR = Path("data/rfm/reports/day09")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
}
OUTCOME_DAYS = 90
TREND_WINDOW_DAYS = 90
TEST_SIZE = 0.25
SEED = 42
TOPK_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]


# -------------------------
# Data
# -------------------------
def load_transactions():
    online_retail = pd.read_parquet('data/rfm/online_retail.parquet')
    df = online_retail.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["NonProduct"] = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)

    # customer-level prediction needs a known customer
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)

    purchases = df[
        (~df["IsCancellation"]) & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0) & (~df["NonProduct"])
    ].copy()
    cancellations = df[
        (df["IsCancellation"]) & (df["Quantity"] < 0)
        & (df["UnitPrice"] > 0) & (~df["NonProduct"])
    ].copy()
    return purchases, cancellations


# -------------------------
# 1-3) Temporal design + target
# -------------------------
def choose_cutoff(purchases):
    """Cutoff leaves a COMPLETE 90-day outcome window inside the data."""
    last_day = purchases["InvoiceDate"].max().normalize()
    cutoff = last_day - pd.Timedelta(days=OUTCOME_DAYS)
    outcome_end = cutoff + pd.Timedelta(days=OUTCOME_DAYS)
    return cutoff, outcome_end


def build_target(purchases, cutoff, outcome_end):
    known = purchases.loc[purchases["InvoiceDate"] < cutoff, "CustomerID"].unique()
    window = purchases[(purchases["InvoiceDate"] >= cutoff) & (purchases["InvoiceDate"] < outcome_end)]
    active_future = set(window["CustomerID"].unique())

    target = pd.DataFrame({"CustomerID": known})
    target["Inactive90"] = (~target["CustomerID"].isin(active_future)).astype(int)
    return target.set_index("CustomerID")


# -------------------------
# 4) Leakage-safe features (only rows < cutoff)
# -------------------------
def build_features(purchases, cancellations, cutoff):
    hp = purchases[purchases["InvoiceDate"] < cutoff].copy()
    hc = cancellations[cancellations["InvoiceDate"] < cutoff].copy()

    # leakage guard
    assert hp["InvoiceDate"].max() < cutoff and (hc.empty or hc["InvoiceDate"].max() < cutoff)

    orders = (
        hp.groupby(["CustomerID", "InvoiceNo"])
        .agg(InvoiceDate=("InvoiceDate", "min"),
             OrderValue=("Revenue", "sum"),
             Units=("Quantity", "sum"))
        .reset_index()
        .sort_values(["CustomerID", "InvoiceDate"])
    )

    g = orders.groupby("CustomerID")
    feat = pd.DataFrame({
        "Recency": (cutoff - g["InvoiceDate"].max()).dt.days,
        "Frequency": g["InvoiceNo"].nunique(),
        "GrossValue": g["OrderValue"].sum(),
        "AvgOrderValue": g["OrderValue"].mean(),
        "ItemsPerOrder": g["Units"].mean(),
        "CustomerTenure": (g["InvoiceDate"].max() - g["InvoiceDate"].min()).dt.days,
    })
    feat["ProductDiversity"] = hp.groupby("CustomerID")["StockCode"].nunique()

    # Monetary = net of cancellations that happened before the cutoff
    cancel_value = hc.groupby("CustomerID")["Revenue"].sum()  # negative
    feat["Monetary"] = feat["GrossValue"] + cancel_value.reindex(feat.index).fillna(0)

    cancel_invoices = hc.groupby("CustomerID")["InvoiceNo"].nunique().reindex(feat.index).fillna(0)
    feat["CancellationRate"] = cancel_invoices / (feat["Frequency"] + cancel_invoices)

    # purchase intervals (NaN if <2 orders for mean, <3 orders for std)
    orders["GapDays"] = orders.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400
    gaps = orders.groupby("CustomerID")["GapDays"]
    feat["MeanDaysBetweenOrders"] = gaps.mean()
    feat["StdDaysBetweenOrders"] = gaps.std()

    # behavioural change: last 90 days vs the 90 days before that
    recent_start = cutoff - pd.Timedelta(days=TREND_WINDOW_DAYS)
    prior_start = cutoff - pd.Timedelta(days=2 * TREND_WINDOW_DAYS)
    recent = orders[orders["InvoiceDate"] >= recent_start]
    prior = orders[(orders["InvoiceDate"] >= prior_start) & (orders["InvoiceDate"] < recent_start)]

    r = recent.groupby("CustomerID").agg(Orders90=("InvoiceNo", "nunique"), Spend90=("OrderValue", "sum"))
    p = prior.groupby("CustomerID").agg(OrdersPrior90=("InvoiceNo", "nunique"), SpendPrior90=("OrderValue", "sum"))
    feat = feat.join(r).join(p)
    for c in ["Orders90", "Spend90", "OrdersPrior90", "SpendPrior90"]:
        feat[c] = feat[c].fillna(0)

    feat["OrdersTrend"] = feat["Orders90"] - feat["OrdersPrior90"]
    feat["SpendTrend"] = feat["Spend90"] - feat["SpendPrior90"]
    aov_recent = (feat["Spend90"] / feat["Orders90"].replace(0, np.nan))
    aov_prior = (feat["SpendPrior90"] / feat["OrdersPrior90"].replace(0, np.nan))
    feat["AOVChange"] = aov_recent - aov_prior  # NaN when either window has no orders

    feat["SingleOrderCustomer"] = (feat["Frequency"] == 1).astype(int)
    return feat.drop(columns=["GrossValue"])


# -------------------------
# 5-7) Models
# -------------------------
def signed_log(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def make_linear_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def make_forest_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=SEED)),
    ])


# -------------------------
# 8) Evaluation
# -------------------------
def evaluate(name, y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision@0.5": precision_score(y, pred, zero_division=0),
        "recall@0.5": recall_score(y, pred, zero_division=0),
        "f1@0.5": f1_score(y, pred, zero_division=0),
    }


def topk_table(name, y, p):
    y = np.asarray(y)
    order = np.argsort(-p, kind="mergesort")
    base = y.mean()
    rows = []
    for f in TOPK_FRACTIONS:
        k = int(np.ceil(f * len(y)))
        captured = y[order[:k]].sum()
        precision = captured / k
        rows.append({
            "model": name, "top_fraction": f, "k_customers": k,
            "inactive_captured": int(captured),
            "recall_at_k": captured / y.sum(),
            "precision_at_k": precision,
            "lift_vs_random": precision / base,
        })
    return pd.DataFrame(rows)


# -------------------------
# Plots
# -------------------------
def plot_curves(y, probs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(np.mean(y), ls="--", color="gray")
    ax[1].set(title="Precision-Recall curve", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.grid(alpha=0.25); a.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_pr_curves.png", dpi=160)
    plt.close()


def plot_gain(y, probs):
    y = np.asarray(y)
    plt.figure(figsize=(7, 5))
    for name, p in probs.items():
        order = np.argsort(-p, kind="mergesort")
        cum = np.cumsum(y[order]) / y.sum()
        plt.plot(np.arange(1, len(y) + 1) / len(y), cum, label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random")
    plt.axvline(0.10, color="red", alpha=0.3)
    plt.xlabel("Share of customers contacted (highest risk first)")
    plt.ylabel("Share of inactive customers captured")
    plt.title("Cumulative gain (targeting)")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "cumulative_gain.png", dpi=160)
    plt.close()


def plot_bar(series, title, fname, xlabel):
    s = series.sort_values()
    plt.figure(figsize=(8, max(4, 0.35 * len(s))))
    plt.barh(s.index, s.values)
    plt.title(title); plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=160)
    plt.close()


# -------------------------
# 9) Explanations
# -------------------------
def explain_logistic(lr_pipe, X_test, p_test, top_n=20):
    names = lr_pipe[:-1].get_feature_names_out()
    coef = pd.Series(lr_pipe.named_steps["clf"].coef_[0], index=names)
    table = pd.DataFrame({
        "coef_per_1sd": coef,
        "odds_ratio_per_1sd": np.exp(coef),
        "direction": np.where(coef > 0, "higher -> more likely inactive", "higher -> less likely inactive"),
    }).sort_values("coef_per_1sd", key=np.abs, ascending=False)
    table.to_csv(OUT_DIR / "logistic_coefficients.csv")
    plot_bar(coef, "Logistic Regression coefficients (standardised)", "logistic_coefficients.png",
             "Coefficient (log-odds per 1 SD)")

    # per-customer contributions for the highest-risk test customers
    Xs = lr_pipe[:-1].transform(X_test)
    contrib = pd.DataFrame(Xs * coef.values, index=X_test.index, columns=names)
    top_idx = pd.Series(p_test, index=X_test.index).sort_values(ascending=False).head(top_n).index
    rows = []
    for cid in top_idx:
        c = contrib.loc[cid].sort_values(ascending=False)
        rows.append({
            "CustomerID": cid,
            "predicted_inactive_prob": float(pd.Series(p_test, index=X_test.index)[cid]),
            "top_risk_drivers": "; ".join(f"{k} ({v:+.2f})" for k, v in c.head(3).items()),
            "top_protective_factors": "; ".join(f"{k} ({v:+.2f})" for k, v in c.tail(3)[::-1].items()),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "high_risk_customer_explanations.csv", index=False)
    return table


def explain_forest(rf_pipe, X_test, y_test):
    result = permutation_importance(
        rf_pipe, X_test, y_test, scoring="roc_auc",
        n_repeats=20, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }, index=X_test.columns).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT_DIR / "forest_permutation_importance.csv")
    plot_bar(imp["importance_mean"], "Random Forest permutation importance (drop in ROC-AUC)",
             "forest_permutation_importance.png", "Mean ROC-AUC decrease")
    return imp


# -------------------------
# Summary
# -------------------------
def write_summary(cutoff, outcome_end, n_known, prevalence, metrics, topk, lr_table, rf_imp):
    m = metrics.set_index("model")
    t10 = topk[topk["top_fraction"] == 0.10].set_index("model")
    top_lr = lr_table.head(5)
    top_rf = rf_imp.head(5)

    lines = [
        "# Day 09 - 90-Day Inactivity Prediction",
        "",
        "## Design",
        f"- Cutoff: **{cutoff:%Y-%m-%d}**; outcome window: [{cutoff:%Y-%m-%d}, {outcome_end:%Y-%m-%d}).",
        "- Features use ONLY transactions before the cutoff (asserted in code).",
        f"- Customers known before cutoff: **{n_known:,}**; inactive rate (Inactive90=1): **{prevalence:.1%}**.",
        "- `Inactive90 = 1` means no successful purchase in the next 90 days. It is a modelling",
        "  definition, not proof of permanent churn: some customers buy seasonally or irregularly.",
        "- Train/test split is a stratified random split of customers at one shared cutoff.",
        "",
        "## Test metrics",
        m.round(3).to_markdown(),
        "",
        "## Targeting: top 10% highest-risk customers",
        t10[["k_customers", "inactive_captured", "recall_at_k", "precision_at_k", "lift_vs_random"]].round(3).to_markdown(),
        "",
        "## Strongest logistic effects (per 1 SD, after signed-log transform)",
        top_lr[["coef_per_1sd", "odds_ratio_per_1sd", "direction"]].round(3).to_markdown(),
        "",
        "## Top Random Forest drivers (permutation importance)",
        top_rf.round(4).to_markdown(),
        "",
        "## Caveats",
        "- Associations are predictive, not causal; contacting a customer may not change the outcome.",
        "- Single cutoff: results may shift with another cutoff (seasonality, e.g. Q4 effects).",
        "- A customer-level random split shares one calendar period; a later-cutoff test would be stricter.",
        "- Customers without a CustomerID are excluded, and the model cannot see why a customer stops buying.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    purchases, cancellations = load_transactions()
    cutoff, outcome_end = choose_cutoff(purchases)

    target = build_target(purchases, cutoff, outcome_end)
    features = build_features(purchases, cancellations, cutoff)

    data = features.join(target, how="inner")
    data.to_csv(OUT_DIR / "customer_features_target.csv")

    X = data.drop(columns=["Inactive90"])
    y = data["Inactive90"]

    pd.Series({
        "cutoff": str(cutoff.date()),
        "outcome_end_exclusive": str(outcome_end.date()),
        "customers": len(data),
        "inactive_rate": y.mean(),
    }).to_csv(OUT_DIR / "design.csv", header=["value"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    lr = make_linear_pipeline().fit(X_train, y_train)
    rf = make_forest_pipeline().fit(X_train, y_train)

    probs = {
        "Dummy (prior)": dummy.predict_proba(X_test)[:, 1],
        "Logistic Regression": lr.predict_proba(X_test)[:, 1],
        "Random Forest": rf.predict_proba(X_test)[:, 1],
    }

    metrics = pd.DataFrame([evaluate(n, y_test, p) for n, p in probs.items()])
    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    topk = pd.concat([topk_table(n, y_test, p) for n, p in probs.items()], ignore_index=True)
    topk.to_csv(OUT_DIR / "topk_targeting.csv", index=False)

    plot_curves(y_test, probs)
    plot_gain(y_test, probs)

    lr_table = explain_logistic(lr, X_test, probs["Logistic Regression"])
    rf_imp = explain_forest(rf, X_test, y_test)

    pd.DataFrame({
        "CustomerID": X_test.index,
        "Inactive90": y_test.values,
        "p_logistic": probs["Logistic Regression"],
        "p_forest": probs["Random Forest"],
    }).sort_values("p_forest", ascending=False).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    write_summary(cutoff, outcome_end, len(data), y.mean(), metrics, topk, lr_table, rf_imp)

    print(metrics.round(3).to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()