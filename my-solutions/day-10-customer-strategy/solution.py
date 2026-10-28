from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    from ucimlrepo import fetch_ucirepo
except ImportError as e:
    raise ImportError("Please install ucimlrepo first: pip install ucimlrepo") from e


# -------------------------
# Config
# -------------------------
OUT_DIR = Path("reports/day10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY07_SEGMENTS_PATH = Path("reports/day07/rfm_scored_customers.csv")
DAY08_RULES_PATH = Path("reports/day08/commercial_rule_shortlist.csv")

NON_PRODUCT_CODES = {
    "POST", "D", "M", "DOT", "BANK CHARGES", "CRUK", "C2",
    "AMAZONFEE", "S", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"
}
OUTCOME_DAYS = 90
TREND_WINDOW_DAYS = 90
SEED = 42
CV_FOLDS = 5

# Priority matrix thresholds (documented assumptions)
VALUE_Q = 0.75   # "high value" = top 25% by expected 90-day value
RISK_Q = 0.70    # "high risk"  = top 30% by predicted inactivity probability

# Campaign scenario assumptions: NOT causal estimates, change them and re-run
BUDGET_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
PRIMARY_BUDGET = 0.10
CONTACT_COST = 2.0      # cost per contacted customer
SUCCESS_PROB = 0.15     # P(a would-be-inactive contacted customer is retained)
GROSS_MARGIN = 0.30     # share of retained revenue kept as margin
N_SIMS = 500


# -------------------------
# Data (same definitions as Day 9)
# -------------------------
def load_transactions():
    df = fetch_ucirepo(id=352).data.original.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["NonProduct"] = df["StockCode"].str.upper().isin(NON_PRODUCT_CODES)

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


def choose_cutoff(purchases):
    last_day = purchases["InvoiceDate"].max().normalize()
    cutoff = last_day - pd.Timedelta(days=OUTCOME_DAYS)
    return cutoff, cutoff + pd.Timedelta(days=OUTCOME_DAYS)


def build_target(purchases, cutoff, outcome_end):
    known = purchases.loc[purchases["InvoiceDate"] < cutoff, "CustomerID"].unique()
    window = purchases[(purchases["InvoiceDate"] >= cutoff) & (purchases["InvoiceDate"] < outcome_end)]
    target = pd.DataFrame({"CustomerID": known}).set_index("CustomerID")
    target["Inactive90"] = (~target.index.isin(window["CustomerID"].unique())).astype(int)
    # Realised spend in the outcome window: used ONLY for validation, never as a feature
    target["FutureSpend90"] = window.groupby("CustomerID")["Revenue"].sum().reindex(target.index).fillna(0)
    return target


def build_features(purchases, cancellations, cutoff):
    hp = purchases[purchases["InvoiceDate"] < cutoff].copy()
    hc = cancellations[cancellations["InvoiceDate"] < cutoff].copy()
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

    cancel_value = hc.groupby("CustomerID")["Revenue"].sum()  # negative
    feat["Monetary"] = feat["GrossValue"] + cancel_value.reindex(feat.index).fillna(0)
    cancel_invoices = hc.groupby("CustomerID")["InvoiceNo"].nunique().reindex(feat.index).fillna(0)
    feat["CancellationRate"] = cancel_invoices / (feat["Frequency"] + cancel_invoices)

    gaps = orders.copy()
    gaps["GapDays"] = gaps.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400
    gg = gaps.groupby("CustomerID")["GapDays"]
    feat["MeanDaysBetweenOrders"] = gg.mean()
    feat["StdDaysBetweenOrders"] = gg.std()

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
    aov_recent = feat["Spend90"] / feat["Orders90"].replace(0, np.nan)
    aov_prior = feat["SpendPrior90"] / feat["OrdersPrior90"].replace(0, np.nan)
    feat["AOVChange"] = aov_recent - aov_prior
    feat["SingleOrderCustomer"] = (feat["Frequency"] == 1).astype(int)
    return feat.drop(columns=["GrossValue"]), orders, hp


# -------------------------
# 1) Risk model (out-of-fold predictions for every customer)
# -------------------------
def signed_log(X):
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def make_linear_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(signed_log)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def make_forest_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                       n_jobs=-1, random_state=SEED)),
    ])


def out_of_fold_risk(X, y):
    """Each customer is scored by a model that never saw that customer's label."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    preds, rows = {}, []
    for name, pipe in {"Logistic Regression": make_linear_pipeline(),
                       "Random Forest": make_forest_pipeline()}.items():
        p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        preds[name] = p
        rows.append({"model": name, "oof_roc_auc": roc_auc_score(y, p)})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT_DIR / "risk_model_oof_scores.csv", index=False)
    best = scores.sort_values("oof_roc_auc", ascending=False).iloc[0]["model"]
    return pd.Series(preds[best], index=X.index, name="p_inactive"), best, scores


# -------------------------
# 2) Customer value
# -------------------------
def build_value(feat, cutoff, data_start):
    """
    Value90 = expected net spend in a 90-day period, assuming each customer keeps their
    pre-cutoff run rate:  max(Monetary, 0) * 90 / observed_days.
    Monetary is net of cancellations dated before the cutoff.
    """
    observed_days = max((cutoff - data_start).days, 1)
    return (feat["Monetary"].clip(lower=0) * OUTCOME_DAYS / observed_days).rename("Value90")


def fit_clv(orders, feat, cutoff):
    """Optional BG/NBD + Gamma-Gamma. Returns None if `lifetimes` is unavailable or fails."""
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        from lifetimes.utils import summary_data_from_transaction_data
    except ImportError:
        print("lifetimes not installed: skipping optional CLV (pip install lifetimes)")
        return None

    try:
        tx = orders.loc[orders["OrderValue"] > 0, ["CustomerID", "InvoiceDate", "OrderValue"]]
        summ = summary_data_from_transaction_data(
            tx, "CustomerID", "InvoiceDate", monetary_value_col="OrderValue",
            observation_period_end=cutoff - pd.Timedelta(days=1))

        bgf = BetaGeoFitter(penalizer_coef=0.001).fit(summ["frequency"], summ["recency"], summ["T"])
        exp_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            OUTCOME_DAYS, summ["frequency"], summ["recency"], summ["T"])

        repeat = summ[(summ["frequency"] > 0) & (summ["monetary_value"] > 0)]
        corr = repeat[["frequency", "monetary_value"]].corr().iloc[0, 1]
        ggf = GammaGammaFitter(penalizer_coef=0.01).fit(repeat["frequency"], repeat["monetary_value"])

        # Gamma-Gamma needs repeat purchasers; one-order customers fall back to their own order value
        avg_value = feat["AvgOrderValue"].clip(lower=0).copy()
        avg_value.loc[repeat.index] = ggf.conditional_expected_average_profit(
            repeat["frequency"], repeat["monetary_value"])

        clv = pd.DataFrame({
            "ExpectedPurchases90": exp_purchases.reindex(feat.index),
            "ExpectedAvgOrderValue": avg_value.reindex(feat.index),
        })
        clv["ExpectedCLV90"] = (clv["ExpectedPurchases90"] * clv["ExpectedAvgOrderValue"]).clip(lower=0)
        clv.attrs["freq_monetary_corr"] = float(corr)
        clv.to_csv(OUT_DIR / "clv_estimates.csv")
        return clv
    except Exception as e:  # keep the main pipeline alive
        print(f"CLV modelling failed ({type(e).__name__}: {e}); continuing without CLV")
        return None


# -------------------------
# 3) Targeting strategies + 6) campaign simulation
# -------------------------
def evaluate_strategies(df, scores, rng):
    """
    Realised loss for a customer = Inactive90 * Value90 (value that disappears if they go quiet).
    Campaign: contacted AND truly inactive customers are retained with prob SUCCESS_PROB.
    Net benefit = retained loss * margin - contact cost. Customers who would have stayed active
    gain nothing from contact (no incremental effect), but still cost money.
    """
    inactive = df["Inactive90"].to_numpy()
    loss = inactive * df["Value90"].to_numpy()
    n, total_inactive, total_loss = len(df), inactive.sum(), loss.sum()

    rows = []
    for f in BUDGET_FRACTIONS:
        k = int(np.ceil(f * n))
        cost = k * CONTACT_COST
        for name, score in scores.items():
            if score is None:  # random targeting, averaged over many draws
                nets, inc, lc = [], [], []
                for _ in range(N_SIMS):
                    idx = rng.choice(n, k, replace=False)
                    succ = rng.random(k) < SUCCESS_PROB
                    nets.append((succ * loss[idx] * GROSS_MARGIN).sum() - cost)
                    inc.append(inactive[idx].sum())
                    lc.append(loss[idx].sum())
                nets, inc_k, loss_k = np.array(nets), np.mean(inc), np.mean(lc)
            else:
                idx = np.argsort(-score, kind="mergesort")[:k]
                succ = rng.random((N_SIMS, k)) < SUCCESS_PROB
                nets = (succ * (loss[idx] * GROSS_MARGIN)).sum(axis=1) - cost
                inc_k, loss_k = inactive[idx].sum(), loss[idx].sum()
            rows.append({
                "budget_fraction": f, "k_contacted": k, "strategy": name,
                "inactive_captured_share": inc_k / total_inactive,
                "value_at_risk_captured_share": loss_k / total_loss,
                "net_benefit_mean": nets.mean(),
                "net_benefit_p5": np.percentile(nets, 5),
                "net_benefit_p95": np.percentile(nets, 95),
                "campaign_cost": cost,
                "roi": nets.mean() / cost,
            })
    return pd.DataFrame(rows)


def sensitivity(df, scores):
    """Expected net benefit at the primary budget across success-probability and cost assumptions."""
    inactive = df["Inactive90"].to_numpy()
    loss = inactive * df["Value90"].to_numpy()
    n = len(df)
    k = int(np.ceil(PRIMARY_BUDGET * n))
    rows = []
    for name, score in scores.items():
        captured = loss.sum() * PRIMARY_BUDGET if score is None else loss[np.argsort(-score, kind="mergesort")[:k]].sum()
        for s in [0.05, 0.10, 0.15, 0.30]:
            for c in [1, 2, 5, 10, 20]:
                rows.append({"strategy": name, "success_prob": s, "contact_cost": c,
                             "expected_net_benefit": s * GROSS_MARGIN * captured - k * c})
        rows.append({"strategy": name, "success_prob": "break-even at current cost",
                     "contact_cost": CONTACT_COST,
                     "expected_net_benefit": (k * CONTACT_COST) / (GROSS_MARGIN * captured) if captured > 0 else np.nan})
    return pd.DataFrame(rows)


# -------------------------
# 3b) Priority groups
# -------------------------
GROUP_STRATEGY = {
    "Immediate retention priority": "High value + high risk: personal outreach, tailored offer, account manager for the top accounts.",
    "Protect / loyalty": "High value + low risk: loyalty perks and service quality; avoid discounting people who would buy anyway.",
    "Low-cost intervention": "Low value + high risk: automated, cheap channels (email); no costly incentives.",
    "Minimal intervention": "Low value + low risk: standard communication only.",
}


def assign_priority(df):
    v_thr = df["Value90"].quantile(VALUE_Q)
    r_thr = df["p_inactive"].quantile(RISK_Q)
    hv, hr = df["Value90"] >= v_thr, df["p_inactive"] >= r_thr
    df["PriorityGroup"] = np.select(
        [hv & hr, hv & ~hr, ~hv & hr],
        ["Immediate retention priority", "Protect / loyalty", "Low-cost intervention"],
        default="Minimal intervention")

    df["ExpectedVaR"] = df["p_inactive"] * df["Value90"]
    df["RealisedLoss"] = df["Inactive90"] * df["Value90"]

    prof = df.groupby("PriorityGroup").agg(
        customers=("p_inactive", "size"),
        mean_p_inactive=("p_inactive", "mean"),
        realised_inactive_rate=("Inactive90", "mean"),
        mean_value90=("Value90", "mean"),
        total_value90=("Value90", "sum"),
        expected_var=("ExpectedVaR", "sum"),
        realised_loss=("RealisedLoss", "sum"),
    )
    prof["customer_share"] = prof["customers"] / prof["customers"].sum()
    prof["value_share"] = prof["total_value90"] / prof["total_value90"].sum()
    prof["expected_var_share"] = prof["expected_var"] / prof["expected_var"].sum()
    prof["suggested_strategy"] = prof.index.map(GROUP_STRATEGY)
    prof = prof.sort_values("expected_var", ascending=False)
    prof.to_csv(OUT_DIR / "priority_matrix.csv")
    return df, prof, v_thr, r_thr


# -------------------------
# 7) Product intelligence link
# -------------------------
def product_links(df, hist_purchases):
    hp = hist_purchases.merge(df[["PriorityGroup"]], left_on="CustomerID", right_index=True, how="inner")
    desc = hp.groupby("StockCode")["Description"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "")

    rows = []
    for grp, sub in hp.groupby("PriorityGroup"):
        n_cust = sub["CustomerID"].nunique()
        top = (sub.groupby("StockCode")["CustomerID"].nunique() / n_cust).sort_values(ascending=False).head(10)
        for code, share in top.items():
            rows.append({"PriorityGroup": grp, "StockCode": code, "Description": desc.get(code, ""),
                         "share_of_group_customers_buying": share})
    pd.DataFrame(rows).to_csv(OUT_DIR / "priority_group_top_products.csv", index=False)

    # reactivation offer = each priority customer's most frequently bought product before the cutoff
    imm_ids = df[df["PriorityGroup"] == "Immediate retention priority"].sort_values("ExpectedVaR", ascending=False).head(200).index
    sub = hp[hp["CustomerID"].isin(imm_ids)]
    counts = sub.groupby(["CustomerID", "StockCode"])["InvoiceNo"].nunique().rename("orders").reset_index()
    best = counts.sort_values(["CustomerID", "orders"], ascending=[True, False]).drop_duplicates("CustomerID")
    best["Description"] = best["StockCode"].map(desc)
    best = best.merge(df.loc[imm_ids, ["p_inactive", "Value90", "ExpectedVaR"]], left_on="CustomerID", right_index=True)
    best.sort_values("ExpectedVaR", ascending=False).to_csv(OUT_DIR / "reactivation_offer_candidates.csv", index=False)

    if DAY08_RULES_PATH.exists():  # Day 8 rules were mined on the full year: use for ideas, not evaluation
        pd.read_csv(DAY08_RULES_PATH).head(15).to_csv(OUT_DIR / "cross_sell_rules_from_day08.csv", index=False)

    if DAY07_SEGMENTS_PATH.exists():
        seg = pd.read_csv(DAY07_SEGMENTS_PATH, usecols=["CustomerID", "BusinessSegment"])
        seg["CustomerID"] = pd.to_numeric(seg["CustomerID"], errors="coerce").astype("Int64")
        j = df[["PriorityGroup"]].reset_index().merge(seg, on="CustomerID", how="inner")
        pd.crosstab(j["BusinessSegment"], j["PriorityGroup"]).to_csv(OUT_DIR / "segment_vs_priority_crosstab.csv")


# -------------------------
# Plots
# -------------------------
def plot_net_benefit(results):
    plt.figure(figsize=(8, 5))
    for name, sub in results.groupby("strategy"):
        sub = sub.sort_values("budget_fraction")
        plt.plot(sub["budget_fraction"] * 100, sub["net_benefit_mean"], marker="o", label=name)
    plt.axhline(0, color="gray", ls="--")
    plt.xlabel("Share of customers contacted (%)")
    plt.ylabel("Mean net benefit (scenario currency)")
    plt.title("Campaign net benefit by targeting strategy (scenario assumptions)")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "net_benefit_by_budget.png", dpi=160)
    plt.close()


def plot_matrix(df, v_thr, r_thr):
    plt.figure(figsize=(8, 6))
    colors = {"Immediate retention priority": "tab:red", "Protect / loyalty": "tab:green",
              "Low-cost intervention": "tab:orange", "Minimal intervention": "tab:gray"}
    for grp, sub in df.groupby("PriorityGroup"):
        plt.scatter(sub["p_inactive"], sub["Value90"] + 1, s=8, alpha=0.4, label=grp, color=colors[grp])
    plt.axvline(r_thr, color="black", ls="--", alpha=0.5)
    plt.axhline(v_thr + 1, color="black", ls="--", alpha=0.5)
    plt.yscale("log")
    plt.xlabel("Predicted P(inactive in next 90 days)")
    plt.ylabel("Expected 90-day value (log scale, +1)")
    plt.title("Risk x value priority matrix")
    plt.legend(markerscale=2); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(OUT_DIR / "risk_value_matrix.png", dpi=160)
    plt.close()


# -------------------------
# 8) Experiment design
# -------------------------
def write_experiment_design(df):
    imm = df[df["PriorityGroup"] == "Immediate retention priority"]
    p0 = float(1 - imm["Inactive90"].mean())   # observed purchase rate in the group without a campaign
    delta = 0.05
    p1 = min(p0 + delta, 0.99)
    n_arm = int(np.ceil((norm.ppf(0.975) + norm.ppf(0.80)) ** 2
                        * (p0 * (1 - p0) + p1 * (1 - p1)) / delta ** 2))

    text = f"""# Proposed randomized retention experiment

## Why an experiment
Everything in this analysis is observational plus scenario assumptions. The campaign simulation
assumed a {SUCCESS_PROB:.0%} retention success rate; only randomization can measure the real effect.

## Design
- **Eligible population:** customers in the *Immediate retention priority* group at the scoring date
  (top {1 - VALUE_Q:.0%} by expected value AND top {1 - RISK_Q:.0%} by predicted inactivity risk).
  Currently {len(imm):,} customers in this historical snapshot.
- **Randomization:** individual-level, stratified by value tier and country, 1:1:1 across three arms.
- **Arms:**
  1. **Control:** no campaign (holdout). Normal communication only.
  2. **Generic offer:** standard reactivation email with a modest incentive.
  3. **Personalised offer:** reactivation email featuring the customer's most-bought products and
     Day 8 complementary products.
- **Optional second experiment (tests the model, not the offer):** within the whole customer base,
  randomize *model-selected* vs *randomly-selected* contact lists with the same budget.
- **Primary outcome:** at least one successful purchase within 90 days of campaign start
  (same definition as `Inactive90`, so it connects to the model).
- **Secondary outcomes:** net revenue per customer, margin after incentive cost, days to next order, cancellation rate.
- **Guardrails:** unsubscribe/complaint rate, return rate, margin erosion from discounts.
- **Evaluation period:** 90 days after launch, with a pre-registered analysis plan.
- **Business success metric:** incremental net margin per eligible customer
  (margin on incremental revenue minus incentive and contact cost) is above zero, with a 95% CI.

## Sample size (normal approximation, two-sided alpha=0.05, power=0.80)
- Baseline purchase rate in this group without a campaign (historical): **{p0:.1%}**
- Minimum detectable effect: **+{delta * 100:.0f} percentage points**
- Required customers per arm: **~{n_arm:,}** (group size available now: {len(imm):,}, across 3 arms)
- If the eligible group is smaller than the required total, either widen eligibility, run the
  experiment over several weeks of rolling enrolment, or accept a larger minimum detectable effect.

## Analysis
- Intention-to-treat comparison of arms with confidence intervals (bootstrap for revenue metrics).
- Pre-specified subgroup checks (value tier, risk band) with multiple-testing caution.
- Use the results to move from "who is likely to leave" to **uplift**: who is *changed* by the treatment.

## Why this moves us toward causal decisions
Random assignment makes treated and control customers comparable on observed and unobserved traits,
so the outcome difference estimates the effect of the intervention rather than of who we chose to contact.
"""
    (OUT_DIR / "experiment_design.md").write_text(text, encoding="utf-8")
    return n_arm, p0


# -------------------------
# Summary
# -------------------------
def write_summary(df, prof, results, design, best_model, scores, clv, corr_rv, top_overlap, var_conc, value_check, n_arm):
    prim = results[results["budget_fraction"] == PRIMARY_BUDGET].set_index("strategy")
    cols = ["inactive_captured_share", "value_at_risk_captured_share", "net_benefit_mean", "roi"]
    rand_net = prim.loc["Random", "net_benefit_mean"]
    best_strat = prim["net_benefit_mean"].idxmax()

    lines = [
        "# Day 10 - From Prediction to Retention Strategy", "",
        "## Setup",
        f"- Cutoff {design['cutoff']:%Y-%m-%d}; outcome window ends {design['outcome_end']:%Y-%m-%d}; "
        f"{len(df):,} known customers; inactive rate {df['Inactive90'].mean():.1%}.",
        f"- Risk model: **{best_model}** (chosen by out-of-fold ROC-AUC). Every customer is scored out-of-fold.",
        f"- `Value90` = max(net Monetary, 0) x 90 / observed days: a run-rate proxy built only from pre-cutoff data.",
        "- Realised value at risk for evaluation = `Inactive90 x Value90`.", "",
        "## Risk model check", scores.round(3).to_markdown(index=False), "",
        "## Targeting strategies at the primary budget "
        f"({PRIMARY_BUDGET:.0%} of customers; contact cost {CONTACT_COST}, success {SUCCESS_PROB:.0%}, margin {GROSS_MARGIN:.0%})",
        prim[cols].round(3).to_markdown(), "",
        "## Priority matrix", prof.drop(columns=["suggested_strategy"]).round(3).to_markdown(), "",
        "## Answers to the final questions",
        f"1. **Are the highest-risk customers also the most valuable?** Spearman correlation between risk and value is "
        f"**{corr_rv:.2f}**; only **{top_overlap:.0%}** of the top-10% risk customers are also in the top-10% by value.",
        f"2. **Greatest value at risk:** the top 10% of customers by expected value at risk hold **{var_conc[0.10]:.1%}** of total "
        f"expected VaR (top 1%: {var_conc[0.01]:.1%}). See `customer_risk_value_ranking.csv`.",
        f"3. **Risk targeting vs random:** at {PRIMARY_BUDGET:.0%}, highest-risk captures "
        f"{prim.loc['Highest risk', 'inactive_captured_share']:.1%} of inactive customers vs "
        f"{prim.loc['Random', 'inactive_captured_share']:.1%} for random.",
        f"4. **Does combining risk and value help?** Highest value-at-risk captures "
        f"{prim.loc['Highest value at risk', 'value_at_risk_captured_share']:.1%} of realised value at risk vs "
        f"{prim.loc['Highest risk', 'value_at_risk_captured_share']:.1%} (risk only) and "
        f"{prim.loc['Highest value', 'value_at_risk_captured_share']:.1%} (value only). Best net benefit: **{best_strat}** "
        f"({prim.loc[best_strat, 'net_benefit_mean']:,.0f} vs random {rand_net:,.0f}).",
        "5. **Different strategies by group:** see `suggested_strategy` in `priority_matrix.csv`.",
        f"6. **Simulation assumptions:** contact cost, {SUCCESS_PROB:.0%} success rate, {GROSS_MARGIN:.0%} margin, "
        "value proxy = historical run rate, and no effect on customers who would have stayed anyway. None of these are estimated from the data.",
        f"7. **Next experiment:** see `experution_design.md`".replace("experution", "experiment") + f" (~{n_arm:,} customers per arm).", "",
        "## Caveats",
        f"- **Metric bias:** realised loss is defined with the same value proxy used by value-based strategies, which favours them by construction.",
        f"- Value proxy check: among customers who stayed active, Spearman(Value90, realised future spend) = {value_check:.2f}.",
        "- Single cutoff in a Q4-heavy year; the inactivity rate and value proxies will not transfer unchanged to other periods.",
        "- Risk is predictive, not causal. High risk does not mean contact will help; that needs the experiment.",
    ]
    if clv is not None:
        lines += ["", "## Optional CLV (BG/NBD + Gamma-Gamma)",
                  f"- Frequency/monetary correlation among repeat buyers: {clv.attrs.get('freq_monetary_corr', float('nan')):.2f} "
                  "(Gamma-Gamma assumes near-zero).",
                  "- BG/NBD already models dropout, so `P(inactive) x ExpectedCLV` partly double-counts; treat it as an exposure ranking, not a forecast of lost revenue."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    rng = np.random.default_rng(SEED)
    purchases, cancellations = load_transactions()
    cutoff, outcome_end = choose_cutoff(purchases)
    data_start = purchases["InvoiceDate"].min().normalize()

    target = build_target(purchases, cutoff, outcome_end)
    feat, orders, hist_purchases = build_features(purchases, cancellations, cutoff)
    data = feat.join(target, how="inner")

    X, y = data.drop(columns=["Inactive90", "FutureSpend90"]), data["Inactive90"]
    p_inactive, best_model, scores = out_of_fold_risk(X, y)

    df = pd.DataFrame({
        "p_inactive": p_inactive,
        "Value90": build_value(feat, cutoff, data_start).reindex(data.index),
        "Inactive90": y,
        "FutureSpend90": data["FutureSpend90"],
    })

    clv = fit_clv(orders, feat, cutoff)
    if clv is not None:
        df = df.join(clv[["ExpectedCLV90"]])

    # --- analysis of risk vs value ---
    corr_rv = df["p_inactive"].corr(df["Value90"], method="spearman")
    n10 = int(np.ceil(0.10 * len(df)))
    top_risk = set(df["p_inactive"].nlargest(n10).index)
    top_value = set(df["Value90"].nlargest(n10).index)
    top_overlap = len(top_risk & top_value) / n10
    df["ExpectedVaR"] = df["p_inactive"] * df["Value90"]
    var_sorted = df["ExpectedVaR"].sort_values(ascending=False)
    var_conc = {q: var_sorted.head(int(np.ceil(q * len(df)))).sum() / var_sorted.sum() for q in (0.01, 0.10)}
    active = df[df["Inactive90"] == 0]
    value_check = active["Value90"].corr(active["FutureSpend90"], method="spearman")

    # --- priority groups + strategies ---
    df, prof, v_thr, r_thr = assign_priority(df)

    scores_by_strategy = {
        "Random": None,
        "Highest risk": df["p_inactive"].to_numpy(),
        "Highest value": df["Value90"].to_numpy(),
        "Highest value at risk": df["ExpectedVaR"].to_numpy(),
    }
    if "ExpectedCLV90" in df:
        scores_by_strategy["Highest expected VaR (CLV)"] = (df["p_inactive"] * df["ExpectedCLV90"].fillna(0)).to_numpy()

    results = evaluate_strategies(df, scores_by_strategy, rng)
    results.to_csv(OUT_DIR / "targeting_strategy_comparison.csv", index=False)
    sensitivity(df, scores_by_strategy).to_csv(OUT_DIR / "campaign_sensitivity.csv", index=False)

    df.sort_values("ExpectedVaR", ascending=False).to_csv(OUT_DIR / "customer_risk_value_ranking.csv")
    plot_net_benefit(results)
    plot_matrix(df, v_thr, r_thr)
    product_links(df, hist_purchases)
    n_arm, _ = write_experiment_design(df)

    design = {"cutoff": cutoff, "outcome_end": outcome_end}
    write_summary(df, prof, results, design, best_model, scores, clv,
                  corr_rv, top_overlap, var_conc, value_check, n_arm)

    print(results[results["budget_fraction"] == PRIMARY_BUDGET].round(3).to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()