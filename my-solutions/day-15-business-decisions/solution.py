import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ML_FORECASTS_FP = OUTPUT_DIR / "day-14-ml-forecasts.parquet"
BASELINE_FORECASTS_FP = OUTPUT_DIR / "day-13-baseline-forecasts.parquet"  # optional, not strictly required here
TRUST_MAP_FP = OUTPUT_DIR / "day-15-trust-map.md"  # optional context

COST_COMPARISON_FP = OUTPUT_DIR / "day-16-cost-comparison.csv"
RISK_SEGMENTS_FP = OUTPUT_DIR / "day-16-risk-segments.csv"
PRIORITIZATION_FP = OUTPUT_DIR / "day-16-prioritization-framework.md"
FINAL_RECOMMENDATION_FP = OUTPUT_DIR / "day-16-final-recommendation.md"

# Cost assumptions (hypothetical, configurable)
# underforecast usually more expensive than overforecast
STOCKOUT_COST_PER_UNIT = 3.0
HOLDING_COST_PER_UNIT = 1.0

# Optional sensitivity scenarios: (stockout, holding)
SCENARIOS = [
    ("base_3_to_1", 3.0, 1.0),
    ("conservative_2_to_1", 2.0, 1.0),
    ("aggressive_5_to_1", 5.0, 1.0),
]

EPS = 1e-8
TOP_RISK_SEGMENTS = 15


# ----------------------------
# Helpers
# ----------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rmsse_from_frame(df, actual_col="sales", pred_col="yhat", scale_col="scale_denom"):
    err2 = (df[actual_col] - df[pred_col]) ** 2
    return np.sqrt(np.mean(err2 / (df[scale_col] + EPS)))


def compute_error_cost(df, pred_col, stockout_cost, holding_cost):
    """
    Error convention: error = actual - forecast
      > 0 => underforecast (stockout risk)
      < 0 => overforecast (excess/holding risk)
    """
    err = df["sales"] - df[pred_col]
    under = np.where(err > 0, err, 0.0)
    over = np.where(err < 0, -err, 0.0)
    cost = under * stockout_cost + over * holding_cost
    return err, under, over, cost


# ----------------------------
# Load
# ----------------------------
print("Loading Day 14 forecasts...")
df = pd.read_parquet(ML_FORECASTS_FP)
df["date"] = pd.to_datetime(df["date"])

required_cols = [
    "id", "date", "sales", "yhat_ml", "yhat_baseline", "scale_denom",
    "state_id", "store_id", "cat_id", "dept_id", "volume_segment", "intermittency_segment"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in day-14-ml-forecasts.parquet: {missing}")

# Only rows with both preds
df = df.dropna(subset=["yhat_ml", "yhat_baseline"]).copy()

# ----------------------------
# 1) Separate under/over + 2) Cost simulation
# ----------------------------
# Base scenario columns for detailed downstream analysis
err_ml, under_ml, over_ml, cost_ml = compute_error_cost(
    df, "yhat_ml", STOCKOUT_COST_PER_UNIT, HOLDING_COST_PER_UNIT
)
err_bl, under_bl, over_bl, cost_bl = compute_error_cost(
    df, "yhat_baseline", STOCKOUT_COST_PER_UNIT, HOLDING_COST_PER_UNIT
)

df["err_ml"] = err_ml
df["under_ml"] = under_ml
df["over_ml"] = over_ml
df["cost_ml"] = cost_ml

df["err_baseline"] = err_bl
df["under_baseline"] = under_bl
df["over_baseline"] = over_bl
df["cost_baseline"] = cost_bl

# ----------------------------
# 3) Accuracy-vs-cost comparison
# ----------------------------
rows = []

def summarize_model(df_, model_name, pred_col, err_col, under_col, over_col, cost_col):
    return {
        "model": model_name,
        "n_obs": len(df_),
        "MAE": mae(df_["sales"].values, df_[pred_col].values),
        "RMSE": rmse(df_["sales"].values, df_[pred_col].values),
        "RMSSE": rmsse_from_frame(df_, "sales", pred_col, "scale_denom"),
        "mean_error": np.mean(df_[err_col].values),  # >0 underforecast bias
        "under_total_units": df_[under_col].sum(),
        "over_total_units": df_[over_col].sum(),
        "total_cost": df_[cost_col].sum(),
        "avg_cost_per_prediction": df_[cost_col].mean(),
    }

# base scenario comparison
rows.append(summarize_model(df, "baseline", "yhat_baseline", "err_baseline", "under_baseline", "over_baseline", "cost_baseline"))
rows.append(summarize_model(df, "ml", "yhat_ml", "err_ml", "under_ml", "over_ml", "cost_ml"))

base_cmp = pd.DataFrame(rows)

# sensitivity scenarios (cost only; accuracy unchanged)
sens_rows = []
for scen_name, stock_c, hold_c in SCENARIOS:
    _, _, _, c_ml = compute_error_cost(df, "yhat_ml", stock_c, hold_c)
    _, _, _, c_bl = compute_error_cost(df, "yhat_baseline", stock_c, hold_c)

    sens_rows.append({
        "scenario": scen_name,
        "stockout_cost_per_unit": stock_c,
        "holding_cost_per_unit": hold_c,
        "model": "baseline",
        "total_cost": float(np.sum(c_bl)),
        "avg_cost_per_prediction": float(np.mean(c_bl)),
    })
    sens_rows.append({
        "scenario": scen_name,
        "stockout_cost_per_unit": stock_c,
        "holding_cost_per_unit": hold_c,
        "model": "ml",
        "total_cost": float(np.sum(c_ml)),
        "avg_cost_per_prediction": float(np.mean(c_ml)),
    })

sens_cmp = pd.DataFrame(sens_rows)

# Merge into one export-friendly table
# We'll mark table_type so one csv can hold both base metrics and sensitivity.
base_cmp_out = base_cmp.copy()
base_cmp_out["table_type"] = "base_accuracy_and_cost"
base_cmp_out["scenario"] = f"base_{int(STOCKOUT_COST_PER_UNIT)}_to_{int(HOLDING_COST_PER_UNIT)}"

sens_cmp_out = sens_cmp.copy()
sens_cmp_out["table_type"] = "cost_sensitivity"

cost_comparison = pd.concat([base_cmp_out, sens_cmp_out], ignore_index=True, sort=False)
cost_comparison.to_csv(COST_COMPARISON_FP, index=False)
print(f"Saved: {COST_COMPARISON_FP}")

# ----------------------------
# 4) High-risk segments
# ----------------------------
# Define risk as total simulated cost (base scenario)
seg_cols = ["volume_segment", "intermittency_segment", "cat_id", "store_id"]

risk = (
    df.groupby(seg_cols, dropna=False)
    .agg(
        n_obs=("id", "size"),
        n_series=("id", "nunique"),
        sales_total=("sales", "sum"),

        baseline_cost_total=("cost_baseline", "sum"),
        ml_cost_total=("cost_ml", "sum"),

        baseline_under_units=("under_baseline", "sum"),
        baseline_over_units=("over_baseline", "sum"),
        ml_under_units=("under_ml", "sum"),
        ml_over_units=("over_ml", "sum"),
    )
    .reset_index()
)

risk["cost_saving_ml_vs_baseline"] = risk["baseline_cost_total"] - risk["ml_cost_total"]  # + means ML cheaper
risk["risk_rank_cost_baseline"] = risk["baseline_cost_total"].rank(method="dense", ascending=False)
risk["risk_rank_cost_ml"] = risk["ml_cost_total"].rank(method="dense", ascending=False)

# Priority score idea: normalize volume + baseline cost + abs error proxy
risk["baseline_cost_per_obs"] = risk["baseline_cost_total"] / (risk["n_obs"] + EPS)
risk["ml_cost_per_obs"] = risk["ml_cost_total"] / (risk["n_obs"] + EPS)

# sort by highest baseline cost (operational risk before intervention)
risk_sorted = risk.sort_values("baseline_cost_total", ascending=False).reset_index(drop=True)
risk_top = risk_sorted.head(TOP_RISK_SEGMENTS).copy()
risk_top.to_csv(RISK_SEGMENTS_FP, index=False)
print(f"Saved: {RISK_SEGMENTS_FP}")

# ----------------------------
# 5) Prioritization framework
# ----------------------------
# Build a compact matrix on vol/intermittency using baseline risk and ML gain
matrix = (
    df.groupby(["volume_segment", "intermittency_segment"], dropna=False)
    .agg(
        n_obs=("id", "size"),
        n_series=("id", "nunique"),
        sales_total=("sales", "sum"),
        baseline_cost_total=("cost_baseline", "sum"),
        ml_cost_total=("cost_ml", "sum"),
    )
    .reset_index()
)
matrix["cost_saving_ml_vs_baseline"] = matrix["baseline_cost_total"] - matrix["ml_cost_total"]

# Priority rule:
# - High priority: high volume + high baseline cost, especially where ML does NOT already solve it.
# - Medium: moderate risk or ML already offers decent savings.
# - Low: low risk and low volume.
# We'll use quantiles for robustness.
cost_q_hi = matrix["baseline_cost_total"].quantile(0.67)
cost_q_lo = matrix["baseline_cost_total"].quantile(0.33)
sales_q_hi = matrix["sales_total"].quantile(0.67)
sales_q_lo = matrix["sales_total"].quantile(0.33)

def assign_priority(row):
    high_risk = (row["baseline_cost_total"] >= cost_q_hi) or (row["sales_total"] >= sales_q_hi)
    low_risk = (row["baseline_cost_total"] <= cost_q_lo) and (row["sales_total"] <= sales_q_lo)

    ml_not_helping = row["cost_saving_ml_vs_baseline"] <= 0  # ML not cheaper
    ml_helping = row["cost_saving_ml_vs_baseline"] > 0

    if high_risk and ml_not_helping:
        return "P1 - Immediate attention (high risk, ML not yet solving)"
    elif high_risk and ml_helping:
        return "P2 - Maintain ML focus (high risk, ML helping)"
    elif low_risk:
        return "P4 - Low monitoring priority"
    else:
        return "P3 - Monitor and iterate"

matrix["priority_bucket"] = matrix.apply(assign_priority, axis=1)

# ----------------------------
# 6) Final recommendation docs
# ----------------------------
# Winner by base scenario cost
base_cost_view = base_cmp.set_index("model")
ml_cost = base_cost_view.loc["ml", "total_cost"]
bl_cost = base_cost_view.loc["baseline", "total_cost"]
cost_delta = bl_cost - ml_cost
cost_winner = "ML" if cost_delta > 0 else "Baseline"

# Winner by RMSSE
ml_rmsse = base_cost_view.loc["ml", "RMSSE"]
bl_rmsse = base_cost_view.loc["baseline", "RMSSE"]
acc_winner = "ML" if ml_rmsse < bl_rmsse else "Baseline"

# Sensitivity robustness
sens_pivot = sens_cmp.pivot_table(
    index=["scenario", "stockout_cost_per_unit", "holding_cost_per_unit"],
    columns="model",
    values="total_cost",
    aggfunc="first"
).reset_index()
sens_pivot["winner"] = np.where(sens_pivot["ml"] < sens_pivot["baseline"], "ML", "Baseline")
ml_wins_sens = (sens_pivot["winner"] == "ML").sum()
n_sens = len(sens_pivot)

# Prioritization markdown
lines = []
lines.append("# Day 16 Prioritization Framework")
lines.append("")
lines.append("## Scenario assumptions (hypothetical)")
lines.append(f"- Stockout cost per unit: {STOCKOUT_COST_PER_UNIT}")
lines.append(f"- Holding cost per unit: {HOLDING_COST_PER_UNIT}")
lines.append("- These are illustrative assumptions, not real retailer costs.")
lines.append("")
lines.append("## Priority logic")
lines.append("1. Estimate operational risk via simulated cost (under/over forecast).")
lines.append("2. Prioritize high-volume or high-cost segments first.")
lines.append("3. In high-risk segments where ML does not beat baseline on cost, prioritize model improvement.")
lines.append("4. In low-risk segments, use lightweight monitoring.")
lines.append("")
lines.append("## Segment matrix")
for _, r in matrix.sort_values(["priority_bucket", "baseline_cost_total"], ascending=[True, False]).iterrows():
    lines.append(
        f"- vol={r['volume_segment']}, intermit={r['intermittency_segment']}: "
        f"priority={r['priority_bucket']}, baseline_cost={r['baseline_cost_total']:.2f}, "
        f"ml_cost={r['ml_cost_total']:.2f}, saving={r['cost_saving_ml_vs_baseline']:.2f}"
    )

with open(PRIORITIZATION_FP, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved: {PRIORITIZATION_FP}")

# Final recommendation markdown
rec = []
rec.append("# Day 16 Final Recommendation")
rec.append("")
rec.append("## Executive summary")
rec.append(f"- Accuracy winner (RMSSE): **{acc_winner}**")
rec.append(f"- Cost winner (base scenario): **{cost_winner}**")
rec.append(f"- Cost difference (baseline - ML): **{cost_delta:.2f}** (positive means ML cheaper)")
rec.append("")
rec.append("## Key observations")
rec.append(f"- ML RMSSE: {ml_rmsse:.5f}")
rec.append(f"- Baseline RMSSE: {bl_rmsse:.5f}")
rec.append(f"- ML total cost: {ml_cost:.2f}")
rec.append(f"- Baseline total cost: {bl_cost:.2f}")
rec.append(f"- ML wins in {ml_wins_sens}/{n_sens} cost sensitivity scenarios.")
rec.append("")
rec.append("## Operational guidance")
rec.append("1. Use ML in segments where both accuracy and cost improve.")
rec.append("2. Keep baseline fallback for segments where ML cost is not lower.")
rec.append("3. Prioritize high-risk segments from `day-16-risk-segments.csv` for targeted iteration.")
rec.append("4. Track underforecast-heavy segments closely due to stockout risk.")
rec.append("")
rec.append("## What to improve in a second iteration")
rec.append("- Segment-specific modeling strategy (e.g., separate handling for highly intermittent demand).")
rec.append("- Better price/event features and interactions.")
rec.append("- Evaluate direct multi-horizon or ensemble approaches inspired by top M5 solutions.")
rec.append("- Consider hierarchical reconciliation for consistency across store/category totals.")

with open(FINAL_RECOMMENDATION_FP, "w", encoding="utf-8") as f:
    f.write("\n".join(rec))
print(f"Saved: {FINAL_RECOMMENDATION_FP}")

print("\nDay 16 completed.")