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

# Inputs
ML_FORECASTS_FP = OUTPUT_DIR / "day-14-ml-forecasts.parquet"

# Outputs
PERF_SEG_FP = OUTPUT_DIR / "day-15-performance-by-segment.csv"
BIAS_FP = OUTPUT_DIR / "day-15-bias-analysis.csv"
FAIL_CASES_FP = OUTPUT_DIR / "day-15-failure-cases.csv"
TRUST_MAP_FP = OUTPUT_DIR / "day-15-trust-map.md"

EPS = 1e-8
TOP_FAILURES = 5  # number of case studies


# ----------------------------
# Metrics
# ----------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rmsse_from_frame(df, actual_col="sales", pred_col="yhat", scale_col="scale_denom"):
    err2 = (df[actual_col] - df[pred_col]) ** 2
    return np.sqrt(np.mean(err2 / (df[scale_col] + EPS)))


def summarize_pair(g):
    """Return baseline vs ML metrics for a group g."""
    b = {
        "n_obs": len(g),
        "MAE_baseline": mae(g["sales"].values, g["yhat_baseline"].values),
        "MAE_ml": mae(g["sales"].values, g["yhat_ml"].values),
        "RMSE_baseline": rmse(g["sales"].values, g["yhat_baseline"].values),
        "RMSE_ml": rmse(g["sales"].values, g["yhat_ml"].values),
        "RMSSE_baseline": rmsse_from_frame(g, "sales", "yhat_baseline", "scale_denom"),
        "RMSSE_ml": rmsse_from_frame(g, "sales", "yhat_ml", "scale_denom"),
    }
    b["MAE_gain"] = b["MAE_baseline"] - b["MAE_ml"]       # + means ML better
    b["RMSE_gain"] = b["RMSE_baseline"] - b["RMSE_ml"]    # + means ML better
    b["RMSSE_gain"] = b["RMSSE_baseline"] - b["RMSSE_ml"] # + means ML better
    return pd.Series(b)


# ----------------------------
# Load + prep
# ----------------------------
print("Loading Day 14 forecasts...")
df = pd.read_parquet(ML_FORECASTS_FP)
df["date"] = pd.to_datetime(df["date"])

required = [
    "id", "date", "sales", "yhat_ml", "yhat_baseline", "scale_denom",
    "state_id", "store_id", "cat_id", "dept_id", "volume_segment", "intermittency_segment"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in day-14-ml-forecasts.parquet: {missing}")

# Only rows where both model predictions exist
df = df.dropna(subset=["yhat_ml", "yhat_baseline"]).copy()

# Errors: convention = Actual - Forecast
df["err_ml"] = df["sales"] - df["yhat_ml"]
df["err_baseline"] = df["sales"] - df["yhat_baseline"]

df["abs_err_ml"] = df["err_ml"].abs()
df["abs_err_baseline"] = df["err_baseline"].abs()

df["sq_err_ml"] = df["err_ml"] ** 2
df["sq_err_baseline"] = df["err_baseline"] ** 2

# Positive error means underforecast; negative means overforecast
df["under_ml"] = np.where(df["err_ml"] > 0, df["err_ml"], 0.0)
df["over_ml"] = np.where(df["err_ml"] < 0, -df["err_ml"], 0.0)
df["under_baseline"] = np.where(df["err_baseline"] > 0, df["err_baseline"], 0.0)
df["over_baseline"] = np.where(df["err_baseline"] < 0, -df["err_baseline"], 0.0)

# ----------------------------
# 1) Performance by segments
# ----------------------------
segment_frames = []

segment_specs = {
    "volume_segment": ["volume_segment"],
    "intermittency_segment": ["intermittency_segment"],
    "state": ["state_id"],
    "store": ["store_id"],
    "category": ["cat_id"],
    "department": ["dept_id"],
    "vol_x_intermit": ["volume_segment", "intermittency_segment"],
}

for slice_name, cols in segment_specs.items():
    tmp = (
        df.groupby(cols, dropna=False)
        .apply(summarize_pair)
        .reset_index()
    )
    tmp["slice_type"] = slice_name
    segment_frames.append(tmp)

perf_seg = pd.concat(segment_frames, ignore_index=True)

# Reorder columns
front = ["slice_type"] + [c for c in ["volume_segment","intermittency_segment","state_id","store_id","cat_id","dept_id"] if c in perf_seg.columns]
metric_cols = [c for c in perf_seg.columns if c not in front]
perf_seg = perf_seg[front + metric_cols]

perf_seg.to_csv(PERF_SEG_FP, index=False)
print(f"Saved: {PERF_SEG_FP}")

# ----------------------------
# 2) Bias analysis
# ----------------------------
def bias_stats(g, err_col, under_col, over_col):
    err = g[err_col].values
    under = g[under_col].sum()
    over = g[over_col].sum()
    return pd.Series({
        "n_obs": len(g),
        "mean_error": np.mean(err),  # >0 underforecast bias, <0 overforecast bias
        "median_error": np.median(err),
        "under_total": under,
        "over_total": over,
        "under_share_of_abs_error": under / (under + over + EPS),
        "over_share_of_abs_error": over / (under + over + EPS),
    })

bias_rows = []

# Overall
overall_ml = bias_stats(df, "err_ml", "under_ml", "over_ml").to_dict()
overall_ml.update({"slice_type": "overall", "slice_value": "all", "model": "ml"})
bias_rows.append(overall_ml)

overall_bl = bias_stats(df, "err_baseline", "under_baseline", "over_baseline").to_dict()
overall_bl.update({"slice_type": "overall", "slice_value": "all", "model": "baseline"})
bias_rows.append(overall_bl)

# By key segments
for seg_col in ["volume_segment", "intermittency_segment", "cat_id", "state_id"]:
    for val, g in df.groupby(seg_col, dropna=False):
        r_ml = bias_stats(g, "err_ml", "under_ml", "over_ml").to_dict()
        r_ml.update({"slice_type": seg_col, "slice_value": val, "model": "ml"})
        bias_rows.append(r_ml)

        r_bl = bias_stats(g, "err_baseline", "under_baseline", "over_baseline").to_dict()
        r_bl.update({"slice_type": seg_col, "slice_value": val, "model": "baseline"})
        bias_rows.append(r_bl)

bias_df = pd.DataFrame(bias_rows)
bias_df.to_csv(BIAS_FP, index=False)
print(f"Saved: {BIAS_FP}")

# ----------------------------
# 3) Failure case studies (largest ML misses)
# ----------------------------
# rank by absolute error on single predictions
fail = df.copy()
fail["ml_rank_key"] = fail["abs_err_ml"]

# Avoid duplicates on same series-day already unique; take top N
top_fail = fail.sort_values("ml_rank_key", ascending=False).head(TOP_FAILURES).copy()

# Add useful diagnostics
top_fail["error_type_ml"] = np.where(top_fail["err_ml"] > 0, "underforecast", "overforecast")
top_fail["error_type_baseline"] = np.where(top_fail["err_baseline"] > 0, "underforecast", "overforecast")
top_fail["ml_minus_baseline_abs_error"] = top_fail["abs_err_ml"] - top_fail["abs_err_baseline"]

# Keep concise columns
fail_cols = [
    "id", "date", "state_id", "store_id", "cat_id", "dept_id",
    "volume_segment", "intermittency_segment",
    "sales", "yhat_ml", "yhat_baseline",
    "err_ml", "abs_err_ml", "error_type_ml",
    "err_baseline", "abs_err_baseline", "error_type_baseline",
    "ml_minus_baseline_abs_error"
]
top_fail = top_fail[fail_cols].sort_values("abs_err_ml", ascending=False)
top_fail.to_csv(FAIL_CASES_FP, index=False)
print(f"Saved: {FAIL_CASES_FP}")

# ----------------------------
# 4) Build trust map markdown
# ----------------------------
overall = summarize_pair(df)
overall_rmsse_gain = overall["RMSSE_gain"]

# Segment trust by vol x intermittency
trust_grid = (
    df.groupby(["volume_segment", "intermittency_segment"], dropna=False)
    .apply(summarize_pair)
    .reset_index()
)
trust_grid["winner"] = np.where(trust_grid["RMSSE_gain"] > 0, "ML", "Baseline")
trust_grid = trust_grid.sort_values(["volume_segment", "intermittency_segment"])

# Core findings for concise narrative
ml_better_share = (trust_grid["winner"] == "ML").mean()
best_segments = trust_grid.sort_values("RMSSE_gain", ascending=False).head(3)
worst_segments = trust_grid.sort_values("RMSSE_gain", ascending=True).head(3)

# Bias summary
overall_bias_ml = bias_df[(bias_df["slice_type"] == "overall") & (bias_df["model"] == "ml")].iloc[0]
overall_bias_bl = bias_df[(bias_df["slice_type"] == "overall") & (bias_df["model"] == "baseline")].iloc[0]

md_lines = []
md_lines.append("# Day 15 Trust Map")
md_lines.append("")
md_lines.append("## Overall performance")
md_lines.append(f"- RMSSE gain (baseline - ML): **{overall_rmsse_gain:.5f}**")
md_lines.append(f"- Interpretation: {'ML better overall' if overall_rmsse_gain > 0 else 'Baseline better overall'}")
md_lines.append("")
md_lines.append("## Where ML tends to help")
for _, r in best_segments.iterrows():
    md_lines.append(
        f"- vol={r['volume_segment']}, intermit={r['intermittency_segment']}: "
        f"RMSSE gain={r['RMSSE_gain']:.5f} ({r['winner']})"
    )
md_lines.append("")
md_lines.append("## Where baseline still wins")
for _, r in worst_segments.iterrows():
    md_lines.append(
        f"- vol={r['volume_segment']}, intermit={r['intermittency_segment']}: "
        f"RMSSE gain={r['RMSSE_gain']:.5f} ({r['winner']})"
    )
md_lines.append("")
md_lines.append("## Coverage")
md_lines.append(f"- Share of vol×intermittency segments where ML wins: **{ml_better_share:.2%}**")
md_lines.append("")
md_lines.append("## Bias summary (overall)")
md_lines.append(
    f"- ML mean error: {overall_bias_ml['mean_error']:.4f} "
    f"(under-share={overall_bias_ml['under_share_of_abs_error']:.2%}, "
    f"over-share={overall_bias_ml['over_share_of_abs_error']:.2%})"
)
md_lines.append(
    f"- Baseline mean error: {overall_bias_bl['mean_error']:.4f} "
    f"(under-share={overall_bias_bl['under_share_of_abs_error']:.2%}, "
    f"over-share={overall_bias_bl['over_share_of_abs_error']:.2%})"
)
md_lines.append("")
md_lines.append("## Recommended trust policy")
md_lines.append("1. Prefer ML in segments with positive and meaningful RMSSE gain.")
md_lines.append("2. Keep baseline fallback in segments where ML repeatedly loses.")
md_lines.append("3. Monitor segments with strong underforecast bias (stockout risk).")
md_lines.append("4. Use top failure cases (`day-15-failure-cases.csv`) for root-cause review.")
md_lines.append("")
md_lines.append("## Files")
md_lines.append("- `day-15-performance-by-segment.csv`")
md_lines.append("- `day-15-bias-analysis.csv`")
md_lines.append("- `day-15-failure-cases.csv`")
md_lines.append("- `day-15-trust-map.md`")

with open(TRUST_MAP_FP, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))
print(f"Saved: {TRUST_MAP_FP}")

print("\nDay 15 completed.")