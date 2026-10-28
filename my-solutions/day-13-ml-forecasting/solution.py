import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = Path("data/m5-forecasting-accuracy/day-13-ml-global/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/m5-forecasting-accuracy/processed/")
SALES_LONG_FP = PROCESSED_DIR / "sales_long.parquet"
SERIES_PROFILE_FP = PROCESSED_DIR / "series_profile.csv"
BASELINE_COMP_FP = Path("data/m5-forecasting-accuracy/day-12-baseline-challenge/day-12-baseline-comparison.csv")

ML_FORECASTS_FP = OUTPUT_DIR / "day-13-ml-forecasts.parquet"
ML_VS_BASELINE_FP = OUTPUT_DIR / "day-13-ml-vs-baseline.csv"
FEATURE_IMPORTANCE_FP = OUTPUT_DIR / "day-13-feature-importance.csv"
ML_SUMMARY_FP = OUTPUT_DIR / "day-13-ml-summary.md"

HORIZON = 28
EPS = 1e-8

# choose one baseline name from Day 13 output
# expected: naive_1, seasonal_naive_7, moving_avg_7, moving_avg_28
DEFAULT_BASELINE_NAME = "seasonal_naive_7"


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


# ----------------------------
# Load
# ----------------------------
print("Loading inputs...")
df = pd.read_parquet(SALES_LONG_FP)
profile = pd.read_csv(SERIES_PROFILE_FP)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# Optional: reduce scope for faster iteration (comment out for full run)
# keep_states = ["CA"]
# df = df[df["state_id"].isin(keep_states)].copy()

max_date = df["date"].max()
val_start = max_date - pd.Timedelta(days=HORIZON - 1)

print(f"Validation window: {val_start.date()} -> {max_date.date()}")

# ----------------------------
# Create leakage-safe features
# ----------------------------
# Lags
for l in [1, 7, 28]:
    df[f"lag_{l}"] = df.groupby("id")["sales"].shift(l)

# Rolling means (shift first!)
df["rmean_7"] = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
df["rmean_28"] = df.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())

# Calendar features
df["dow"] = df["date"].dt.dayofweek
df["month_num"] = df["date"].dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)

# Ensure SNAP/event columns exist (if missing from your day 12 table, create defaults)
for c in ["snap_CA", "snap_TX", "snap_WI", "event_type_1", "event_type_2"]:
    if c not in df.columns:
        df[c] = np.nan if "event" in c else 0

# Price features
if "sell_price" in df.columns:
    df["sell_price"] = df["sell_price"].astype("float32")
    grp = df.groupby("id")["sell_price"]
    df["price_rel_item_mean"] = df["sell_price"] / (grp.transform("mean") + EPS)
    df["price_change_1"] = grp.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
else:
    df["sell_price"] = 0.0
    df["price_rel_item_mean"] = 1.0
    df["price_change_1"] = 0.0

# ----------------------------
# Target encoding prep (simple category codes)
# ----------------------------
cat_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1", "event_type_2"]
for c in cat_cols:
    if c not in df.columns:
        df[c] = "unknown"
    df[c] = df[c].astype("category")

# ----------------------------
# Train/validation split
# ----------------------------
train_df = df[df["date"] < val_start].copy()
val_df = df[df["date"] >= val_start].copy()

# RMSSE denominator from training
train_df["diff1"] = train_df.groupby("id")["sales"].diff()
scale = (
    train_df.groupby("id", as_index=False)["diff1"]
    .apply(lambda s: np.mean(np.square(s.dropna())) if s.dropna().shape[0] > 0 else 0.0)
    .rename(columns={"diff1": "scale_denom"})
)
scale["scale_denom"] = scale["scale_denom"].fillna(0.0).clip(lower=EPS)

# Feature columns
feature_cols = [
    # lag/rolling
    "lag_1", "lag_7", "lag_28", "rmean_7", "rmean_28",
    # calendar
    "dow", "month_num", "is_weekend",
    # snap
    "snap_CA", "snap_TX", "snap_WI",
    # price
    "sell_price", "price_rel_item_mean", "price_change_1",
    # identities
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    # event types
    "event_type_1", "event_type_2",
]

# Keep rows where required lag/rolling features are available
required_feats = ["lag_1", "lag_7", "lag_28", "rmean_7", "rmean_28"]
train_model = train_df.dropna(subset=required_feats).copy()
val_model = val_df.dropna(subset=required_feats).copy()

X_train = train_model[feature_cols]
y_train = train_model["sales"].astype(float)

X_val = val_model[feature_cols]
y_val = val_model["sales"].astype(float)

# ----------------------------
# Train LightGBM
# ----------------------------
print("Training LightGBM global model...")
model = LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=[c for c in cat_cols if c in feature_cols]
)

val_model["yhat_ml"] = model.predict(X_val)
val_model["yhat_ml"] = np.clip(val_model["yhat_ml"], 0, None)

# ----------------------------
# Build baseline predictions on same validation rows
# ----------------------------
# Build from full df to guarantee same dates/ids, then merge to val_model keys
tmp = df[["id", "date", "sales"]].copy().sort_values(["id", "date"])
tmp["naive_1"] = tmp.groupby("id")["sales"].shift(1)
tmp["seasonal_naive_7"] = tmp.groupby("id")["sales"].shift(7)
tmp["moving_avg_7"] = tmp.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
tmp["moving_avg_28"] = tmp.groupby("id")["sales"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())

# choose baseline from day-13 comparison if available
baseline_name = DEFAULT_BASELINE_NAME
if BASELINE_COMP_FP.exists():
    comp = pd.read_csv(BASELINE_COMP_FP)
    if "model" in comp.columns and len(comp) > 0:
        baseline_name = comp.sort_values("RMSSE").iloc[0]["model"]

if baseline_name not in ["naive_1", "seasonal_naive_7", "moving_avg_7", "moving_avg_28"]:
    baseline_name = DEFAULT_BASELINE_NAME

print(f"Using Day 13 benchmark baseline: {baseline_name}")

val_model = val_model.merge(
    tmp[["id", "date", baseline_name]].rename(columns={baseline_name: "yhat_baseline"}),
    on=["id", "date"],
    how="left"
)

# attach scale denom and segments
val_model = val_model.merge(scale, on="id", how="left")
val_model["scale_denom"] = val_model["scale_denom"].fillna(EPS).clip(lower=EPS)

seg_cols = ["id", "volume_segment", "intermittency_segment"]
if all(c in profile.columns for c in seg_cols):
    val_model = val_model.merge(profile[seg_cols], on="id", how="left")
else:
    val_model["volume_segment"] = "unknown"
    val_model["intermittency_segment"] = "unknown"

# keep output columns
out_cols = [
    "id", "date", "sales", "yhat_ml", "yhat_baseline", "scale_denom",
    "state_id", "store_id", "cat_id", "dept_id",
    "volume_segment", "intermittency_segment"
]
val_out = val_model[out_cols].copy()
val_out.to_parquet(ML_FORECASTS_FP, index=False)
print(f"Saved: {ML_FORECASTS_FP}")

# ----------------------------
# Compare metrics: ML vs baseline
# ----------------------------
def summarize_metrics(d, pred_col, label):
    return {
        "model": label,
        "n_obs": len(d),
        "MAE": mae(d["sales"].values, d[pred_col].values),
        "RMSE": rmse(d["sales"].values, d[pred_col].values),
        "RMSSE": rmsse_from_frame(d, actual_col="sales", pred_col=pred_col, scale_col="scale_denom")
    }

rows = [
    summarize_metrics(val_out.dropna(subset=["yhat_baseline"]), "yhat_baseline", f"baseline_{baseline_name}"),
    summarize_metrics(val_out.dropna(subset=["yhat_ml"]), "yhat_ml", "ml_lightgbm_global"),
]
cmp = pd.DataFrame(rows)
cmp.to_csv(ML_VS_BASELINE_FP, index=False)
print(f"Saved: {ML_VS_BASELINE_FP}")

# ----------------------------
# Feature importance
# ----------------------------
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
fi.to_csv(FEATURE_IMPORTANCE_FP, index=False)
print(f"Saved: {FEATURE_IMPORTANCE_FP}")

# ----------------------------
# Segment-level gain check
# ----------------------------
seg = (
    val_out.dropna(subset=["yhat_baseline", "yhat_ml"])
    .groupby(["volume_segment", "intermittency_segment"], dropna=False)
    .apply(lambda g: pd.Series({
        "n_obs": len(g),
        "MAE_baseline": mae(g["sales"].values, g["yhat_baseline"].values),
        "MAE_ml": mae(g["sales"].values, g["yhat_ml"].values),
        "RMSSE_baseline": rmsse_from_frame(g, "sales", "yhat_baseline", "scale_denom"),
        "RMSSE_ml": rmsse_from_frame(g, "sales", "yhat_ml", "scale_denom"),
    }))
    .reset_index()
)
seg["RMSSE_gain_vs_baseline"] = seg["RMSSE_baseline"] - seg["RMSSE_ml"]  # positive means ML better

# ----------------------------
# Summary markdown
# ----------------------------
overall = cmp.set_index("model")
ml_row = overall.loc["ml_lightgbm_global"]
bl_row = overall.loc[f"baseline_{baseline_name}"]

impr = bl_row["RMSSE"] - ml_row["RMSSE"]
winner = "ML" if impr > 0 else "Baseline"

top_feats = fi.head(10)["feature"].tolist()

summary = f"""# Day 13 ML Summary

## Setup
- Validation horizon: {HORIZON} days
- Validation period: {val_start.date()} to {max_date.date()}
- Baseline benchmark from Day 13: `{baseline_name}`

## Overall comparison
- Baseline RMSSE: {bl_row['RMSSE']:.5f}
- ML RMSSE: {ml_row['RMSSE']:.5f}
- RMSSE improvement (baseline - ML): {impr:.5f}
- Winner: **{winner}**

## Interpretation
Top features by importance:
{chr(10).join([f"- {f}" for f in top_feats])}

## Segment insights
Positive `RMSSE_gain_vs_baseline` means ML outperforms baseline in that segment.
(Use downstream in Day 15 failure analysis.)
"""

with open(ML_SUMMARY_FP, "w", encoding="utf-8") as f:
    f.write(summary)
print(f"Saved: {ML_SUMMARY_FP}")

print("\nDay 13 completed.")