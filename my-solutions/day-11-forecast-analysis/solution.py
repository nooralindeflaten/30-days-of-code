import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FORECAST_DAYS = 28

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ----------------------------
# Config
# ----------------------------
RAW_DIR = Path("data/m5-forecasting-accuracy/")
PROCESSED_DIR = RAW_DIR / "processed/"
OUTPUT_DIR = RAW_DIR / "day-11/"

SALES_FP = RAW_DIR / "sales_train_validation.csv"
CAL_FP = RAW_DIR / "calendar.csv"
PRICES_FP = RAW_DIR / "sell_prices.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional scope reduction for faster iteration
USE_SUBSET = False
SUBSET_STATE_IDS = ["CA"]  # e.g. ["CA"]
SUBSET_STORE_IDS = None    # e.g. ["CA_1", "CA_2"]


# ----------------------------
# Utilities
# ----------------------------
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def safe_cat(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


# ----------------------------
# Load data
# ----------------------------
print("Loading raw files...")
sales = pd.read_csv(SALES_FP)
calendar = pd.read_csv(CAL_FP)
prices = pd.read_csv(PRICES_FP)

# Parse date
calendar["date"] = pd.to_datetime(calendar["date"])

# Optional subset for faster day challenge
if USE_SUBSET:
    if SUBSET_STATE_IDS is not None:
        sales = sales[sales["state_id"].isin(SUBSET_STATE_IDS)].copy()
    if SUBSET_STORE_IDS is not None:
        sales = sales[sales["store_id"].isin(SUBSET_STORE_IDS)].copy()

# ----------------------------
# Basic hierarchy/data quality
# ----------------------------
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
d_cols = [c for c in sales.columns if c.startswith("d_")]

overview = {
    "n_products": sales["item_id"].nunique(),
    "n_departments": sales["dept_id"].nunique(),
    "n_categories": sales["cat_id"].nunique(),
    "n_stores": sales["store_id"].nunique(),
    "n_states": sales["state_id"].nunique(),
    "n_item_store_series": sales["id"].nunique(),
    "history_days": len(d_cols),
    "start_date": calendar.loc[calendar["d"] == d_cols[0], "date"].iloc[0],
    "end_date": calendar.loc[calendar["d"] == d_cols[-1], "date"].iloc[0],
}

print("\nDataset overview:")
for k, v in overview.items():
    print(f"{k}: {v}")

# Missingness quick check
missing_summary = pd.DataFrame({
    "table": ["sales", "calendar", "prices"],
    "missing_cells": [
        int(sales.isna().sum().sum()),
        int(calendar.isna().sum().sum()),
        int(prices.isna().sum().sum()),
    ]
})

# ----------------------------
# Melt sales (wide -> long)
# ----------------------------
print("\nTransforming sales to long format...")
sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=d_cols,
    var_name="d",
    value_name="sales"
)

sales_long = reduce_mem_usage(sales_long)

# Merge calendar
sales_long = sales_long.merge(
    calendar[[
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "snap_CA", "snap_TX", "snap_WI", "event_name_1", "event_type_1",
        "event_name_2", "event_type_2"
    ]],
    on="d",
    how="left"
)

# Merge prices (price available by item/store/week)
sales_long = sales_long.merge(
    prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

# Fill missing price forward/backward within series if needed
sales_long["sell_price"] = sales_long.groupby(["store_id", "item_id"])["sell_price"].transform(
    lambda s: s.ffill().bfill()
)

sales_long = reduce_mem_usage(sales_long)

# Save processed long table
sales_long_fp = PROCESSED_DIR / "sales_long.parquet"
sales_long.to_parquet(sales_long_fp, index=False)
print(f"Saved: {sales_long_fp}")

# ----------------------------
# Aggregate demand behavior
# ----------------------------
daily_total = (
    sales_long.groupby("date", as_index=False)["sales"].sum()
    .sort_values("date")
)
daily_total["ma_7"] = daily_total["sales"].rolling(7, min_periods=1).mean()
daily_total["ma_28"] = daily_total["sales"].rolling(28, min_periods=1).mean()

plt.figure(figsize=(14, 6))
plt.plot(daily_total["date"], daily_total["sales"], alpha=0.35, label="Daily sales")
plt.plot(daily_total["date"], daily_total["ma_7"], linewidth=2, label="7-day MA")
plt.plot(daily_total["date"], daily_total["ma_28"], linewidth=2, label="28-day MA")
plt.title("Total Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Units")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_demand_trend.png", dpi=140)
plt.close()

# ----------------------------
# Seasonality views
# ----------------------------
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# calendar has weekday names, ensure consistent labels
weekday_map = {
    "Sat": "Saturday", "Sun": "Sunday", "Mon": "Monday", "Tue": "Tuesday",
    "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"
}

tmp = sales_long[["weekday", "month", "sales"]].copy()
tmp["weekday"] = tmp["weekday"].replace(weekday_map)

weekday_season = tmp.groupby("weekday", as_index=False)["sales"].mean()
weekday_season["weekday"] = pd.Categorical(weekday_season["weekday"], categories=weekday_order, ordered=True)
weekday_season = weekday_season.sort_values("weekday")

month_season = tmp.groupby("month", as_index=False)["sales"].mean().sort_values("month")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=weekday_season, x="weekday", y="sales", ax=axes[0], color="#4c78a8")
axes[0].set_title("Average Sales by Weekday")
axes[0].tick_params(axis="x", rotation=25)

sns.lineplot(data=month_season, x="month", y="sales", marker="o", ax=axes[1], color="#f58518")
axes[1].set_title("Average Sales by Month")
axes[1].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "seasonality_weekday_month.png", dpi=140)
plt.close()

# ----------------------------
# Series profile (volume + intermittency)
# ----------------------------
series_profile = (
    sales_long.groupby(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"], as_index=False)
    .agg(
        total_sales=("sales", "sum"),
        avg_daily_sales=("sales", "mean"),
        zero_days=("sales", lambda s: (s == 0).sum()),
        n_days=("sales", "size")
    )
)

series_profile["zero_rate"] = series_profile["zero_days"] / series_profile["n_days"]

# Volume segments: tertiles
v_q1, v_q2 = series_profile["total_sales"].quantile([0.33, 0.66]).values
series_profile["volume_segment"] = safe_cat(
    series_profile["total_sales"],
    bins=[-np.inf, v_q1, v_q2, np.inf],
    labels=["low", "medium", "high"]
)

# Intermittency segments: fixed intuitive thresholds
series_profile["intermittency_segment"] = safe_cat(
    series_profile["zero_rate"],
    bins=[-0.01, 0.2, 0.5, 1.0],
    labels=["low", "medium", "high"]
)

# Save profile
series_profile_fp = PROCESSED_DIR / "series_profile.csv"
series_profile.to_csv(series_profile_fp, index=False)
print(f"Saved: {series_profile_fp}")

# ----------------------------
# Sales concentration
# ----------------------------
contrib = series_profile[["id", "total_sales"]].sort_values("total_sales", ascending=False).copy()
contrib["cum_sales_share"] = contrib["total_sales"].cumsum() / contrib["total_sales"].sum()
contrib["series_rank_pct"] = (np.arange(1, len(contrib) + 1)) / len(contrib)

plt.figure(figsize=(8, 6))
plt.plot(contrib["series_rank_pct"], contrib["cum_sales_share"], linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
plt.title("Sales Concentration Curve")
plt.xlabel("Cumulative share of series")
plt.ylabel("Cumulative share of sales")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sales_concentration_curve.png", dpi=140)
plt.close()

# ----------------------------
# External signal checks (associational only)
# ----------------------------
# Event day vs non-event day
event_flag = sales_long["event_name_1"].notna() | sales_long["event_name_2"].notna()
event_summary = pd.DataFrame({
    "group": ["event_day", "non_event_day"],
    "avg_sales": [
        sales_long.loc[event_flag, "sales"].mean(),
        sales_long.loc[~event_flag, "sales"].mean()
    ]
})

# SNAP effect by state (matching relevant SNAP column)
snap_col_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
snap_rows = []
for state, snap_col in snap_col_map.items():
    st = sales_long[sales_long["state_id"] == state]
    if len(st) == 0:
        continue
    snap_rows.append({
        "state_id": state,
        "snap_day_avg_sales": st.loc[st[snap_col] == 1, "sales"].mean(),
        "non_snap_day_avg_sales": st.loc[st[snap_col] == 0, "sales"].mean(),
    })
snap_summary = pd.DataFrame(snap_rows)

# Price-sales association (coarse, per-series correlation then median)
tmp_price = sales_long.dropna(subset=["sell_price"]).copy()
series_corr = (
    tmp_price.groupby("id")[["sales", "sell_price"]]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
    .rename(columns={"sell_price": "sales_price_corr"})
)
median_sales_price_corr = series_corr["sales_price_corr"].median()

# Save external checks
event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)
snap_summary.to_csv(OUTPUT_DIR / "snap_summary.csv", index=False)
series_corr.to_csv(OUTPUT_DIR / "series_sales_price_corr.csv", index=False)

# ----------------------------
# Write short markdown summary
# ----------------------------
top20_share = contrib.iloc[int(len(contrib) * 0.2) - 1]["cum_sales_share"] if len(contrib) > 10 else np.nan
high_intermit_share = (series_profile["intermittency_segment"] == "high").mean()

summary_md = f"""# Day 12 Demand Diagnosis Summary

## Dataset overview
- Products: {overview['n_products']}
- Departments: {overview['n_departments']}
- Categories: {overview['n_categories']}
- Stores: {overview['n_stores']}
- States: {overview['n_states']}
- Item-store series: {overview['n_item_store_series']}
- Historical days: {overview['history_days']} ({overview['start_date'].date()} to {overview['end_date'].date()})

## Data quality
- Missing cells (sales/calendar/prices): {missing_summary['missing_cells'].tolist()}

## Key findings
- Demand trend and volatility are shown in `total_demand_trend.png`.
- Weekly/monthly seasonality is shown in `seasonality_weekday_month.png`.
- Sales concentration is shown in `sales_concentration_curve.png`.
- Share of highly intermittent series: {high_intermit_share:.3f}

## External signal checks (associational)
- Event vs non-event average sales: see `event_summary.csv`
- SNAP day comparison by state: see `snap_summary.csv`
- Median within-series sales/price correlation: {median_sales_price_corr:.3f}

## Forecasting implications for Day 13
1. Use temporal holdout (28 days), not random split.
2. Include seasonal baseline (t-7) because weekday structure is visible.
3. Evaluate by volume + intermittency segments from `series_profile.csv`.
4. Prefer scale-aware metrics (e.g., RMSSE) due to concentration and volume imbalance.
"""

summary_fp = OUTPUT_DIR / "day-11-demand-diagnosis-summary.md"
with open(summary_fp, "w", encoding="utf-8") as f:
    f.write(summary_md)

print(f"Saved: {summary_fp}")
print("\nDay 11 completed.")