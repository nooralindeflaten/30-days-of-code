from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
import pandas as pd

DATA_PATH = "data/m5-forecasting-accuracy/"
PROCESSED_DATA_PATH = DATA_PATH + "processed/"
FORECAST_DAYS = 28

sales = pd.read_parquet(
    PROCESSED_DATA_PATH + "sales.parquet"
)
calendar = pd.read_parquet(
    PROCESSED_DATA_PATH + "calendar.parquet"
)

day_cols = sorted(
    [
        col for col in sales.columns
        if col.startswith("d_")
    ],
    key=lambda x: int(x.split("_")[1])
)

train_days = day_cols[:-28]
validation_days = day_cols[-28:]

sales_df = sales.set_index("id")[day_cols]

# Work directly with the wide M5 matrix.
# This is much more memory-efficient than melting the full dataset.
train = sales_df[train_days].to_numpy(dtype=np.float32)
actual = sales_df[validation_days].to_numpy(dtype=np.float32)

n_series = train.shape[0]

print(f"Number of series: {n_series:,}")
print(f"Training days:    {train.shape[1]:,}")
print(f"Forecast horizon: {actual.shape[1]}")


# Dates for plotting
date_lookup = calendar.set_index("d")["date"]

train_dates = pd.DatetimeIndex(
    date_lookup.loc[train_days].values
)

validation_dates = pd.DatetimeIndex(
    date_lookup.loc[validation_days].values
)


# ============================================================
# WHY NOT RANDOM SPLITTING?
# ============================================================

"""
Random train/test splitting is inappropriate because observations in a
time series are ordered.

A random split could allow observations from the future to appear in
the training set while earlier observations appear in the test set.

This creates temporal leakage and does not represent the real business
problem, where future demand must be predicted using only information
available at the forecast origin.

The final 28 days are therefore held out as one chronological block.
"""


# ============================================================
# 3. BASELINE FUNCTIONS
# ============================================================


def naive_forecast(history, horizon=28):
    """
    Assume future demand remains equal to the final observed value.
    """

    last_value = history[:, -1:]

    return np.repeat(
        last_value,
        horizon,
        axis=1
    )


naive_pred = naive_forecast(
    train,
    FORECAST_DAYS
)


# ============================================================
# 4. BASELINE 2 — WEEKLY SEASONAL NAIVE
# ============================================================

def seasonal_naive_forecast(
    history,
    horizon=28,
    season_length=7
):
    """
    Repeat the final observed weekly pattern.
    """

    last_week = history[
        :,
        -season_length:
    ]

    repeats = int(
        np.ceil(
            horizon / season_length
        )
    )

    forecast = np.tile(
        last_week,
        (1, repeats)
    )

    return forecast[
        :,
        :horizon
    ]


seasonal_pred = seasonal_naive_forecast(
    train,
    FORECAST_DAYS,
    season_length=7
)


# ============================================================
# 5. BASELINE 3 — 7-DAY MOVING AVERAGE
# ============================================================

def moving_average_forecast(
    history,
    horizon=28,
    window=7
):
    """
    Use average demand over the final observed week.
    """

    recent_mean = (
        history[
            :,
            -window:
        ]
        .mean(axis=1)
        .reshape(-1, 1)
    )

    return np.repeat(
        recent_mean,
        horizon,
        axis=1
    )


moving_average_pred = moving_average_forecast(
    train,
    FORECAST_DAYS,
    window=7
)


# ============================================================
# 6. EVALUATION METRICS
# ============================================================

def mae(y_true, y_pred):
    return np.mean(
        np.abs(
            y_true - y_pred
        )
    )


def rmsse(
    y_true,
    y_pred,
    y_train
):
    """
    RMSSE is calculated independently for each item-store series.

    Scale =
    mean squared change in historical demand.

    Final score =
    mean RMSSE across valid series.
    """

    forecast_mse = np.mean(
        (y_true - y_pred) ** 2,
        axis=1
    )

    historical_changes = np.diff(
        y_train,
        axis=1
    )

    scale = np.mean(
        historical_changes ** 2,
        axis=1
    )

    # Series with no historical variation
    # cannot produce a valid RMSSE denominator.
    valid = scale > 0

    series_rmsse = np.sqrt(
        forecast_mse[valid]
        / scale[valid]
    )

    return np.mean(
        series_rmsse
    )


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

models = {
    "Naive": naive_pred,
    "Seasonal Naive": seasonal_pred,
    "7-Day Moving Average": moving_average_pred
}


results = []


for model_name, prediction in models.items():

    results.append({
        "Model": model_name,
        "MAE": mae(
            actual,
            prediction
        ),
        "RMSSE": rmsse(
            actual,
            prediction,
            train
        )
    })


comparison = (
    pd.DataFrame(results)
    .sort_values("RMSSE")
    .reset_index(drop=True)
)


print("\nBASELINE COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)