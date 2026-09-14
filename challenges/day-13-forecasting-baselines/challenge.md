# Challenge 02 — Forecasting Baselines

## Scenario

The business now understands the major demand patterns in its historical sales data.

Before introducing machine learning, the forecasting team needs a set of simple benchmarks.

A complex model is only valuable if it can outperform a reasonable simple forecasting strategy.

---

## Business Question

**How difficult is it to beat simple retail forecasting rules?**

---

## Tasks

### 1. Create a temporal validation setup

Create a 28-day validation period representing the forecasting horizon.

All model inputs must come from information available before the beginning of the validation period.

Do not use random train/test splitting.

Explain why random splitting would be inappropriate for this problem.

---

### 2. Implement naive forecasting

Create a simple naive forecast using previous observed demand.

For example:

`forecast(t) = sales(t - 1)`

Evaluate its performance.

---

### 3. Implement seasonal naive forecasting

Create forecasts based on historical weekly patterns.

For example:

`forecast(t) = sales(t - 7)`

Consider additional seasonal lags such as:

- 14 days
- 28 days

---

### 4. Implement moving-average forecasting

Experiment with historical averages such as:

- 7-day mean
- 28-day mean

Ensure that forecasts use only historical information.

---

### 5. Evaluate the baselines

Compare the forecasting approaches using appropriate error metrics.

At minimum investigate:

- MAE
- RMSE
- RMSSE

Explain why scale-dependent metrics can be misleading when comparing thousands of products with very different sales volumes.

---

### 6. Analyze where each baseline fails

Break forecast performance down by:

- state
- store
- category
- department
- sales volume
- intermittency

Determine whether the best overall baseline is also best for every type of series.

---

## Deliverables

Produce:

1. A clearly documented temporal validation design.
2. At least three simple forecasting baselines.
3. A model comparison table.
4. Forecast-vs-actual plots for representative series.
5. Error analysis across different types of demand.
6. A written recommendation for the baseline that future ML models should be required to beat.

---

## Final Questions

Answer:

1. Which simple baseline performs best overall?
2. Which types of demand are easiest to forecast?
3. Which types are hardest?
4. Does weekly seasonality provide meaningful predictive value?
5. Where do simple forecasting rules fail?
6. What minimum performance should an ML model beat?

---

## Constraint

Do not optimize complex machine-learning models.

The purpose of this challenge is to establish a credible benchmark.