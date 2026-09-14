# Challenge 03 — Global Machine Learning Forecasting

## Scenario

Simple forecasting baselines have established the minimum level of performance required.

The forecasting team now wants to know whether a machine-learning model can learn demand patterns across products and stores.

Rather than training thousands of independent models, investigate whether a global model can learn across the retail network.

---

## Business Question

**Can shared patterns across products, stores and calendar conditions improve demand forecasts beyond simple historical rules?**

---

## Tasks

### 1. Create a model-ready dataset

Transform the required sales history into a long-format dataset suitable for machine learning.

Include identifiers such as:

- item
- department
- category
- store
- state

Be mindful of memory usage.

---

### 2. Engineer lag features

Create historical demand features such as:

- lag 1
- lag 7
- lag 14
- lag 28
- lag 56

Every feature must contain only information that would have been available at prediction time.

---

### 3. Engineer rolling features

Investigate features such as:

- 7-day rolling mean
- 28-day rolling mean
- rolling standard deviation
- recent maximum/minimum demand

Use appropriate shifting to prevent target leakage.

---

### 4. Add calendar features

Consider:

- day of week
- month
- year
- weekend
- events
- event types
- SNAP

---

### 5. Add price features

Investigate features such as:

- current sell price
- price change
- relative price
- price compared with recent historical price

---

### 6. Train a global model

Train a gradient-boosting model such as LightGBM.

Compare it against the strongest baseline from Challenge 02.

Do not assume that increased model complexity automatically produces better forecasts.

---

### 7. Analyze feature importance

Investigate which variables contribute most to predictions.

Ask whether the model appears to rely primarily on:

- recent demand
- weekly seasonality
- product identity
- store identity
- calendar conditions
- price

---

## Deliverables

Produce:

1. A documented feature-engineering pipeline.
2. A leakage-safe training dataset.
3. A global ML forecasting model.
4. Comparison against the strongest simple baseline.
5. Feature-importance analysis.
6. Example forecasts for several different types of series.
7. A discussion of where ML provides the largest improvement.

---

## Final Questions

Answer:

1. Does the ML model outperform the baseline?
2. By how much?
3. Which features matter most?
4. Which products benefit most from ML?
5. Which products still remain difficult?
6. Is the additional model complexity justified?