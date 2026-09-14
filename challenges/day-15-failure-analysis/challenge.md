# Challenge 04 — Forecast Evaluation & Failure Analysis

## Scenario

The forecasting model performs well on average.

However, an average metric does not tell the business where the model can actually be trusted.

Your task is to move beyond a single performance score and investigate the model's failure modes.

---

## Business Question

**Where does our forecasting system work, where does it fail, and why?**

---

## Tasks

### 1. Evaluate across the retail hierarchy

Calculate forecast performance across:

- state
- store
- category
- department

Determine whether improvements are consistent throughout the business.

---

### 2. Evaluate by demand volume

Divide item-store series into groups such as:

- low volume
- medium volume
- high volume

Compare forecast performance.

---

### 3. Evaluate by intermittency

Group series according to their zero-sales rate.

Determine how forecast accuracy changes as demand becomes increasingly sparse.

---

### 4. Analyze forecast bias

Investigate whether the model systematically:

- overforecasts
- underforecasts

Determine whether bias differs across products or stores.

---

### 5. Investigate large errors

Identify the largest forecast failures.

For selected examples, investigate possible explanations such as:

- sudden demand spikes
- events
- changing prices
- intermittent demand
- structural changes
- unusual historical patterns

---

### 6. Compare ML improvements against the baseline

Calculate the improvement of the ML model relative to the strongest baseline.

Identify:

- where ML adds substantial value
- where improvements are marginal
- where the baseline still wins

---

## Deliverables

Produce:

1. Performance by hierarchy level.
2. Performance by demand volume.
3. Performance by intermittency.
4. Forecast-bias analysis.
5. Several case studies of major forecasting failures.
6. Baseline-vs-ML improvement analysis.
7. A written assessment of where the forecasting system should and should not be trusted.

---

## Final Questions

Answer:

1. Is model performance consistent across the business?
2. What characteristics make a series difficult to forecast?
3. Does ML help intermittent products?
4. Where does the baseline outperform ML?
5. Is the model systematically biased?
6. Which failures would matter most operationally?