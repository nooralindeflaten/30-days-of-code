# Challenge 05 — From Forecast Accuracy to Business Decisions

## Scenario

The forecasting team has produced a model with measurable improvements over simple baselines.

But the operations team does not manage RMSE.

They manage inventory.

A forecast that is statistically better does not automatically produce a better operational decision.

Your final task is to translate forecast performance into a simplified business decision framework.

---

## Business Question

**Does a more accurate forecast lead to better inventory decisions, and where does forecast error create the greatest business risk?**

---

## Tasks

### 1. Separate underforecasting and overforecasting

For each prediction calculate:

`Forecast Error = Actual Demand - Forecast Demand`

Distinguish between:

- underforecasting
- overforecasting

Interpret their different operational consequences.

---

### 2. Create an inventory-cost scenario

Define clearly documented hypothetical costs for:

- underforecasting / stockouts
- overforecasting / excess inventory

These values are scenario assumptions and must not be presented as real Walmart costs.

For example:

`Estimated Cost = Underforecast × Stockout Cost + Overforecast × Holding Cost`

---

### 3. Compare models using operational cost

Compare:

- naive baseline
- seasonal baseline
- ML model

using both:

- forecast accuracy
- simulated operational cost

Determine whether the most accurate model is also the least costly model.

---

### 4. Identify high-risk forecasts

Investigate which combinations of:

- products
- categories
- stores
- demand levels

produce the greatest simulated operational risk.

---

### 5. Prioritize forecasting effort

Create a simple prioritization framework.

For example:

- high-volume + high-error → high priority
- low-volume + high-error → potentially lower priority
- high-volume + low-error → monitor
- low-volume + low-error → low priority

The exact framework should be justified by the analysis.

---

### 6. Compare your approach with M5 competition solutions

Only after completing your own analysis, investigate how high-ranking M5 solutions approached the forecasting problem.

Compare concepts such as:

- global vs local models
- feature engineering
- pooling
- direct vs recursive forecasting
- hierarchical forecasting
- ensembling

Focus on what you would change in a second iteration rather than simply reproducing winning code.

---

## Deliverables

Produce:

1. Underforecast vs overforecast analysis.
2. A clearly documented inventory-cost simulation.
3. Accuracy-vs-cost model comparison.
4. Identification of high-risk products or groups.
5. A forecast-prioritization framework.
6. A short comparison with high-ranking M5 approaches.
7. Final recommendations to the hypothetical retail stakeholder.

---

## Final Reflection

Conclude the five-day project by answering:

1. What did simple baselines teach you?
2. Where did machine learning create value?
3. What made demand difficult to forecast?
4. Did higher accuracy always produce better simulated business outcomes?
5. Which parts of the business deserve the most forecasting attention?
6. What would you improve with more time?
7. What did successful M5 competitors do that your approach did not?

---

## Final Principle

The goal is not simply:

**"Build the model with the lowest forecasting error."**

The goal is:

**"Use forecasting to support better decisions under uncertainty."**