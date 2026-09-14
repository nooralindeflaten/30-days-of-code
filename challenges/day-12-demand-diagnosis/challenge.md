# Challenge 01 — Retail Demand Diagnosis

## Scenario

You are working as a Data Scientist for a large retailer with thousands of products sold across multiple stores.

The forecasting team wants to improve its demand forecasts, but before selecting a model, the business needs to understand what makes its demand predictable — and what makes it difficult.

Your task is to investigate the historical sales data and identify the major patterns that should influence the forecasting strategy.

---

## Business Question

**What patterns in retail demand are likely to make forecasting difficult, and where does demand behave differently across products, stores and time?**

---

## Dataset

Use the **M5 Forecasting — Accuracy** dataset.

For this challenge, focus on:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

Do not use the future evaluation period when drawing conclusions about historical demand.

The raw competition data should not be committed to the repository.

---

## Tasks

### 1. Understand the forecasting hierarchy

Explore the structure of the dataset.

Determine:

- number of products
- number of departments
- number of categories
- number of stores
- number of states
- number of item-store time series
- length of the historical period

Explain why forecasting an individual item-store combination is different from forecasting total company sales.

---

### 2. Analyze overall demand

Investigate total unit sales over time.

Look for:

- long-term trends
- volatility
- structural changes
- recurring patterns

Compare daily sales with smoothed 7-day and 28-day trends.

---

### 3. Investigate seasonality

Analyze demand by:

- weekday
- month

Determine whether clear weekly or annual patterns exist.

Consider what these patterns imply for future lag and calendar features.

---

### 4. Compare the retail hierarchy

Analyze demand across:

- states
- stores
- categories
- departments

Identify groups with noticeably different demand levels or trends.

Ask:

> Would one forecasting strategy be equally appropriate for every part of the business?

---

### 5. Investigate intermittent demand

For every item-store series, calculate the proportion of historical days with zero sales.

Analyze the distribution of this metric.

Compare intermittency across categories and departments.

Identify examples of:

- high-volume demand
- regular demand
- highly intermittent demand

---

### 6. Analyze sales concentration

Determine whether sales are evenly distributed across item-store series.

Calculate cumulative sales contribution and investigate whether a relatively small proportion of series accounts for a large proportion of demand.

Consider why this matters when evaluating forecast errors.

---

### 7. Explore external signals

Investigate whether demand appears related to:

- calendar events
- event types
- SNAP days
- price variation

Do not interpret observational differences as causal effects.

The goal is to determine whether these variables may contain useful predictive information.

---

## Deliverables

Produce:

1. A short data-quality and dataset overview.
2. A visualization of total demand over time.
3. A visualization demonstrating weekly or monthly seasonality.
4. A comparison across at least one level of the retail hierarchy.
5. An intermittency analysis.
6. A sales-concentration analysis.
7. A short investigation of events, SNAP or prices.
8. A written summary of the main forecasting challenges discovered.

---

## Final Questions

Conclude the analysis by answering:

1. Is demand stable over time?
2. Is there meaningful seasonality?
3. Does demand differ geographically?
4. Does demand differ across product groups?
5. How severe is intermittent demand?
6. Is sales volume concentrated among a small number of series?
7. Which external variables appear potentially useful?
8. What should these findings mean for the forecasting approach?

---

## Constraint

Do **not** build a forecasting model yet.

The purpose of this challenge is to understand the problem before choosing a solution.