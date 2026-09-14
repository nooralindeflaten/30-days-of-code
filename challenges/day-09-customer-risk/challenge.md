# Challenge 04 — Customer Inactivity Prediction

## Scenario

Customer segmentation describes historical behaviour, but the business also wants to identify valuable customers whose behaviour may be changing.

The retention team wants an early-warning system that can identify customers who are likely to become inactive.

Your task is to formulate customer inactivity as a temporal prediction problem.

---

## Business Question

**Which currently known customers are most likely to become inactive over the next 90 days?**

---

## Tasks

### 1. Define inactivity

Create a clear operational definition.

For this project:

> A customer is considered inactive if they make no successful purchase during the following 90-day period.

Explain why this is a modeling definition rather than proof that a customer has permanently churned.

---

### 2. Create a temporal cutoff

Choose a historical cutoff date that leaves enough future data to observe the 90-day outcome.

Split the data into:

`Historical period → cutoff → 90-day outcome window`

All predictive features must be calculated using transactions occurring before the cutoff.

---

### 3. Construct the target

For customers known before the cutoff, determine whether they make a successful purchase during the following 90 days.

Create:

`Inactive90 = 1`

if no purchase occurs.

Otherwise:

`Inactive90 = 0`

---

### 4. Engineer historical customer features

Using only information available before the cutoff, create features such as:

- Recency
- Frequency
- Monetary
- Average order value
- customer tenure
- product diversity
- items per order
- cancellation rate
- average purchase interval
- variability in purchase intervals

Optional behavioural-change features:

- recent spending trend
- recent order-frequency trend
- change in average order value

---

### 5. Establish a baseline

Train a simple baseline such as:

`DummyClassifier`

Determine what performance can be achieved without meaningful predictive information.

---

### 6. Train an interpretable model

Train Logistic Regression.

Use appropriate preprocessing.

Interpret:

- coefficients
- direction of effects
- predicted probabilities

---

### 7. Train a nonlinear model

Compare Logistic Regression with a tree-based model such as:

- Random Forest
- XGBoost
- LightGBM

Do not add models purely to increase the model count.

---

### 8. Evaluate appropriately

Consider:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1

Also evaluate the model from a targeting perspective.

For example:

**Recall@10%**

> If the retention team can contact only the highest-risk 10% of customers, what percentage of future inactive customers are captured?

---

### 9. Explain predictions

Use feature importance or SHAP to understand:

- global drivers of inactivity
- individual high-risk predictions

Distinguish predictive relationships from causal explanations.

---

## Deliverables

Produce:

1. A documented inactivity definition.
2. Temporal train/outcome design.
3. Leakage-safe customer features.
4. Baseline model.
5. Logistic Regression.
6. Nonlinear comparison model.
7. Classification metrics.
8. Top-k targeting analysis.
9. Model explanation.
10. Retention-oriented interpretation.

---

## Final Questions

Answer:

1. How predictable is 90-day inactivity?
2. Which behaviours are most associated with future inactivity?
3. How much does ML improve over the baseline?
4. How many inactive customers can be identified within a limited targeting budget?
5. Which customers appear highest risk?
6. What can the model predict, and what can it not tell us?

---

## Constraint

Do not use future transactions when constructing historical customer features.

Preventing temporal leakage is more important than achieving a higher model score.