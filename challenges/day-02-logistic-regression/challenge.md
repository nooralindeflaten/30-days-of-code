# Day 02: Logistic Regression & Classification Metrics From Scratch

**Model 1 — Supervised Learning Baseline (prep day 2 of 4)**

## Why this day exists
Day 5's final model is a binary classifier. Before reaching for
`LogisticRegression()` on Day 5, you should understand exactly what it's doing
and how to judge whether it's any good — accuracy alone lies to you constantly,
especially on imbalanced data (which the real dataset later this week has).

## Data
Two datasets today — one to prove your implementation works, one to prove it
generalizes:

1. **Synthetic** (to test your implementation against a known-separable case):
```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=1.8, random_state=7)
```

2. **Real, small, no internet required** (to sanity-check on something with
   real structure): `sklearn.datasets.load_breast_cancer()` — 30 features,
   binary target, built into scikit-learn.

## Objective
Implement logistic regression trained via gradient descent, plus the core
binary classification metrics, all from scratch.

## Tasks
1. Implement `sigmoid(z)` (numerically stable — clip `z` before `np.exp`).
2. Implement binary cross-entropy loss.
3. Implement `LogisticRegressionGD` with `.fit`, `.predict_proba`, `.predict`.
4. Implement, from scratch (no `sklearn.metrics`): a `confusion_matrix`
   function, then `accuracy`, `precision`, `recall`, `f1_score` built on top
   of it.
5. Train on the synthetic blobs, plot the decision boundary.
6. Train on `load_breast_cancer` (features will need scaling — reuse or write
   a simple standardization step), and report all 4 metrics plus the
   confusion matrix.

## Bonus (optional)
- Verify every one of your metric functions against the equivalent
  `sklearn.metrics` function on the same predictions — they should match
  exactly.
- Try shifting your classification threshold away from 0.5 and see how
  precision and recall trade off against each other.

## Constraints
- No `sklearn.linear_model.LogisticRegression` — the model itself must be
  yours. `sklearn.datasets` for loading data is fine.

## Done when
- Decision boundary plot on the blobs visibly separates the two classes.
- On breast cancer, you get all 4 metrics without an error, and can explain
  the difference between precision and recall using this specific dataset's
  classes as the example (what does a false negative actually mean here?).
