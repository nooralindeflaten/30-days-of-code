# Day 03: Real-World Data Preprocessing

**Model 1 — Supervised Learning Baseline (prep day 3 of 4)**

## Why this day exists
This is the dataset you'll be using through Day 5. Real data is never clean —
missing values, mixed types, class imbalance. The choices you make today
(how you impute, how you encode) directly become part of Friday's model.

## Data: UCI Adult Census Income
Predict whether a person's income is `>50K` or `<=50K` from census attributes.
This is one of the most commonly used classification benchmarks in ML — you
will see it again in other people's portfolios, so understand it well.

**Option A — via scikit-learn (needs internet, easiest):**
```python
from sklearn.datasets import fetch_openml
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame
```

**Option B — direct download (also needs internet):**
https://archive.ics.uci.edu/dataset/2/adult
(`adult.data` — no header row; column names are listed on that page)

**Option C — no internet available:**
Use this synthetic generator instead, which deliberately injects the same
kinds of problems (missing values, mixed types, imbalance) as the real thing:
```python
import numpy as np, pandas as pd

def make_income_like(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(38, 13, n), 17, 90).round().astype(int)
    education = rng.choice(
        ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"],
        size=n, p=[0.35, 0.25, 0.25, 0.10, 0.05])
    hours_per_week = np.clip(rng.normal(40, 10, n), 5, 99).round().astype(int)
    occupation = rng.choice(
        ["Tech", "Sales", "Admin", "Service", "Exec-managerial", None],
        size=n, p=[0.2, 0.15, 0.15, 0.2, 0.2, 0.1])
    marital_status = rng.choice(["Married", "Single", "Divorced"], size=n, p=[0.5, 0.35, 0.15])
    edu_score = pd.Series(education).map(
        {"HS-grad": 0, "Some-college": 1, "Bachelors": 2, "Masters": 3, "Doctorate": 4}).values
    logit = (-3.5 + 0.05 * age + 0.55 * edu_score + 0.03 * hours_per_week
              + 0.3 * (marital_status == "Married") + rng.normal(0, 0.6, n))
    prob = 1 / (1 + np.exp(-logit))
    income = np.where(rng.uniform(size=n) < prob, ">50K", "<=50K")
    df = pd.DataFrame({"age": age, "education": education, "hours_per_week": hours_per_week,
                        "occupation": occupation, "marital_status": marital_status, "income": income})
    # inject missing values into a numeric column too
    idx = rng.choice(df.index, size=int(0.05 * n), replace=False)
    df.loc[idx, "hours_per_week"] = np.nan
    return df
```

## Objective
Turn this raw, messy dataset into a clean, model-ready feature matrix.

## Tasks
1. Load the data (whichever option applies to you) and inspect it: `.info()`,
   `.isna().sum()`, `.describe()`, and check the class balance of the target
   (`income`). Write down the imbalance ratio — you'll need it Friday.
2. Handle missing values: numeric columns → your choice of strategy (median
   is reasonable), categorical columns → your choice (mode, or an explicit
   "Unknown" category — pick one and justify it in a comment).
3. Encode categorical columns (one-hot, or another approach you can justify).
4. Scale/standardize numeric columns.
5. Engineer at least 2 new features from the raw columns (e.g. something
   derived from age, education, or hours worked — your call).
6. Split into train/test (80/20), **stratified on the target** given the
   class imbalance you noted in step 1.
7. Save your cleaned train/test splits to disk (`.csv` or `.parquet`) — Day 5
   will load these directly rather than redoing this work.

## Bonus (optional)
- Wrap steps 2–4 as a proper `sklearn.pipeline.Pipeline` +
  `ColumnTransformer` instead of manual pandas — this is how you'd actually
  do it on a real team, and it prevents train/test leakage automatically.
- Look for outliers in the numeric columns (IQR rule) and decide, with a
  written justification, whether to cap, remove, or leave them.

## Constraints
- Whatever you do to the training set, do it in a way that would generalize
  to unseen data at prediction time — e.g. compute imputation values from
  train only, then apply them to test.

## Done when
- `X_train`, `X_test`, `y_train`, `y_test` exist, contain no NaNs, and are
  fully numeric.
- You have a one-paragraph note (in your solution's `notes.md`) on the class
  imbalance you found and why it might matter for the metrics you choose on
  Day 5.
