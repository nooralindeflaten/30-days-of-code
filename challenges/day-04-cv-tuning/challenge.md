# Day 04: Cross-Validation & Hyperparameter Tuning

**Model 1 — Supervised Learning Baseline (prep day 4 of 4)**

## Why this day exists
A single train/test split can lie to you — get a lucky or unlucky split and
your accuracy estimate is noise. Tomorrow you pick a final model and
hyperparameters for real; today you build the tool that makes that choice
trustworthy instead of a guess.

## Data
Your cleaned train split from Day 3 (`X_train`, `y_train`). Load it from the
file you saved yesterday — don't redo the preprocessing.

## Objective
Implement K-fold cross-validation from scratch, then use it to compare models
and tune hyperparameters properly.

## Tasks
1. Implement `k_fold_split(n_samples, k, seed)` from scratch (no
   `sklearn.model_selection.KFold`) — it should yield `(train_idx, val_idx)`
   for each of the `k` folds.
2. Implement `cross_validate(model_fn, X, y, k=5)` that trains a fresh model
   per fold and returns the mean and standard deviation of a chosen metric
   (given the class imbalance from Day 3, think about whether accuracy is
   actually the right metric here, or whether F1/ROC-AUC would serve you
   better).
3. Using your cross-validation function, compare three model types on the
   same folds: `LogisticRegression`, `DecisionTreeClassifier`, `KNeighborsClassifier`
   (using `sklearn` implementations is fine here — the CV *mechanism* is what
   you're building, not the models).
4. Pick the most promising model type from step 3, then run a small grid
   search over its key hyperparameters (e.g. for a decision tree:
   `max_depth`, `min_samples_split`) using your own cross-validation function.
5. Report the best hyperparameter combination and its mean CV score.

## Bonus (optional)
- Make your K-fold splitter **stratified** (preserve the class ratio in every
  fold) — given yesterday's imbalance, this matters more than it would on a
  balanced dataset.
- Compare your from-scratch grid search result against
  `sklearn.model_selection.GridSearchCV` run with the same folds, and confirm
  they agree.

## Constraints
- The K-fold splitting logic and the cross-validation loop must be yours.
  Using `sklearn` model classes themselves (not their CV/search tools) is fine.

## Done when
- You can state, in one sentence, which model type and which hyperparameters
  you're carrying into tomorrow, and why — backed by a specific CV score, not
  a vibe.
