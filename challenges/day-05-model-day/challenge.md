# Day 05 — MODEL DAY: Ship the Income Classifier

**Model 1 — Supervised Learning Baseline (final day)**

## Why this day exists
This is where the previous four days stop being isolated exercises and
become one finished thing: a trained model you evaluated properly, that
you could actually explain in an interview.

## Data
Your cleaned Day 3 splits + the winning model/hyperparameters from Day 4.

## Objective
Train the final model, evaluate it honestly, and package it like a real
project — not just a script that prints one accuracy number.

## Tasks
1. Load your Day 3 train/test splits.
2. Train your Day 4 winning model configuration on the **full training set**.
3. Evaluate on the **held-out test set** (never touched until now) with:
   accuracy, precision, recall, F1, and a confusion matrix. Given the class
   imbalance, report which metric you're treating as the "headline" number
   and justify why.
4. Plot an ROC curve and report AUC.
5. If your model type supports it, extract and plot feature importances (or
   coefficients) — which features actually drove predictions? Does it match
   your intuition from Day 3's EDA?
6. Save the trained model to disk (`joblib.dump`).
7. Write a **model card** — use the template below, filled out for real —
   and save it as `MODEL_CARD.md` in this model's folder.

## Model card template
```markdown
# Model Card: Income Classifier (Model 1)

## Problem
What are we predicting, and why would anyone care?

## Data
Source, size, feature list, class balance, and anything notable/messy about it
(this is where your Day 3 notes pay off).

## Approach
Model type, key hyperparameters, and *why* — reference your Day 4 CV results.

## Results
Headline metric + why you chose it. Full metric breakdown. Confusion matrix.
ROC-AUC. One sentence on what the top features tell you about the problem.

## Limitations
Where would this model likely fail or be unfair? (Class imbalance, any
demographic features involved, dataset age/representativeness, etc.)

## What I'd do next
If you had another week on this, what's the next improvement?
```

## Bonus (optional)
- Compare your final model's test performance against a trivial baseline
  (always predict the majority class) to make sure it's actually adding value
  beyond the class imbalance.
- Write a short "how to use this model" snippet showing how someone else
  would load `model.joblib` and get a prediction on a new record.

## Constraints
- The test set must not have influenced any decision before this day (no
  peeking at test metrics during Day 4's tuning — if you did, that's worth
  noting honestly in the model card's limitations section).

## Done when
- `MODEL_CARD.md` exists, reads like something you'd actually put in a
  portfolio, and the numbers in it are real numbers from your own run, not
  round/suspicious ones.
- You could explain this whole project to a recruiter in under 90 seconds
  using only the model card.
