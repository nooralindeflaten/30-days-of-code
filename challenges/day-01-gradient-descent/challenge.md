# Day 01: Gradient Descent From Scratch

**Model 1 — Supervised Learning Baseline (prep day 1 of 4)**

## Why this day exists
Every model you'll train for the rest of this challenge — logistic regression,
a neural net, even the PyTorch CNN in Model 3 — is fit using some flavor of
gradient descent under the hood. Today you build the actual loop by hand once,
so it stops being a black box.

## Data
Synthetic — generate it yourself, no download needed:
```python
import numpy as np
np.random.seed(0)
X = np.random.uniform(-5, 5, size=(200, 1))
true_w, true_b = 3.0, 7.0
y = true_w * X[:, 0] + true_b + np.random.normal(scale=1.0, size=200)
```

## Objective
Implement linear regression trained via **batch gradient descent**, using only
NumPy (no `sklearn.linear_model`).

## Tasks
1. Implement a `LinearRegressionGD` class with `.fit(X, y)` and `.predict(X)`.
2. Inside `.fit`, run batch gradient descent: compute predictions, compute MSE
   loss, compute gradients w.r.t. weight and bias, update both, repeat.
3. Print the loss every 100 iterations — confirm it's monotonically decreasing.
4. After training, compare your learned `w` and `b` against the true values
   (3.0 and 7.0) — they should be close but not exact (there's noise in the data).
5. Plot the data as a scatter plot with your fitted line drawn on top.

## Bonus (optional)
- Support multiple features (not just 1D `X`) — same math, just matrix shapes change.
- Add L2 regularization (ridge) as an optional constructor argument.
- Try 3 different learning rates (too low, good, too high) and plot all three
  loss curves on one chart to see divergence firsthand.

## Constraints
- No `sklearn`, no `scipy.optimize` — this one has to be by hand.
- You may use `matplotlib` for the plot.

## Done when
- Your loss curve strictly decreases (or plateaus) — if it's bouncing around or
  increasing, your learning rate is too high.
- Your learned `w` is within ~0.3 of 3.0, and `b` within ~0.3 of 7.0.
- You can explain, out loud, what the gradient of MSE w.r.t. `w` represents,
  without looking it up.
