"""
Day 01: Gradient Descent From Scratch — Starter Template

Fill in the TODOs. See challenge.md for the full spec, data generation code,
and "done when" criteria.
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, l2=0.0):
        """
        learning_rate: step size for each gradient descent update.
        n_iterations: how many passes of gradient descent to run.
        l2: L2 (ridge) regularization strength. 0.0 = no regularization.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the model on X (shape: n_samples x n_features) and y (shape: n_samples,)
        using batch gradient descent.

        By the end of this method:
        - self.w and self.b should hold the learned parameters.
        - self.loss_history should have one entry per iteration, so you can
          plot it afterward and confirm it's decreasing.
        """
        n_samples, n_features = X.shape

        # TODO: initialize self.w (zeros, shape (n_features,)) and self.b (0.0)

        for i in range(self.n_iterations):
            # TODO: compute predictions: y_predicted = X @ w + b

            # TODO: compute MSE loss (+ L2 penalty term), append to self.loss_history

            # TODO: compute gradients dw, db
            #   dw = (-2/n_samples) * X.T @ (y - y_predicted) + 2 * l2 * w
            #   db = (-2/n_samples) * sum(y - y_predicted)

            # TODO: update self.w and self.b using self.learning_rate

            # Optional but recommended: print loss every 100 iterations
            # to visually confirm it's decreasing as you develop this.
            pass

        return self

    def predict(self, X):
        """Return predictions for X using the learned w and b."""
        # TODO
        pass


def truncate_at_divergence(history):
    """
    Returns the prefix of `history` up to (not including) the first
    non-finite value (inf/nan). Useful when plotting a run that diverged
    (e.g. too-high learning rate) so matplotlib doesn't choke on inf/nan.
    """
    # TODO
    pass


def main():
    np.random.seed(42)
    X = np.random.uniform(-5, 5, size=(200, 1))
    true_w, true_b = 3.0, 7.0
    y = true_w * X[:, 0] + true_b + np.random.normal(scale=1.0, size=200)

    # TODO: train a model with a reasonable learning rate (e.g. 0.01)
    # and print learned vs. true w/b once done.

    # Bonus: train a few more models with different learning rates
    # (e.g. too low, good, too high) to compare convergence behavior.
    # Careful: a too-high learning rate can diverge to inf/nan — see
    # truncate_at_divergence() above if you plot it.

    # TODO: plot two panels side by side using plt.subplots(1, 2, ...):
    #   left  = scatter of the data + your fitted line
    #   right = loss curve(s) vs iteration, one line per learning rate tried
    pass


if __name__ == "__main__":
    main()