"""
Day 02: Logistic Regression & Classification Metrics — Starter Template

Fill in the TODOs. See challenge.md for the full spec and "done when" criteria.
"""
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix as sklearn_confusion_matrix,
    f1_score as sklearn_f1_score,
    accuracy_score as sklearn_accuracy_score,
    precision_score as sklearn_precision_score,
    recall_score as sklearn_recall_score,
)
from sklearn.datasets import load_breast_cancer


def sigmoid(z):
    """
    Numerically stable sigmoid. Clip z before exponentiating so large
    magnitude inputs don't overflow np.exp.
    """
    # TODO
    pass


def CE_loss(y_true, y_pred):
    """
    Binary cross-entropy loss. Clip y_pred away from exactly 0 or 1
    first, or log(0) will blow this up.
    """
    # TODO
    pass


class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Train via batch gradient descent on the binary cross-entropy loss.
        By the end: self.weights, self.bias learned; self.loss_history has
        one entry per iteration.
        """
        n_samples, n_features = X.shape
        # TODO: initialize self.weights (zeros) and self.bias (0)

        for _ in range(self.n_iterations):
            # TODO: z = X @ weights + bias
            # TODO: y_predicted = sigmoid(z)

            # TODO: dw = (1/n_samples) * X.T @ (y_predicted - y)
            # TODO: db = (1/n_samples) * sum(y_predicted - y)

            # TODO: update self.weights, self.bias using self.learning_rate

            # TODO: compute CE_loss(y, y_predicted), append to self.loss_history
            pass

    def predict_proba(self, X):
        """Return predicted probabilities (sigmoid of the linear combination)."""
        # TODO
        pass

    def predict(self, X, threshold=0.5):
        """Return hard 0/1 predictions using the given threshold."""
        # TODO
        pass


def confusion_matrix(y_true, y_pred):
    """
    Returns a 2x2 array. Any consistent layout works as long as your
    precision/recall/f1 functions below index into it consistently —
    just note in a comment which convention you picked, since it likely
    won't match sklearn's [[tn, fp], [fn, tp]] layout by default.
    """
    # TODO
    pass


def accuracy_score(y_true, y_pred):
    """Correct predictions / total predictions."""
    # TODO
    pass


def precision_score(y_true, y_pred):
    """Of everything predicted positive, how much was actually positive?
    Watch for division by zero when there are no positive predictions."""
    # TODO
    pass


def recall_score(y_true, y_pred):
    """Of everything actually positive, how much did we catch?
    Watch for division by zero when there are no actual positives."""
    # TODO
    pass


def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall. Watch for division by zero
    when both precision and recall are 0."""
    # TODO
    pass


def plot_decision_boundary(model, X, y):
    """
    Only meaningful for 2D feature data (like the blobs dataset) — build
    a mesh grid over the feature space, predict on every grid point, and
    contourf the result under a scatter of the real points.
    """
    # TODO
    pass


def run_blobs(X, y):
    """
    Train on the synthetic blobs, print your 4 metrics + confusion matrix,
    and show the decision boundary plot.
    """
    # TODO
    pass


def results(X, y):
    """
    Train your model AND sklearn's LogisticRegression on the same data,
    then print your metrics next to sklearn's equivalents so you can
    compare them directly. Use this on breast cancer (30 features, no
    boundary plot — that only makes sense in 2D).
    """
    # TODO
    pass


if __name__ == "__main__":
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=1.8, random_state=7)
    X_bc, y_bc = load_breast_cancer(return_X_y=True)
    X_bc = StandardScaler().fit_transform(X_bc)  # logistic regression needs scaled features

    print("Blobs results: ")
    run_blobs(X, y)

    print("\nBreast Cancer results: ")
    results(X_bc, y_bc)