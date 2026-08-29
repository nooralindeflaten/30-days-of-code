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
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

def CE_loss(y_true, y_pred):
    # Clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = CE_loss(y, y_predicted)
            self.loss_history.append(loss)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        y_predicted_proba = self.predict_proba(X)
        return np.where(y_predicted_proba >= threshold, 1, 0)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])

def accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return (cm[0, 0] + cm[1, 1]) / np.sum(cm)

def precision_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0

def recall_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) != 0 else 0

def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) != 0 else 0

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def results(X,y):
    my_model = LogisticRegressionGD(learning_rate=0.01, n_iterations=1000)
    my_model.fit(X, y)

    sklearn = LogisticRegression(max_iter=1000)
    sklearn.fit(X,y)

    # make predictions
    y_pred = my_model.predict(X)
    y_pred_sklearn = sklearn.predict(X)
    # evaluate the model

    cm = confusion_matrix(y, y_pred)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print("Confusion Matrix Mine:\n", cm)
    print("Confusion Matrix Sklearn:\n", sklearn_confusion_matrix(y, y_pred_sklearn))
    print("Accuracy:", acc)
    print("Accuracy Sklearn:", sklearn_accuracy_score(y, y_pred_sklearn))
    print("Precision:", prec)
    print("Precision Sklearn:", sklearn_precision_score(y, y_pred_sklearn))
    print("Recall:", rec)
    print("Recall Sklearn:", sklearn_recall_score(y, y_pred_sklearn))
    print("F1 Score:", f1)
    print("F1 Score Sklearn:", sklearn_f1_score(y, y_pred_sklearn))



def run_blobs(X,y):
    model = LogisticRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    plot_decision_boundary(model, X, y)
    
    
if __name__ == "__main__":
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=1.8, random_state=7)
    X_bc, y_bc = load_breast_cancer(return_X_y=True)
    X_bc = StandardScaler().fit_transform(X_bc)
    print("Blobs results: ")
    run_blobs(X, y)
    print("\nBreast Cancer results: ")
    results(X_bc, y_bc)