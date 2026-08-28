import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.loss_history = []
        self.l2 = 0.1  # L2 regularization parameter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []

        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.w) + self.b
            
            # MSE 
            loss = (1 / n_samples) * np.sum((y - y_predicted) ** 2) + self.l2 * np.sum(self.w ** 2)
            self.loss_history.append(loss)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss}")
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_predicted)) + 2 * self.l2 * self.w
            db = (-2 / n_samples) * np.sum(y - y_predicted)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db


    def predict(self, X):
        return np.dot(X, self.w) + self.b

def main():
    np.random.seed(42)
    X = np.random.uniform(-5, 5, size=(200, 1))
    true_w, true_b = 3.0, 7.0
    y = true_w * X[:, 0] + true_b + np.random.normal(scale=1.0, size=200)
    

    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    model_low_lr = LinearRegressionGD(learning_rate=0.001, n_iterations=1000)
    model_low_lr.fit(X, y)
    
    model_high_lr = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
    model_high_lr.fit(X, y)
    print(f"Learned weights: {model.w}, Learned bias: {model.b}")
    print(f"Final loss: {model.loss_history[-1]}")
    print(f"True weights: {true_w}, True bias: {true_b}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X, y, color='blue', label='Data points')
    axes[0].plot(X, model.predict(X), color='red', label='Fitted line')
    axes[0].set_title('Linear Regression with Gradient Descent')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].legend()

    axes[1].plot(range(len(model.loss_history)), model.loss_history, color='blue', label='LR=0.01')
    axes[1].plot(range(len(model_low_lr.loss_history)), model_low_lr.loss_history, color='orange', label='LR=0.001')
    axes[1].plot(range(len(model_high_lr.loss_history)), model_high_lr.loss_history, color='green', label='LR=0.1')
    axes[1].set_title('Loss vs Iteration by Learning Rate')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
        
    
if __name__ == "__main__":
    main()