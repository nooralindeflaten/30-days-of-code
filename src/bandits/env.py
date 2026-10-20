import random
import numpy as np


class BernoulliBandit:
    def __init__(self, probs, costs, seed=42):
        probs = np.asarray(probs, dtype=float)
        costs = np.asarray(costs, dtype=float)
        if probs.shape != costs.shape:
            raise ValueError("probs and costs must have same shape")
        if np.any((probs < 0) | (probs > 1)):
            raise ValueError("all probabilities must be in [0,1]")

        self.probs = probs
        self.costs = costs
        self.n_arms = len(probs)
        self.rng = np.random.default_rng(seed)

        # True expected reward per arm: E[success - cost] = p - cost
        self.true_means = self.probs - self.costs
        self.best_arm = int(np.argmax(self.true_means))
        self.best_mean = float(self.true_means[self.best_arm])

    def pull(self, arm: int):
        """Return (reward, success)."""
        p = self.probs[arm]
        c = self.costs[arm]
        success = 1.0 if self.rng.random() < p else 0.0
        reward = success - c
        return float(reward), int(success)

    def expected_reward(self, arm: int):
        return float(self.true_means[arm])

    def regret(self, arm: int):
        """Instant expected regret for selecting `arm` instead of optimal arm."""
        return float(self.best_mean - self.true_means[arm])

class ContextualLinearBandit():
    def __init__(self, n_arms, n_features, noise_std=0.1, seed=42):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Randomly generate true parameter vectors for each arm
        self.theta = self.rng.normal(size=(n_arms, n_features))

    def pull(self, arm: int, context: np.ndarray):
        """Return (reward, success)."""
        if context.shape[0] != self.n_features:
            raise ValueError(f"Context must have shape ({self.n_features},)")

        mean_reward = np.dot(self.theta[arm], context)
        reward = mean_reward + self.rng.normal(scale=self.noise_std)
        return float(reward), int(reward > 0)  # success if reward > 0

    def expected_reward(self, arm: int, context: np.ndarray):
        return float(np.dot(self.theta[arm], context))

    def regret(self, arm: int, context: np.ndarray):
        """Instant expected regret for selecting `arm` instead of optimal arm."""
        optimal_arm = int(np.argmax([np.dot(self.theta[a], context) for a in range(self.n_arms)]))
        optimal_mean = float(np.dot(self.theta[optimal_arm], context))
        selected_mean = float(np.dot(self.theta[arm], context))
        return optimal_mean - selected_mean
    
    def get_optimal_arm(self, context: np.ndarray):
        """Return the index of the optimal arm for a given context."""
        if context.shape[0] != self.n_features:
            raise ValueError(f"Context must have shape ({self.n_features},)")
        return int(np.argmax([np.dot(self.theta[a], context) for a in range(self.n_arms)]))
    