import numpy as np


class EpsilonGreedyAgent:
    def __init__(self, n_arms, epsilon=0.1, seed=42):
        self.n_arms = n_arms
        self.seed = seed
        self.epsilon = epsilon
        
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms)
        self.q_values = np.zeros(n_arms)
    
    
    def select_arm(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.q_values))
    
    def update_epsilon(self,new_epsilon):
        self.epsilon = new_epsilon
        return 
    
    def select(self, x=None):
        return self.select_arm()
    
    
    def update(self, k_arm, reward, x=None):
        self.counts[k_arm] = self.counts[k_arm] + 1
        n = self.counts[k_arm]
        
        value = self.q_values[k_arm]
        new_value = ((n-1)/float(n)) * value + (1 / float(n)) * reward
        self.q_values[k_arm] = new_value
    
    
class UCBAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.c = 1
    
    def select_arm(self):
        for a in range(self.n_arms):          # pull each arm once first
            if self.counts[a] == 0:
                return a
        t = self.counts.sum()
        bonus = np.sqrt(self.c * np.log(t) / self.counts)
        return int(np.argmax(self.values + bonus))
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
        return

class ThompsonSamplingBernoulliAgent:
    """
    Thompson Sampling for Bernoulli outcomes.

    IMPORTANT:
    - Update should use binary success (0/1), not net reward.
    - If your environment returns reward = success - cost, pass success separately.
    """

    def __init__(self, n_arms, costs,seed=42, alpha0=1.0, beta0=1.0):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.costs = np.asarray(costs, dtype=float)

        self.alpha = np.ones(n_arms, dtype=float) * alpha0
        self.beta = np.ones(n_arms, dtype=float) * beta0

        # optional running reward stats (for plotting/comparison)
        self.counts = np.zeros(n_arms, dtype=int)
        self.reward_means = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        return int(np.argmax(self.rng.beta(self.alpha, self.beta) - self.costs))

    def update(self, arm, reward, success=None):
        """
        For correctness in Bernoulli TS, provide success in {0,1}.
        If success is None, we fallback to thresholding reward > 0 (not ideal if cost shifts reward).
        """
        if success is None:
            s = 1 if reward > 0 else 0
        else:
            s = int(success)

        self.alpha[arm] += s
        self.beta[arm] += (1 - s)

        # optional reward mean tracking
        self.counts[arm] += 1
        n = self.counts[arm]
        q = self.reward_means[arm]
        self.reward_means[arm] = q + (reward - q) / n