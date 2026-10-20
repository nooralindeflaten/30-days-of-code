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
        if np.random.random() > self.epsilon:
            return np.argmax(self.n_arms)
        else:
            return np.random.choice(len(self.q_values))
    
    def update_epsilon(self,new_epsilon):
        self.epsilon = new_epsilon
        return 
    
    def update_arm(self, k_arm, reward):
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
        self.c = 2
    
    def init_arms_ucb(self, k_arms):
        self.counts = np.ones(k_arms)
        self.values = np.ones(k_arms)
        return
    
    def select_arm(self):
        total_counts = sum(self.counts)
        ucb_values = np.ones(self.n_arms)
        for arm in range(self.n_arms):
            ucb_values[arm] = self.values[arm] + np.sqrt(self.c * np.log(total_counts/self.counts[arm]))
        return np.argmax(ucb_values)
    
    def update_arm(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value
        return
    
class EpsilonGreedyAgentCopilot:
    """
    Classic epsilon-greedy for bounded rewards.
    """

    def __init__(self, n_arms, epsilon=0.1, seed=42):
        self.n_arms = n_arms
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

        self.counts = np.zeros(n_arms, dtype=int)
        self.q_values = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.q_values))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        q = self.q_values[arm]
        # incremental mean
        self.q_values[arm] = q + (reward - q) / n


class UCB1AgentCopilot:
    """
    Upper Confidence Bound (UCB1):
    pick arm maximizing q_i + sqrt(2 log t / n_i)
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms, dtype=int)
        self.q_values = np.zeros(n_arms, dtype=float)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # pull each arm once first
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a

        bonus = np.sqrt((2.0 * np.log(self.t)) / self.counts)
        ucb = self.q_values + bonus
        return int(np.argmax(ucb))

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        q = self.q_values[arm]
        self.q_values[arm] = q + (reward - q) / n

class ThompsonSamplingBernoulliAgent:
    """
    Thompson Sampling for Bernoulli outcomes.

    IMPORTANT:
    - Update should use binary success (0/1), not net reward.
    - If your environment returns reward = success - cost, pass success separately.
    """

    def __init__(self, n_arms, seed=42, alpha0=1.0, beta0=1.0):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

        self.alpha = np.ones(n_arms, dtype=float) * alpha0
        self.beta = np.ones(n_arms, dtype=float) * beta0

        # optional running reward stats (for plotting/comparison)
        self.counts = np.zeros(n_arms, dtype=int)
        self.reward_means = np.zeros(n_arms, dtype=float)

    def select_arm(self):
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

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