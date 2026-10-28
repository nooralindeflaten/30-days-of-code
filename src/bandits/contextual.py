import numpy as np

class LinUCB:
    """
    Per-arm linear model with confidence bonus:
      p_a(x) = x^T theta_a + alpha * sqrt(x^T A_a^{-1} x)
    """
    def __init__(self, n_arms, d, alpha=1.0):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha

        self.A = np.array([np.eye(d) for _ in range(n_arms)])    # (K,d,d)
        self.b = np.zeros((n_arms, d), dtype=float)

    def _score(self, arm, x):
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        exploit = float(x @ theta)
        explore = float(np.sqrt(x @ A_inv @ x))
        return exploit + self.alpha * explore

    def select(self, x):
        scores = [self._score(a, x) for a in range(self.n_arms)]
        return int(np.argmax(scores))

    def update(self, arm, reward, x):
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def expected_reward_hat(self, x):
        # model-based expected reward estimates for DR
        out = np.zeros(self.n_arms, dtype=float)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            out[a] = x @ theta
        return out
