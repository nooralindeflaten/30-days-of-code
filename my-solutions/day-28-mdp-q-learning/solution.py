"""
Day 29 — MDP + Q-learning vs SARSA (lean, CliffWalking)

Outputs:
- reports/day28/learning_curves.csv
- reports/day28/learning_curves.png
- reports/day28/policy_grid_qlearning.txt
- reports/day28/policy_grid_sarsa.txt
- reports/day28/summary.md

Run:
  python -m day-28-mdp-q-learning.solution
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

# -------------------------
# Config
# -------------------------
OUT = Path("data/rfl/reports/day28")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SEEDS = 10
EPISODES = 600
MAX_STEPS = 300

ALPHA = 0.1
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995  # multiplicative per episode

# CliffWalking defaults are fine
ENV_NAME = "CliffWalking-v1"

# action symbols for policy print
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


# -------------------------
# RL helpers
# -------------------------
def epsilon_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s]))


def run_q_learning(seed):
    env = gym.make(ENV_NAME)
    rng = np.random.default_rng(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    episode_returns = []
    eps = EPS_START

    for ep in range(EPISODES):
        s, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0

        for _ in range(MAX_STEPS):
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # Q-learning target: max_a' Q(s', a')
            td_target = r + (0.0 if done else GAMMA * np.max(Q[s2]))
            Q[s, a] += ALPHA * (td_target - Q[s, a])

            ep_ret += r
            s = s2
            if done:
                break

        episode_returns.append(ep_ret)
        eps = max(EPS_END, eps * EPS_DECAY)

    env.close()
    return np.array(episode_returns), Q


def run_sarsa(seed):
    env = gym.make(ENV_NAME)
    rng = np.random.default_rng(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    episode_returns = []
    eps = EPS_START

    for ep in range(EPISODES):
        s, _ = env.reset(seed=seed + ep)
        a = epsilon_greedy(Q, s, eps, rng)
        done = False
        ep_ret = 0.0

        for _ in range(MAX_STEPS):
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            a2 = epsilon_greedy(Q, s2, eps, rng) if not done else 0

            # SARSA target: Q(s', a') from actual next action
            td_target = r + (0.0 if done else GAMMA * Q[s2, a2])
            Q[s, a] += ALPHA * (td_target - Q[s, a])

            ep_ret += r
            s, a = s2, a2
            if done:
                break

        episode_returns.append(ep_ret)
        eps = max(EPS_END, eps * EPS_DECAY)

    env.close()
    return np.array(episode_returns), Q


def moving_avg(x, w=20):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def policy_grid_text(Q):
    """
    Renders greedy policy for CliffWalking 4x12 grid.
    S=start, G=goal, C=cliff cells.
    """
    n_rows, n_cols = 4, 12
    grid = np.full((n_rows, n_cols), "", dtype=object)

    # CliffWalking indexing is row-major: s = r * n_cols + c
    start = (3, 0)
    goal = (3, 11)
    cliff = {(3, c) for c in range(1, 11)}

    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) == start:
                grid[r, c] = "S"
            elif (r, c) == goal:
                grid[r, c] = "G"
            elif (r, c) in cliff:
                grid[r, c] = "C"
            else:
                s = r * n_cols + c
                a = int(np.argmax(Q[s]))
                grid[r, c] = ARROWS[a]

    lines = []
    for r in range(n_rows):
        lines.append(" ".join(f"{grid[r, c]:>2}" for c in range(n_cols)))
    return "\n".join(lines)


def main():
    # Multi-seed runs
    q_curves = []
    s_curves = []
    q_last_Q = None
    s_last_Q = None

    for i in range(N_SEEDS):
        seed_i = SEED + i

        q_ret, qQ = run_q_learning(seed_i)
        s_ret, sQ = run_sarsa(seed_i)

        q_curves.append(q_ret)
        s_curves.append(s_ret)

        q_last_Q = qQ
        s_last_Q = sQ

    q_arr = np.vstack(q_curves)  # [seeds, episodes]
    s_arr = np.vstack(s_curves)

    q_mean = q_arr.mean(axis=0)
    s_mean = s_arr.mean(axis=0)
    q_std = q_arr.std(axis=0)
    s_std = s_arr.std(axis=0)

    # Save curves table
    df = pd.DataFrame({
        "episode": np.arange(1, EPISODES + 1),
        "qlearning_mean_return": q_mean,
        "qlearning_std_return": q_std,
        "sarsa_mean_return": s_mean,
        "sarsa_std_return": s_std,
        "qlearning_mean_return_smooth": moving_avg(q_mean, 20),
        "sarsa_mean_return_smooth": moving_avg(s_mean, 20),
    })
    df.to_csv(OUT / "learning_curves.csv", index=False)

    # Plot
    x = df["episode"].values
    plt.figure(figsize=(8, 5))

    plt.plot(x, df["qlearning_mean_return_smooth"], label="Q-learning", linewidth=2)
    plt.fill_between(
        x,
        df["qlearning_mean_return"] - df["qlearning_std_return"],
        df["qlearning_mean_return"] + df["qlearning_std_return"],
        alpha=0.15
    )

    plt.plot(x, df["sarsa_mean_return_smooth"], label="SARSA", linewidth=2)
    plt.fill_between(
        x,
        df["sarsa_mean_return"] - df["sarsa_std_return"],
        df["sarsa_mean_return"] + df["sarsa_std_return"],
        alpha=0.15
    )

    plt.title("CliffWalking: Q-learning vs SARSA (mean ± std over seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "learning_curves.png", dpi=160)
    plt.close()

    # Policy grids
    q_grid = policy_grid_text(q_last_Q)
    s_grid = policy_grid_text(s_last_Q)
    (OUT / "policy_grid_qlearning.txt").write_text(q_grid, encoding="utf-8")
    (OUT / "policy_grid_sarsa.txt").write_text(s_grid, encoding="utf-8")

    # Summary
    tail = 50
    q_tail = float(q_mean[-tail:].mean())
    s_tail = float(s_mean[-tail:].mean())

    summary = f"""# Day 29 Summary (CliffWalking)

Config:
- seeds: {N_SEEDS}
- episodes: {EPISODES}
- alpha: {ALPHA}
- gamma: {GAMMA}
- epsilon: start={EPS_START}, end={EPS_END}, decay={EPS_DECAY}

Final-window average return (last {tail} episodes):
- Q-learning: {q_tail:.3f}
- SARSA: {s_tail:.3f}

Interpretation:
- Q-learning is off-policy and tends to learn greedy targets (often near-cliff shortest path).
- SARSA is on-policy and tends to learn safer behavior under exploration.

Bandit bridge:
- Bandits are 1-step MDPs (no state transition consequences).
- Full RL is needed when current actions influence future states/rewards.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    print("Saved:")
    print(OUT / "learning_curves.csv")
    print(OUT / "learning_curves.png")
    print(OUT / "policy_grid_qlearning.txt")
    print(OUT / "policy_grid_sarsa.txt")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()