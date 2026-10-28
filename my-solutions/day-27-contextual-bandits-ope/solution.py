"""
Day 28 runner — Contextual bandits + offline policy evaluation (lean).

What it does:
1) Simulates contextual interactions where best arm depends on context
2) Logs randomized-policy data with propensities
3) Trains LinUCB online
4) Compares:
   - best fixed arm
   - non-contextual epsilon-greedy
   - contextual LinUCB
5) Offline policy evaluation on logged data:
   - IPS
   - Doubly Robust (DR)
   and compares each estimate to simulator "true" value

Outputs:
- data/logged_interactions.csv
- models/policy.joblib
- reports/day28/learning_curves.png
- reports/day28/ope_vs_truth.csv
- reports/day28/summary.csv

Run:
PYTHONPATH=. python my-solutions/day-28-contextual-bandits-ope/solution.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os, sys
sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.bandits.contextual import LinUCB
from src.bandits.agents import EpsilonGreedyAgent
from src.bandits.env import ContextualLinearBandit
# ---------------------------
# Config
# ---------------------------
SEED = 42
N_ARMS = 4
D = 5                    # context dimension
T = 4000                 # timeline per run
N_SEEDS = 15             # for learning curves comparison
LOG_N = 12000            # size of randomized log for OPE
ALPHA = 1.0              # LinUCB exploration scale
EPS = 0.1                # epsilon-greedy for non-contextual baseline
LOGGER_EPS = 0.20        # logging policy exploration (must be > 0 for support)

OUT_REPORT = Path("data/rfl/reports/day28")
OUT_DATA = Path("data/rfl")
OUT_MODEL = Path("models")
OUT_REPORT.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_MODEL.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities
# ---------------------------
def one_run(env_seed, agent_type):
    env = ContextualLinearBandit(n_arms=N_ARMS, d=D, seed=env_seed)
    if agent_type == "linucb":
        agent = LinUCB(n_arms=N_ARMS, d=D, alpha=ALPHA)
    elif agent_type == "egreedy_noncontext":
        agent = EpsilonGreedyAgent(n_arms=N_ARMS, epsilon=EPS, seed=env_seed)
    elif agent_type == "best_fixed":
        # oracle best fixed arm by Monte Carlo on contexts
        # pick one arm and stick to it
        xs = np.array([env.sample_context() for _ in range(3000)])
        arm_means = np.array([np.mean([env.expected_rewards(x)[a] for x in xs]) for a in range(N_ARMS)])
        best_a = int(np.argmax(arm_means))
        agent = None
    else:
        raise ValueError(agent_type)

    regrets = np.zeros(T, dtype=float)
    rewards = np.zeros(T, dtype=float)

    for t in range(T):
        x = env.sample_context()
        best_a, best_mu = env.best_arm(x)

        if agent_type == "best_fixed":
            a = best_a if t == -1 else best_a  # intentional constant arm
            # Actually use fixed arm selected from MC:
            a = best_a  # replaced immediately below for clarity
            # recompute once:
            if t == 0:
                xs = np.array([env.sample_context() for _ in range(2000)])
                arm_means = np.array([np.mean([env.expected_rewards(xi)[a_] for xi in xs]) for a_ in range(N_ARMS)])
                fixed_arm = int(np.argmax(arm_means))
            a = fixed_arm
        else:
            a = agent.select(x)

        r, _ = env.step(x, a)

        if agent_type != "best_fixed":
            agent.update(a, r, x)

        regrets[t] = best_mu - env.expected_rewards(x)[a]
        rewards[t] = r

    return np.cumsum(regrets), np.cumsum(rewards) / (np.arange(T) + 1)


def generate_logged_data(seed=123, n=12000, logger_eps=0.2):
    """
    Logging policy: epsilon-mixture of greedy wrt true expected reward and uniform random.
    We store propensity explicitly for OPE.
    """
    env = ContextualLinearBandit(n_arms=N_ARMS, d=D, seed=seed)
    rows = []

    for _ in range(n):
        x = env.sample_context()
        mu = env.expected_rewards(x)
        greedy = int(np.argmax(mu))

        if env.rng.random() < logger_eps:
            a = int(env.rng.integers(0, N_ARMS))
            pi_b = logger_eps * (1.0 / N_ARMS) + (1 - logger_eps) * (1.0 if a == greedy else 0.0)
        else:
            a = greedy
            pi_b = logger_eps * (1.0 / N_ARMS) + (1 - logger_eps)

        r, s = env.step(x, a)

        row = {
            "action": a,
            "reward": r,
            "success": s,
            "propensity": pi_b,
        }
        for j in range(D):
            row[f"x{j}"] = x[j]
        rows.append(row)

    return pd.DataFrame(rows), env


def train_policy_from_log_linucb(df):
    pol = LinUCB(n_arms=N_ARMS, d=D, alpha=ALPHA)
    xcols = [f"x{j}" for j in range(D)]
    for r in df.itertuples(index=False):
        x = np.array([getattr(r, c) for c in xcols], dtype=float)
        a = int(r.action)
        rew = float(r.reward)
        pol.update(a, rew, x)
    return pol


def policy_action(policy, x):
    return policy.select(x)


def estimate_ips(log_df, target_policy):
    xcols = [f"x{j}" for j in range(D)]
    vals = []
    for r in log_df.itertuples(index=False):
        x = np.array([getattr(r, c) for c in xcols], dtype=float)
        a = int(r.action)
        rew = float(r.reward)
        pb = float(r.propensity)

        at = policy_action(target_policy, x)
        if a == at:
            vals.append(rew / pb)
        else:
            vals.append(0.0)
    return float(np.mean(vals))


def estimate_dr(log_df, target_policy, reward_model_policy):
    """
    DR with a direct-method baseline from reward_model_policy.expected_reward_hat(x)
    """
    xcols = [f"x{j}" for j in range(D)]
    vals = []
    for r in log_df.itertuples(index=False):
        x = np.array([getattr(r, c) for c in xcols], dtype=float)
        a = int(r.action)
        rew = float(r.reward)
        pb = float(r.propensity)

        at = policy_action(target_policy, x)
        qhat_all = reward_model_policy.expected_reward_hat(x)
        qhat_at = float(qhat_all[at])
        qhat_a = float(qhat_all[a])

        corr = ((1.0 if a == at else 0.0) / pb) * (rew - qhat_a)
        vals.append(qhat_at + corr)
    return float(np.mean(vals))


def true_policy_value(env_seed, policy, n=10000):
    env = ContextualLinearBandit(n_arms=N_ARMS, d=D, seed=env_seed)
    vals = []
    for _ in range(n):
        x = env.sample_context()
        a = policy_action(policy, x)
        vals.append(env.expected_rewards(x)[a])  # expected reward, low-noise truth estimate
    return float(np.mean(vals))


# ---------------------------
# Main
# ---------------------------
def main():
    # 1) Learning curves across seeds
    agents = ["best_fixed", "egreedy_noncontext", "linucb"]
    curves = {a: [] for a in agents}
    rewards = {a: [] for a in agents}

    for s in range(N_SEEDS):
        for a in agents:
            reg, avg_r = one_run(env_seed=SEED + s, agent_type=a)
            curves[a].append(reg)
            rewards[a].append(avg_r)

    # aggregate
    summary_rows = []
    plt.figure(figsize=(8, 5))
    x = np.arange(1, T + 1)

    for a, color in zip(agents, ["tab:gray", "tab:blue", "tab:green"]):
        arr = np.vstack(curves[a])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        plt.plot(x, mean, label=a, color=color, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

        summary_rows.append({
            "agent": a,
            "final_cum_regret_mean": float(mean[-1]),
            "final_cum_regret_std": float(std[-1]),
            "final_avg_reward_mean": float(np.vstack(rewards[a]).mean(axis=0)[-1]),
        })

    plt.title("Contextual bandit learning: cumulative regret (mean ± std)")
    plt.xlabel("Step")
    plt.ylabel("Cumulative regret")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_REPORT / "learning_curves.png", dpi=160)
    plt.close()

    pd.DataFrame(summary_rows).to_csv(OUT_REPORT / "summary.csv", index=False)

    # 2) Create logged data for OPE
    log_df, env_for_note = generate_logged_data(seed=SEED + 999, n=LOG_N, logger_eps=LOGGER_EPS)
    log_df.to_csv(OUT_DATA / "logged_interactions.csv", index=False)

    # 3) Train target policy from log and save it
    target_policy = train_policy_from_log_linucb(log_df)
    joblib.dump(target_policy, OUT_MODEL / "policy.joblib")

    # 4) OPE estimates vs simulator truth
    ips = estimate_ips(log_df, target_policy)
    dr = estimate_dr(log_df, target_policy, reward_model_policy=target_policy)
    truth = true_policy_value(env_seed=SEED + 999, policy=target_policy, n=12000)

    ope_df = pd.DataFrame([
        {"estimator": "IPS", "estimate": ips, "true_value": truth, "abs_error": abs(ips - truth)},
        {"estimator": "DR",  "estimate": dr,  "true_value": truth, "abs_error": abs(dr - truth)},
    ])
    ope_df.to_csv(OUT_REPORT / "ope_vs_truth.csv", index=False)

    print("Saved:")
    print(OUT_DATA / "logged_interactions.csv")
    print(OUT_MODEL / "policy.joblib")
    print(OUT_REPORT / "learning_curves.png")
    print(OUT_REPORT / "summary.csv")
    print(OUT_REPORT / "ope_vs_truth.csv")
    print("\nOPE vs truth:")
    print(ope_df.to_string(index=False))


if __name__ == "__main__":
    main()