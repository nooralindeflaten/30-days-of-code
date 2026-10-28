from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.bandits.agents import EpsilonGreedyAgent, UCBAgent, ThompsonSamplingBernoulliAgent
from src.bandits.env import BernoulliBandit


HORIZON = 2000
N_SEEDS = 30
EPSILON = 0.1


PROBS = [0.80, 0.67, 0.57, 0.42, 0.30]
COSTS = [0.36, 0.22, 0.12, 0.06, 0.01]
OUT = Path("data/rfl/reports/day26")

def run_one(agent_name, seed, probs, costs):
    env = BernoulliBandit(probs=probs, costs=costs, seed=seed)

    if agent_name == "epsilon_greedy":
        agent = EpsilonGreedyAgent(n_arms=env.n_arms, epsilon=EPSILON, seed=seed)
        needs_success = False
    elif agent_name == "ucb1":
        agent = UCBAgent(n_arms=env.n_arms)
        needs_success = False
    elif agent_name == "thompson":
        agent = ThompsonSamplingBernoulliAgent(n_arms=env.n_arms, costs=costs, seed=seed + 1000)
        needs_success = True
    else:
        raise ValueError(agent_name)

    regrets = np.zeros(HORIZON, dtype=float)
    actions = np.zeros(HORIZON, dtype=int)
    rewards = np.zeros(HORIZON, dtype=float)

    for t in range(HORIZON):
        a = agent.select_arm()
        r, s = env.pull(a)

        if needs_success:
            agent.update(a, r, success=s)
        else:
            agent.update(a, r)

        actions[t] = a
        rewards[t] = r
        regrets[t] = env.regret(a)

    out = {
        "regret": regrets,
        "cum_regret": np.cumsum(regrets),
        "actions": actions,
        "rewards": rewards,
        "best_arm": env.best_arm,
        "best_mean": env.best_mean,
        "true_means": env.true_means.copy(),
    }
    return out





def main():
    OUT.mkdir(parents=True, exist_ok=True)
    AGENTS = {"epsilon_greedy": "Epsilon-Greedy", "ucb1": "UCB1", "thompson": "Thompson Sampling"}
    rows, curves = [], {k: [] for k in AGENTS}

    for name in AGENTS:
        for seed in range(N_SEEDS):
            res = run_one(name, seed, PROBS, COSTS)
            curves[name].append(res["cum_regret"])
            rows.append({"agent": name, "seed": seed,
                        "final_regret": res["cum_regret"][-1],
                        "best_arm_share": np.mean(res["actions"] == res["best_arm"])})

    pd.DataFrame(rows).groupby("agent").agg(["mean", "std"]).to_csv(OUT / "results.csv")

    plt.figure(figsize=(10, 6))
    for name, label in AGENTS.items():
        arr = np.vstack(curves[name])
        m, s = arr.mean(0), arr.std(0)
        plt.plot(m, label=label)
        plt.fill_between(range(HORIZON), m - s, m + s, alpha=0.15)
    plt.xlabel("Time step"); plt.ylabel("Cumulative regret")
    plt.title("Cumulative regret (mean ± std over seeds)")
    plt.legend(); plt.grid(alpha=0.25)
    plt.savefig(OUT / "regret_curves.png", dpi=160); plt.close()
    
if __name__ == "__main__":
    main()
        