"""
Day 31 — simulate traffic against Day 30 API + drift experiment (lean).

What it does:
1) Calls /decide for many contexts
2) Simulates reward with a known environment
3) Sends /feedback
4) Produces rolling reward plots:
   - static environment
   - drift environment (payoff shift midway)

Outputs:
- reports/day30/static_traffic.csv
- reports/day30/drift_traffic.csv
- reports/day30/drift_experiment.png
- reports/day30/latency.csv
- reports/day30/monitor_alerts.csv

Run service first:
  uvicorn service.app:app --port 8000

Run script:
PYTHONPATH=. python my-solutions/day-30-safe-deployment/solution.py
"""

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# -------------------------
# Config
# -------------------------
BASE_URL = "http://127.0.0.1:8000"
SEED = 42
N_STATIC = 3000
N_DRIFT = 3000
DRIFT_STEP = 1500
WINDOW = 100

# "True world" used only by this simulator
N_ARMS = 4
D = 5
COSTS = np.array([0.35, 0.22, 0.11, 0.03], dtype=float)

# Base environment weights
W1 = np.array([
    [ 0.9,  0.4, -0.2,  0.1,  0.0],
    [ 0.3, -0.2,  0.7,  0.1,  0.2],
    [-0.4,  0.8,  0.3, -0.1,  0.2],
    [ 0.1,  0.1, -0.1,  0.9, -0.4],
], dtype=float)

# Drifted weights (arm payoffs change)
W2 = np.array([
    [ 0.2, -0.3,  0.8,  0.2,  0.1],
    [ 0.8,  0.1, -0.2,  0.3, -0.2],
    [-0.2,  0.2,  0.1, -0.6,  0.8],
    [ 0.3,  0.7, -0.4,  0.1, -0.1],
], dtype=float)

OUT = Path("data/rfl/reports/day30")
OUT.mkdir(parents=True, exist_ok=True)


# -------------------------
# Env helpers
# -------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sample_context(rng):
    x = rng.normal(0, 1, size=D)
    return np.clip(x, -3, 3)


def expected_reward(x, arm, W):
    p = sigmoid(W[arm] @ x)
    return p - COSTS[arm]


def sample_reward(x, arm, W, rng):
    p = sigmoid(W[arm] @ x)
    s = 1.0 if rng.random() < p else 0.0
    return float(s - COSTS[arm]), int(s)



def decide(x):
    payload = {f"x{i}": float(x[i]) for i in range(D)}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/decide", json=payload, timeout=5)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    body = r.json()
    return body, dt_ms


def feedback(decision_id, reward):
    payload = {"decision_id": decision_id, "reward": float(reward)}
    t0 = time.perf_counter()
    r = requests.post(f"{BASE_URL}/feedback", json=payload, timeout=5)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return dt_ms



def run_traffic(n_steps, drift=False, drift_step=None, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    lat_decide = []
    lat_feedback = []

    for t in range(n_steps):
        x = sample_context(rng)
        W = W1 if (not drift or t < drift_step) else W2

        # API decision
        d, ms_decide = decide(x)
        a = int(d["action"])
        did = d["decision_id"]
        prop = float(d["propensity"])

        # Simulated "world" reward
        rew, success = sample_reward(x, a, W, rng)
        mu = expected_reward(x, a, W)

        # best possible under current world
        mu_all = np.array([expected_reward(x, j, W) for j in range(N_ARMS)])
        best_mu = float(np.max(mu_all))
        regret = best_mu - mu

        # send feedback
        ms_fb = feedback(did, rew)

        lat_decide.append(ms_decide)
        lat_feedback.append(ms_fb)

        rows.append({
            "t": t,
            "phase": "pre_drift" if (not drift or t < drift_step) else "post_drift",
            "action": a,
            "propensity": prop,
            "reward": rew,
            "success": success,
            "expected_reward_for_action": mu,
            "best_expected_reward": best_mu,
            "instant_regret": regret,
            "decide_latency_ms": ms_decide,
            "feedback_latency_ms": ms_fb,
            "decision_id": did,
            "x0": x[0], "x1": x[1], "x2": x[2], "x3": x[3], "x4": x[4],
        })

    df = pd.DataFrame(rows)
    # rolling metrics
    df["rolling_reward"] = df["reward"].rolling(WINDOW, min_periods=WINDOW).mean()
    df["rolling_regret"] = df["instant_regret"].rolling(WINDOW, min_periods=WINDOW).mean()
    df["cum_reward"] = df["reward"].cumsum()
    df["cum_regret"] = df["instant_regret"].cumsum()

    lat = {
        "decide_p50_ms": float(np.percentile(lat_decide, 50)),
        "decide_p95_ms": float(np.percentile(lat_decide, 95)),
        "feedback_p50_ms": float(np.percentile(lat_feedback, 50)),
        "feedback_p95_ms": float(np.percentile(lat_feedback, 95)),
    }
    return df, lat


def monitor_alerts(df, drop_threshold=0.08):
    """
    Simple monitor:
    alert when rolling reward drops more than drop_threshold
    below baseline (baseline = mean rolling_reward over first stable window).
    """
    rr = df["rolling_reward"].copy()
    base = rr.dropna().iloc[:max(20, WINDOW // 2)].mean()
    alerts = []

    for i, v in rr.items():
        if np.isnan(v):
            continue
        if (base - v) > drop_threshold:
            alerts.append({
                "t": int(df.loc[i, "t"]),
                "rolling_reward": float(v),
                "baseline": float(base),
                "drop": float(base - v),
                "alert": "reward_drop",
            })
    return pd.DataFrame(alerts)


def main():
    # quick health check
    h = requests.get(f"{BASE_URL}/health", timeout=5)
    h.raise_for_status()
    print("Service health:", h.json())

    # 1) static traffic
    static_df, lat_static = run_traffic(
        n_steps=N_STATIC, drift=False, drift_step=None, seed=SEED
    )
    static_df.to_csv(OUT / "static_traffic.csv", index=False)

    # 2) drift traffic
    drift_df, lat_drift = run_traffic(
        n_steps=N_DRIFT, drift=True, drift_step=DRIFT_STEP, seed=SEED + 1
    )
    drift_df.to_csv(OUT / "drift_traffic.csv", index=False)

    # 3) drift plot
    plt.figure(figsize=(9, 5))
    plt.plot(static_df["t"], static_df["rolling_reward"], label="static env rolling reward", alpha=0.8)
    plt.plot(drift_df["t"], drift_df["rolling_reward"], label="drift env rolling reward", alpha=0.9)
    plt.axvline(DRIFT_STEP, linestyle="--", linewidth=1.5, color="red", label="drift starts")
    plt.title(f"Rolling reward (window={WINDOW}) — static vs drift traffic")
    plt.xlabel("Step")
    plt.ylabel("Rolling reward")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT / "drift_experiment.png", dpi=160)
    plt.close()

    # 4) monitor alerts on drift run
    alerts = monitor_alerts(drift_df, drop_threshold=0.08)
    alerts.to_csv(OUT / "monitor_alerts.csv", index=False)

    # 5) latency summary
    lat_rows = [
        {"run": "static", **lat_static},
        {"run": "drift", **lat_drift},
    ]
    pd.DataFrame(lat_rows).to_csv(OUT / "latency.csv", index=False)

    # 6) tiny summary markdown
    pre = drift_df[drift_df["t"] < DRIFT_STEP]["reward"].mean()
    post = drift_df[drift_df["t"] >= DRIFT_STEP]["reward"].mean()
    summary = f"""# Day 31 Drift Summary

- Service: `{BASE_URL}`
- Static run steps: {N_STATIC}
- Drift run steps: {N_DRIFT}
- Drift starts at step: {DRIFT_STEP}
- Rolling window: {WINDOW}

## Measured reward shift
- Mean reward pre-drift: {pre:.4f}
- Mean reward post-drift: {post:.4f}
- Delta (post - pre): {post - pre:.4f}

## Monitor
- Alert count: {len(alerts)}

## Latency
See `latency.csv` for p50/p95 decide and feedback latency.

## Notes
- This is a simulated environment.
- No-detection style caveat equivalent here: absence of negative feedback does not prove policy optimality.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")

    print("Saved:")
    print(OUT / "static_traffic.csv")
    print(OUT / "drift_traffic.csv")
    print(OUT / "drift_experiment.png")
    print(OUT / "latency.csv")
    print(OUT / "monitor_alerts.csv")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()