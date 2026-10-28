# RL + Deployment Block (Days 27–31) — Overview

This document is the index for the final challenge block:
- Day 27: Multi-armed bandits
- Day 28: Contextual bandits + offline policy evaluation
- Day 29: Q-learning + SARSA (Gymnasium)
- Day 30: Policy serving API (FastAPI)
- Day 31: Drift simulation + monitoring

---

## Folder structure (expected)

```text
src/
  bandits/
    env.py
    agents.py
    contextual.py
  serving/
    guardrails.py

my-solutions/
  day-27-bandits/
    challenge.md
    day27_bandits.py
  day-28-contextual-bandits-ope/
    challenge.md
    day28_contextual_bandits_ope.py
  day-29-mdp-q-learning/
    challenge.md
    day29_mdp_qlearning.py
  day-30-serve-policy-api/
    challenge.md
    app.py
    test_api.py
    requirements.txt
    Dockerfile
  day-31-safe-deployment/
    challenge.md
    day31_simulate_traffic.py

models/
logs/
data/
reports/
```

---

## Environment setup (conda recommended)

From repo root:

```bash
conda activate <your-env>
pip install -r my-solutions/day-30-serve-policy-api/requirements.txt
pip install numpy pandas matplotlib gymnasium requests scikit-learn joblib pytest
```

---

## Run order (important)

## 1) Day 27
```bash
PYTHONPATH=. python my-solutions/day-27-bandits/day27_bandits.py
```

Expected key artifacts:
- `reports/day27/results.csv`
- `reports/day27/regret_curves.png`

---

## 2) Day 28
```bash
PYTHONPATH=. python my-solutions/day-28-contextual-bandits-ope/day28_contextual_bandits_ope.py
```

Expected key artifacts:
- `data/logged_interactions.csv`
- `models/policy.joblib` ✅ required by Day 30
- `reports/day28/ope_vs_truth.csv`

---

## 3) Day 29
```bash
PYTHONPATH=. python my-solutions/day-29-mdp-q-learning/day29_mdp_qlearning.py
```

Expected key artifacts:
- `reports/day29/learning_curves.png`
- `reports/day29/policy_grid_qlearning.txt`
- `reports/day29/policy_grid_sarsa.txt`

---

## 4) Day 30 (start API)
```bash
PYTHONPATH=. uvicorn app:app --app-dir my-solutions/day-30-serve-policy-api --reload --port 8000
```

Smoke test:
```bash
curl http://127.0.0.1:8000/health
```

Run tests:
```bash
PYTHONPATH=. pytest my-solutions/day-30-serve-policy-api -q
```

Expected runtime artifacts:
- `logs/decisions.jsonl`
- `logs/feedback.jsonl`

---

## 5) Day 31 (while Day 30 API is running)
```bash
PYTHONPATH=. python my-solutions/day-31-safe-deployment/day31_simulate_traffic.py
```

Expected key artifacts:
- `reports/day31/drift_experiment.png`
- `reports/day31/latency.csv`
- `reports/day31/monitor_alerts.csv`
- `reports/day31/summary.md`

---

## Troubleshooting quick guide

## API won’t start
Most common cause:
- `models/policy.joblib` missing or stale

Fix:
1. Re-run Day 28
2. Ensure `src/bandits/contextual.py` exists and is imported by service (`LinUCB` must be importable during joblib load)

Debug:
```bash
PYTHONPATH=. python -c "import app" 
# if running from day-30 folder, else:
PYTHONPATH=. python -c "import my-solutions.day-30-serve-policy-api.app"
```

(If import path with hyphens breaks, use the first style with `--app-dir` and direct `app` import.)

## Day 31 fails requests
- Verify Day 30 is still running on port 8000
- Check `/health` response
- Update `BASE_URL` in Day 31 script if using a different port

## Tests fail due to paths
Run tests from repo root and keep `PYTHONPATH=.`.

---

## What “done” looks like

- [ ] Bandit comparison completed with regret curves (Day 27)
- [ ] Contextual policy + OPE vs truth completed (Day 28)
- [ ] Q-learning vs SARSA curves + policy grids completed (Day 29)
- [ ] Decision API serving policy with propensity logging (Day 30)
- [ ] Drift simulation with monitoring + latency report (Day 31)

---

## Notes / constraints

- This block is **simulation-based**, not production deployment.
- Propensity logging is required for trustworthy offline evaluation.
- Keep model/data/log directories in `.gitignore` when needed.
- If short on time, Docker is optional; local API run is enough.