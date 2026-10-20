"""
Day 30 — Lean FastAPI policy service.

Endpoints:
- GET  /health
- POST /decide
- POST /feedback

Artifacts:
- loads models/policy.joblib
- writes logs/decisions.jsonl
- writes logs/feedback.jsonl

Run:
  uvicorn app.service.app:app --reload --port 8000
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import uuid
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat
from typing import Annotated
import os, sys

# -------------------------
# Config (edit here)
# -------------------------
MODEL_PATH = Path("models/policy.joblib")
LOG_DIR = Path("data/rfl/logs")
DECISION_LOG = LOG_DIR / "decisions.jsonl"
FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"
POLICY_VERSION = "day28-linucb-v1"

N_ARMS = 4   # must match day28 policy
D = 5        # context size x0..x4
EXPLORATION_FLOOR = 0.05  # minimum non-zero propensity support

LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load policy once at startup
# -------------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Missing {MODEL_PATH}. Run Day 28 first.")
POLICY = joblib.load(MODEL_PATH)

# In-memory index for simple feedback validation
DECISION_INDEX = {}  # decision_id -> record


# -------------------------
# Schemas
# -------------------------

class DecideRequest(BaseModel):
    x0: Annotated[float, Field(strict=True, ge=-10, le=10)]
    x1: Annotated[float, Field(strict=True, ge=-10, le=10)]
    x2: Annotated[float, Field(strict=True, ge=-10, le=10)]
    x3: Annotated[float, Field(strict=True, ge=-10, le=10)]
    


class DecideResponse(BaseModel):
    decision_id: str
    action: int
    propensity: float
    policy_version: str


class FeedbackRequest(BaseModel):
    decision_id: str = Field(min_length=8)
    reward: Annotated[float, Field(strict=True, ge=-5, le=5)]


class FeedbackResponse(BaseModel):
    ok: bool
    decision_id: str


# -------------------------
# Helpers
# -------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def as_context(req: DecideRequest):
    return np.array([req.x0, req.x1, req.x2, req.x3, req.x4], dtype=float)


def safe_propensity(action: int):
    """
    We expose stochastic propensity with exploration floor so logs remain usable for OPE.
    This is an epsilon-floor mixture around deterministic policy action:
      P(chosen action)=1-floor + floor/K
      others=floor/K
    """
    return (1.0 - EXPLORATION_FLOOR) + (EXPLORATION_FLOOR / N_ARMS), EXPLORATION_FLOOR / N_ARMS


def append_jsonl(path: Path, obj: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# App
# -------------------------
app = FastAPI(title="Day30 Policy Service", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "policy_version": POLICY_VERSION,
        "model_path": str(MODEL_PATH),
        "timestamp": now_iso(),
    }


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    x = as_context(req)

    # deterministic policy action from Day 28 object
    try:
        action = int(POLICY.select(x))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"policy select failed: {e}")

    if not (0 <= action < N_ARMS):
        raise HTTPException(status_code=500, detail=f"policy returned invalid action {action}")

    p_action, _ = safe_propensity(action)

    decision_id = str(uuid.uuid4())
    record = {
        "timestamp": now_iso(),
        "decision_id": decision_id,
        "policy_version": POLICY_VERSION,
        "context": {
            "x0": req.x0, "x1": req.x1, "x2": req.x2, "x3": req.x3, "x4": req.x4
        },
        "action": action,
        "propensity": p_action,
    }

    append_jsonl(DECISION_LOG, record)
    DECISION_INDEX[decision_id] = record

    return DecideResponse(
        decision_id=decision_id,
        action=action,
        propensity=p_action,
        policy_version=POLICY_VERSION,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if req.decision_id not in DECISION_INDEX:
        raise HTTPException(status_code=404, detail="unknown decision_id")

    rec = {
        "timestamp": now_iso(),
        "decision_id": req.decision_id,
        "reward": float(req.reward),
    }
    append_jsonl(FEEDBACK_LOG, rec)

    return FeedbackResponse(ok=True, decision_id=req.decision_id)