
from fastapi.testclient import TestClient
import os, sys
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from service.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "policy_version" in body


def test_decide_valid():
    payload = {"x0": 0.1, "x1": -0.3, "x2": 1.2, "x3": 0.0, "x4": -1.0}
    r = client.post("/decide", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "decision_id" in body
    assert 0 <= body["action"] < 4
    assert 0.0 < body["propensity"] <= 1.0


def test_decide_invalid():
    # x0 out of accepted range [-10,10]
    payload = {"x0": 100.0, "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0}
    r = client.post("/decide", json=payload)
    assert r.status_code == 422


def test_feedback_unknown_id():
    r = client.post("/feedback", json={"decision_id": "missing-id", "reward": 1.0})
    assert r.status_code == 404


def test_feedback_roundtrip():
    d = client.post("/decide", json={"x0": 0.0, "x1": 0.1, "x2": -0.2, "x3": 0.3, "x4": 0.0})
    did = d.json()["decision_id"]

    f = client.post("/feedback", json={"decision_id": did, "reward": 0.8})
    assert f.status_code == 200
    assert f.json()["ok"] is True