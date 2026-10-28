# Day 31 Drift Summary

- Service: `http://127.0.0.1:8000`
- Static run steps: 3000
- Drift run steps: 3000
- Drift starts at step: 1500
- Rolling window: 100

## Measured reward shift
- Mean reward pre-drift: 0.3991
- Mean reward post-drift: 0.3903
- Delta (post - pre): -0.0088

## Monitor
- Alert count: 134

## Latency
See `latency.csv` for p50/p95 decide and feedback latency.

## Notes
- This is a simulated environment.
- No-detection style caveat equivalent here: absence of negative feedback does not prove policy optimality.
