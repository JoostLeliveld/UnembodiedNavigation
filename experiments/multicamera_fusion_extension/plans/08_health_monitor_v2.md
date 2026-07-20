# Plan 08 — Online camera-health monitor v2 (§11, RQ4/H4, E6)

## Purpose
Detect a camera that has physically degraded (moved, delayed, dropped) — the
thing spatial reliability cannot see. Continuous health `h ∈ (0,1)` + debounced
state machine, from operational signals only.

## What exists / reuse
- `reliability.health.CameraHealthMachine` — per-camera TRACKING/DEGRADED/
  LOST/REACQUIRING from misses/age/NIS/reliability. Keep as the availability
  layer; v2 adds the *calibration-health* layer.
- `reliability.overlap` — pairwise disagreement stats (p90, outlier rate,
  systematic bias) = the cross-camera evidence feed.
- NIS gating precedent: 2-dof 0.99 threshold 9.21 (frozen, do not retune).

## New code
`src/reliability/reliability/health_ewma.py` + tests.

```python
class InnovationHealthMonitor:      # per camera
    update(nis, innovation_uv, dropped: bool, cross_disagreement: float|None)
    # EWMA of NIS:      m ← (1−ρ)m + ρ·d²
    # EWMA of innovation MEAN (bias detector): b ← (1−ρb)b + ρb·ν
    # health h = σ(η0 − η1·max(0, m−m0) − η2·|b| − η3·drop_rate − η4·cross)
class HealthDebouncer:              # HEALTHY→SUSPECT→DEGRADED→RECOVERING→HEALTHY
    # M_s/M_d/M_r/M_h window counts; response policy:
    # SUSPECT = covariance inflation only; DEGRADED = hard reject; RECOVERING = slow re-entry
def isolate_suspect_camera(pairwise_d2: mapping) -> camera_id|None
    # ≥3 cameras: odd-one-out from pairwise consistency; 2 cameras: ambiguous → None + caveat flag
```

Persistent innovation mean `b` is the calibration-bias detector (a biased
camera has consistent-direction innovations even when NIS looks tolerable).

## Gates (§21 health-monitor gate, E6 criteria)
- No GT signal anywhere (firewall test).
- On controlled calibration-ablation replays (plan 11 drift sweep): detection
  delay acceptable, isolation accuracy high, false-isolation of healthy
  cameras LOW — **critical failure rule**: if it removes healthy cameras more
  often than it isolates faulty ones, it is not used downstream.
- Single rejected observation never flips state (debounce test).
- Unit tests: EWMA algebra, bias detection on synthetic biased stream,
  state-machine transition table, 2-camera ambiguity returns None,
  monotonicity (worse evidence never raises h).
