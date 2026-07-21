# Plan 01 — Toro-style calibrated-covariance baseline (B2a/B2b)

## Purpose
Exact re-implementation of the comparison method (Toro Diz et al.): per-camera
2×2 measurement covariance from the **nearest static calibration location**,
validated-FOV gating, constant-velocity Kalman fusion, 0.08 s time binning.
Every headline claim is "vs this baseline", so it must be reproduced faithfully
and share detections/timestamps/process model with all other methods.

## What exists / reuse
- `reliability.fusion.sequential_kalman_update_2d` + 2×2 helpers (matrix ops, SPD checks).
- `reliability.replay` frame stream (shared detections for all conditions).
- CV process model: add here once, reuse for every replay condition.

## New code
`src/reliability/reliability/toro_baseline.py` + `tests/reliability/test_toro_baseline.py`.

API sketch:
```python
@dataclass(frozen=True)
class CalibrationPoint:          # one static calibration location
    camera_id: str
    position_xy: tuple[float, float]
    covariance_xy: 2x2            # axis-specific, measured at standstill
    sample_count: int

class ToroCovarianceModel:
    def covariance_for(camera_id, position_xy) -> 2x2   # nearest point, no interp
    def in_validated_fov(camera_id, position_xy) -> bool # convex hull / declared region

def bin_observations(observations, bin_width_s=0.08) -> list[list[MapObservation]]  # B2a
# B2b = same covariance model, timestamped sequential updates (reuse fusion path)
```

Two variants are mandatory (§14): **B2a** exact 0.08 s binning, **B2b**
sequential timestamped updates — separates "reliability modelling" gains from
"better temporal handling" gains.

## Calibration-point data
Static standstill captures at surveyed grid poses per camera in the 4-cam world
(reuse teleport-sweep tooling from the commissioning study). Each point stores
mean residual + axis-specific covariance + count. Points and their measurement
uncertainty are documented — not treated as perfect poses (§6.3). Blocked on
projection_calibration_v3; the module itself is data-independent and testable
with synthetic points now.

## Gates
- Unit: nearest-point lookup correctness, FOV gate, binning boundary cases,
  PSD covariance passthrough, B2a vs B2b divergence on asynchronous fixtures.
- Fusion gate (§21): B2 reproduced before any comparison is quoted; same
  detections for every method; process noise Q tuned on validation data only
  and **shared** across all conditions (no per-method Q — the known
  "gains are just smoothing from a different Q" trap).
- Report their metrics too: RMSE, MAE, displacement std.
