# Baselines for multi-camera handover & fusion

[Back to module 07](../README.md)

External methods reimplemented on **our** setup (identical detections,
timestamps, process model, evaluation trajectories) so improvements are
attributable. This is where "recreating a published method to see how relevant
it is to our setup" lives for this contribution.

## Toro-Diz et al. — static per-camera calibrated covariance
Assigns each measurement a 2×2 covariance from the nearest static calibration
point, rejects measurements outside the validated FOV, fuses with a
constant-velocity Kalman filter.

- Reimplementation: [`../../../src/reliability/reliability/toro_baseline.py`](../../../src/reliability/reliability/toro_baseline.py) (nearest-point lookup, convex-hull FOV gate, 0.08 s binning `B2a` / sequential `B2b`, CV process model) — unit-tested in [`../../../tests/reliability/test_toro_baseline.py`](../../../tests/reliability/test_toro_baseline.py).
- Reproduction requirements + the two Toro variants: [ROADMAP §14](../../../experiments/multicamera_fusion_extension/plans/ROADMAP.md) and [plan 01](../../../experiments/multicamera_fusion_extension/plans/01_toro_baseline.md).

## The B0–B9 comparison ladder
The full baseline table (best-single / constant-R / Toro / confidence-only /
GP-only / GP+confidence / full / selection variants / oracle) and its mapping to
existing replay condition IDs is pre-registered in
[ROADMAP §14 + baseline table](../../../experiments/multicamera_fusion_extension/plans/ROADMAP.md).
Head-to-head numbers land here once real commissioning data exist.
