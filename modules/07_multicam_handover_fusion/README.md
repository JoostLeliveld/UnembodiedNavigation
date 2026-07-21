# 07 — Multi-camera handover & fusion (paper extension)

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | Static per-camera error calibration is insufficient because camera usefulness changes with position, individual detections, availability and calibration health; we combine a spatial GP prior, calibrated frame-level detector evidence and online camera-health monitoring, mapped into camera-specific covariance for fusion and planning. |
| **Status** | Status report — data-independent library implemented + unit-tested; empirical results blocked on the detector retrain (real multi-camera data). No multi-camera result is claimed yet. |
| **Chapter** | [08 — large-warehouse scaling](../../research_story/08_large_warehouse_scaling/) (ACTIVE) / [09 — multicamera handover & fusion](../../research_story/09_multicamera_handover_fusion/) (PLUMBING+PILOT) |

This is the home of the **paper extension** over Toro-Diz et al. The full plan,
roadmap, and per-module implementation plans live in the owning study:
[`../../experiments/multicamera_fusion_extension/`](../../experiments/multicamera_fusion_extension/)
([ROADMAP](../../experiments/multicamera_fusion_extension/plans/ROADMAP.md)).

## What it computes (three reliability quantities kept separate)
1. Spatial availability `a_i(s)` — GP classifier.
2. Conditional usability `q_i(s)` — GP classifier.
3. Anisotropic conditional covariance `R_cond_i(s)`.
Plus instantaneous evidence: calibrated confidence, EWMA camera health, stacked
trust — fused via robust/Joseph sequential updates and information-aware
selection. Planning uses only spatially predictable reliability, never the
current frame's confidence copied across the horizon.

## Where it lives (runtime library — all unit-tested)
- [`toro_baseline.py`](../../src/reliability/reliability/toro_baseline.py) · [`conditional_covariance.py`](../../src/reliability/reliability/conditional_covariance.py) · [`confidence_calibration.py`](../../src/reliability/reliability/confidence_calibration.py) · [`trust_stacker.py`](../../src/reliability/reliability/trust_stacker.py) · [`health_ewma.py`](../../src/reliability/reliability/health_ewma.py) · [`fusion.py`](../../src/reliability/reliability/fusion.py) (v2 primitives) · [`planning_covariance.py`](../../src/reliability/reliability/planning_covariance.py)
- Statistics backbone: [`campaign_statistics.py`](../../src/reliability/reliability/campaign_statistics.py)
- Evaluators / replay drivers: [`../../experiments/multicamera_fusion_extension/tools/`](../../experiments/multicamera_fusion_extension/tools/)

## baselines/
External methods reimplemented as comparisons for this contribution. The
Toro-Diz et al. baseline (static nearest-point covariance + CV Kalman) is coded
in [`toro_baseline.py`](../../src/reliability/reliability/toro_baseline.py); see
[`baselines/`](baselines/) for the paper notes and the B0–B9 comparison table.

## framings/
Candidate framings of our own paper centered on this contribution live in
[`framings/`](framings/).
