# 07 — Multi-camera handover & fusion (paper extension)

[Back to modules index](../README.md)

| | |
|---|---|
| **Claim** | Static per-camera error calibration is insufficient because camera usefulness changes with position, individual detections, availability and calibration health; we combine a spatial GP prior, calibrated frame-level detector evidence and online camera-health monitoring, mapped into camera-specific covariance for fusion and planning. |
| **Status** | Data-independent library implemented + unit-tested. First REAL Gazebo pilot done (single pass, not paper evidence): both camera fixes verified live — camera_A works in batched GPU mode; camera_C's v2 projection calibration cuts its error 0.156→0.077 m vs GT. See [REAL_RUN_FINDINGS](../../experiments/multicamera_fusion_extension/REAL_RUN_FINDINGS_2026-07-21.md). Full multi-camera campaign still pending (needs a full traverse for camera_B + repeated runs). |
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

## Real-run findings (2026-07-21)
- [REAL_RUN_FINDINGS](../../experiments/multicamera_fusion_extension/REAL_RUN_FINDINGS_2026-07-21.md) — brick-by-brick live Gazebo: sim RTF, detector rate/accuracy tradeoff, and both camera fixes verified vs ground truth.
- [THROUGHPUT_DIAGNOSIS](../../experiments/multicamera_fusion_extension/THROUGHPUT_DIAGNOSIS_2026-07-21.md) — why 3 Hz is inference-bound on the P2000 (corrected by the real runs).

## baselines/
External methods reimplemented as comparisons for this contribution. The
Toro-Diz et al. baseline (static nearest-point covariance + CV Kalman) is coded
in [`toro_baseline.py`](../../src/reliability/reliability/toro_baseline.py); see
[`baselines/`](baselines/) for the paper notes and the B0–B9 comparison table.

## framings/
Candidate framings of our own paper centered on this contribution live in
[`framings/`](framings/).
