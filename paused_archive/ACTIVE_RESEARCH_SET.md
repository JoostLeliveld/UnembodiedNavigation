# Active research set — ICRA focus

## One active question

> Does accounting for persistent, camera-specific correlated error produce an honest
> belief, and does that honesty improve closed-loop safety or progress compared with a
> conventional overconfident filter?

## Active evidence chain

```text
four-camera recorded observations
→ projection residual audit
→ gated v2/v3 calibration comparison
→ correlated-bias / self-confirmation diagnosis
→ per-camera correlation floor + leave-one-camera-out check
→ calibration-expiry monitor
→ achievable-precision field
→ one matched closed-loop campaign
```

## Active experiment directories

- `experiments/external_camera_bias_model/`
- `experiments/projection_amplification/`
- `experiments/operational_residual_rcond/`
- `experiments/network_commissioning_realism/`
- `experiments/bayesian_filter_showcase/`
- `experiments/calibration_drift_lifecycle/`
- `experiments/achievable_precision_map/`
- `experiments/planner_covariance_branching/`
- `experiments/efe_hit_miss_mixture/` — supporting correctness only
- `experiments/closed_loop_calibration/` — immediate campaign
- `experiments/multicamera_commissioning_bigwarehouse/` — source captures/calibration
- `experiments/multicamera_fusion_extension/` — Toro baseline, manager, and replay support

## Active runtime code

- `src/reliability/reliability/observation_model.py`
- `src/reliability/reliability/operational_residual.py`
- `src/reliability/reliability/projection.py`
- `src/reliability/reliability/{fusion,handover,health,health_ewma,camera_manager}.py`
- `src/reliability/reliability/{contracts,firewall,campaign_statistics}.py`
- `src/planning/planning/core/{belief_correction,casadi_efe}.py`
- `src/planning/planning/{nodes,planners}/`
- `src/experiments/` and the frozen campaign runner/metrics paths
- `src/perception/` only to the extent required by the frozen evidence runtime
- `src/sim/` for `warehouse_aws` and `warehouse_full_4cam`

## Active data/artifacts retained in `logs/`

- `logs/visibility_comparison/honest_campaign_v1/`
- `logs/visibility_comparison/whitenoise_campaign_v1/`
- `logs/visibility_comparison/_paper_runs/`
- `logs/visibility_comparison/warehouse_visibility_capture_v1/`
- `logs/visibility_comparison/warehouse_visibility_targets_v1/`
- `logs/visibility_comparison/warehouse_visibility_gp_v1/`
- `logs/visibility_comparison/warehouse_visibility_campaign_v1/`
- `logs/visibility_comparison/spawn_grid_20260727/` — required by the current closed-loop
  configs
- `logs/perception_models/warehouse_yolo_detector_v1/`
- `logs/perception_models/warehouse_yolo_detector_4cam_v3_960/`
- all core August study result folders under `logs/studies/`
- `paper_artifacts/` and all evidence manifests

## Frozen predecessor material

- `RobotControlExternalCamera/` is Paper 1 and its 60-run C0/C1/C2 campaign.
- `thesis-report/` is the historical submitted-paper source.
- `research_story/00_problem_and_existing_baseline/` remains a cited anchor, not an active
  experiment programme.

## Current stop condition

No new research branch opens until:

1. the two detector-runtime contract tests pass;
2. the causal closed-loop arms are frozen;
3. the 30-run campaign is complete or returns a documented null;
4. the ICRA results table and six essential figures are generated.

