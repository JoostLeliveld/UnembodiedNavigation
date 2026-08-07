# Figures

Every plot this project makes decisions from. Grouped by `registry.yaml` experiment ID.

Raw output stays in the ignored `logs/studies/`; this tree is the tracked, browsable
copy. Each figure carries a `.provenance.json` naming its source and SHA-256.

Regenerate with `python3 scripts/research/promote_figures.py`.

| Experiment | Study | Figures |
|---|---|---|
| EXP-PRECISION | `achievable_precision_map` | 2 |
| EXP-BELIEF | `bayesian_filter_showcase` | 6 |
| EXP-DRIFT | `calibration_drift_lifecycle` | 4 |
| EXP-HIT-MISS | `efe_hit_miss_mixture` | 4 |
| EXP-BIAS | `external_camera_bias_model` | 18 |
| _unmapped_ | `fused_observation_model` | 8 |
| EXP-COMMISSION | `multicamera_commissioning_bigwarehouse` | 4 |
| EXP-CAM-MGMT | `multicamera_fusion_extension` | 2 |
| EXP-NET-COMMISSION | `network_commissioning_realism` | 6 |
| EXP-RCOND | `operational_residual_rcond` | 6 |
| EXP-PIXEL-GROUND | `pixel_ground_path` | 4 |
| EXP-PLANNER-BRANCH | `planner_covariance_branching` | 6 |
| EXP-PROJ-AMP | `projection_amplification` | 32 |
| _unmapped_ | `single_camera_uigp_reliability` | 5 |

## Experiments with no figures

- `EXP-CL-CAL`
- `EXP-USABLE`
