# Phase 0 Reliability Contract Schema

Phase 0 introduces a narrow, additive schema layer. The purpose is to stop leakage
before learned reliability models or dataset exporters are added.

## OperationalReliabilitySample

This record is allowed to become a model feature row.

| Field | Unit | Notes |
| --- | --- | --- |
| `schema_version` | text | Must be `phase0.v1`. |
| `sample_id` | text | Stable sample key. |
| `timestamp_s` | s | Capture or log time. |
| `run_id`, `task_id`, `seed` | ids | Run provenance. |
| `image_ref`, `image_hash` | text | Image reference and content/artifact hash. |
| `detector_result` | mapping | Detection flag, score, bbox/mask metrics, source code, timing. |
| `selected_pixel` | px | Optional `(u, v)` selected by the detector pipeline. |
| `projection_valid` | bool | Whether selected pixel projection succeeded. |
| `measurement_age_s`, `measurement_stale` | s/bool | Runtime staleness state. |
| `odometry` | mapping | Online odometry summary, no GT. |
| `state_estimate` | mapping | Online state estimate summary, no GT. |
| `belief` | mapping | Planner belief summary and covariance terms, no GT. |
| `camera_relative_range_m` | m | Camera-relative range feature. |
| `image_location` | mapping | Normalized image-location features such as `u_norm`, `v_norm`. |
| `recent_detector_history` | bool list | Recent detector availability booleans. |
| `config_hash`, `artifact_hashes` | text/mapping | Runtime provenance. |
| `metadata` | mapping | Non-oracle implementation metadata only. |

Unknown fields are rejected. Keys containing GT, oracle, collision, clearance,
geometry-breach, localization-error, or final-outcome tokens are rejected recursively.

## EvaluationOnlySample

This record is a label/audit row. It must never be joined into operational features
without explicitly selecting target labels for offline training/evaluation.

| Field | Unit | Notes |
| --- | --- | --- |
| `schema_version` | text | Must be `phase0.v1`. |
| `sample_id` | text | Joins to operational row only in labeled offline contexts. |
| `timestamp_s` | s | Evaluation timestamp. |
| `run_id`, `task_id`, `seed` | ids | Run provenance. |
| `gazebo_ground_truth_pose` | mapping | Evaluation-only true pose. |
| `ground_truth_projected_pixel` | px | Evaluation-only projected GT pixel. |
| `ground_truth_localization_error_m` | m | Evaluation label. |
| `clearance_m` | m | Evaluation-only clearance metric. |
| `collision`, `geometry_breach` | bool | Safety outcome labels. |
| `final_task_outcome` | text | Run outcome label. |
| `metrics` | mapping | Additional evaluation-only metrics. |

## CameraObservation

This record is the Phase 1 camera-facing contract. It lets the current single
external camera be represented as one member of a future camera network without
adding a second camera or changing the active campaign.

Array-like inputs are normalized to JSON-safe immutable tuples.

| Field | Unit | Notes |
| --- | --- | --- |
| `schema_version` | text | Must be `phase0.v1` until the contract is intentionally bumped. |
| `camera_id` | text | Use `camera_A` for the current single camera adapter. |
| `timestamp_s` | s | Observation timestamp. |
| `pixel_uv` | px | Optional `(u, v)` selected localization pixel. Required when `detection_valid` is true. |
| `detection_valid` | bool | Detector produced a usable candidate. |
| `detector_score` | probability | Detector confidence in `[0, 1]`. |
| `bbox_xyxy` | px | Optional `(x_min, y_min, x_max, y_max)` bbox. |
| `measurement_age_s` | s | Non-negative observation age. |
| `calibration_id` | text | Calibration/homography provenance. |
| `image_frame_id` | text | Camera/image frame identifier. |
| `conditional_cov_uv` | `px^2` | 2x2 SPD pixel covariance conditioned on this observation path. |
| `availability_probability` | probability | Per-camera availability probability. |
| `association_probability` | probability | Per-camera association/identity confidence. |

Unknown evaluation-only fields such as `oracle_visible`, `ground_truth_pose`, or
`localization_error_m` are rejected.

## CameraQuality

This is the compact quality output consumed by future camera selection/fusion
policies.

| Field | Unit | Notes |
| --- | --- | --- |
| `camera_id` | text | Camera identifier. |
| `p_available` | probability | Estimated availability. |
| `conditional_cov_uv` | `px^2` | 2x2 SPD conditional covariance. |
| `association_confidence` | probability | Association confidence in `[0, 1]`. |
| `epistemic_score` | scalar | Non-negative uncertainty/novelty score. |
| `stale` | bool | Whether the observation stream is stale. |
| `source_model` | text | Provenance such as `fixed_R`, `detector_score`, or `single_camera_gp`. |

## CameraReliabilityProvider

The operational provider interface is:

```python
query(camera_id: str, belief_xy: Sequence[float], timestamp_s: float) -> CameraQuality
```

Implemented provider types:

| Provider | Inputs | Leakage status |
| --- | --- | --- |
| `FixedCameraReliabilityProvider` | Camera ID, fixed probability, fixed covariance | Operational-safe. |
| `GridMapReliabilityProvider` | Camera ID, `belief_xy`, `.npz` grid with `xs`, `ys`, probability map | Operational-safe when artifact was trained without evaluation fields. |
| `MultiCameraReliabilityProvider` | Camera ID dispatch over provider map | Operational-safe. |

The grid provider follows the current GP artifact convention:
`P_conservative_plan_map` has shape `(len(ys), len(xs))`. Out-of-support queries
are explicit: `min`, `clamp`, or `raise`.

## ReliabilityPrediction

`ReliabilityPrediction` is model output before covariance adaptation.

| Field | Unit | Notes |
| --- | --- | --- |
| `p_available` | probability | Estimated observation availability. |
| `raw_probability` | probability | Uncalibrated score when present. |
| `calibrated_probability` | probability | Calibrated availability score. |
| `conditional_quality` | mapping | Optional non-oracle quality summary. |
| `epistemic_uncertainty` | scalar | Non-negative uncertainty score. |
| `staleness_s` | s | Prediction age/staleness. |

## UpdateCovariance

`UpdateCovariance` is for estimator updates that consume an available pixel
measurement.

| Field | Unit | Notes |
| --- | --- | --- |
| `matrix_px2` | `px^2` | 2x2 symmetric positive definite covariance. |
| `available` | bool | Whether the update observation exists. |
| `adapter_id` | text | Adapter/model provenance. |
| `epistemic_inflation` | scalar | Must be at least 1. |
| `staleness_inflation` | scalar | Must be at least 1. |

## PlanningCovariance

`PlanningCovariance` is planner-facing and separate from filter-update covariance
or no-go costs.

| Field | Unit | Notes |
| --- | --- | --- |
| `matrix_px2` | `px^2` | 2x2 symmetric positive definite planning covariance. |
| `p_available` | probability | Availability probability. |
| `conditional_covariance_px2` | `px^2` | Optional 2x2 SPD covariance conditioned on visibility. |
| `epistemic_score`, `staleness_score` | scalar | Non-negative diagnostics. |

## Leakage Firewall

The firewall configuration is
`src/reliability/config/leakage_firewall.yaml`. It lists forbidden feature columns,
forbidden source topics/paths, allowed oracle/evaluation contexts, planner-facing
forbidden imports, and normal-runtime config constraints.

The active normal runtime config remains
`scripts/visibility_comparison/warehouse_visibility_campaign.yaml`; Phase 0 adds a
regression test that it does not enable GT state or reliability sources.

## Replay And Oracle Surfaces

`reliability_tools replay` runs the required R0-R4 baselines on split exports.
R4 may be backed by an explicit operational provider such as:

```bash
--gp-artifact paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz
```

`reliability_tools benchmark` adds the wider suite wrapper and can include M5/M6
multi-camera policies when replay frames contain multiple camera observations.

`reliability_tools export-multicamera` accepts one perception CSV per camera and
writes the same split `operational/` and `evaluation_only/` records. Replay frames
group near-synchronous camera observations by rounded timestamp so M5/M6 policies
can consume A/B observations from the same interval.

`reliability_tools validate-overlap` computes A/B map-estimate disagreement from
operational replay frames. It does not use GT; GT-based residual plots remain an
evaluation-only analysis layer.

`reliability.oracle` is evaluation-only. Its records are labeled
`evaluation_only_oracle` and are for feasibility questions such as whether a
hypothetical camera B would cover known camera-A dropouts. These labels cannot be
used as operational features or normal runtime reliability sources.
