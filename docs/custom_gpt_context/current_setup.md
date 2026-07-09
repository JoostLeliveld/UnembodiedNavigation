# Current Setup

This file summarizes the active current setup. If in doubt, the source of truth
is [`../current_runtime_contract.yaml`](../current_runtime_contract.yaml) plus
[`../experiment_registry.md`](../experiment_registry.md).

## Active Runtime Surface

Status: active current honest-campaign surface.

World:

- `warehouse_aws.world.sdf`
- fixed external camera at approximately `(0.0, -5.5, 4.8)`

Tasks:

- `route_apron_to_a3_mid`
- `route_apron_to_a2_mid`
- `route_west_to_a1_upper`
- `control_west_to_a1_low`

Seeds:

- `0, 1, 2, 3, 4`

Conditions:

| Condition | Meaning |
| --- | --- |
| C1 | constant camera covariance, no GP-scaled visibility |
| C2 | visibility-aware planner using GP-scaled camera covariance |

## Perception

Current detector:

- `warehouse_yolo_detector_v1`
- local runtime checkpoint: `logs/perception_models/warehouse_yolo_detector_v1/model.pt`
- trained image size: 960
- runtime image size: 640
- confidence threshold: 0.05
- selected pixel source: bounding-box bottom centre
- mask usage in current runtime: false

Important caveat:

YOLO mAP or confidence is not enough. The detector must be judged by the
downstream localization point: the selected bottom-centre pixel and its
projected BEV error.

## GP Reliability Artifact

Current GP artifact:

- `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`

The GP represents detector reliability/trust over the map. It does not learn
`R` directly. At planning time, reliability is mapped to an effective
image-space measurement covariance `R_plan(s)`.

## Planning Runtime

Important locked settings:

- heading update mode: `camera_xy_only`
- no visual/keypoint heading in current campaign
- global horizon: 75
- global dt: 0.4
- lookahead: 30 s
- NIS gate: chi-square 2 DOF at 0.99, threshold 9.21
- no-go mode: `keep_in`
- no-go belief kappa: 1.0
- visibility reward weight: 0.0 in the locked campaign

This means the GP is not a direct visibility reward. It enters through
planner-facing camera covariance and expected belief behavior.

## Current Result Surface

Current packaged result surface:

- `docs/paper_vs_current/current/`

Headline:

- C1: 15/20 clean goals
- C2: 20/20 clean goals
- C1 GT geometry breaches: 4/20
- C2 GT geometry breaches: 0/20
- physics contacts: 0

These are current-result claims, not the submitted-paper historical claims.

## Ground Truth In Current Runtime

Ground truth is used for evaluation metrics:

- goal distance,
- clearance,
- geometry breaches,
- collision/physics-contact cross-checks,
- path/outcome auditing.

Ground truth is not an operational state source for the method.

