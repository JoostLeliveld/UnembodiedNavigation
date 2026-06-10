# Paper Alignment

This file is the maintained code-to-paper contract. It keeps the scientific
claim narrow enough that the implementation, figures, and text can stay aligned.

## Paper-Facing Runtime

The primary runtime entry point is:

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/aws_f31b1_final_config.yaml \
  --log-root logs/visibility_comparison/aws_f31b1_final_v1
```

That config pins the world (`warehouse_aws.world.sdf`), detector
(`aws_yolo_simseg_v2/model.pt`), and GP (`aws_gp_v7/yolo_score_raw_gp.npz`).

## Current Paper Claim Surface

| Role | Code name | Paper meaning |
| --- | --- | --- |
| Core world | `warehouse_aws.world.sdf` | AWS route-choice benchmark |
| Main task | `F31_b1_apron_a3_mid` | apron → A3-mid route choice (occluded mid vs visible lower-sweep) |
| Saved secondary | `a0_west_to_a1_upper_blocked_mid` | saved a0 line for a future multi-task run |
| Baseline (C1) | `constant_R_efe` | EFE with spatially uniform detector-observation covariance |
| Method (C2) | `visibility_aware_efe` | EFE with GP-derived state-dependent detector-observation covariance |
| Optional ablation (C3) | `risk_only_ablation` | GP covariance active, ambiguity disabled |
| GP artifact | `logs/visibility_comparison/aws_gp_v7/yolo_score_raw_gp.npz` | planner-facing reliability artifact (camera z=4.8, y=-5.5) |

`warehouse_aws.world.sdf` is the paper-facing candidate world. Do not claim AWS
results as paper evidence until the full chain is complete (seeded logs, metrics,
figures). The F31_b1 closed-loop route-split is currently OPEN — see
`active_research_state.md`. The former compact-benchmark line
(`warehouse_occ_light`, `shadow_tradeoff_*`, `current_gp`) is retired/archived.

## Method Layers

Keep these layers separate in wording, figures, and code comments:

- Known driveable / forbidden-zone layer: 2D planner constraints shared across
  conditions.
- Physical scene geometry: walls, shelves, racks, boxes, and other simulated 3D
  objects.
- Learned observation reliability: GP-derived reliability that changes
  planner-facing camera `(x, y)` covariance.
- Planner costs: risk, ambiguity, feasibility/no-go, and total EFE.

Low learned reliability is not a physical obstacle. A floor region may be
driveable but visually unreliable.

## Odometry And Dead Reckoning

The simulated robot is a TurtleBot3-style differential-drive robot. Gazebo
publishes odometry from continuous diff-drive velocity integration in the
physics simulation, not discrete wheel-tick counting. `/odom_noisy` adds slip
and noise and is used for dead reckoning when external-camera updates are weak
or absent.

There is no IMU/gyro in the current paper-facing setup. Heading is odometry
backed. The external camera provides position updates only.

## Projection Geometry

The camera observation pipeline is:

```text
image detection (u, v)
-> calibrated projection / homography
-> planar ground position (x, y)
-> EKF camera update
```

The robot is assumed to move on a planar ground surface. The current
implementation treats the projection model as exact. Projection uncertainty,
homography calibration residuals, and first-order projection Jacobian error are
not propagated into the EKF or EFE covariance. This limitation matters most near
image edges and oblique viewpoints.

## GP Observability

The GP represents learned observation reliability for camera-derived `(x, y)`
updates. It does not model traversability, does not directly improve heading,
and is not itself the planner objective.

For paper figures, the GP field is useful as setup context, but the main
explanatory result should show planner-facing quantities: ambiguity and total
EFE cost, with risk and feasibility/no-go terms separated.

The conservative planner field should be described as posterior mean discounted
by GP uncertainty. Sampled regions are evidence; unsampled regions are
extrapolation toward a conservative prior.

## Figure Priorities

Before submission, the paper should show:

- known driveable floor and forbidden zones separately from learned reliability;
- GP sample coverage or an explicit statement about unsampled extrapolation;
- an ambiguity field or EFE cost decomposition for representative C1/C2 plans;
- trajectories generated from real run traces, not schematic behavior claims.

## Future Work Boundaries

Future-work ideas may include online GP updating, projection/calibration
uncertainty propagation, random A-to-B generalisation, real warehouse
validation, and hierarchical sparse route-candidate scoring. These should not be
described as implemented unless a validated code path and artifact chain exist.

Sparse planning must be framed as fair candidate-route scoring. It must not be
implemented as mission waypoints that force route choice.

## Alignment Rules

1. Every main-paper result maps to a run directory, campaign config, GP
   artifact, and generated figure/table script.
2. Every method claim is visible in launch parameters or run manifests.
3. Code paths not used in the paper are labeled exploratory, diagnostic, or
   legacy.
4. Paper figures are generated from real run traces unless explicitly labeled as
   schematics.
5. The no-go term is a shared feasibility component, not a visibility reward.
6. Heading source and camera `(x, y)` source are reported separately.
7. Homography/projection uncertainty is acknowledged as a limitation.
