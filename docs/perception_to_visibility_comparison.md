# Perception To Visibility Comparison

This document explains the clean comparison framework used to study how different perception-derived observability definitions change the visibility GP, the ambiguity map, and potentially the planned path.

## Purpose

The point of the framework is not to argue that one detector is universally best.

The point is:

- the same sampled workspace is observed under several perception-derived observability definitions
- each definition produces a scalar target over workspace
- each target is fitted into the same planner-facing GP contract
- each GP induces a different observation-covariance landscape
- those different landscapes can change the ambiguity map and the planned trajectory

## Why Perception Is Separated From Visibility Modeling

Perception outputs are noisy and method-specific. Planning, however, needs one common interface.

So the framework separates:

1. raw observations from the camera or geometry
2. method-specific scalar observability targets
3. GP visibility fields
4. planner use of those fields

This keeps the comparison fair. Different methods may define observability differently, but they all must eventually provide the same planner input:

`p_vis(x,y) -> R_plan(x,y)`

## Why Bottom-Center Is Used Consistently

All perception methods use the same observation rule:

- selected pixel = bottom-center of the detected mask, blob, or bbox

This is intentional.

- it reduces method-to-method differences that come only from choosing different pixel summaries
- it stays close to the downstream use case, where an image point is projected to BEV
- it avoids comparing a mask-based method using one geometric proxy against a blob-based method using another

## Why Teleport Sampling Is Used

Teleport sampling is the active capture mode for this comparison framework.

Why:

- it gives dense, even spatial coverage
- it is easier to compare methods on the same sample set
- it avoids path-dependent sampling bias from a moving robot trajectory
- it produces a cleaner basis for GP fitting than opportunistic runtime logs

The canonical raw capture output is:

- `logs/visibility_comparison/current_capture/samples.csv`
- `logs/visibility_comparison/current_capture/images/`
- `logs/visibility_comparison/current_capture/previews/`
- `logs/visibility_comparison/current_capture/capture_manifest.json`

## Compared Observability Definitions

The framework reserves these method ids:

- `red_binary`
- `red_area_corrected`
- `yolo_binary`
- `yolo_confidence`
- `oracle_visibility`

At the current stage, the implemented methods are:

- `oracle_visibility`
- `red_binary`
- `yolo_binary`
- `yolo_confidence`

The remaining deferred method is:

- `red_area_corrected`

### Red Binary

Target:

- `y = 1` if a usable red blob is detected, else `0`

Meaning:

- can the simple color detector observe the robot at all?

### Red Corrected Area

Target:

- `r = A / A_ref(x,y)`
- `y = clip((r - r_min) / (r_ok - r_min), 0, 1)`

Default shaping:

- `r_min = 0.05`
- `r_ok = 0.35`

Meaning:

- how much of the expected robot appearance is visible after compensating for perspective scale?

### YOLO Binary

Target:

- `y = 1` if YOLO detects a usable robot mask/bbox above threshold, else `0`

Meaning:

- can the learned detector observe the robot at all?

### YOLO Confidence

Target:

- the raw YOLO score, clipped into `[0,1]` if needed

Meaning:

- a soft detector-reliability signal

Important:

- this score is treated as an uncalibrated detector score
- it is not claimed to be a calibrated probability

### Oracle / Reference Visibility

Target:

- `y = 1` if the robot is visible according to geometry/reference visibility, else `0`

Meaning:

- a simulation-geometry reference signal

Important:

- this is not a runtime perception method
- it is a reference field for comparison

## Why Raw Blob Area Is Corrected Using A_ref(x,y)

Raw blob area is strongly affected by perspective:

- close robot -> larger image footprint
- far robot -> smaller image footprint

So a naive global area threshold confounds:

- perspective size change
- actual partial occlusion

To reduce that problem, the framework uses a simple position-dependent reference area:

- spatially bin samples over `x,y`
- in each bin, estimate `A_ref(x,y)` using a high percentile of observed blob area
- normalize measured area by that local reference

This is not a perfect physical area model. It is only meant to stop perspective from dominating the red-area target.

## Why Positional-Error GP Is Not The Primary Target Yet

A GP on localization error is interesting, but it mixes several effects:

- detector quality
- bottom-pixel choice
- homography geometry
- estimator behavior

That is useful later, but it is not the cleanest first comparison target.

The primary comparison in this framework is therefore:

- observability / detection-derived targets first
- position-error modeling later if needed

## Shared Planner Interface

Every method must end in the same interface:

- GP visibility field: `p_vis(x,y)`
- planner covariance field: `R_plan(x,y)`

The planner mechanics are not changed per method.

Only the fitted visibility field changes.

That means differences in ambiguity maps or planned paths can be attributed to:

- different observability definitions

not to:

- different planner logic

## Full Detector-Stack Comparison Convention

When the comparison is run in full detector-stack mode, the live runtime perception backend is paired with the method family:

- `red_binary` and `red_area_corrected` use the live `image_markers` backend
- `yolo_binary` and `yolo_confidence` use the live `yolo` backend
- `oracle_visibility` remains a reference visibility field; its live backend must be stated explicitly
- `visibility_unaware_baseline` disables the visibility GP in the planner; its live backend must also be stated explicitly

This keeps the detector family aligned with the visibility field family without changing planner mechanics per method.

## Why Ambiguity Maps Are Plotted In Addition To GP Value Maps

The GP value map shows:

- where the robot is predicted to be more or less observable

The ambiguity map shows:

- what that means after the planner converts visibility into observation covariance

Those are not the same thing.

Two fields with similar `p_vis` ranges can still induce different ambiguity structure once mapped into `R_plan(x,y)`.

So multiple fields are plotted:

1. GP visibility field `p_vis(x,y)`
2. induced ambiguity field
3. `r_plan_uv_std(x,y)` representing the planner-facing observation-noise proxy

Note: The framework explicitly does NOT plot a static field-level 'risk map' because the true `risk_cost` within the EFE planner is goal-dependent. Planner objectives like `efe_risk` are instead plotted dynamically over time since the first command.

## Run Completion Contract

Planner execution is governed by a strict, deterministic completion contract enforced by the logger. Runs are terminated only for three mutually exclusive reasons:

- `goal_reached`: Target reached and held.
- `timeout_after_first_cmd`: A strict time budget evaluated *after* the first physical movement command, avoiding startup bias.
- `stuck`: A rolling window detects when the command rate is high but the robot has stopped making measurable progress toward the goal.

Only runs that output a valid `run_summary.json` with one of these reasons are ingested into the final path analysis and report. Incomplete runs are explicitly excluded.

## Current Shared Backbone

The active shared scripts are:

- `scripts/visibility_comparison/capture_visibility_samples.py`
- `scripts/visibility_comparison/extract_perception_targets.py`
- `scripts/visibility_comparison/compute_area_reference.py`
- `scripts/visibility_comparison/build_gp_targets.py`
- `scripts/visibility_comparison/fit_visibility_gps.py`
- `scripts/visibility_comparison/plot_gp_and_ambiguity_maps.py`
- `scripts/visibility_comparison/run_planner_method_sweep.py`
- `scripts/visibility_comparison/plot_planned_paths.py`
- `scripts/visibility_comparison/make_visibility_comparison_report.py`

Current status:

- the shared raw/target/GP/report contracts are implemented
- the geometry-based oracle/reference path is fully implemented
- the red binary path is fully implemented
- red and YOLO target extraction logic are implemented later as separate method passes

## First Finished Methods

The first completed methods in the new framework are:

- `oracle_visibility`
- `red_binary`

What is already implemented for it:

- teleport capture writes `oracle_visible`, `oracle_bottom_u`, `oracle_bottom_v`, and `oracle_occlusion_reason`
- perception target extraction carries oracle values into `perception_targets.csv`
- perception target extraction also runs the red blob detector and fills:
  - `red_detected`
  - `red_area`
  - `red_bbox_xyxy`
  - `red_bottom_u`
  - `red_bottom_v`
- GP target construction writes:
  - `oracle_visibility`
  - `red_binary`
- GP fitting already produces:
  - `oracle_visibility_gp.npz`
  - `red_binary_gp.npz`
  when data are available
- GP and ambiguity plotting already works for both fields
- planner runs can use either artifact through explicit `visibility_artifact_path`

So `oracle_visibility` is the clean geometry/reference anchor, and `red_binary` is the first finished perception method on top of that same backbone.

## Safe Claims

- Different perception-derived observability definitions can induce different GP fields, ambiguity landscapes, and planned trajectories under the same planner.
- The comparison framework uses one consistent bottom-center observation rule across methods.
- The planner interface is shared across methods.
- Teleport sampling is used to obtain dense, comparable spatial evidence.
- Oracle visibility is a reference field, not a runtime perception method.
- YOLO confidence is treated as an uncalibrated detector score.

## Unsafe Claims

- YOLO confidence is a calibrated probability.
- Red blob area is a universal visibility measure.
- Oracle visibility is runtime perception.
- The current estimator is a fully visual pose estimator.
- One method is universally best without broader experiments.
