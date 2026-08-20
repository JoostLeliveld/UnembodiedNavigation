# Temporal ground-anchor sequence protocol

This protocol was frozen before the full 84-frame depth inference and evaluation.

## Question

Across repeated four-camera map updates, does the optional Bayesian affine anchor
filter preserve visibility accuracy while reducing scale/shift jitter, and does it
continue producing maps through isolated failures of the current floor-anchor mask?

## Frozen source sequence

- Dataset: `logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01`
- `records.jsonl` SHA-256:
  `eb1ec37bee880e3d41928f2077c55978652123fe845288bcc70e2929ae242205`
- `manifest.json` SHA-256:
  `5726de71428f4352210d3ff6f439033e55037f52ad536135c0b716695780e5cd`
- Scenario YAML SHA-256:
  `38837a55f5d4d6b32aefee7e2b27a61fb3d54748e5b28f922c71041f43f8b3e4`
- Cameras: `external_camera`, `external_camera_b`, `external_camera_c`,
  `external_camera_d`
- 21 synchronized capture times from 0.4 s through 8.0 s; exactly 84 records.
- Events: clear aisle, pallet spawn, pallet motion, stop, removal.
- Monocular model: `dav2_relative_small`, adapter uncertainty disabled.
- Operational replay cadence: one update every 10 s. Source simulator timestamps remain
  recorded separately. This exercises 21 filter updates over a 200 s operational horizon;
  it does not manufacture 200 s of lighting or hardware drift.

The method may read only RGB-derived depth, fixed calibration, the planner's traversable
regions, the floor plane, and the predeclared stress mask. Gazebo depth, visibility grids,
and obstacle state are opened only after each method result exists, by the evaluator.

## Arms

1. `enhanced_single`: enhanced floor-anchor selection and a fresh robust affine per frame.
2. `enhanced_temporal`: identical current-frame anchors plus the per-camera Bayesian
   scale/shift filter.

Both arms run first on the untouched sequence.

## Predeclared anchor-dropout stress test

At source timestamps `2.4`, `4.4`, `6.0`, and `8.0` s, for every camera, the evaluator
passes an empty externally supplied floor-segmentation mask. These are update indices
6, 11, 16, and 21 (one-based), selected before looking at any prediction or oracle score.
This represents an isolated failure of the floor-selector input; it is not claimed to be
a naturally observed segmentation failure.

The same predictions and masks go to both arms. The single-frame arm must refuse those
updates. The temporal arm may reuse only a prior younger than its configured 30 s stale
limit. No RGB/depth pixels or obstacle map are carried between frames.

## Metrics and units

- Output availability: valid camera-frame outputs out of 84, and valid outputs at the 16
  injected camera-frame dropouts.
- Visibility: balanced accuracy and visible-class IoU against the geometry oracle,
  computed per camera-frame and summarized first across cameras within each synchronized
  update, then descriptively across the 21 updates.
- Metric structure depth: per-frame MAE on pixels whose Gazebo depth lies at least 0.10 m
  in front of the analytic floor, summarized in the same hierarchy.
- Affine stability: per-camera absolute successive changes in raw DA-V2 relative-depth
  scale and shift. These are parameter-stability diagnostics, not localization error.
- Runtime: `estimate_visibility` only; saved prediction I/O and neural inference excluded.

The synchronized update is the reporting unit. The 21 sequential updates are temporally
dependent, so the report is descriptive and does not attach an iid confidence interval.

