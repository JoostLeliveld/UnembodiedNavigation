# F74 YOLO / State / Belief Audit

This audit uses the F73 boxside route-choice run:

- Log root: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1`
- Task: `probe_a4_boxside_north_to_a3top`
- Config: `scripts/visibility_comparison/aws_probe_boxside_north_route_choice_config.yaml`
- Output folder: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F74`

All metrics below exclude launch/global-solve idle time and start at the first non-trivial command.

## Figures

- `F74_yolo_audit_summary.png`: BEV truth/belief/projected detections plus time-series of detection, state, and belief error.
- `F74_yolo_image_space_reconstruction.png`: reconstructed image-space YOLO boxes and selected pixels from logged diagnostics. These are **not raw RGB screenshots** because F73 did not save camera frames.
- `F74_pixel_correction_effect.png`: whether pixel corrections improved or worsened the planner belief relative to truth.

### C1 constant-R
- Run directory: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1/probe_a4_boxside_north_to_a3top/C1/seed0/experiment_20260604_144614`
- Outcome: `goal_reached`; path `4.835 m`; min goal `0.077 m`.
- Runtime perception rows: `62` after first command.
- YOLO detection rate: `0.452` (`28/62`).
- Pixel-pose within runtime timeout (`age <= 1.25 s`): `0.532`. Important: `pixel_pose_available=1` can still mean latest stale pose exists.
- YOLO selected score: mean=0.322, p50=0.000, p95=0.789, max=0.851.
- YOLO BEV localization error: mean=1.107, p50=0.624, p95=2.729, max=2.863 m.
- `/state` position error: mean=1.197, p50=0.761, p95=2.819, max=2.939 m.
- Planner truth-belief error: mean=0.308, p50=0.170, p95=1.061, max=1.951 m.
- YOLO latency: mean=0.064, p50=0.060, p95=0.101, max=0.109 s; inference: mean=63.742, p50=60.517, p95=101.569, max=109.295 ms.
- Latest pixel-correction diagnostic accepted flag rate: `0.963`; latest reject reasons: `{'dt_implausible': 4}`. This is a sampled diagnostic stream, not a count of unique correction events.


### C2 GP-aware
- Run directory: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1/probe_a4_boxside_north_to_a3top/C2/seed0/experiment_20260604_144802`
- Outcome: `goal_reached`; path `5.837 m`; min goal `0.131 m`.
- Runtime perception rows: `80` after first command.
- YOLO detection rate: `0.975` (`78/80`).
- Pixel-pose within runtime timeout (`age <= 1.25 s`): `1.000`. Important: `pixel_pose_available=1` can still mean latest stale pose exists.
- YOLO selected score: mean=0.745, p50=0.769, p95=0.877, max=0.885.
- YOLO BEV localization error: mean=0.288, p50=0.326, p95=0.473, max=0.593 m.
- `/state` position error: mean=0.371, p50=0.422, p95=0.534, max=0.642 m.
- Planner truth-belief error: mean=0.107, p50=0.105, p95=0.168, max=0.567 m.
- YOLO latency: mean=0.063, p50=0.058, p95=0.102, max=0.128 s; inference: mean=62.757, p50=57.853, p95=102.519, max=128.120 ms.
- Latest pixel-correction diagnostic accepted flag rate: `1.000`; latest reject reasons: `{}`. This is a sampled diagnostic stream, not a count of unique correction events.


## What Is Going Wrong?

The high `/state` error is not evidence that every fresh YOLO detection is bad. It is mostly a
freshness/blackout problem:

1. The detector publishes `/perception/pixel_pose` only when `yolo_detected_after_threshold=1`.
2. The logger records `pixel_pose_available=1` whenever a latest pixel pose exists, even if the current frame is a miss.
3. During C1's weak-visibility segment, many frames are below threshold. The latest pixel/state can remain stale while the robot keeps moving.
4. That stale or low-confidence camera state is then a poor representation of the robot's current truth pose.

So the right reading is:

- C2: YOLO is useful. It stays in a visible route, detections remain high-confidence, and belief error stays small.
- C1: YOLO becomes unavailable/unreliable in the chosen route. The state stream can look available while effectively stale, and pixel corrections are neutral or slightly harmful. This is exactly the failure mode the visibility-aware planner should avoid.

## Does YOLO Add Something?

Yes, but only when the robot remains in regions where detections are fresh and geometrically valid.
For C2, the detector provides dense, high-score updates and keeps the planner belief close to truth.
For C1, the detector blackout makes `/state` misleading unless freshness is handled explicitly.

## Implementation Notes

- Raw camera frames were not stored in this run; future perception audits should enable optional image snapshots for selected frames.
- `frame_age_at_publish_s` is not reliable in this log because it mixes time bases. Prefer `yolo_latency_s`, `yolo_inference_ms`, and explicit pixel/correction ages.
- Paper/result metrics should distinguish:
  - fresh YOLO detection error,
  - stale latest-state error,
  - planner truth-belief error,
  - and final task outcome.

## Implemented Follow-Up For Future Runs

This audit motivated an explicit freshness field for camera-derived state:

- `experiment.csv` now logs `state_age_s` and `state_fresh`.
- `perception.csv` now logs `pixel_pose_fresh`.
- `run_manifest.json` records `pixel_timeout_s`.
- The planner no longer resets belief from stale `/state/bev` after an implausible delayed correction.

Future dashboards should use these fields to separate fresh camera-state error from stale latest-state error.

Do not remove `dt_implausible`; it is a stale-correction guard. The issue is how the system behaves after a rejected or missing update.
