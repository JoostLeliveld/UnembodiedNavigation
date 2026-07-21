# DATA SOURCE — dedicated commissioning coverage drive (NOT the navigation campaign)

Decision (2026-07-21, user-directed, supersedes the first draft of `B` and
`REUSE_MAP`). Reliability-map training data for this study comes from a
**dedicated camera-commissioning coverage drive**, not from the EFE navigation
campaign (`run_visibility_campaign.py` / C1/C2 runs).

## Why the navigation campaign was the wrong source

- **Biased sampling.** The EFE planner chooses where the robot goes, so the
  training locations are whatever trajectories the planner preferred — sparse
  and skewed. The C1 (85% detection) vs C2 (100% detection) gap seen in the
  aborted capture is the symptom: C2 steers toward camera-visible routes, so it
  barely samples the camera-weak regions the map most needs to learn.
- **Map↔planner circularity.** Training the reliability map on runs produced by
  a planner that will later *consume* that map couples the two. A commissioning
  drive that is agnostic to the planner breaks the loop.
- **Not how a real system commissions cameras.** When fixed cameras are
  installed in a real warehouse you drive the robot around the operating area
  once (teleop or a coverage routine), camera + detector running, and record
  where detection succeeds/fails. The map is built from that survey, then frozen
  and used. This is `research_story` ch.10 (`active_commissioning`) territory.

## What the commissioning drive is

A robot driven around the **whole drivable region** of `warehouse_aws` "as if a
human were teleoperating it" — systematic serpentine coverage through every
aisle (A1–A4), the lower main aisle, upper cross-aisle, west service lane and
the connectors — deliberately visiting **both** camera-strong (near, centred)
and camera-weak (far, image-edge, shelf-occluded) locations so the log contains
genuine detection *misses*, not just hits. Driven by a coverage controller
publishing `/cmd_vel` (waypoint / pure-pursuit) — **the EFE planner does not
choose the path**.

While it drives, the same runtime as the navigation stack runs: external camera
→ YOLO detector → pixel-to-BEV → belief EKF (pixel-correction) → `experiment_logger`.
Every frame logs the identical tuple the GP pipeline expects:
`(belief m_x/m_y, belief cov S_xx/S_xy/S_yy, det_hit, yolo_score_raw, GT eval-only)`.
So `build_belief_gp_events.py → fit_belief_aware_gp.py` run **unchanged**.

## Design requirements

1. **Coverage, not goals.** Serpentine/lawnmower over the drivable prisms; visit
   each region; several passes and both directions for repeated outcomes per cell.
2. **Belief must match the runtime EKF.** The logged `(μ, P)` must be the same
   pixel-correction belief the navigation stack uses — run that estimator
   decoupled from planning (mechanism per the architecture map; do NOT hand-roll
   a new filter).
3. **Drive by odom/waypoints in map frame, never GT.** GT stays evaluation-only.
   Drivable-lane geometry is an allowed operational input (the planner already
   uses `driveable_geometry_json`); CAD *shelf* geometry stays eval-only.
4. **Region-disjoint held-out for the GP.** A real commissioning pass is one
   thorough drive; for the held-out NLL/Brier eval, split the covered area into
   contiguous spatial blocks and hold blocks out (leave-region-out), so we test
   interpolation into unvisited area — not memorization of adjacent frames.

## Reuse (per NO_SHORTCUTS)

- Driver template: `experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py`
  (the bigwarehouse commissioning drive) — adapt its `/cmd_vel` waypoint-follower
  + logging pattern to `warehouse_aws`, single camera.
- Sim + detector + estimator + logger: the existing launch stack with EFE
  planning disabled (exact toggles per the architecture map).
- GP fit + events + metrics: unchanged (`fit_belief_aware_gp.py`,
  `build_belief_gp_events.py`, `scripts/shared/metrics.py`).

Outputs → `logs/visibility_comparison/single_cam_commissioning_v1/` (fresh; the
aborted `single_cam_uigp_capture_v1` navigation runs are NOT used for the fit).
