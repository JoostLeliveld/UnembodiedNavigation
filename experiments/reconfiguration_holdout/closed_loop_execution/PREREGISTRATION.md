# Prospective closed-loop execution of L2-selected routes

Frozen 2026-08-19 before L2 was selected, rendered, or scored and before any run in
this campaign existed. L1 route results are discovery inputs used only to choose two
fixed start--goal tasks. This protocol does not alter the L2 prediction/routing
confirmation or rescue it if either registered hypothesis fails.

## Question and claim boundary

E3 selects routes offline. This successor asks whether executing those different
routes through the same belief-based local tracker changes realised time without an
accepted external-camera correction and belief error. It is not an online route-choice
comparison: the global routes are selected once by E3 and hash-bound before execution.
The deployed EFE objective is not used or rehabilitated.

## Frozen tasks and route cell

The network is four cameras and the maximum detour budget is 20%, matching L2 H2.
Two tasks are fixed from the L1 discovery analysis:

1. `dock_sw__aisle_wn`, selected because it had the largest L1 measured blind-distance
   advantage of the recomputed route over the GP route among the declared tasks;
2. `mid_ne__mid_sw`, selected because its two L1 routes had the largest Hausdorff
   separation.

The endpoints and all 89-task definitions remain those in E3. For each task, the two
executed L2 routes are exactly E3's `gp` and `mono_depth` polylines. Detector outcomes
and measured blind distance do not choose, alter, smooth, or gate a route.

Before any execution, one campaign manifest records the L2 world hash, E3 manifest
and route-geometry hashes, exact route coordinates and per-route hashes, endpoints,
controller/filter/detector/noise configuration, five seeds, and expected run IDs.
There are exactly `2 tasks x 2 route arms x 5 seeds = 20` valid paired runs.

## Route-execution gate

The campaign runs only if, for each task:

- both routes exist and satisfy the registered 20% length constraint;
- every point lies on the frozen driveable mask with at least the planner's declared
  clearance;
- both routes start and end at the registered task endpoints (within one 0.25 m grid
  cell); and
- GP and monocular routes differ by at least 0.50 m Hausdorff distance.

This gate reads only planned coordinates and the L2 lane map. It does not read detector
outcomes, correction events, ground truth, or belief error. If either task fails, that
task is reported as a failed route-separation gate and is not replaced.

## Execution method

Add `global_planner_mode=preselected_route`. It accepts exactly one JSON polyline and
its expected SHA-256, validates the route-execution gate, persists the same hash and
coordinates in the run directory, and passes the polyline to the existing belief-based
local waypoint tracker without running a global EFE or shortest-path solve. Control is
computed from the planner belief, never Gazebo odometry or ground truth. Ground truth is
logger-only evaluation data.

The two arms differ only in the preselected global polyline. Camera topics, detector,
measurement model, Bayesian filter, local tracker, gains, collision geometry, motion
noise, speed limits, start/goal, timeout, and seed are identical within every pair.
The campaign configuration is dry-run validated and one smoke pair must complete before
the 20-run campaign; smoke runs are excluded.

## Runs, invalidity, and censoring

Seeds are exactly `{0,1,2,3,4}` and are paired by `(task, seed)`. An infrastructure
failure is a run with no first command, missing logger output, wrong world/route hash,
simulator/bridge contamination, or malformed/nonmonotone timestamps. It is rerun with
the same identifier/seed after cleanup. Collision, failure to reach the goal, filter
divergence, or timeout after the first command are outcomes, not infrastructure
invalidity, and are never dropped.

The analysis window starts at the first nonzero command and ends at mission completion
or the frozen timeout. Start and end are censoring boundaries for correction gaps.

## Outcomes and inference

An accepted correction event is a logger row with `pixel_corr_accepted == 1`, deduplicated
by `pixel_corr_apply_stamp`; repeated cached diagnostic rows are not new events.

**Primary:** longest elapsed interval between distinct accepted corrections, including
the start-to-first and last-to-end censored intervals. For pair `u=(task,seed)`, the
contrast is `gap_GP(u) - gap_mono(u)`; positive favours the route recomputed from current
imagery. Report all ten pair values, mean, median, a deterministic paired bootstrap 95%
interval (10,000 resamples, seed 20260819), and an exact two-sided sign test with exact
zero ties dropped. This is one confirmatory test at alpha 0.05.

**Secondary:** per-run belief RMSE
`sqrt(mean((planner_belief_x-gt_x)^2 + (planner_belief_y-gt_y)^2))`, computed after the
first command and then paired by task/seed; time fraction since the last distinct accepted
correction exceeding 1 s and 2 s; total distinct corrections; completion; collision;
elapsed time; and path length. Never use legacy `truth_x/y`, never pool logger rows across
runs, and never report the logger's mean `belief_error_gt_m` as RMSE.

## Interpretation

A positive, significant primary contrast supports only that these preselected
availability-aware routes changed realised correction continuity in this simulated L2
world. A null leaves the paper's contribution at offline route choice. Belief RMSE is
secondary and cannot replace a null primary. No physical navigation claim is made.
