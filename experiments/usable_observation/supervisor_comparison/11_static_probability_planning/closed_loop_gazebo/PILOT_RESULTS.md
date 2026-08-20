# Closed-loop Gazebo pilot result

## Outcome

This is a real four-camera Gazebo smoke test, but it is **inconclusive for the observation
probability hypothesis**. All three conditions selected `solver:warm_or_cold`, produced a
near-stationary global plan, executed about one metre, and triggered the 20 s stuck detector.
They never entered the low-availability region where the observation models differ.

| Condition | Planning observation model | Outcome | Path [m] | Closest goal distance [m] | Belief RMSE vs Gazebo GT [m] | Collision |
|---|---|---:|---:|---:|---:|---:|
| C1 | constant R | stuck | 1.081 | 12.436 | 0.091 | no |
| C2 | frozen four-camera `p_use`, deployed `R/p` approximation | stuck | 1.084 | 12.460 | 0.099 | no |
| C3 | same `p_use`, explicit Bernoulli hit/miss propagation | stuck | 0.970 | 12.624 | 0.099 | no |

Experimental unit: one complete navigation run. Sample size: one seed per condition. Metric
object: planner belief during navigation. Position reference: `gt_x/gt_y` from the Gazebo GT
bridge, used only for post-run evaluation. Position frame: `map_bev`. The exact run directories
are frozen in `closed_loop_metrics.json` and the campaign log.

The 2-D covariance was also overconfident on this short segment: median NEES was 8.74, 10.88,
and 10.88 for C1--C3, with empirical 95% coverage 0.0%, 1.7%, and 0.0%. This is estimator
diagnostic evidence only; it is not evidence that one planning observation model is better.

## What the run did prove

- Gazebo started the `warehouse_full_4cam` world for every condition.
- Camera A--D image bridges were active.
- The detector used one batched four-image YOLO path with batch order A, B, C, D.
- The filter, camera manager, EFE planner, noisy actuation/odometry, local feedback controller,
  stuck detector and Gazebo-GT logger all ran together.
- C1, C2 and C3 used the same task, seed, dynamics, detector and estimator settings.
- The C3 planner runtime reported `use_hit_miss_mixture=True`; C1/C2 reported `False`.

## Blocking failure and attempted correction

The pilot global plans ended roughly 12.9 m from the goal. The local controller then repeatedly
reported `driveable_clearance_violation_step_0` and safe-stopped.

A second diagnostic run moved the start 0.30 m inside the south boundary, supplied only the two
explicit route candidates, and required terminal goal error <= 0.5 m. In corrected C1, all three
optimizer candidates were properly marked `goal_feasible=False`, but the planner still selected
the least-bad infeasible `warm_or_cold` solution. The local controller again rejected step zero.
The remaining matched reruns were stopped because they could not test the scientific variable.

## Offline correction prepared for the next execution

The route-discovery failure has now been isolated from the scientific comparison with an offline
solver. It returns only exact-start, exact-goal, clearance-validated routes and raises an error
instead of returning a partial path. It also evaluates C1--C3 on a common candidate set and
scores the simplified polyline that the controller would actually receive.

| Condition | Selected offline route | Length [m] | Minimum clearance [m] | Mean planning `p_use` | Minimum planning `p_use` |
|---|---:|---:|---:|---:|---:|
| C1 | candidate 0 | 14.24 | 0.263 | 0.662 | 0.001 |
| C2 | candidate 1 | 14.50 | 0.278 | 1.000 | 0.994 |
| C3 | candidate 1 | 14.50 | 0.278 | 1.000 | 0.994 |

This is the intended first result: the probability-aware objectives accept a 0.26 m detour to
avoid the modeled observation shadow. It is not yet evidence of better navigation or
localization. The next Gazebo campaign should feed the stored `controller_waypoints_xy` directly
to the local controller, bypass continuous global EFE discovery, while retaining the fail-closed
endpoint and clearance checks. Gazebo was deliberately left untouched for this offline solve.
