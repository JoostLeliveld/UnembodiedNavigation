# F73 Tracker Accuracy Diagnostic

Log root: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/probe_boxside_north_route_choice_gpu_v1`

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F73/F73_tracker_accuracy_diagnostic.png`

All means, percentiles, and maxima below are computed after the first
non-trivial command. Pre-command global-solve / warm-up rows are excluded.

## C1 constant-R

- outcome `goal_reached`, path=4.84 m, min_goal=0.077 m.

- truth-to-initial-plan distance: mean=0.076 m, p95=0.147 m, max=0.160 m.

- active waypoint distance: mean=0.363 m, p95=0.605 m, max=0.634 m.

- truth-belief error: mean=0.308 m, p95=1.061 m, max=1.951 m.

- truth-state error: mean=1.169 m, p95=2.803 m, max=2.930 m.

- absolute execution yaw error: mean=0.425 rad, p95=2.502 rad.

- min obstacle clearance=0.136 m, max cmd age=0.100 s.

## C2 GP-aware

- outcome `goal_reached`, path=5.84 m, min_goal=0.131 m.

- truth-to-initial-plan distance: mean=0.107 m, p95=0.424 m, max=0.424 m.

- active waypoint distance: mean=0.371 m, p95=0.636 m, max=0.646 m.

- truth-belief error: mean=0.107 m, p95=0.168 m, max=0.567 m.

- truth-state error: mean=0.336 m, p95=0.514 m, max=0.642 m.

- absolute execution yaw error: mean=0.366 rad, p95=1.765 rad.

- min obstacle clearance=0.190 m, max cmd age=0.103 s.


Conclusion: this diagnostic separates three things. Distance from initial global plan measures whether local execution abandons the selected homotopy; waypoint distance measures tracker convergence; truth-belief error measures what the planner actually bases future decisions on. The external-camera `/state` error can be large without directly steering the robot if the planner belief has already been propagated/corrected differently.
