# F49 Tracking/Yaw Diagnostic

Generated from existing run root:
`logs/visibility_comparison/f45_b1_tracking_yaw_v2`.

Files:
- PNG: `timing_presentation/figures/F49/F49_tracking_yaw_diagnosis.png`
- PDF: `timing_presentation/figures/F49/F49_tracking_yaw_diagnosis.pdf`
- Generic dashboard: `timing_presentation/figures/F49/F49_dashboard.png`

## Configuration Tested

F49 is a controlled Gazebo smoke after wiring the local-tracking parameters into
the runtime and logger:

- global route: `H=80`, multistart route candidates enabled for both C1 and C2.
- local tracker: EFE local controller, `H=15`, `local_optimizer_maxiter=20`.
- `local_tracking_use_odom_yaw=true`.
- `latency_compensate_plan_handoff=true`.
- `cmd_publish_rate=10 Hz`.
- command noise and encoder noise remain enabled.

## Run Summary

| condition | outcome | path [m] | min goal [m] | min obs [m] | mean state err [m] | solve mean [ms] | solve p90 [ms] | ms/eval | yaw p90 [rad] | pixel-age max [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | collision | 5.10 | 1.86 | -0.017 | 0.342 | 829 | 1280 | 23 | 1.82 | 6.28 |
| C2 | goal_reached | 6.29 | 0.07 | 0.239 | 0.187 | 756 | 959 | 24 | 1.99 | 0.94 |

## Diagnosis

This is not a successful behavior run. It is useful because it separates three
failure modes that were previously mixed together.

1. **Initial yaw/spawn is not the root cause here.** Both C1 and C2 start with a
   clean frame sanity check (`truth_start_yaw_error=0`). If yaw becomes bad, it
   happens during runtime tracking/perception rather than at Gazebo spawn.

2. **The global route choice remains mostly sensible.** The first route and local
   endpoint overlays show the route family chosen before execution. The weird
   behavior still appears during tracking and replanning, not because the first
   global planner has no idea where to go.

3. **Local EFE is under-budgeted in F49.** `H=15, maxiter=20` reduced horizon and
   should have helped timing, but the logs repeatedly report
   `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT`. That means the local controller
   is often executing solver-returned sequences that have not actually converged.
   This can explain jagged heading corrections and poor waypoint tracking.

4. **C1 has a perception freshness failure before collision.** The C1 console log
   reported stale pixel belief ages of multiple seconds and an implausible
   correction gap of about 7.5 s before belief reset. That makes C1 a localization
   failure case, not a clean local-controller-only case.

5. **C2 is the more revealing controller failure.** C2 keeps lower state error
   than C1 but still collides. When localization is comparatively good and the
   route is plausible, the remaining failure is local tracking/clearance/timing:
   waypoint following plus local barrier optimization is not robust enough yet.

## Next Iteration

Do not change maximum speed or the route story first. The next controlled
iteration should keep the same task and route candidates, but make the local
tracker numerically honest:

- try `local_horizon=20` again with `local_optimizer_maxiter=35` or `40`;
- keep `local_tracking_use_odom_yaw=true`;
- keep the new waypoint/yaw diagnostics;
- compare optimizer success fraction and ms/eval against F49;
- only after local convergence improves, test whether the remaining problem is
  waypoint acceptance radius or driveable-region clearance.


## F49-specific interpretation

F49 keeps the F47/F48 execution contract: global multistart route selection, local multistart disabled, local H=12, odom yaw for local tracking, command and encoder noise enabled. The difference is a tight local waypoint prior (`local_goal_prior_*_std_start=4`, `local_goal_prior_*_std_final=2`) plus the F48 code fix that appends the actual task goal as the final waypoint. This is the first run in this chain with the desired qualitative outcome: C1 collides after taking the visibility-poor/risky behavior, while C2 reaches the goal after following the visibility-aware route.
