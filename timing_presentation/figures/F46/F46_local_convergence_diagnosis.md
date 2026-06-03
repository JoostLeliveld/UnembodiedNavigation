# F46 Local-Convergence Diagnostic

Generated from existing run root:
`logs/visibility_comparison/f46_b1_local_convergence_v1`.

Files:
- PNG: `timing_presentation/figures/F46/F45_tracking_yaw_diagnosis.png`
- PDF: `timing_presentation/figures/F46/F45_tracking_yaw_diagnosis.pdf`
- Generic dashboard: `timing_presentation/figures/F45/F45_dashboard.png`

## Configuration Tested

F46 is a controlled Gazebo smoke after F45 showed the local tracker was
under-budgeted:

- global route: `H=80`, multistart route candidates enabled for both C1 and C2.
- local tracker: EFE local controller, `H=20`, `local_optimizer_maxiter=40`.
- `local_tracking_use_odom_yaw=true`.
- `latency_compensate_plan_handoff=true`.
- `cmd_publish_rate=10 Hz`.
- command noise and encoder noise remain enabled.

## Run Summary

| condition | outcome | path [m] | min goal [m] | min obs [m] | mean state err [m] | solve mean [ms] | solve p90 [ms] | ms/eval | yaw p90 [rad] | pixel-age max [s] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | collision | 6.94 | 1.37 | -0.055 | 0.282 | 3000 | 4433 | 62 | 1.81 | 10.85 |
| C2 | stuck | 10.23 | 0.66 | 0.038 | 0.290 | 2831 | 4048 | 66 | 2.99 | 2.83 |

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

3. **Local EFE is under-budgeted in F45.** `H=15, maxiter=20` reduced horizon and
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
- compare optimizer success fraction and ms/eval against F45;
- only after local convergence improves, test whether the remaining problem is
  waypoint acceptance radius or driveable-region clearance.


## Added F46 Interpretation

Compared with F45, local convergence improved enough that C2 avoided immediate collision, but it ended as `stuck` while still commanding motion. This means more local iterations alone do not solve the tracker. The next fair ablation is to keep global multistart for route selection, disable local multistart for waypoint tracking, and keep a short single warm-started local solve.
