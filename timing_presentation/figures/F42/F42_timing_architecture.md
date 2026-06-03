# F42 - Timing Architecture Diagnosis

This figure checks whether the current instability is mainly caused by small tuning changes, a weak GPU, or the runtime architecture.

## Files

- Dashboard: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F42/F42_dashboard.png`
- Timing plot: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F42/F42_timing_architecture.png`
- Planner-only sweep: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f42_planner_only_timing/initial_rollout_sweep.png`
- Gazebo log root: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f42_b1_timing_architecture_v1`

## Key Numbers

Planner-only initial solves:
- C1 H20: 3.1 s mean solve, 2.11 m mean terminal goal distance.
- C1 H80: 11.2 s mean solve, 0.03 m mean terminal goal distance.
- C2 H20: 2.7 s mean solve, 2.98 m mean terminal goal distance.
- C2 H80: 12.3 s mean solve, 0.15 m mean terminal goal distance.

C1 Gazebo run:
- completion: `stuck`
- path: `3.55 m`, minimum goal distance: `2.41 m`
- local solve mean / median / max: `2510 / 2197 / 4637 ms`
- active control index mean / max: `0.00 / 0`
- YOLO inference mean / max: `73 / 133 ms`
- detector total latency mean / max: `0.56 / 0.64 s`

C2 Gazebo run:
- completion: `collision`
- path: `4.37 m`, minimum goal distance: `2.50 m`
- local solve mean / median / max: `3301 / 2761 / 8935 ms`
- active control index mean / max: `0.00 / 0`
- YOLO inference mean / max: `101 / 1995 ms`
- detector total latency mean / max: `0.59 / 1.01 s`

## Conclusion

The workstation is part of the wall-time problem, but it is not the only problem. YOLO is running with CUDA and typical inference is on the order of 0.07-0.11 s, while local EFE solves are multi-second and H80 initial solves are tens of seconds in the full Gazebo stack.

F42 also exposed an instrumentation gap: the existing active-control fields are sampled when a planner result is published, not at every command tick. Continuous command-tape diagnostics have now been added for the next run, so we can distinguish genuine tape resets from normal handoff-time samples.

Next fix should target architecture: reduce synchronous local EFE replanning, preserve command-tape phase across replans, and only replace an active tape when the new plan is fresh and meaningfully better. The next smoke run should use the new `exec_*` columns to verify whether the controller actually advances through the tape.
