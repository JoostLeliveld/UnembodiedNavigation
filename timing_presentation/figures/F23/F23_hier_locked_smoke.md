# F23 Locked Hierarchical Offline Smoke

- figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F23/F23_hier_locked_smoke.png`
- pdf: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F23/F23_hier_locked_smoke.pdf`
- samples: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F23/F23_hier_locked_smoke.csv`

## Purpose

Show the original long-horizon global plan for C1 and C2 under the locked
AWS contract, including cost decomposition, planner-derived waypoints,
global solve time, and short-horizon local tracker solve-time samples.

This is an offline smoke test, not Gazebo evidence.

## Locked Values

- task: `B1_apron_a4_to_uppermid_a3`
- global horizon: `80`
- local horizon: `20`
- final goal-prior std: `8.0` px
- visible/missed observation std: `2.5` / `40.0` px
- command noise in Gazebo contract: `True`
- encoder noise in Gazebo contract: `True`
- route seeds: `mid_cross_lane`, `lower_sweep_lane`

## Summary

- C1 constant-R: global solve `21.12s`, J `2203.8` (risk `207.5`, amb `1497.1`, drive `499.2`), terminal d `0.05 m`, 8 waypoints, local median `0.89s`.
  - waypoints: `[[3.119, -0.426], [3.089, 0.572], [2.922, 1.37], [2.055, 1.738], [1.06, 1.738], [0.943, 1.717], [1.03, 1.71], [0.991, 1.702]]`
  - local H20 segment solve times: `[1.379, 0.82, 0.748, 0.892, 0.927]`
- C2 visibility-aware: global solve `39.74s`, J `2506.4` (risk `342.8`, amb `1666.7`, drive `496.9`), terminal d `0.09 m`, 9 waypoints, local median `1.00s`.
  - waypoints: `[[3.012, -1.462], [2.328, -2.113], [1.392, -1.89], [1.024, -0.987], [0.997, 0.046], [1.076, 0.981], [1.038, 1.783], [1.101, 1.658], [0.969, 1.669]]`
  - local H20 segment solve times: `[1.359, 0.973, 1.03, 1.084, 0.678, 0.565]`

## Interpretation

The local tracker timing is sampled on planner-derived waypoints. The
waypoints are not mission waypoints: each condition first solves its own
global EFE problem and the tracker follows that resulting plan.
