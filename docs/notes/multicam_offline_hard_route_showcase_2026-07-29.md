# Multicamera hard-route offline showcase — 2026-07-29

## Scope

This is the required offline gate before another Gazebo campaign. It uses:

- `warehouse_full_4cam.world.sdf`;
- the fused four-camera GP artifact from `_rob_val.yaml`;
- the real `UnicyclePlannerBase` objective and optimizer;
- the `rob_hardA` and `rob_hardB` start, yaw, and final goal only.

Task mission waypoints are deliberately ignored. The route initializer is
generated deterministically from driveable and collision geometry and does not
consume the GP, detector output, camera identity, or condition label.

## Estimator fixes established offline

The recorded pre-fix hardA run
`logs/visibility_comparison/rsweep_r08/rob_hardA/C2/seed0/experiment_20260729_142121`
contains the two estimator failures:

- 5 metric per-camera corrections were rejected by the 0.5 m *pixel* jump gate;
- 164/345 deduplicated corrections reported negative NIS;
- 811/854 logged full belief covariance rows were indefinite, with minimum
  eigenvalue -127.85.

Two deterministic regression tests now cover the fixes:

1. metric measurements select the metric gate even when per-camera re-anchoring
   is disabled;
2. `camera_xy_only` commits a PSD covariance by zeroing XY-heading cross terms
   rather than combining posterior XY variance with prior cross-covariance.

These are estimator-invariant tests, not synthetic navigation evidence.

## Planner failures found offline

The hard-route solve had four independent problems:

1. the full-camera driveable map is one open-floor rectangle, so the old
   corridor-only lane generator returned zero seeds;
2. the 75 × 0.4 s horizon has only 18.0 m kinematic reach, while the safe hard
   routes are 20.0–21.1 m before turn time;
3. the route initializer used a fixed 0.18 m waypoint radius despite moving
   0.24 m per global step, so it could skip and oscillate around corners;
4. it continued chasing the final waypoint after arrival and allowed motion
   with 0.65 rad heading error, producing overshoot and corner cutting.

The offline candidate fixes are:

- collision-aware, condition-neutral A* seed generation with line-of-sight
  simplification;
- robot radius + no-go margin + one global translation step of seed clearance;
- step-scaled waypoint arrival, stop-on-final-arrival, and a 0.25 rad
  turn-before-drive gate;
- global horizon 120 and optimizer budget 120 for the hard routes.

## Results

| Task | Setting | Converged | Valid | Goal gap | Clearance | Path length | Straight-line deviation | Heading change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rob_hardA | old H75 / iter60 | no | yes | 5.62 m | 0.007 m | 15.07 m | 5.68 m | 1.12 rad |
| rob_hardA | offline fix H120 / iter120 | **yes** | **yes** | **0.17 m** | **0.153 m** | 19.68 m | **5.90 m** | **3.13 rad** |
| rob_hardB | old H75 / iter60 | no | yes | 3.21 m | 0.019 m | 17.54 m | 6.78 m | 2.14 rad |
| rob_hardB | offline fix H120 / iter120 | **yes** | **yes** | **0.12 m** | **0.209 m** | 20.88 m | **7.13 m** | **5.45 rad** |

Both fixed routes pass the 0.25 m terminal goal gate, are collision-valid, and
are visibly non-straight. The optimizer selected the automatically generated
`collision_astar_shortest` basin for both tasks and formally converged.

![Offline hard-route comparison](../logs/studies/multicam_nav_demo/offline_hard_routes/offline_multicam_hard_route_gate.png)

Machine-readable results:
`logs/studies/multicam_nav_demo/offline_hard_routes/offline_multicam_hard_route_gate.json`.

Reproduce:

```bash
source install/setup.bash
MPLCONFIGDIR=/tmp/mpl_multicam \
python3 scripts/visibility_comparison/diag/offline_multicam_hard_route_gate.py
```

## Interpretation

The offline gate now passes. This does **not** yet claim real Gazebo success:
perception timing, correction freshness, local tracking, and collision outcomes
still require waypoint-free hard-route runs after freezing these settings.
