# F21 Initial Planning Behavior

- figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F21_initial_planning_behavior.png`
- samples: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F21_initial_planning_behavior.csv`
- source run: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/initial_rollout_diagnostics/F21_initial_planning_behavior_v1/`

## Task

- world: `warehouse_aws.world.sdf`
- task: `B1_apron_a4_to_uppermid_a3`
- start: `(3.20, -1.00, yaw=0.0)`; robot faces east as if it just serviced the right shelf
- goal: `(1.00, 1.75)`
- no mission waypoints; route seeds are optimizer initializations only
- driveable-region rule: 2-sigma belief-tube log barrier

## What The Optimizer Currently Converges To

| condition | horizon | best moving basin | total J | risk | ambiguity | drive/barrier | terminal d | path |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| C1 constant-R | 80 | A3 detour seed | 1518.6 | 463.5 | 0.0 | 1055.1 | 0.05 m | 10.0 m |
| C1 constant-R | 120 | A3 detour seed | 1879.3 | 656.1 | 0.0 | 1223.2 | 0.05 m | 11.0 m |
| C2 visibility-aware | 80 | A3 detour seed | 3252.6 | 547.7 | 1659.8 | 1045.1 | 0.15 m | 9.3 m |
| C2 visibility-aware | 120 | none; safe stop | 27488.2 | 24371.6 | 1796.8 | 1319.8 | 3.52 m | 0.0 m |

The H80 C2 A3-detour seed is the useful behavior: it reaches near the goal
while keeping the belief tube inside the known driveable floor. The cold C2
initialization still safe-stops, so this is not yet a clean autonomous route
choice from a neutral initialization.

The H120 C2 result is currently not useful as route-choice evidence. Under the
new 2-sigma barrier plus current goal/risk schedule, it collapses to the
safe-stop basin for all tested initializations.

## Interpretation

F21 is not paper evidence yet. It is the first "actual initial planning"
snapshot after the driveable-region barrier change:

- the objective can score a valid C2 detour path at H80;
- the long-horizon C2 landscape is still too brittle/conservative;
- the next target is not world design, but objective/optimizer tuning so the
  C2 moving basin is found without relying on a named route seed.

The desired next F22 should compare **neutral multistart candidates** after
tuning the goal schedule and barrier strength, and should show whether C2
selects the visible moving route while C1 prefers a less reliable/direct route
or accumulates higher risk.

