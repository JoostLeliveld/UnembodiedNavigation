# F22 Realistic Neutral Multistart Choice

- figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F22_realistic_multistart_choice.png`
- samples: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F22_realistic_multistart_choice.csv`

## Setup

- task: `B1_apron_a4_to_uppermid_a3`
- start: `(3.20, -1.00, yaw=0.0)`, shelf-facing east
- goal: `(1.00, 1.75)`
- horizon: `80`
- candidates are condition-neutral optimizer initializations, not mission waypoints.
- floor routes are condition-neutral route initializations through known driveable corridors only.
- these labels are not claims about globally shortest graph paths.

## Selected Initial Plans

- C1 constant-R: `zero_hold` with J=2300.9, risk=320.3, ambiguity=1497.1, barrier=483.4, terminal d=0.06 m.
- C2 visibility-aware: `floor_route_upper_cross` with J=2648.2, risk=471.3, ambiguity=1657.6, barrier=519.3, terminal d=0.15 m.

## Interpretation

This figure is closer to the intended comparison than F21: both C1 and C2 receive the same realistic candidate set. The selected path is the best valid optimized rollout under each condition's objective. If C2 chooses a different route, that difference comes from planner-facing observation covariance and the ambiguity/risk terms, not from route scripting.
