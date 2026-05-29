# F27 - R01 Gazebo Smoke Diagnostic

Config: `scripts/visibility_comparison/aws_f27_r01_gazebo_smoke_config.yaml`
Changes vs F25: `nogo_safe_distance 0.13→0.20`, `local_optimizer_maxiter 60→25`.

## Results

| condition | outcome | note |
|---|---|---|
| C1 | no completed run |
| C2 | timeout (>260s) | global solve never completed |

## Fixed vs F25

- Obstacle avoidance: `min_obstacle_distance_m = +0.119 m` — no rack/crate penetration.
- Local solve time: mean ~730 ms vs 1897 ms in F25.

## New failures

### C1: Belief-y divergence (homography outliers)
As the robot approached y≈4.5 (near north wall at 4.92), `planner_belief_y` oscillated
between ~0.3 m and 14+ m. The planner believed robot was at y≈0.3 while truth was y≈4.8.
Root causes: (1) outer walls not in known_2d_regions no-go layer, (2) homography
back-projection gives invalid y for robot positions near the northern camera boundary.

### C2: Global solve timeout
visibility_aware_efe with ambiguity_weight=8.0 did not complete global solve in 260 s.
F23 offline: C2 took ~40 s. Gazebo adds >6x overhead on first solve.

## Next steps (F28)

1. Add outer wall bounds to world_profiles.yaml known_2d_regions as non-driveable,
   OR increase run_timeout to allow C2 global solve (try 400 s).
2. Clip homography outliers: reject pixel corrections where projected y > world_ymax
   or < world_ymin (world bounds ±5 m). Already partially handled but threshold too wide.
3. Investigate C2 global solve Gazebo overhead: check if GP artifact load happens on
   first callback and adds ~200s init cost.

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F27/F27_r01_gazebo_smoke.png`
PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F27/F27_r01_gazebo_smoke.pdf`
