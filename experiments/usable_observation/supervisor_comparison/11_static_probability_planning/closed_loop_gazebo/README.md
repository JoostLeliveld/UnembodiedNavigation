# Closed-loop Gazebo campaign

This is a separate evidence level from the parent folder's stochastic replay. Each run starts
the actual `warehouse_full_4cam` Gazebo world, four camera streams, YOLO detector, camera
manager, Bayesian filter, EFE planner and local closed-loop controller.

Conditions are matched on task and seed:

- `C1`: constant-observability EFE;
- `C2`: static detector-probability field with the deployed single-covariance mapping;
- `C3`: the same static field with explicit Bernoulli hit/miss propagation.

The completed pilot contains one prespecified discriminating task and one matched seed. It is
a closed-loop feasibility experiment, not a powered navigation campaign. All three runs were
started in the simulator and executed commands; none reached the intended route-choice region.
Read [PILOT_RESULTS.md](PILOT_RESULTS.md) before interpreting the figures.

Outputs in this folder:

- `offline_routes/01_executable_routes.png`: complete, clearance-validated offline global
  routes for C1--C3, scored on the exact simplified polylines intended for the controller;
- `offline_routes/routes.json`: machine-readable controller waypoints and their route contract;
- `offline_routes/selected.csv`, `candidates.csv`, and `route_points.csv`: selected-route,
  candidate-set, and sampled-path data;
- `01_closed_loop_routes.png`: planned, Gazebo-GT and belief paths over the frozen four-camera
  planning field;
- `02_closed_loop_metrics.png`: belief error and navigation outcome diagnostics;
- `closed_loop_metrics.csv` / `.json`: exact per-run values and run directories;
- `campaign_pilot.yaml`: frozen configuration of the completed matched pilot;
- `campaign.yaml`: corrected diagnostic configuration for the next run after fixing planner
  infeasible-fallback behavior.

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
source install/setup.bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config experiments/usable_observation/supervisor_comparison/11_static_probability_planning/closed_loop_gazebo/campaign_pilot.yaml \
  --log-root logs/visibility_comparison/static_puse_closed_loop_gazebo_v1 \
  --run-timeout 480 --first-cmd-timeout 300 --cleanup-delay 8
```

The corrected configuration should not be promoted to a result until the planner refuses to
execute when every global candidate fails the terminal-goal feasibility gate.

## Offline route solve completed after the pilot

`solve_offline_routes.py` bypasses the failed continuous global EFE discovery step. It generates
a common set of complete Dijkstra candidates on a clearance-eroded driveable grid, simplifies
each candidate to controller waypoints, then revalidates and scores that exact polyline. A route
is rejected unless it starts and ends at the prescribed task endpoints and maintains at least
0.25 m driveable clearance when sampled every 0.04 m. Failure is explicit: no partial or
goal-infeasible route is returned.

For `route_tall_shadow_west_safe_start`, C1 selected the 14.24 m shortest route through a region
with minimum planning `p_use=0.001`. C2 and C3 selected the same 14.50 m alternative, whose
minimum planning `p_use=0.994`. All three offline paths have zero endpoint error and pass the
clearance contract. These are offline model predictions, not navigation outcomes; Gazebo was not
started, stopped, or queried while producing them.

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
python3 experiments/usable_observation/supervisor_comparison/11_static_probability_planning/closed_loop_gazebo/solve_offline_routes.py
```
