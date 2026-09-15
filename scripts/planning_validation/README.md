# Global route objective validation (offline, no Gazebo)

Everything here runs without ROS, without Gazebo and without the detector: it
reads the committed campaign config, tasks, world geometry and the fitted GP
artifact, and exercises the planner's own code. Nothing is retrained and no
existing run, log, route or manifest is modified.

Background and results: [`docs/global_planner_objective_correction.md`](../../docs/global_planner_objective_correction.md).

| script | what it does |
|---|---|
| `minimal_duration_bias_repro.py` | Minimal two-route reproduction of the duration-dependent accounting defect: identical constant q and R, one route 4 m longer, and the legacy ranking flips with a pure change of measurement units. Exits non-zero if the reproduction fails. |
| `run_global_route_validation.py` | The validation grid (tasks x q/R conditions x objectives x footprints x gate sets). Writes a **new** versioned artifact directory and refuses to overwrite an existing one. |
| `check_global_solve.py` | Runs the DEPLOYED solve path (`UnicyclePlannerBase.plan`, CasADi + L-BFGS-B, multistarted from the lane-graph seeds) with the corrected objective, and reports convergence, winning seed and cost decomposition. Prints only. |
| `freeze_corrected_routes.py` | Turns a validation artifact directory into `frozen_routes.json` for a provisional campaign. Refuses to overwrite. |
| `offline_planner_setup.py` | Shared construction of the global-stage planner, safety model and route candidates from the campaign config. Run it directly for a smoke check of the task/candidate wiring. |

## Typical order

```bash
python3 scripts/planning_validation/minimal_duration_bias_repro.py
python3 -m pytest tests/planning -q
python3 scripts/planning_validation/run_global_route_validation.py --tag <tag>
python3 scripts/planning_validation/check_global_solve.py
python3 scripts/planning_validation/freeze_corrected_routes.py \
    paper_artifacts/planning/global_objective_v2_<tag>
```

Only after those gates pass should the simulator campaign be started, using the
frozen routes and the command recorded in `frozen_routes.json`.

## Requirements

`numpy < 2`, `scipy`, `casadi`, `pyyaml`. The planner's `risk_components` uses a
scalar conversion that NumPy 2 rejects elsewhere in the stack, so pin NumPy 1.x
to match the runtime.
