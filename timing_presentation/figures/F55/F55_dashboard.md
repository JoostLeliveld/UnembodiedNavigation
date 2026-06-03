# F55 - Runtime Outcome Classification Check

Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F55/F55_dashboard.png`

PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F55/F55_dashboard.pdf`

Config: `/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/aws_f55_runtime_outcome_classification_config.yaml`

Log root: `/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/f55_runtime_outcome_classification_v2`

## What Changed

F55 keeps the F50/F49 runtime method contract and changes only the campaign/logger semantics. The corrected v2 run starts the stuck-detection window after the first real command, so long global solves are not counted as idle motion before execution begins.

This is diagnostic, not paper evidence. It checks whether the runtime can distinguish `stuck`, `collision`, `timeout`, `goal_reached`, and `infra_invalid` before planner changes.

## Outcomes

| Condition | Seed | Outcome | Path m | Min goal m | Mean truth error m |
|---|---:|---|---:|---:|---:|
| C1 | 0 | stuck | 2.72 | 1.900 | 0.485 |
| C1 | 1 | stuck | 3.44 | 1.899 | 1.051 |
| C1 | 2 | collision | 3.82 | 2.202 | 2.568 |
| C2 | 0 | goal_reached | 6.58 | 0.184 | 0.179 |
| C2 | 1 | goal_reached | 6.44 | 0.211 | 0.174 |
| C2 | 2 | goal_reached | 5.97 | 0.108 | 0.182 |

Earlier comparison:

| Run | Outcomes |
|---|---|
| `f50_b1_tight_local_goal_3seed_v1` | 3 collision, 2 infra_invalid, 1 goal_reached |
| `f54_b1_visual_heading_ablation_v1` | 1 collision, 2 infra_invalid |
| `f55_runtime_outcome_classification_v1` | 6 stuck, but invalid because stuck fired immediately after global solve |
| `f55_runtime_outcome_classification_v2` | 2 stuck, 1 collision, 3 goal_reached, 0 infra_invalid |

## Decision

Keep the logger semantics fix. F55 v2 passes the classification gate: wall-clock infra-invalids are removed, stable goal reaching counts as success, and no-progress safe stops are labeled `stuck` rather than collision or infra failure.

The next single-change test is F56: keep the global EFE route choice unchanged, but replace local EFE waypoint tracking with the robust simple local tracker to isolate route selection from local tracking failure.
