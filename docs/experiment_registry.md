# Experiment Registry

This registry separates paper evidence from exploratory diagnostics. A run is
not paper evidence unless the full artifact chain is present.

## Evidence Chain

| Link | Required Record |
| --- | --- |
| World | fixed world file and camera pose |
| Detector | detector artifact trained/validated for that world |
| Visibility data | capture directory and manifest from that world |
| GP | fitted artifact with `P_conservative_plan_map` |
| Config | campaign config with explicit detector and GP paths |
| Logs | seeded run directories with manifests and summaries |
| Figures | generated from logs, not hand-drawn behavior claims |

## Paper-Core Runs

| Status | Asset / Run Family | Notes |
| --- | --- | --- |
| valid | `logs/visibility_comparison/current_capture` | compact benchmark visibility capture |
| valid | `logs/visibility_comparison/current_targets` | compact benchmark detector targets |
| valid | `logs/visibility_comparison/current_gp` | compact benchmark planner-facing GP |
| valid | `logs/visibility_comparison/paper_taskA_model_selection_c2_v1` | model-selection evidence |
| valid | `logs/visibility_comparison/paper_taskA_mc_nominal_c1_vs_c2_v1` | nominal C1/C2 comparison |
| valid | `logs/visibility_comparison/paper_taskA_mc_highnoise_c1_vs_c2_v1` | high-noise C1/C2 comparison |

## Exploratory AWS Line

| Status | Asset / Run Family | Notes |
| --- | --- | --- |
| exploratory | `logs/perception_datasets/aws_simseg_v2` | AWS detector data line |
| exploratory | `logs/perception_models/aws_yolo_simseg_v2` | AWS detector line |
| exploratory | `logs/visibility_comparison/aws_gp_v5` | Latest retained AWS GP line for the old AWS exploratory world; capture images/targets were cleaned to keep only the fitted artifact and current diagnostic plots |
| exploratory | `logs/visibility_comparison/experiment_b_aws_v33_smoke` | smoke/diagnostic only |

## Invalid Or Rejected Lines

| Status | Line | Reason |
| --- | --- | --- |
| invalid | visible-goal AWS route-choice probe | baseline behavior already used the detour-like route; learned condition stalled at high ambiguity weight |
| invalid | dark-final-goal AWS route-choice probe | final goal was itself camera-poor, confounding route-choice interpretation |
| invalid | waypoint-driven mission behavior | route sequence is externally imposed rather than emerging from EFE |

## Adding A New Evidence Line

Add a row here only after the chain is complete. Until then, mark the run
`exploratory` or `diagnostic` and avoid paper-result language.
