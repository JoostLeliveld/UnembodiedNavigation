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
| archived (pre-code-fix) | `logs/visibility_comparison/paper_taskA_model_selection_c2_v1` | model-selection evidence |
| archived (pre-code-fix) | `logs/visibility_comparison/paper_taskA_mc_nominal_c1_vs_c2_v1` | nominal C1/C2 comparison |
| archived (pre-code-fix) | `logs/visibility_comparison/paper_taskA_mc_highnoise_c1_vs_c2_v1` | high-noise C1/C2 comparison |

## Exploratory AWS Line

| Status | Asset / Run Family | Notes |
| --- | --- | --- |
| exploratory | `logs/perception_datasets/aws_simseg_v2` | AWS detector data line |
| exploratory | `logs/perception_models/aws_yolo_simseg_v2` | AWS detector line |
| exploratory | `logs/visibility_comparison/aws_gp_v5` | Latest retained AWS GP line; capture/targets cleaned, only fitted artifact and diagnostic plots remain |
| candidate (IN PROGRESS) | `logs/visibility_comparison/paper_final_v1` | Current AWS candidate runtime: one-shot global EFE route choice (H80/dt0.25, neutral multistart) followed by a shared simple proportional local tracker (`use_simple_local_controller:true`). The local tracker is execution plumbing, not the GP contribution; it uses odom yaw, configurable yaw gate, and a predicted mean-clearance gate. Config `aws_paper_final_config.yaml`. Evidence status remains candidate until the full artifact chain and final figures/tables are complete. |
| diagnostic | `logs/visibility_comparison/localEFE_paper_v1` | Low-rate local belief-space EFE variant (`use_simple_local_controller:false`). Useful for diagnosing local solver behavior and belief-closed control, but not the active AWS candidate unless explicitly re-registered with a complete artifact chain. |
| diagnostic | `logs/visibility_comparison/f24_r01_gazebo_smoke_v2` | F25 Gazebo smoke: both conditions crashed (geometry penetration); config aws_f24_r01_gazebo_smoke_config.yaml |
| diagnostic | `logs/visibility_comparison/initial_rollout_diagnostics/` | F26 config (aws_f26_r01_gazebo_smoke_config.yaml) addresses F25 root causes: nogo_safe_distance 0.13→0.30, local_optimizer_maxiter 60→25 |

## Timing And Initial-Plan Diagnostics

| Status | Asset / Run Family | Notes |
| --- | --- | --- |
| diagnostic | `timing_presentation/figures/` | horizon, runtime, yaw, tracker, and route-choice diagnostic figures (F1+); useful for method development, not evidence unless tied to final logs |
| diagnostic | `logs/visibility_comparison/initial_rollout_diagnostics/` | initial-plan sweeps showing objective/optimizer behavior before Gazebo validation |

## Invalid Or Rejected Lines

| Status | Line | Reason |
| --- | --- | --- |
| invalid | visible-goal AWS route-choice probe | baseline behavior already used the detour-like route; learned condition stalled at high ambiguity weight |
| invalid | dark-final-goal AWS route-choice probe | final goal was itself camera-poor, confounding route-choice interpretation |
| invalid | waypoint-driven mission behavior | route sequence is externally imposed rather than emerging from EFE |
| invalid | oversized-ambiguity-only demonstration | route difference is not scientifically persuasive if it only appears by overwhelming the rest of the objective |

## Adding A New Evidence Line

Add a row here only after the chain is complete. Until then, mark the run
`exploratory` or `diagnostic` and avoid paper-result language.

For the current final AWS experiment plan, see
`docs/final_experiment_definition.md`.
