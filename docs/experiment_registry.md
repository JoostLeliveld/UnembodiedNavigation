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
| exploratory | `logs/visibility_comparison/aws_gp_v5` | Prior AWS GP line. NOTE: fit WITHOUT the R4 occluder ("C1-ONLY diagnostic" per old SDF comment) — not a C2-valid GP. Superseded by aws_gp_v6. |
| exploratory (CURRENT) | `logs/visibility_comparison/aws_capture_v7` + `aws_targets_v7` + `aws_gp_targets_v7` + `aws_gp_v7` | F87 GP line on the LOCKED raised+back camera (z=4.8, y=−5.5; south/side walls moved with it). 912 frames, 647/912 detected (71%), driveable-only sample filter, tuned length_scale=0.90 / noise_var=0.05 / beta=0.5. A1 made observable by the camera move + length-scale (A1-west raw YOLO 0.74–0.83; ρ_plan ~0.55) WITHOUT de-occlusion; A4 rack-shadow stays low (~0.005). **F87 offline gate PASS: C1→NW-blind (d=0.19), C2→south-visible through A1 (d=0.35, clear).** Config `gp_artifact` points here. Figures: `gp_pipeline_aws_v7.pdf`, `F87_offline_rollout_v7.png`, `problem_setup_camera.pdf`. v6/v6b are STALE (old camera z=4.5, y=−4.9). |
| stale (old camera) | `logs/visibility_comparison/aws_capture_v6` + `aws_targets_v6` + `aws_gp_targets_v6` + `aws_gp_v6` (+ `aws_gp_v6b`) | F86a v4 GP line: first C2-VALID GP on the cleaned geometry (R4 occluding crate stack, continuous R1 left shelf). 912 frames (24x20x4), 622/912 detected, camera [0,-4.9,4.5], 18-prism geometry embedded. Strong discriminating field: NW-blind band 0.013 vs south-visible 0.589 (`aws_gp_v6/Pmap_sanity.png`). FINDING: A1-upper goal is camera-poor with no observable approach → visibility-aware C2 stops safely before the blind mid-A1 band (principled; confirmed robust to 4x tighter goal prior), constant-R C1 commits. Offline route-split gate FAILs by design (C2 does not traverse a distinct both-reach route) — this is the stability/safe-stop regime, not a both-reach route-split. Config `aws_f86a_camera_xy_config.yaml` (`nogo_penalty_type: warning_band`, `nogo_weight: 2000`). DECISION (user): restore a both-reach route-split → needs world/goal adjustment so an observable approach to the goal exists, then GP re-recapture + Gazebo v4 campaign (DEFERRED). |
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
