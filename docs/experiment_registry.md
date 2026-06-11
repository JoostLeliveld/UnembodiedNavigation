# Experiment Registry

Separates paper evidence from exploratory diagnostics. A run is not paper
evidence unless the full artifact chain is present. Aligned 2026-06-10.

NOTE: pass-1/2 cleanup moved most historical run families to the sibling archive
`/home/joostleliveld/Thesis/_archive_nonpaper/` (see its `ARCHIVE_MANIFEST.md`).
Only the paper KEEP set lives in the repo now.

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

## Current Paper-Facing Artifacts (in repo, KEEP)

| Status | Asset | Notes |
| --- | --- | --- |
| CURRENT | `logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz` | Paper GP on the LOCKED camera (z=4.8, y=-5.5). = the v7 capture (912 frames, 647/912 detected, 71%) **plus an added A0 west-corridor column at x=-4.61**; same fit params length_scale 0.90 / noise_var 0.05 / beta 0.5; driveable-only sample filter. A1 made observable by camera move + length-scale (not de-occlusion); A4 rack-shadow stays low (~0.005). Config `gp_artifact` points here. (`aws_gp_v7` archived to `_archive_nonpaper/` 2026-06-11.) |
| CURRENT | `logs/perception_models/aws_yolo_simseg_v2/model.pt` | Paper detector (sim seg). Used for both capture and runtime. |
| world | `src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf` | Locked geometry + camera pose. |
| config | `scripts/visibility_comparison/aws_f31b1_final_config.yaml` | MAIN F31_b1 comparison runtime (v7 GP, camera_xy_only, warning_band). |
| saved a0 line | `scripts/visibility_comparison/aws_f86a_camera_xy_config.yaml` + `logs/visibility_comparison/f86a_camera_xy_v1/2/3` | Saved secondary `a0_west_to_a1_upper_blocked_mid` line for a future multi-task run. |

## Evidence Status

- **a0 / F87 (saved secondary, OFFLINE):** offline gate PASS on aws_gp_v7 —
  C1→NW-blind reaches (d≈0.19), C2→south-visible through A1 (d≈0.35, clear). This is
  an OFFLINE global-plan result for the a0 task, not a closed-loop campaign.
- **F31_b1 (MAIN): route-split OPEN.** Under the locked runtime both C1 and C2
  currently optimize to the lower-sweep; the objective has no path-length term
  (`control_weight=0`, `goal_progress=0`), so C1 has no incentive to take the shorter
  occluded route. Connector seam artifact fixed. No closed-loop F31_b1 campaign is
  valid as route-split evidence yet. See `active_research_state.md`.

## Superseded / Archived (moved out of repo)

| Asset family | Reason |
| --- | --- |
| `aws_gp_v5`, `aws_gp_v6`, `aws_gp_v6b` | Superseded GPs (no-occluder / old camera z=4.5–4.9). |
| `aws_capture_v6/v7`, `aws_targets_*`, perception_datasets | Raw capture + training data; fitted v7 GP + detector kept, raw archived. |
| `current_capture/targets/gp`, `paper_taskA_*`, `paper_final_v1` | Compact-benchmark + pre-v7 candidate runs (old runtime H80/odom-yaw/log_barrier). |
| `localEFE_paper_v1` | Abandoned local belief-space EFE variant (`use_simple_local_controller:false`). |
| f24–f85 smokes/route-choice/timing, probes, dry runs | Exploratory tuning history. |

## Invalid / Rejected Lines (do not revive as evidence)

| Line | Reason |
| --- | --- |
| visible-goal AWS route-choice probe | baseline already took the detour route; learned condition stalled at high ambiguity weight |
| dark-final-goal AWS route-choice probe | final goal itself camera-poor, confounding route-choice |
| waypoint-driven mission behavior | route externally imposed, not emergent from EFE |
| oversized-ambiguity-only demonstration | not persuasive if it only appears by overwhelming the objective |

## Adding a new evidence line

Add a row only after the chain is complete. Until then mark `exploratory` /
`diagnostic` and avoid paper-result language. The runtime contract is
`docs/paper_runtime_contract.yaml`; current status is `docs/active_research_state.md`.
