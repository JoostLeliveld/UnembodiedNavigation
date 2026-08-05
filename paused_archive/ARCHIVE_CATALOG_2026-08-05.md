# Complete cleanup and archive catalog — 2026-08-05

Status vocabulary:

- **MOVED** — physically relocated to the external archive.
- **PAUSED IN PLACE** — source is tracked or entangled with active code; do not develop it,
  but move it only in a clean Git commit after references are rewritten.
- **KEEP AS EVIDENCE** — not active development, but required to support a claim/null.
- **DELETE AFTER VERIFY** — reproducible or duplicate material; deletion is safe after the
  canonical copy/hash is checked.

## A. Research directions paused in place

These are scientifically paused now. Their source remains temporarily at its old path to
avoid turning the current dirty worktree into hundreds of ambiguous deletions.

| Path | Status | Why paused | Resume only if |
|---|---|---|---|
| `experiments/optionA_commissioning/` | PAUSED IN PLACE | Earlier uncertain-input/initialisation programme; result no longer paper headline | New paper explicitly studies uncertain pose inputs |
| `experiments/single_camera_uigp_reliability/` | PAUSED IN PLACE / KEEP AS EVIDENCE | UIGP null and health experiments; WP5 is cited evidence | Only WP5 reproduction or thesis appendix needs it |
| `experiments/usable_observation/` | PAUSED IN PLACE / KEEP AS EVIDENCE | GP loses/ties geometry; factorisation is not headline | A new dataset has meaningful in-FOV failures or variable `p_qual` |
| `experiments/fused_observation_model/` | PAUSED IN PLACE / KEEP AS EVIDENCE | Availability-fusion analysis is supporting only | Reviewer requires the held-out fusion-rule ablation |
| `experiments/geometric_baseline/` | PAUSED IN PLACE / KEEP AS EVIDENCE | Paper 1 conventional baseline | Paper 1 must be reproduced |
| `experiments/single_camera_current/` | PAUSED IN PLACE / KEEP AS EVIDENCE | Frozen Paper 1 summary | Predecessor result is audited |
| `experiments/warehouse_layout_sketches/` | PAUSED IN PLACE | World-design exploration and Meerhoven are outside the critical path | Core paper is frozen and an external-validity test is affordable |
| `modules/10_active_commissioning/` | PAUSED IN PLACE | Future paper; no implementation/evidence | Current model and per-camera maps are frozen |
| `research_story/01_*` through `research_story/08_*` | PAUSED IN PLACE / KEEP AS EVIDENCE | July thesis programme has been superseded by August results | Thesis integration, not ICRA implementation |
| `research_story/10_active_commissioning/` | PAUSED IN PLACE | Explicit future work | Separate commissioning paper begins |
| `research_story/11_final_thesis_campaign/` | PAUSED IN PLACE | Old all-programme freeze checklist | Thesis-wide final campaign begins |

## B. Headlines retired but evidence retained

Do not delete these results; their nulls justify the focused method.

1. Uncertain-input GP beats point/smoothing at the real uncertainty level.
2. Learned service GP beats geometry/FOV/distance on held-out routes.
3. `p_det × p_qual` provides a useful separate quality field in the single-camera data.
4. Per-camera spatial `R_cond` beats a pooled constant.
5. More cameras or uniform fusion beats best-camera selection.
6. The hit/miss mixture changes route choice.
7. Geometry/occlusion modelling is a headline contribution.
8. Active commissioning is part of the current paper.
9. Meerhoven is required evidence or “twelve cameras beat four.”
10. Infrastructure localisation replaces modern onboard SLAM.
11. The paper provides formal safety guarantees or hardware validation.

Evidence folders to keep even while paused:

- `logs/studies/optionA_commissioning/`
- `logs/studies/single_camera_uigp_reliability/`
- `logs/studies/usable_observation/`
- `logs/studies/fused_observation_model/`
- `logs/studies/geometry_visibility_prior/`
- `logs/studies/multicamera_fusion_extension/`
- every corresponding `RESULTS.md`, JSON/CSV summary, figure, and preregistration

## C. World and simulator branches paused in place

| Path | Status | Reason |
|---|---|---|
| `src/sim/gazebo_worlds/worlds/warehouse_full_4cam_asym.world.sdf` | PAUSED IN PLACE | Layout experiment; not an evidence world |
| `src/sim/gazebo_worlds/worlds/warehouse_full_4cam_mixed.world.sdf` | PAUSED IN PLACE | Layout experiment; not an evidence world |
| `src/sim/gazebo_worlds/worlds/warehouse_full_4cam_staggered.world.sdf` | PAUSED IN PLACE | Layout experiment; not an evidence world |
| `src/sim/gazebo_worlds/worlds/warehouse_meerhoven.world.sdf` | PAUSED IN PLACE | Optional external-validity world |
| `src/sim/models/external_camera_e/` through `external_camera_l/` | PAUSED IN PLACE | Meerhoven-only cameras |
| Meerhoven entries in `tasks.yaml`, `world_profiles.yaml`, and `bringup_sim.launch.py` | PAUSED IN PLACE | Interleaved with active uncommitted changes; remove in one isolated commit later |
| `tests/experiments/test_meerhoven_world_contract.py` | PAUSED IN PLACE | Tests paused world, not the core campaign |
| Meerhoven bridge check in `tests/sim/test_gazebo_version_contract.py` | PAUSED IN PLACE | Same reason |

Active evidence worlds remain only `warehouse_aws.world.sdf` and
`warehouse_full_4cam.world.sdf`.

## D. Detector datasets archived as cold data

The entire `logs/perception_datasets/` tree is eligible for cold storage because campaign
runtime needs the chosen model, not its raw training images. Preserve dataset manifests,
split lists, audit summaries, and selected contact sheets in `paper_artifacts` or the
archive.

Top-level payloads:

- `fourcam_capture_v1/` — superseded capture, ~1.2 GB
- `warehouse_yolo_dataset_4cam_v3_20260724/` — chosen four-camera training set, ~3.2 GB
- `warehouse_yolo_dataset_v1/` — Paper 1 training set, ~356 MB
- `warehouse_meerhoven_yolo_v1/` — paused-world capture, ~188 MB
- `warehouse_yolo_dataset_4cam_v2_train_A_smoke_20260722/` — smoke training set, ~134 MB
- `_smoke_4cam_capA/` — smoke capture, ~21 MB
- `warehouse_yolo_dataset_4cam_v1_diag/` — diagnostic set, ~9 MB
- every `*.failed_*` directory
- the empty/superseded `archive/` folder

Status after relocation: **MOVED**. Restore only for detector retraining/audit.

## E. Detector model cleanup

Keep active:

- `logs/perception_models/warehouse_yolo_detector_v1/`
- `logs/perception_models/warehouse_yolo_detector_4cam_v3_960/`

Archive:

- `logs/perception_models/archive/`
- `logs/perception_models/warehouse_yolo_detector_4cam_v2_640_diag/`

Inside every retained model directory, `ultralytics_run/weights/last.pt` and duplicate
copies of `model.pt`/`best.pt` are **DELETE AFTER VERIFY** candidates. Retain one canonical
checkpoint, its SHA-256, manifest, training configuration, and metrics.

## F. Visibility/campaign logs kept active

These paths are not moved:

- `_paper_runs/`
- `honest_campaign_v1/`
- `whitenoise_campaign_v1/`
- `warehouse_visibility_campaign_v1/`
- `warehouse_visibility_capture_v1/`
- `warehouse_visibility_targets_v1/`
- `warehouse_visibility_gp_v1/`
- `spawn_grid_20260727/` — final-campaign configs directly reference its GP artifacts

## G. Visibility/campaign logs moved to cold archive

### Old Paper 1 staging and diagnostics

- `_shot_capture/`
- `_smoke_c0_v2/`
- `aws_targets_v7b_col461/`
- `belief_aware_gp_score_v1/`
- `belief_gp_events/`
- `optionA_whitenoise_events/`
- `single_cam_commissioning_v1/`
- `single_cam_uigp_capture_v1/`
- `whitenoise_validation_v1/`
- `paired_lowlat_others/`
- `paired_lowlat_taskA/`
- `paired_lowlat_west/`
- `paired_mechanism_current_taskA/`
- `paired_mechanism_current_west/`

### Retired or exploratory worlds/captures

- `big_capture_v1/`
- `big_targets_v1/`
- `warehouse_big_zeroshot_v1/`
- `warehouse_big_zeroshot_v1b/`
- `warehouse_full_4cam_showcase_v1/`
- `warehouse_full_4cam_showcase_v1_runner.log`
- `full2cam_capA_v1/`
- `full2cam_capB_v1/`
- `full2cam_demo_v3/`
- `full2cam_tgtA_v1/`
- `full2cam_tgtB_v1/`
- `stack_capture/`
- `stack_capture2/`
- `stack_targets/`
- `stack_targets2/`
- `depth_sensed_initial_gp_v1/`
- `fourcam_actual_20260715/`
- `single_camera_c_long_20260716/`
- `single_camera_c_multi_aisle_20260716/`

### Multi-camera bringup/scheduling/localisation probes

- `mc_2x2/`
- `mc_blind/`
- `mc_fusion/`
- `mc_geomsel/`
- `mc_gt/`
- `mc_loc/`
- `mc_locC/`
- `mc_missions/`
- `mc_missions_smoke/`
- `mc_percam/`
- `mc_ratecheck/`
- `mc_sched/`, `mc_sched2/`, `mc_sched3/`
- `mc_sf/`
- `mc_slow/`
- `mc_smoke/`
- `mc_smooth/`
- `mc_tour_one/`
- `mc_tours/`
- `mc_verify/`
- `mc_video/`
- `mc_wp13/`, `mc_wp13c/`, `mc_wp13_fffb/`, `mc_wp13_multi/`

### Hard-route/showcase tuning and sweeps

- `multicam_hard_free_complete_lanes_ready/`
- `multicam_hard_free_fused_terminal_gate_exec/`
- `multicam_hard_free_fused_terminal_gate_v2/`
- `multicam_hard_free_smoke_fc0c670/`
- `multicam_hard_free_smoke_fc0c670_real/`
- `multicam_hard_free_terminal_gate/`
- `multicam_hard_free_terminal_gate_dryrun/`
- `multicam_hard_free_terminal_gate_exec/`
- `multicam_hard_free_terminal_gate_real/`
- `multicam_solve_showcase/`
- `multicam_solve_showcase_fixed/`
- `multicam_solve_showcase_real/`
- `rob_campaign/`
- `rob_val/`
- `rsweep_r08/`, `rsweep_r14/`, `rsweep_r21/`
- `clv2_smoke/`, `clv3_smoke/`

Status after relocation: **MOVED**. These are recoverable from the external archive, but
they are not inputs to the focused campaign.

## H. Study payloads moved to cold archive

The following result families are useful historical work but do not need to sit in the
active `logs/studies` search surface:

- `fourcam_detector_audit/`
- `geometry_visibility_prior/`
- `multicam_nav_demo/`
- `optionA_commissioning/`
- `usable_observation/`
- `warehouse_layout_sketches/`

Their code/results remain documented in this catalog. Status after relocation: **MOVED**.

The following stay active because the focused paper cites them directly:

- `achievable_precision_map/`
- `bayesian_filter_showcase/`
- `calibration_drift_lifecycle/`
- `efe_hit_miss_mixture/`
- `external_camera_bias_model/`
- `multicamera_commissioning_bigwarehouse/`
- `multicamera_fusion_extension/`
- `network_commissioning_realism/`
- `operational_residual_rcond/`
- `planner_covariance_branching/`
- `projection_amplification/`
- `residual_logging_schema/`
- `single_camera_uigp_reliability/` — retained because WP5/self-monitoring is cited
- `fused_observation_model/` — retained temporarily because Module 07 links it directly

## I. Generated local material

| Path | Status | Action |
|---|---|---|
| `log/` | MOVED | Colcon build logs; archive once, then normally delete future copies |
| `build/` | DELETE AFTER VERIFY | Rebuilt by `colcon build`; keep temporarily so active runtime is not interrupted |
| `install/` | DELETE AFTER VERIFY | Same; rebuild before the campaign |
| `.pytest_cache/` and every `__pycache__/` | DELETE AFTER VERIFY | Pure cache |
| `rec_launch.log` | MOVED | One-off log |
| ignored root `yolo26n.pt` | MOVED | Generic checkpoint, not a frozen artifact |
| `local_artifacts/` duplicate models | DELETE AFTER VERIFY | Remove after comparing hashes with canonical runtime models |
| `paper_summary.txt` | DELETE AFTER VERIFY | Auto-generated |
| `logs/paper_figures/_archive_nonpaper_20260612/` | DELETE AFTER VERIFY | Duplicate non-paper/staging figures |

## J. Script/config cleanup after the campaign

Pause now; physically move in a dedicated Git cleanup commit after the final campaign so
the worktree remains reviewable:

- `_mc_*.yaml`, `_rsweep_*.yaml`, and hard-route debug configs under
  `scripts/visibility_comparison/`
- `_clv2_smoke.yaml` and `_clv3_smoke.yaml` after the final configs pass
- `render_multicam_*showcase*.py` and offline hard-route renderers
- old world/showcase capture scripts that directly reference archived `stack_capture*`
- legacy perception audit scripts no longer used by the frozen detector
- superseded paper-figure scripts once their final artifact is promoted

Do not move `run_visibility_campaign.py`, `_clv2.yaml`, `_clv3.yaml`, campaign metrics,
the projection/calibration tools, or the core paper analysis scripts.

## K. Repositories and presentation packages

| Path | Status | Rule |
|---|---|---|
| `../RobotControlExternalCamera/` | KEEP AS EVIDENCE / FROZEN | Paper 1 only; no new features |
| `../thesis-report/` | KEEP AS EVIDENCE / FROZEN | Historical submitted paper |
| `../midterm_presentation/` | PAUSED IN PLACE | Presentation artifact, not research source |
| `../meeting_results_update_2026-07-27/` | PAUSED IN PLACE | Meeting snapshot |
| `../side_projects/` | PAUSED IN PLACE | Unrelated to ICRA critical path |
| `../_archive/` | EXTERNAL COLD STORAGE | Exclude from normal repository searches |

## L. Never clean up as “clutter”

- Any canonical campaign or raw capture used for a quoted result unless a verified cold
  copy exists.
- `RESULTS.md`, preregistrations, manifests, artifact hashes, summary JSON/CSV, and analysis
  scripts.
- Null-result evidence.
- v2 and v3 calibration artifacts; both are immutable experiment arms.
- Current detector checkpoints and the exact GP artifacts referenced by final configs.
- The GT firewall, operational/evaluation schema, metrics code, and runtime contracts.

