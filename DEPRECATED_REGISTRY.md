# Registry of Deprecated & Legacy Components

This document compiles all files, folders, binary assets, and specific code blocks that have been marked as legacy under the `# [DEPRECATED_LEGACY_CLEANUP]` sweep. These components are deprecated because they support older, exploratory, or diagnostic paths and do not belong to the final paper-facing benchmark runs (F85–F88).

---

## STATUS: files/dirs MOVED to archive (2026-06-10)

The file/folder/binary items listed in sections 1–3 and 5 below have been **moved out
of the repo** (not just tagged) to the sibling archive
`/home/joostleliveld/Thesis/_archive_nonpaper/` (path-preserving mirror). 325 items,
~2.8 GB; repo 4.9 GB → 2.1 GB. See `_archive_nonpaper/ARCHIVE_MANIFEST.md` for the full
list and restore instructions, and `_archive_nonpaper/_move_log.txt` for the per-item log.

Beyond Gemini's original flags, the move also archived: GP versions v5/v6/v6b, the raw
captures (aws_capture_v6/v7) + training datasets (aws_simseg_v2, keypoint sets) per
"archive raw, keep fitted", the full f24–f85 campaign suite + probe/dry runs,
~61 legacy `*_config.yaml`, ~35 diagnostic scripts, 5 legacy worlds + 3 world backups,
old perception models, root build logs/`runs`/`timing_presentation`, and the **stale
`PAPER_ASSET_INDEX.md`** (it referenced a superseded detector/GP).

Paper KEEP set retained: world `warehouse_aws.world.sdf`, fitted GP `aws_gp_v7`, detector
`aws_yolo_simseg_v2`, configs `aws_f31b1_final_config.yaml` + `aws_f86a_camera_xy_config.yaml`,
the active pipeline + F88 engine `efe_offline_lab.py`, `scripts/paper_figures/`, the
GP/detector capture+fit pipeline, the whole `src/` tree, `docs/`, `tests/`. Verified after
the move: KEEP-set intact, F88 offline smoke (C1+C2) PASS, figure-generator smoke PASS.

## Archive sweep + method update (2026-06-11)

Moved to `_archive_nonpaper/logs/visibility_comparison/` (reversible, path-preserving):
`aws_gp_v7` (superseded by **aws_gp_v7b** = v7 targets + added A0 west-corridor column),
`aws_gp_targets_v7b_col461` (intermediate), `aws_f31b1_final_v2`, and the column-shift-tainted
`aws_f31b1_final_v3`. The active paper GP is now **aws_gp_v7b**; the KEEP-set GP reference
above (`aws_gp_v7`) is historical — read it as v7b.

Source/dead-logic changes (offline-verified, build + tests green):
- Removed the dead `Q` field from `CasadiEfeParams` (process noise is rebuilt per step as the
  exact `Q_d` — see `docs/uncertainty_propagation.md`).
- `nogo_cost.py` keep-in clearance switched to the true union-boundary signed distance.

STILL DEFERRED (Gazebo-gated, not yet removed): keypoint-heading code in
`pixel_to_bev_state_node.py`, local-EFE reference-segment branch in `efe_agent_node.py`,
belief-nogo + NIS-gate code paths, and the `odom_measurement`/`visual_heading` heading-mode
validation strings in `unicycle_planner_node.py`. These are inert under the locked config but
live in ROS nodes the offline smoke cannot exercise; remove only with a Gazebo smoke.

## Honesty/dead-logic audit pass 3 (2026-06-10)

Findings from a 3-way audit (objective terms / config knobs / old-logic+story). Most of the
codebase is already clean (project_to_driveable gone; no visibility_weight reward; belief-nogo,
NIS gate, keypoint/visual/displacement heading, lateral offsets all disabled-by-flag; GP never
touches heading; simple tracker never queries GP; no silent stale/profile-GP fallback). Actions:

- **Docs corrected (story ↔ code):** selected pixel source is `bbox_bottom` (not `mask_bottom`
  — that was a pass-2 error); fixed in `paper_runtime_contract.yaml` + `runtime_dataflow.md`.
  Added caveat: NIS pixel-gate is OFF (threshold 0) ⇒ use `yolo_detected_after_threshold`, never
  `pixel_corr_accepted`, for availability/rejection metrics.
- **Config de-misled (behavior-neutral):** in `aws_f31b1_final_config.yaml` — fixed the
  "a0/F87 VERBATIM" header and the copy-pasted `nw_blind/south_visible` route comment
  (real seeds: mid_cross_lane/lower_sweep_lane); `local_nogo_penalty_type: log_barrier → warning_band`
  (inert; local-EFE not called); annotated `horizon: 20` as legacy/offline-default; added
  `[DEAD under <mode>]` markers to the local-EFE / softplus-logbarrier / camera_xy_only heading
  knob groups; deleted 2 truly-unread keys (`heading_min_displacement_m`, `heading_bev_noise_sigma_m`).
- **OPEN DECISION — `observation_risk_scale = 1.25`:** an unexplained 25% amplifier on the risk
  term (not from EFE; applied equally to C1/C2). Recommend → 1.0 (or justify in paper). NOT changed
  (behavior-changing; awaiting user decision).
- **DEFERRED (ROS-node, Gazebo-gated):** legacy heading-mode strings (`odom_measurement`/
  `visual_heading`) in `unicycle_planner_node.py`; keypoint code in `pixel_to_bev_state_node.py`;
  belief-nogo + NIS-gate code; the now-dead `goal_progress_weight` param plumbing. All inert; remove
  with a Gazebo smoke.

## Objective cleanup (2026-06-10) — goal_progress reward REMOVED

- The metric goal-distance reward `goal_progress_weight * ||mean-goal||^2` was removed from
  both objective implementations (`casadi_efe.py`, `base_planner.py::_evaluate_controls`). It is
  a non-EFE goal attractor (goal-seeking must emerge from the EFE goal-prior in the risk term),
  and its own code comment showed it had been added to mask the zero-velocity/stop-short local
  minimum. Verified behavior-neutral: it was weight 0 in every active config; offline C1/C2
  solves are byte-identical after removal (`goal_progress_cost=0`). `goal_progress_n_steps` is
  KEPT (it is the legit goal-prior covariance anneal schedule, not a goal reward).
- DEFERRED (inert plumbing, needs build+Gazebo to remove safely): the `goal_progress_weight`
  param declarations in `unicycle_planner_node.py`, launch defaults in `visibility_launch_common.py`,
  the EFEParams dataclass field, and the logged `terminal_goal_progress_m` column — all now dead.

## Cleanup pass 2 (2026-06-10) — REMOVED / DONE

- **Section 5 config entries REMOVED.** `tasks.yaml` now keeps only the
  `warehouse_aws.world.sdf` block with tasks {F31_b1_apron_a3_mid, a0_west_to_a1_upper_blocked_mid,
  b2/b3/b4 generality, b5_a4_apron_to_a2_mid, visible_aisle_sanity_aws}; the occ_light/longshadow
  task blocks + exploratory AWS probe/F24/B1 tasks are deleted. `world_profiles.yaml` keeps only
  `warehouse_aws.world.sdf` (occ_light/longshadow/putaway/parcel profiles deleted). Verified:
  `colcon build experiments` passes, F88 offline smoke (C1+C2) PASS, a0 load PASS.
- **Docs consolidated**: 3 runtime contracts → `paper_runtime_contract.yaml` v0.5; stale v5/compact
  docs deleted; survivors realigned to v7/F31_b1/warning_band with honest OPEN F31_b1 status.
- **logs/ trimmed** to the functional floor (aws_gp_v7 + aws_yolo_simseg_v2 + a0 logs); handoff folder deleted.
- Build note: `experiments/setup.py` globs `data/visibility_gp/*` (now just README.md after the
  pass-1 npz archival); a stale `build/experiments` tree had to be cleaned once.

## DEFERRED — ROS-node dead code (audited safe, NOT yet removed)

**Section 4** dead code blocks inside active nodes — confirmed off the active path
(`camera_xy_only` + `use_simple_local_controller:true`) by audit, but **deferred** because the
available offline smoke (`build_planner`) does NOT exercise the ROS nodes; only a Gazebo campaign
would validate their removal, and the user is mid-F88. Remove these once a Gazebo smoke is approved:
- `pixel_to_bev_state_node.py`: `_yaw_from_keypoints()` + keypoint-yaw fallback (KEEP the
  `keypoint_marker_world_z` param declaration — the active config still passes it).
- `efe_agent_node.py`: `_build_local_reference_segment()` + local-ref-tracking branch/params.
- `unicycle_planner_node.py`: `odom_measurement`/`visual_heading` from heading-mode validation.
- `experiment_logger.py`: keypoint heading params (KEEP any still passed by active configs).
- `src/perception/.../pose_keypoints.py` + `pose_extraction.py` — still imported by the live
  detector node; remove only with a detector refactor.

---

## 1. Duplicate & Obsolete Python Scripts
The following Python modules and scripts have been annotated with a header tag:

- **Root duplicate module**: `casadi_efe_0dadf30.py` (Duplicate of active symbolic CasADi planner).
- **Diagnostics scripts** (`scripts/diagnostics/`):
  - `rollout_covariance_trace.py` (Offline covariance trace analysis, depends on `efe_offline_lab.py`).
  - `route_cost_crossover.py` (Offline cost evaluation tool, depends on `efe_offline_lab.py`).
  - `route_feasibility_offline.py` (Offline feasibility optimizer, depends on `efe_offline_lab.py`).
  - `sanity_offline_track.py` (Offline closed-loop sanity checker, depends on `efe_offline_lab.py`).
- **Keypoint/Pose perception scripts** (`scripts/perception/`):
  - `annotate_existing_capture_as_keypoints.py` (Legacy keypoint projection script).
  - `capture_projected_keypoint_dataset.py` (Legacy keypoint simulation capture script).
  - `train_yolo_pose.py` (Legacy YOLO pose/keypoint trainer).
- **Legacy package modules**:
  - `src/perception/perception/core/pose_keypoints.py` (Keypoint helper definitions).

---

## 2. Binary Models & GP Visibility Maps
These binary files are marked using adjacent sidecar `.deprecated` descriptor text files:

- **Root YOLO weights**:
  - `yolo11n-pose.pt` -> `yolo11n-pose.pt.deprecated`
  - `yolo26n.pt` -> `yolo26n.pt.deprecated`
- **Legacy GP visibility fields** (`src/experiments/data/visibility_gp/`):
  - `warehouse_open_shelves_empirical_visibility_gp.npz` -> `warehouse_open_shelves_empirical_visibility_gp.npz.deprecated`
  - `warehouse_putaway_empirical_visibility_gp.npz` -> `warehouse_putaway_empirical_visibility_gp.npz.deprecated`

---

## 3. Directory & Folder Annotations
Entire folders or large directories designated as legacy:

- **Archive directory**: Prepend legacy tag to `archive/README.md`.
- **Obsolete run logs**: Manifest file `logs/visibility_comparison/.deprecated_manifest` lists all legacy/exploratory campaign directories.
- **Obsolete datasets & models**: `.deprecated` sidecars created in:
  - `logs/projected_keypoint_dataset_aws_v1/`
  - `logs/projected_keypoint_dataset_aws_v2/`
  - `logs/projected_keypoint_dataset_aws_v3/`
  - `logs/keypoint_dataset_occ_light_v1/`
  - `logs/perception_models/yolo_pose_aws_v2/`

---

## 4. Specific Code Blocks in Active Nodes
Deprecated code regions inside active runtime files are demarcated with start and end markers:

- **`pixel_to_bev_state_node.py`**:
  - Keypoint yaw/heading parameter declarations.
  - Visual yaw fusion math block.
- **`efe_agent_node.py`**:
  - Local tracking trajectory builder (`_build_local_reference_segment`).
  - Local tracking parameter declarations.
- **`unicycle_planner_node.py`**:
  - Legacy validation modes (`odom_measurement` and `visual_heading`).
- **`experiment_logger.py`**:
  - Legacy keypoint parameters, retrievals, and log outputs.

---

## 5. Configuration Files
- **`tasks.yaml`**: Tagged all task profiles other than F31, F32, F33, F37, F44, F65, and F85–F88.
- **`world_profiles.yaml`**: Tagged legacy world profiles `warehouse_occ_longshadow`, `warehouse_putaway`, and `parcel_sortation`.
