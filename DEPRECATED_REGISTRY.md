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

## DEFERRED — surgical pass NOT yet done (needs `colcon build` + retest)

These are in-place edits to ACTIVE files, intentionally left for a separate pass:
- **Section 4** dead code blocks inside active nodes (`pixel_to_bev_state_node.py`,
  `efe_agent_node.py`, `unicycle_planner_node.py`, `experiment_logger.py`): keypoint-heading,
  local-EFE reference-segment tracking, and `odom_measurement`/`visual_heading` validation modes.
- **Section 5** legacy ENTRIES inside `tasks.yaml` and `world_profiles.yaml` (keep F31_b1, a0,
  b5, warehouse_aws; drop occ_light/longshadow/putaway/parcel + exploratory AWS tasks).
- `src/perception/.../pose_keypoints.py` (+ its test) — archive once heading surgery lands.

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
