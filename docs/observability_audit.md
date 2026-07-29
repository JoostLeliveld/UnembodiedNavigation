# Observability audit — Phase 0 of the spatial-observability refocus

**Date:** 2026-07-23 · **Branch:** `agent/reliability-contracts-multicamera` (HEAD `9cf9664`)
**Purpose:** repository audit before refocusing on the per-camera spatial observability map
`p_c(x, y) = P(camera c produces a usable robot observation | robot state)`.
No runtime behavior was modified for this audit.

The refocus explicitly makes the contribution the *observability field itself*, not
learned measurement covariance and not covariance-blending rules. This audit maps what
exists, what the current GP actually predicts, where ground truth touches the pipeline,
and what is missing relative to the new target.

---

## 1. Current data flow

### 1.1 Runtime (live Gazebo run)

```
Gazebo camera(s)
  → YOLO detector node                         [single-cam or 4-cam batched]
      src/perception/perception/nodes/yolo_robot_detector_node.py
      src/perception/perception/nodes/batched_four_camera_yolo_node.py
      selection + threshold: src/perception/perception/core/yolo_selection.py
        (yolo_selection.py:144  detected_after_threshold = raw_best_score >= confidence_threshold)
  → pixel → BEV projection
      src/state/state/core/pixel_to_bev.py, nodes/pixel_to_bev_state_node.py
      camera model: src/unav_common/unav_common/camera_model.py (ObliqueCameraModel)
  → belief filter + planner (EFE)
      src/planning/planning/nodes/unicycle_planner_node.py
      core: src/planning/planning/core/casadi_efe.py
  → per-run CSV logging
      src/experiments/experiments/nodes/experiment_logger.py   (single-cam campaign)
      experiments/multicamera_commissioning_bigwarehouse/tools/record_operational_logs.py
      + record_evaluation_truth.py                             (4-cam world)
```

### 1.2 Offline (training / evaluation)

```
Campaign logs (logs/visibility_comparison/<campaign>/, logs/studies/<study>/)
  → belief-stamped detector events
      scripts/visibility_comparison/build_belief_gp_events.py
      (coordinates = planner_belief_x/y + planner covariance; GT columns copied for audit only)
  → GP fit
      scripts/visibility_comparison/fit_belief_aware_gp.py     (THE canonical GP code)
  → GP artifact (.npz)  → planner C2 via visibility_artifact_path param
      unicycle_planner_node.py:120,274  → casadi_efe.expected_visibility_ca → R_plan
```

4-cam world variant (newer, contract-governed):

```
<run>/raw/experiment.csv (odom_noisy…) + <run>/raw/camera_*_perception.csv
  + <run>/evaluation_only/ground_truth.csv        [firewalled]
  → experiments/multicamera_fusion_extension/tools/load_commissioning_run.py
      (operational MapObservation frames vs EvaluationFrame — GT only in the latter)
  → opportunity rows
      src/reliability/reliability/opportunity.py  (build_opportunity_row)
      experiments/multicamera_fusion_extension/tools/build_opportunity_dataset.py
  → per-camera GP
      experiments/multicamera_fusion_extension/tools/train_factorized_gp.py
      (wrappers: train_availability_gp.py, train_quality_gp.py; delegates to fit_belief_aware_gp.py)
  → offline replay / fusion evaluation
      src/reliability/reliability/replay.py  (ReplayMode R0–R4, M5–M8, B6_health_aware_fusion)
```

---

## 2. File and function reference map

| Area | Where | Notes |
|---|---|---|
| Detection (single cam) | `src/perception/perception/nodes/yolo_robot_detector_node.py` | `confidence_threshold` param, default 0.25 (node:61) |
| Detection (4-cam batched) | `src/perception/perception/nodes/batched_four_camera_yolo_node.py`, `core/four_camera_batch.py` | validates threshold ∈ [0,1] (node:219) |
| Detection selection/threshold | `src/perception/perception/core/yolo_selection.py:144` | the single gate producing `detected_after_threshold` |
| 4-cam runtime contract | `src/perception/perception/core/four_camera_runtime_contract.py` | `BATCHED_CAMERA_ORDER = ("camera_A","camera_B","camera_C","camera_D")` (line 27); `confidence_threshold` is a contract field |
| Camera selection / handover | `src/reliability/reliability/camera_manager.py` | `CameraManager.select`, eligibility reasons, contender hysteresis |
| Camera manager node | `src/reliability/reliability/nodes/camera_manager_node.py` | live shadow = replay parity |
| Operational/eval contracts | `src/reliability/reliability/contracts.py` | `OperationalReliabilitySample`, `EvaluationOnlySample`, `CameraObservation`; `LeakageError` on eval-key intrusion (`reject_evaluation_only_keys`, line 203) |
| GT firewall | `src/reliability/reliability/firewall.py` | validates feature columns, training-loader sources, planner-facing imports, config sources |
| Contract schema docs | `docs/reliability_contracts/schema.md` + example JSONs | |
| Opportunity rows | `src/reliability/reliability/opportunity.py` | `OpportunityRow`, `build_opportunity_row` (line 188), `availability_label = int(association_valid)` (line 245); `LOOReference` + `label_loo_usability` (line 249) for non-circular usability |
| Opportunity exporter | `experiments/multicamera_fusion_extension/tools/build_opportunity_dataset.py` | CSV: sample_id, camera_id, run_id, timestamp_s, belief xy+cov, predicted uv+cov, ellipse_inside_fraction, stream_healthy, detection_received, association_valid, raw_confidence, availability_label |
| LOO labels | `experiments/multicamera_fusion_extension/tools/build_loo_labels.py` | leave-one-camera-out reference |
| GP events (single cam) | `scripts/visibility_comparison/build_belief_gp_events.py` | coords = `planner_belief_x/y` (line 193–194); manifest states GT is audit-only (line 322–323) |
| GP fit (canonical) | `scripts/visibility_comparison/fit_belief_aware_gp.py` | targets: `hit` (det_hit) or `score` (yolo_score_raw) (lines 113–120); bin-aggregate + Beta smoothing + expected-kernel latent GP over belief cov |
| Per-camera GP trainer | `experiments/multicamera_fusion_extension/tools/train_factorized_gp.py` | one camera per invocation (`--camera-id`); features `m_x, m_y` = belief xy; target forced to `hit` |
| GP held-out validation | `scripts/visibility_comparison/validate_gp_heldout.py`; `scripts/geometry_visibility/gp_usability_validation.py` | Brier/AUROC/calibration exist |
| Geometry/FOV prior | `scripts/geometry_visibility/` (module 05) | day-zero FOV+occlusion prior; `build_full4cam_planner_prior.py` records `"ground_truth_used": False` provenance |
| Planner visibility hook | `src/planning/planning/core/casadi_efe.py:185` | `expected_visibility_ca(mean, cov, prob_state)` — UT sigma points over belief cov |
| Visibility → R_plan | `casadi_efe.py:355–364` | `p_vis → _visibility_effective_score_ca → _blend_observation_covariance_ca → R_plan` |
| Trust→R mapping (source of truth) | `src/reliability/reliability/covariance_mapping.py` | precision blend; `gate_decision` accept/inflate/reject; `MissEndpointPolicy.require_reconciled()` blocks quoting the 40 px vs 120 px `r_miss_uv` mismatch |
| Planner params | `src/planning/planning/nodes/unicycle_planner_node.py:120,247–274` | `use_visibility_model`, `visibility_artifact_path`, `r_visible_uv`, `visibility_sigma_kappa` |
| Campaign runner | `scripts/visibility_comparison/run_visibility_campaign.py` | run matrix = task × condition × seed; per-run ROS domain |
| Campaign conditions | `scripts/visibility_comparison/warehouse_visibility_campaign_honest_v2.yaml` | C0 `geometric_shortest_path`, C1 `constant_R_efe`, C2 `visibility_aware_efe` + `gp_artifact` |
| Run logger (single cam) | `src/experiments/experiments/nodes/experiment_logger.py` | buffers `/ground_truth_tf` (line 761) for eval-only error columns |
| Replay / fusion modes | `src/reliability/reliability/replay.py:23` | `ReplayMode`: R0 odom-only … R4 current-GP-R, M5 sequential fusion, M6/M7/M8 selection, B6 health-aware fusion |
| Health monitoring | `reliability` health_ewma (WP5) + `experiments/single_camera_uigp_reliability/tools/wp5_drift_detection.py` | innovation-driven per-camera health |
| Metrics (canonical) | `scripts/shared/metrics.py` | Brier/logloss/AUC/Spearman/ECE — never hand-rolled |
| Log loading (canonical) | `scripts/geometry_visibility/campaign_metrics.py` | `load_run/load_detections` assert canonical columns |
| Tests | `tests/` (root pytest via `pyproject.toml`/`conftest.py`) | contract, firewall, manifest, replay tests exist |

---

## 3. Current labels and their sources

| Label | Defined at | Source signals | Used by |
|---|---|---|---|
| `det_hit` (binary detection) | campaign event logs → `build_belief_gp_events.py` | detector output after frozen threshold | `fit_belief_aware_gp.py --target hit` |
| `yolo_score_raw` (continuous) | same | raw detector confidence (pre-threshold best score) | `fit_belief_aware_gp.py --target score` — **this trained the artifact the C2 planner currently consumes** (`paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`) |
| `availability_label` | `opportunity.py:245` = `int(association_valid)` where `association_valid = detected AND association_window_valid` | detection existence + association timing, gated by predicted-visibility opportunity | `train_availability_gp.py` |
| LOO usability | `opportunity.py:249 label_loo_usability` | residual of the camera's measurement vs a leave-one-camera-out operational reference (`max_residual_px`) | `train_quality_gp.py` (non-circular quality label) |
| Eval-only labels (`collision`, `geometry_breach`, `ground_truth_localization_error_m`, …) | `contracts.py EvaluationOnlySample` | Gazebo GT | evaluation scripts only; firewalled |

**Key fact:** the *planner-facing* GP artifact is trained on **raw detector confidence**
(`yolo_score_raw`), which the refocus explicitly forbids as the target. The `hit` target and
the availability/LOO machinery already point in the new direction but are not what the
planner consumes today.

---

## 4. Every use of ground truth

Enforced policy already in place: `gt_*`/`eval_*` keys raise `LeakageError` inside
operational contracts (`contracts.py:183–210`), and `firewall.py` validates feature
tables, loader sources, and planner-facing imports. `docs/metric_definitions_and_gt_audit.md`
documents the historical odom-as-truth contamination incident and its fix.

| Site | Use | Class |
|---|---|---|
| `experiment_logger.py:628–940` | subscribes `/ground_truth_tf`, buffers truth, writes `*_gt` error columns | **evaluation only** (explicit comments; belief/state columns are the operational ones) |
| `<run>/evaluation_only/ground_truth.csv` + `record_evaluation_truth.py`, `attach_evaluation_truth.py` (4-cam tools) | GT stream captured to a separate eval-only file | **evaluation only** |
| `load_commissioning_run.py` | reads GT exclusively into `EvaluationFrame`; operational frames use `odom_noisy_*` | **evaluation only** (docstring: "Firewall discipline") |
| `contracts.py EvaluationOnlySample` | holds `gazebo_ground_truth_pose`, GT-projected pixel, GT error | **evaluation only** by construction |
| `compute_paper_metrics.py`, `scripts/paper_figures/*` | GT for reported errors/figures | **evaluation only** |
| `scripts/visibility_comparison/capture_visibility_samples.py:125` | calls `/world/<w>/set_pose` to teleport the robot over a sample grid; commanded grid pose ≙ sample coordinate | **sim control — but effectively GT-as-input** for the legacy teleport-grid dataset (commanded pose = true pose after convergence check). Superseded by belief-stamped events for the honest campaign, but any artifact still derived from teleport captures must not feed the new observability map. |
| `scripts/perception/capture_yolo_dataset.py`, `legacy/spin_capture.py`, `legacy/capture_projected_bbox_dataset.py` | sim-side auto-labels for **detector training** (segmentation-derived boxes) | detector-training concern, outside the observability-map firewall; acceptable (the detector is a fixed sensor), but must be stated as an assumption |
| `build_full4cam_planner_prior.py:119` | records `"ground_truth_used": False` provenance flag | good pattern to standardize |
| `scripts/reliability/probe_multicamera_accuracy.py`, misc diag scripts | GT to probe calibration accuracy | **calibration check / debugging** — allowed under the new rules |

**No planner or training input reads GT** in the current honest-campaign or 4-cam contract
paths. The one legacy exception is the teleport-grid capture family (and any GP artifact
descended from it, including the original `warehouse_visibility_gp_v1` lineage — the
paper-era GP was fit on teleport-grid data; the honest-campaign belief-stamped refit is
the compliant lineage).

---

## 5. Single-camera or genuinely multicamera?

**Both exist; the planner-facing path is single-camera, the contract/replay path is genuinely multicamera.**

- Genuinely multicamera: 4-cam batched detector (`camera_A..D`, `four_camera_runtime_contract.py`),
  per-camera perception CSVs, `CameraObservation.camera_id`, `CameraManager` handover state
  machine, per-camera GP training (`train_factorized_gp.py --camera-id`), offline fusion
  replay modes (M5–M8, B6), per-camera health (health_ewma), 4-cam world
  `warehouse_full_4cam.world.sdf` + `warehouse_full4cam_commissioning.launch.py`.
- Single-camera: the entire honest-campaign planner loop (C0/C1/C2), the GP artifact the
  planner loads, `expected_visibility_ca` (one scalar visibility field, no camera index),
  and the R_plan blend. There is **no per-camera probability vector reaching the planner**
  today; `visibility_artifact_path` points at one aggregate field.
- Two-world rule (CLAUDE.md): method development in `warehouse_aws` (single cam),
  frozen-method evaluation in `warehouse_full_4cam`.

---

## 6. Current GP target

- Canonical fitter `fit_belief_aware_gp.py` supports `hit` (binary `det_hit`) and `score`
  (`yolo_score_raw` clipped to [0,1]); events are aggregated on a spatial grid, Beta-smoothed,
  and fit with a latent GP whose expected-RBF kernel integrates over the **belief covariance**
  of each event (inputs `m_x, m_y, S_xx, S_xy, S_yy` = planner belief, never truth).
- The **deployed** planner artifact is the `score` target — raw detector confidence — i.e. the
  current field is "expected YOLO confidence", not `P(usable observation)`.
- The newer factorized per-camera trainers force `hit` and consume opportunity rows
  (`availability_label`), which is the closest existing thing to `p_c(x,y)` — but the
  probability semantics are conditioned on the opportunity gate (see §8, risk 2).

---

## 7. Current planner interface

- Params (`unicycle_planner_node.py`): `use_visibility_model`, `visibility_artifact_path`
  (GP .npz), `visibility_geometry_json`, `r_visible_uv`, `r_miss` (120 px runtime default),
  `visibility_sigma_kappa`.
- Core: `casadi_efe.expected_visibility_ca(mean, cov, prob_state)` — unscented transform of a
  scalar state→probability function over the belief covariance → effective score →
  `_blend_observation_covariance_ca` → `R_plan` per horizon step.
- Single source of truth for the trust→covariance formula:
  `reliability/covariance_mapping.py` (`trust_to_update_covariance`, precision blend, bounded,
  monotone; offline `geometry_visibility.trust_to_r_plan` proven identical to ~1e-9).
- Conditions are selected per-run by the campaign yaml (`planner: geometric_shortest_path |
  constant_R_efe | visibility_aware_efe`); C2 alone receives `gp_artifact`.

This interface already matches the refocus requirement "keep observability→R_plan fixed as an
implementation adapter": the only research-variable entering the planner is *which
`prob_state` field* backs `expected_visibility_ca`.

---

## 8. Risks and missing evidence (relative to the refocus)

1. **Planner-facing target is detector confidence.** The deployed C2 artifact regresses
   `yolo_score_raw`. Under the refocus this must be replaced by `p_c(x,y)` trained on a
   usable-observation label. (Phase 4 replaces the artifact, not the adapter.)
2. **Opportunity gating silently conditions the probability.** `build_opportunity_row`
   returns `None` (no record) when `inside_valid_region`/`stream_healthy`/association-window/
   scale gates fail (`opportunity.py:220–221`). So `availability_label` estimates
   `P(usable | opportunity)`, where "opportunity" itself depends on a *predicted* ellipse
   fraction from the belief — a model-dependent choice baked into the label. The refocus
   contract requires a record **per camera per synchronized opportunity including misses,
   with `failure_reason`** — the current builder cannot produce that without modification.
3. **No `failure_reason`, `route_id`, `yaw`, `state_source`, `frame_received`,
   `confidence_threshold` fields** in `OpportunityRow`. `OperationalReliabilitySample` has
   most raw ingredients (`measurement_age_s`, `measurement_stale`, `projection_valid`,
   `detector_result`, per-camera via `metadata.camera_id`) but no unified usable/failure
   verdict.
4. **Confidence threshold is not frozen across launch files**: 0.05 in
   `warehouse_full4cam_commissioning.launch.py:380` and `warehouse_multicamera_extension.launch.py:112`
   vs 0.25 in `warehouse_primary_comparison.launch.py:187`, `warehouse_visibility_capture.launch.py:140`,
   and the detector-node default. A "frozen confidence threshold" is a stated requirement of
   the usable-observation definition — one value must be chosen and recorded per dataset.
5. **Legacy teleport-grid lineage.** Any artifact descending from
   `capture_visibility_samples.py` embeds commanded (≙ true) poses as inputs. Compliant
   lineage = belief-stamped events / odom-noisy 4-cam logs. Dataset manifests must carry a
   `ground_truth_used: false` provenance flag (pattern already exists in
   `build_full4cam_planner_prior.py`).
6. **No multicamera products layer.** Nothing computes `p_any`, expected usable-camera count,
   ≥2-camera probability, dominant camera, handover regions, or bottlenecks; and no empirical
   joint-outcome check of the conditional-independence approximation exists. `CameraManager`
   makes *decisions* but publishes no probability field.
7. **Baseline gap.** Named baselines exist in pieces (constant-R planner condition; geometry
   FOV prior module 05; distance features inside the GP scripts) but there is no evaluated
   baseline ladder (global constant / distance-logistic / calibrated FOV-range / grid-frequency)
   on a held-out, spatially-blocked usable-observation dataset. Prior evidence
   (`logs`/memory: leave-region-out nulls) suggests the GP may **not** beat simple baselines —
   the Phase 4 gate must be taken seriously.
8. **Detection ≠ usable observation.** Current labels ignore latency/freshness at the *label*
   level (freshness is an opportunity gate, not part of the outcome) and ignore projection
   validity of the *measured* point. The Phase 1 contract must decide, per gate, whether it
   defines the opportunity (denominator) or the outcome (numerator) — this is presently
   implicit and inconsistent between the single-cam (`det_hit`) and 4-cam (`availability_label`)
   paths.
9. **Known constant mismatch** `r_miss_uv` 40 px (offline) vs 120 px (runtime) — reconciled
   formula, unreconciled endpoint; `MissEndpointPolicy.require_reconciled()` blocks quoting it.
   Irrelevant to the observability map itself but relevant to holding the adapter "fixed".
10. **Data coverage evidence is missing** for spatial-block evaluation in the 4-cam world:
    detector v2 misses the strict mid-range gate (0.40/0.36 at 8–12/12–16 m), so per-camera
    usable rates will be range-skewed; the coverage summary of Phase 2 must quantify this
    before any GP claims.

---

## 9. What can be reused directly (do not rebuild)

- `OperationalReliabilitySample` / `EvaluationOnlySample` + `LeakageError` firewall — extend,
  don't replace.
- `build_opportunity_row` — refactor to *always* emit a record with `usable_observation` +
  `failure_reason` instead of returning `None`.
- `fit_belief_aware_gp.py` — already per-camera-capable via `train_factorized_gp.py`; force
  target `hit`-style binary labels.
- `scripts/shared/metrics.py` (Brier/NLL/ECE/AUC), `campaign_metrics.load_run`.
- `covariance_mapping.py` + `expected_visibility_ca` — the frozen adapter of Phase 7.
- Geometry FOV/occlusion prior (module 05) — Phase 3 baseline #3 nearly for free.
- `run_visibility_campaign.py` run matrix + condition plumbing — Phase 7 conditions slot in
  as new `planner`/artifact values.
- `ReplayMode` replay harness — Phase 8 stress tests (dropout already exists as a concept in
  fusion replay).
