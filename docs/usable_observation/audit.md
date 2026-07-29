# Usable-observation audit — p_det / p_qual / p_use refocus

**Date:** 2026-07-24 · **Branch:** `agent/reliability-contracts-multicamera` (HEAD `9cf9664`)
**Scope:** repository audit before redesigning how detector *availability* and detector
*quality* enter planning, under the decomposition

```
p_det,c(s)  = P(D_c = 1 | s)                     detection availability
p_qual,c(s) = P(U_c = 1 | D_c = 1, s)            conditional operational quality
p_use,c(s)  = p_det,c(s) · p_qual,c(s) = P(D_c=1 ∧ U_c=1 | s)   usable observation
```

No runtime behavior was modified for this audit (read-only inspection only).

**Relationship to the prior audit.** `docs/observability_audit.md` (2026-07-23, Phase 0 of the
`p_c(x,y)` refocus) covers the same repository and remains valid. This document is *not* a
duplicate: it re-reads the same code through the sharper **p_det / p_qual / p_use**
decomposition the current task requires, and it verifies the specific claims that decomposition
depends on (does the pipeline separate "a detection existed" from "the detection was usable"?
does a record exist for misses? is the quality gate frozen?). Where a finding is unchanged it is
cited, not restated. Line numbers below were re-verified on 2026-07-24.

Every field below is tagged with its source domain: **GT** (Gazebo truth, evaluation-only),
**ODOM** (wheel odometry), **BELIEF** (planner belief/EKF), **STATE** (state-estimate node),
**PIXEL** (image-plane detector output), **MODEL** (a fitted artifact).

---

## A. Detection pipeline

| Question | Answer (verified) | Source |
|---|---|---|
| Where YOLO inference happens | single-cam: `src/perception/perception/nodes/yolo_robot_detector_node.py`; 4-cam batched: `src/perception/perception/nodes/batched_four_camera_yolo_node.py` (+ `core/four_camera_batch.py`) | PIXEL |
| Where confidence is thresholded | `src/perception/perception/core/yolo_selection.py:144` — `detected_after_threshold = raw_best_score >= confidence_threshold`. This single line is the D_c gate. | PIXEL |
| Confidence threshold value | **NOT frozen.** Node defaults `0.25` (`yolo_robot_detector_node.py:61`, `batched_four_camera_yolo_node.py:125`). Launch overrides: `0.05` in `warehouse_full4cam_commissioning.launch.py:380` and `warehouse_multicamera_extension.launch.py:112`; `0.25` in `warehouse_visibility_capture.launch.py:140`. A single frozen value per dataset is a task requirement and does not exist today. | — |
| Whether non-detections are logged (runtime) | Yes at the raw sample level: `OperationalReliabilitySample.detector_result` carries `detected: bool` (a miss is `detected=False`), and `recent_detector_history` is a bool tuple. So a per-frame miss is representable in the *operational contract*. See §C for where misses are then **dropped**. | PIXEL |
| Selected localization pixel | `OperationalReliabilitySample.selected_pixel: (u,v)` (`contracts.py:241`); the runtime bottom-centre selection lives in the perception node / `pixel_to_bev`. | PIXEL |
| Bounding-box / image-coordinate outputs | present in `detector_result` and `image_location` mappings (`contracts.py:240,249`), but **not** promoted to the flat, named bbox fields the target `ObservationOpportunity` contract lists (`bbox_*`, `edge_distance_px`). | PIXEL |
| Frame freshness | `measurement_age_s`, `measurement_stale` on the operational sample (`contracts.py:243–244`); consumed as an *opportunity* gate in `opportunity.py:217`, not as an *outcome*. | PIXEL |
| Camera ID | `OperationalReliabilitySample.metadata["camera_id"]` (read at `opportunity.py:203`); canonical order `("camera_A","camera_B","camera_C","camera_D")` in `four_camera_runtime_contract.py:27`. | — |
| Tracker / association | `CameraObservation` + `CameraManager` (`camera_manager.py`) hold association/handover state; `opportunity.py` uses an `association_delta_s` window, not a full tracker. | — |

---

## B. Projection / localization pipeline

| Question | Answer (verified) | Source |
|---|---|---|
| Detection → camera measurement | `src/state/state/core/pixel_to_bev.py`, node `nodes/pixel_to_bev_state_node.py`; camera model `src/unav_common/unav_common/camera_model.py` (`ObliqueCameraModel`). | PIXEL→STATE |
| Projection validity check | `OperationalReliabilitySample.projection_valid: bool` (`contracts.py:242`); geometric validity also computed in `reliability/projection.py`. | STATE |
| Measurement rejection logic | Two layers: (i) opportunity gates in `opportunity.py:212–221` (inside-region, stream-healthy, association-window, scale) decide whether a row exists at all; (ii) trust→covariance gate `reliability/covariance_mapping.py` `gate_decision` (accept / inflate / reject) at fusion time. | STATE |
| Operational state source | `belief` (planner belief, BELIEF) and `state_estimate` (STATE) mappings; `odometry` (ODOM). GP events use `planner_belief_x/y` (`build_belief_gp_events.py:193–194`). **No GT** in the operational path. | BELIEF/STATE/ODOM |
| GT leak into data-gen or runtime? | **None found in the current honest-campaign / 4-cam contract paths.** `contracts.py:183–210` raises `LeakageError` on any `gt_*`/`eval_*` key; `firewall.py` validates feature columns, loader sources, planner-facing imports. GT reaches disk only via `<run>/evaluation_only/` and `EvaluationOnlySample`. One **legacy** exception persists — see GT-firewall section below. | — |

---

## C. Existing reliability / GP pipeline

**Three parallel target definitions exist, and they disagree.** This is the central finding.

| Target | Defined at | Semantics | Consumed by |
|---|---|---|---|
| `yolo_score_raw` (continuous) | `fit_belief_aware_gp.py --target score` | **raw detector confidence** ∈ [0,1] | **the deployed C2 planner artifact** `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`. This is the "GP learns R"/"confidence ≈ covariance" lineage the task forbids. **MODEL** |
| `det_hit` (binary) | `fit_belief_aware_gp.py --target hit`; `train_factorized_gp.py` forces `hit` | `P(detection after threshold | s)` — closest existing thing to **p_det** | offline per-camera GP; not deployed to planner. **MODEL** |
| `availability_label = int(association_valid)` | `opportunity.py:245` | `P(detected ∧ association-window-valid | opportunity)` — a *gated* p_use-like quantity | `train_availability_gp.py`. **MODEL** |
| `usable_label = int(residual ≤ max_residual_px)` | `opportunity.py:284`, non-circular LOO reference | conditional quality label, only on `association_valid` rows | `train_quality_gp.py`. **MODEL** |

- **Input features:** `m_x, m_y` = belief xy (`train_factorized_gp.py`); the canonical fitter's
  expected-RBF kernel integrates over belief covariance `S_xx,S_xy,S_yy` (BELIEF). Never truth.
- **Missing detections representation — the key gap.** `build_opportunity_row` **returns `None`**
  (`opportunity.py:220–221`) whenever the *opportunity* gate fails (outside predicted valid
  region, stale stream, association window, or predicted height < min). Consequences:
  1. The current opportunity dataset contains **only opportunities**, not "every camera
     opportunity including misses" — it silently drops a whole class of negatives.
  2. `availability_label` estimates `P(usable | opportunity)`, and "opportunity" itself depends on
     a **belief-predicted** ellipse-inside fraction (`min_ellipse_inside_fraction=0.8`,
     `opportunity.py:216`) — a model-dependent choice baked into the denominator.
  3. There is **no `failure_reason`** field, so the *reason* a camera failed (no frame / stale /
     no detection / low conf / bad projection / bad association / clipped / rejected) is not
     recorded — exactly the enumeration the new contract requires.
- **det vs qual not cleanly separated.** `p_det` (`det_hit`) and a quality-ish label
  (`usable_label` via LOO residual) both exist, but there is no single record carrying
  `detection_label`, `quality_label`, and `usable_label = detection_label ∧ quality_label`
  together, and no config-frozen quality gate. `usable_label` is `None` when `association_valid`
  is false, i.e. it is a *conditional-on-detection* quality label, not the product p_use.
- **Artifact format:** `.npz` GP (mean field + latent params). Calibration metrics (Brier, AUROC,
  ECE, reliability) already exist in `validate_gp_heldout.py`,
  `scripts/geometry_visibility/gp_usability_validation.py`, and canonically in
  `scripts/shared/metrics.py`.
- **Classification vs regression:** both. `--target hit`/`availability` are binary (classification);
  `--target score` is regression on confidence (the forbidden deployed path).

---

## D. Planner interface

| Question | Answer (verified) | Source |
|---|---|---|
| Where `expected_visibility_ca` is called | `src/planning/planning/core/casadi_efe.py:185`; wired via `unicycle_planner_node.py` params `use_visibility_model`, `visibility_artifact_path`. | — |
| How p_vis is computed | Unscented transform: 5 sigma points over the **xy belief covariance** (`casadi_efe.py:190–195`), each evaluated through `prob_state(state)` and clipped to `[1e-4, 1−1e-4]`, weighted sum. So `p̄_vis(b) ≈ Σ w_i p(χ_i)`. | BELIEF→MODEL |
| Where covariance blending occurs | `casadi_efe.py:213–223` `_blend_observation_covariance_ca`: **precision-weighted** blend `blended_prec = p·prec_visible + (1−p)·prec_miss`, `R_plan = diag(1/blended_prec)`. Single source of truth for the trust→R math: `reliability/covariance_mapping.py`. | — |
| Dimensions / units of R_plan | 2×2 diagonal, image-plane (uv) covariance in px²; `R_visible` from `r_visible_uv`, `R_miss` from `r_miss` (120 px runtime default; 40 px offline — the reconciled-formula / unreconciled-endpoint mismatch, `MissEndpointPolicy.require_reconciled()`). | PIXEL |
| One camera or multiple? | **Planner-facing path is single-camera.** `prob_state` is one scalar field, no camera index; `visibility_artifact_path` points at one aggregate GP. | — |
| Simultaneous cameras fused? | Not in the planner. Offline only: `replay.py` fusion modes (M5 sequential, M6–M8 selection, B6 health-aware). No `p_any` / expected-camera-count field is computed anywhere. | — |
| Sigma-point usage | See above; `_xy_visibility_sigma_points_ca` (`casadi_efe.py:~160`), `kappa=1.0`. This is exactly the behavior the task says to **preserve and test**, not change. | — |

**This interface already satisfies the "frozen adapter" requirement:** the only research variable
that needs to enter the planner is *which `prob_state` field* backs `expected_visibility_ca`.
Swapping the artifact from `yolo_score_raw_gp.npz` to a `p_use` field is a data change, not a
covariance-blending change.

---

## E. Experiment infrastructure

- **Active world / two-world rule:** method dev in `warehouse_aws` (single cam); frozen-method
  eval in `warehouse_full_4cam.world.sdf` (CLAUDE.md hard rule). `honest_campaign_v1` is locked.
- **Conditions:** `scripts/visibility_comparison/warehouse_visibility_campaign_honest_v2.yaml` —
  C0 `geometric_shortest_path`, C1 `constant_R_efe`, C2 `visibility_aware_efe` (+ `gp_artifact`).
  Run matrix = task × condition × seed via `run_visibility_campaign.py`.
- **Runtime contract:** `docs/current_runtime_contract.yaml`, `docs/paper_runtime_contract.yaml`.
- **Registry:** `docs/experiment_registry.md`; storyline `research_story/` ch.00–11 +
  `registry.yaml`.
- **Logs / artifacts:** single-cam `logs/visibility_comparison/<campaign>/`; studies
  `logs/studies/<study>/`; 4-cam `logs/multicamera_commissioning_bigwarehouse/`,
  `logs/reliability_bigworld_multicamera_story/`. Locked media in `paper_artifacts/`.
- **Current vs historical:** deployed C2 GP = `yolo_score_raw` (historical/forbidden target);
  factorized `hit`/availability trainers exist but are **not** deployed to the planner.

---

## Confirmed current behavior

1. Detection gate = one line, `yolo_selection.py:144`, threshold-on-raw-score. p_det has a clean
   operational definition (`det_hit`) already; a GP for it exists (`train_factorized_gp.py`).
2. The **deployed planner** consumes a GP of **raw YOLO confidence** (`yolo_score_raw`), i.e. it
   currently treats expected confidence as the observability field. This is the practice the task
   is redesigning away from.
3. The planner adapter (sigma-point UT → precision blend → R_plan) is clean, single-camera, and
   is the correct thing to hold **fixed**.
4. GT is firewalled in all current operational paths (`contracts.py`, `firewall.py`).
5. Genuine multicamera **infrastructure** exists (4-cam detector, per-camera CSVs, CameraManager,
   per-camera GP, fusion replay).

## Uncertain behavior requiring tests

- Whether `train_factorized_gp` per-camera `hit` fields ever beat a constant / distance / FOV
  baseline on a **spatially-blocked, run-grouped** holdout (prior leave-region-out nulls suggest
  they may not — Phase-4 gate must be taken seriously).
- Whether `p_det · p_qual` (product) is calibrated, or needs a final isotonic/Platt stage.
- Exact bottom-centre pixel-selection convention and edge-clip handling at the *sample* level
  (needs a unit test on real records to freeze `edge_distance_px` / `CLIPPED_OR_EDGE`).
- The empirical validity of the `p_any` conditional-independence approximation vs joint 4-cam
  outcomes (detector-limited; see below).

## Ground-truth firewall violations (relative to the new contribution)

- **None in the current operational/training/planner paths.**
- **One quarantined legacy lineage:** `scripts/visibility_comparison/capture_visibility_samples.py:125`
  teleports the robot via `/world/<w>/set_pose` over a sample grid — commanded pose ≙ true pose,
  i.e. GT-as-input. Any artifact descended from it (including the original
  `warehouse_visibility_gp_v1` teleport-grid lineage) must **not** feed the new p_use map. New
  dataset manifests must carry `ground_truth_used: false` (pattern already in
  `build_full4cam_planner_prior.py:119`).
- **Detector-training auto-labels** (`scripts/perception/capture_yolo_dataset.py`) use sim
  segmentation — acceptable under the rules as *detector-training data generation*, but must be
  declared as an assumption (the detector is a fixed sensor).

## Missing fields (contract gap vs the required `ObservationOpportunity`)

Relative to the required schema, the current `OpportunityRow` / `OperationalReliabilitySample`
are missing or mismatched on:

- `failure_reason` (controlled enum) — **absent**; the single biggest gap.
- A record **per camera per opportunity including misses** — current builder returns `None` on
  gate failure, so misses outside the predicted region are dropped.
- Unified `detection_label`, `quality_label`, `usable_label` on one row with
  `usable = detection ∧ quality` — **absent** (labels are split across two functions and
  conditioned differently).
- `route_id`, `state_source`, `frame_expected`/`frame_received`/`frame_age_ms`,
  `confidence_threshold` (as a recorded field), flat `bbox_*`, `selected_pixel_u/v`,
  `image_width/height`, `edge_distance_px`, `projection_attempted`,
  `association_attempted`/`_valid`, `tracking_available`/`_valid`, `accepted_by_localizer`,
  `source_labels`, `schema_version` on the opportunity row — **absent or nested in free-form maps**.
- A **frozen** confidence threshold recorded per dataset (currently 0.05 vs 0.25 across launches).

## Smallest safe implementation plan (staged, gated)

Phase order mirrors the task's own structure and the existing 9-phase gated plan; each phase
stops if its validation gate fails.

- **P0 (this doc).** Audit. No runtime change. *Gate: findings above are verified.* ✅
- **P1 — Usable-observation contract.** New `observability/contracts.py` (`ObservationOpportunity`,
  `FailureReason` enum), `observability/gates.py` (frozen quality gate), `config/usable_observation_gate.yaml`,
  `schemas/observation_opportunity.schema.json`, `docs/usable_observation/data_contract.md`. Unit
  tests for every gate + combination. **Extends** `OperationalReliabilitySample`; does **not** touch
  runtime nodes or the planner. *Gate 1: misses represented, one record per opportunity, labels
  deterministic from frozen config, no GT, schema tests pass.*
- **P2 — Offline exporter** from existing logs (`observability/exporter.py`, `tools/export_observation_dataset`).
  Split by run/route; misses included; manifest with commit + config hash + GT-firewall result.
  *Gate 2: coverage plots, class balance, run/route split, credible failure reasons.*
- **P3 — Baselines first** (B0 constant, B1 distance-logistic, B2 FOV/range, B3 grid-frequency) on
  a spatially-blocked, run-grouped holdout. *Gate 3: baselines reproduce end-to-end.*
- **P4 — Direct p_use + two-stage p_det·p_qual GP**, with final calibration if the product is
  miscalibrated. *Gate 4: selected model reasonably calibrated on held-out data AND beats/complements
  the simplest baseline — else the simpler model wins and we document it.*
- **P5 — Multicamera products** (`p_any` labelled as conditional-independence approximation,
  expected-camera-count) + empirical joint-outcome check where real 4-cam logs exist.
- **P6 — Planner conditions** P0–P4 differing **only** in the `prob_state` source; adapter fixed.
  *Gate 5: only observability source differs; R_plan/geometry unchanged; limit + monotonicity tests.*
- **P7 — Route + navigation evaluation** (predicted vs realized observability; nav metrics with
  seeds + run-grouped bootstrap CIs). *Gate 6: route differences attributable to predicted
  observability; realized outcomes support it.*

**First minimal change proposed:** this audit document only (done). Next proposed change = P1
contract skeleton + gate config + schema + unit tests, altering no runtime node and no planner.

## Genuine multicamera evidence — status

- Real 4-cam Gazebo runs **exist** (`warehouse_full_4cam`, per-camera perception CSVs +
  `evaluation_only/ground_truth.csv`), so an empirical joint-camera check for `p_any` is possible
  in principle.
- **But it is detector-limited:** the 4-cam detector v2 misses the strict mid-range gate
  (~0.40/0.36 hit rate at 8–12 / 12–16 m), and camera overlap is only ~7–13% of the floor. So
  per-camera usable rates are range-skewed and the *redundancy* evidence is thin. Multi-camera
  demonstrations must be labelled **MODEL-ONLY / DIAGNOSTIC** until the detector gates, per the
  no-synthetic-data rule. The camera-indexed interface should be built regardless.

---

*Deliverable A of the usable-observation task. Companion required docs to follow (method.md,
data_contract.md, confidence_analysis.md, final_report.md). Prior related audit:
`docs/observability_audit.md`.*
