# Repo mapping — where the paper-extension architecture lands (2026-07-17)

Decision: **no restructure of the runtime column.** The paper draft (§19)
proposes a fresh `multicam/` + `offline/` tree; the repo already implements most
of that runtime column under `src/reliability/`, and everything in the proposal
maps onto existing homes (table below). This decision is unchanged.

> **Update (2026-07-21):** a *presentation-layer* restructure has since landed —
> a top-level `modules/` front door (one folder per contribution). It does not
> move any runtime code; the mappings below are all still correct. This
> contribution's landing page is `modules/07_multicam_handover_fusion/`, whose
> `baselines/` holds the Toro-Diz reimplementation notes and `framings/` holds
> the paper drafts. See `CLAUDE.md` for the updated repo map.

## Paper §19 → actual repo locations

| Proposed (§19)                      | Actual home                                                        | Status |
|-------------------------------------|--------------------------------------------------------------------|--------|
| `camera_observation_node`           | `src/perception/.../yolo_robot_detector_node` (`publish_camera_observation_json`) | exists |
| `timestamp_alignment`               | `reliability.replay` frame builder + `export-multicamera`          | exists |
| `camera_projection`                 | `reliability.projection` + `fit_projection_calibration.py` (commissioning study) | exists (v2, v3 pending detector retrain) |
| `observation_association`           | `reliability.export` / opportunity builder                         | partial → plan 02 |
| `reliability_query`                 | `reliability.providers` (grid `.npz` + dispatch)                   | exists |
| `confidence_calibrator`             | `reliability.confidence_calibration`                               | NEW → plan 04 |
| `camera_health_monitor`             | `reliability.health` (state machine) + `reliability.health_ewma`   | extend → plan 08 |
| `covariance_mapper`                 | `reliability.contracts.UpdateCovariance` + `geometry_visibility.trust_to_r_plan` + adapter | reconcile → plan 07 |
| `multicamera_filter`                | `reliability.fusion` (+ robust/Joseph/info-selection additions)    | extend → plan 09 |
| `planner_observation_adapter`       | `reliability.planning_covariance` + `unicycle_planner_node` interface | NEW → plan 10 |
| `offline/build_opportunity_dataset.py` | `experiments/multicamera_fusion_extension/tools/`               | NEW → plan 02 |
| `offline/build_leave_one_camera_out_labels.py` | same tools dir                                          | NEW → plan 02 |
| `offline/train_availability_gp.py` / `train_quality_gp.py` | wrapper over canonical `scripts/visibility_comparison/fit_belief_aware_gp.py` | plan 03 |
| `offline/fit_confidence_calibrator.py` | tools dir, thin CLI over `reliability.confidence_calibration`  | plan 04 |
| `offline/fit_trust_stacker.py`      | tools dir over `reliability.trust_stacker`                         | NEW → plan 05 |
| `offline/fit_conditional_covariance.py` | tools dir over `reliability.conditional_covariance`            | NEW → plan 06 |
| `offline/replay_fusion.py`          | `reliability_tools replay` / `benchmark` (R0–R4, M5–M8)            | exists, extend with new baselines |
| `offline/run_camera_subset_sweep.py`| tools dir (drives `reliability.replay` over camera masks)          | NEW → plan 11 |
| `offline/run_dropout_sweep.py`      | commissioning robustness suite (D5) + tools                        | partial → plan 11 |
| `offline/run_calibration_drift_sweep.py` | tools dir (perturb calibration JSON, re-project, replay)      | NEW → plan 11 |
| `offline/evaluate_navigation.py`    | `scripts/shared/metrics.py` + `campaign_metrics.py` + E8 harness   | plan 11 |
| Toro-style baseline (§14 B2)        | `reliability.toro_baseline`                                        | NEW → plan 01 |
| `docs/multicam_reliability/` evidence bundle (§20) | `research_story/{04,08,09}/evidence.yaml` manifests + `logs/studies/multicamera_fusion_extension/<expN>/` | conventions already stronger → plan 12 |

## Message/contract mapping (§19 YAML → contracts.py)

- Observation message → `CameraObservation` (already has camera_id, stamp,
  pixel/map estimate, detector score, staleness). Add-if-missing:
  `calibration_hash`, `detector_hash` fields (plan 02).
- Reliability response → `ReliabilityPrediction` (has mean/std). Factorised
  availability/quality goes in as two predictions or new optional fields (plan 03).
- Trust message → `UpdateCovariance` (+ `gate_decision`/`reason` fields, plan 07).
- Planner query → `PlanningCovariance` batch adapter (plan 10).

## Placement rules for all new work

1. Library code (pure, unit-tested, no ROS imports): `src/reliability/reliability/<module>.py` + `tests/reliability/test_<module>.py`.
2. Study CLIs/experiment drivers: `experiments/multicamera_fusion_extension/tools/`.
3. Outputs: `logs/studies/multicamera_fusion_extension/<expN_name>/` with `RESULTS.md` each.
4. Metrics: import `scripts/shared/metrics.py` — never hand-roll Brier/NLL/ECE/AUROC.
5. GT firewall: anything touching `gt_*`/oracle goes through `reliability.firewall` patterns; extend `test_leakage_firewall.py` for every new module that reads exports.
6. Storyline: register each study in `research_story/registry.yaml` + the serving chapter's `evidence.yaml` **after** the in-flight commissioning edits land (files currently modified by the parallel workstream — do not touch until committed).

## Two-world rule position

All new fusion/calibration METHOD code is developed offline against unit
fixtures + replay exports (any world's data is fine for plumbing). Frozen
hyperparameters are pre-registered before the confirmatory 4-cam campaigns.
Per-site commissioning (detector retrain, projection v3) follows the
commissioning plan (`experiments/multicamera_commissioning_bigwarehouse/RESEARCH_PLAN_ADDITIONS_2026-07-16.md`)
— that plan's Modules 1–2 are upstream dependencies of everything here.
