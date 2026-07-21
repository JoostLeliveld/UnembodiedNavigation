# Documentation Index

This folder is the deeper reference layer behind the demonstration modules.
Start with the repository [`README.md`](../README.md), then the contribution
front door [`modules/`](../modules/) and the investigation
[`experiments/`](../experiments/).

Files are grouped by purpose below. They are kept flat (not physically
subfoldered) because several are path-anchored — loaded by tests, written by
code generators, or referenced by the LOCKED and workstream-owned chapter
manifests — so their paths are contracts, not free to move.

## Contracts (machine-readable / path-anchored — do not move casually)

| File | Role |
|---|---|
| [`current_runtime_contract.yaml`](current_runtime_contract.yaml) | Machine-readable contract for the active honest-campaign runtime. Loaded by `tests/visibility_comparison/test_current_runtime_contract.py`; pinned by `research_story/registry.yaml` and the ch.00 manifest. |
| [`paper_runtime_contract.yaml`](paper_runtime_contract.yaml) | Historical machine-readable snapshot of the submitted-paper runtime. |
| [`experiment_registry.md`](experiment_registry.md) | Artifact-chain registry: active evidence chain, current results surface, superseded lines, caveats. Named as `artifact_chain_registry` in `registry.yaml`. |
| [`reliability_contracts/`](reliability_contracts/) | Phase-0 operational vs evaluation-only sample schema + ground-truth firewall (`schema.md` + example JSONs). |

## Reference (method + convention documentation)

| File | Role |
|---|---|
| [`contribution_map.md`](contribution_map.md) | The module-chain narrative from camera observation to reliability-aware route behavior (long-form companion to [`../modules/`](../modules/)). |
| [`campaign_log_metrics.md`](campaign_log_metrics.md) | Read before computing any metric: the 200+ log columns and the belief/truth column traps. Cited by the canonical loaders in `scripts/`. |
| [`metric_definitions_and_gt_audit.md`](metric_definitions_and_gt_audit.md) | Metric definitions and the `truth_*` = wheel-odom contamination audit. |
| [`modular_validation_workflow.md`](modular_validation_workflow.md) | Validation-first way of working and the module exit checklists. |
| [`runtime_dataflow.md`](runtime_dataflow.md) | Offline artifact flow and online ROS topic flow. |
| [`uncertainty_propagation.md`](uncertainty_propagation.md) | Process, command, encoder, and belief-covariance conventions. |
| [`perception_details.md`](perception_details.md) | YOLO detector architecture, dataset, inference settings, training performance. |
| [`PLANNER_HYPERPARAMETERS.md`](PLANNER_HYPERPARAMETERS.md) | Planner knobs, intended effects, and tuning cautions. |
| [`warehouse_full_4cam_layout.md`](warehouse_full_4cam_layout.md) | Canonical 4-camera world layout. Generated/checked by `scripts/geometry_visibility/make_warehouse_full.py` and a reliability asset test. |
| [`geometry_visibility_deployment.md`](geometry_visibility_deployment.md) | Real-facility deployment reasoning for the geometry prior. |
| [`reliability_prior_sensing_survey.md`](reliability_prior_sensing_survey.md) | Decision memo on occlusion sensing for the weak-prior chapter. |

## Paper comparison

| Path | Role |
|---|---|
| [`paper_vs_current/`](paper_vs_current/) | Original-paper vs current-honest-runtime diff (markdown + the two frozen config YAMLs). Heavily referenced; media consolidated out on 2026-07-15. |

## Assets

| Path | Role |
|---|---|
| [`assets/`](assets/) | Generated warehouse map figures (`warehouse_full_4cam_map.{png,svg}`, produced by `scripts/geometry_visibility/make_warehouse_full.py`). |

For implementation details, use the package READMEs under `src/`:
[`sim`](../src/sim/README.md), [`perception`](../src/perception/README.md),
[`state`](../src/state/README.md), [`planning`](../src/planning/README.md), and
[`experiments`](../src/experiments/README.md).
