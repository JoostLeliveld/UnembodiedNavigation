# Technical documentation

**Start here: [`METHOD.md`](METHOD.md)** — the whole method start to finish in plain
language, with a glossary (belief, NEES, NIS, `R`, `Q`, IPM, bias) and an explicit list of
what is settled versus open. Everything else in this directory is detail hanging off one of
its steps. Read it before `research/`, which is a bookkeeping layer and assumes you already
know what the system does.

This directory documents runtime contracts, dataflow, geometry, perception, metrics, and
deployment mechanics. Scientific status and paper framing live only in [`research/`](../research/README.md).

## Runtime contracts

| Path | Role |
|---|---|
| [`current_runtime_contract.yaml`](current_runtime_contract.yaml) | Active honest-campaign runtime contract. |
| [`paper_runtime_contract.yaml`](paper_runtime_contract.yaml) | Historical submitted-paper runtime snapshot. |
| [`reliability_contracts/`](reliability_contracts/) | Operational versus evaluation-only schema and truth firewall. |

## Technical references

| Path | Role |
|---|---|
| [`localization_metrics.md`](localization_metrics.md) | **Canonical localization statistics, information timeline, run registry, and comparison rules. Read before quoting any error number.** |
| [`campaign_log_metrics.md`](campaign_log_metrics.md) | Log columns and metric traps. |
| [`metric_definitions_and_gt_audit.md`](metric_definitions_and_gt_audit.md) | Metric definitions and truth audit. |
| [`runtime_dataflow.md`](runtime_dataflow.md) | Offline artifact and online ROS-topic flow. |
| [`uncertainty_propagation.md`](uncertainty_propagation.md) | Process, command, encoder, and covariance conventions. |
| [`perception_details.md`](perception_details.md) | Detector architecture, dataset, settings, and training performance. |
| [`PLANNER_HYPERPARAMETERS.md`](PLANNER_HYPERPARAMETERS.md) | Planner knobs and tuning cautions. |
| [`warehouse_full_4cam_layout.md`](warehouse_full_4cam_layout.md) | Canonical four-camera world layout. |
| [`geometry_visibility_deployment.md`](geometry_visibility_deployment.md) | Geometric-prior deployment mechanics. |
| [`usable_observation/`](usable_observation/) | Usable-observation data contract and technical audits. |
| [`paper_vs_current/`](paper_vs_current/) | Historical paper/runtime configuration comparison. |

Implementation documentation lives beside each package under `src/`. Evidence status,
claims, assumptions, reviewer questions, and validation gates are in `research/registry.yaml`.
