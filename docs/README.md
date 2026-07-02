# Documentation Index

This folder is the deeper reference layer behind the demonstration modules.
Start with the repository [`README.md`](../README.md), then click into the
module landing pages:
[`yolo`](../yolo/), [`gp`](../gp/), [`estimation`](../estimation/),
[`planning`](../planning/), and [`experiments`](../experiments/).

| File | Role |
| --- | --- |
| [`demo_media.md`](demo_media.md) | Visual gallery, README media map, and planned video storyboard. |
| [`experiment_registry.md`](experiment_registry.md) | Paper-facing artifacts, evidence status, superseded lines, and caveats. |
| [`paper_runtime_contract.yaml`](paper_runtime_contract.yaml) | Machine-readable contract for paper-facing runs. |
| [`runtime_dataflow.md`](runtime_dataflow.md) | Offline artifact flow and online ROS topic flow. |
| [`gazebo_compute_request_presentation.md`](gazebo_compute_request_presentation.md) | Slide deck with local and external evidence for Gazebo camera-sensor compute limits. |
| [`perception_details.md`](perception_details.md) | YOLO detector architecture, dataset, inference settings, and training performance. |
| [`PLANNER_HYPERPARAMETERS.md`](PLANNER_HYPERPARAMETERS.md) | Planner knobs, intended effects, and tuning cautions. |
| [`uncertainty_propagation.md`](uncertainty_propagation.md) | Process, command, encoder, and belief-covariance conventions. |

For implementation details, use the package READMEs under `src/`:
[`sim`](../src/sim/README.md), [`perception`](../src/perception/README.md),
[`state`](../src/state/README.md), [`planning`](../src/planning/README.md), and
[`experiments`](../src/experiments/README.md).
