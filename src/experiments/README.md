# `experiments`

This package defines the experiment surface: world, task, planner condition, launch wiring, logging, and campaign reproducibility.

## Central Files

| File | Role |
| --- | --- |
| [`launch/warehouse_primary_comparison.launch.py`](launch/warehouse_primary_comparison.launch.py) | main paper launch for `constant_R_efe`, `visibility_aware_efe`, and optional `risk_only_ablation` |
| [`launch/warehouse_visibility_agent.launch.py`](launch/warehouse_visibility_agent.launch.py) | diagnostic launch for non-primary probing |
| [`launch/warehouse_visibility_capture.launch.py`](launch/warehouse_visibility_capture.launch.py) | offline capture launch for GP fitting |
| [`config/world_profiles.yaml`](config/world_profiles.yaml) | world registry and camera/profile metadata |
| [`config/tasks.yaml`](config/tasks.yaml) | benchmark, support, exploratory, and legacy task definitions |
| [`experiments/core/visibility_launch_common.py`](experiments/core/visibility_launch_common.py) | shared runtime assembly |
| [`experiments/nodes/experiment_logger.py`](experiments/nodes/experiment_logger.py) | run manifest, CSV logging, and summary writing |

## Paper-Facing Use

The paper-facing AWS robustness campaign is configured by
`scripts/visibility_comparison/aws_f31b1_final_config.yaml` and run through
`scripts/visibility_comparison/run_visibility_campaign.py`.

The primary launch enforces the important paper assumptions:

- YOLO perception path
- explicit trained YOLO model path
- explicit GP artifact path for GP-using planner conditions
- shared world/task/obstacle geometry across compared planners
- run manifests that record planner, artifact, estimator, noise, and safety settings

## Current Paper Benchmark

The AWS-style warehouse is the current paper-facing benchmark. It uses the
curated artifacts under `paper_artifacts/` plus a local YOLO checkpoint under
`local_artifacts/`.

See also:

- [`../../docs/experiment_registry.md`](../../docs/experiment_registry.md)
- [`../../docs/runtime_dataflow.md`](../../docs/runtime_dataflow.md)
- [`config/README.md`](config/README.md)
