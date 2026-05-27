# `scripts`

This folder contains offline tooling around the ROS runtime. Treat scripts as paper-facing only when they are part of the locked campaign, metric, figure, or artifact pipeline.

## Paper-Facing Visibility Pipeline

| Purpose | File |
| --- | --- |
| Capture visibility samples | [`visibility_comparison/capture_visibility_samples.py`](visibility_comparison/capture_visibility_samples.py) |
| Extract YOLO perception targets | [`visibility_comparison/extract_perception_targets.py`](visibility_comparison/extract_perception_targets.py) |
| Build GP targets | [`visibility_comparison/build_gp_targets.py`](visibility_comparison/build_gp_targets.py) |
| Fit GP artifacts | [`visibility_comparison/fit_visibility_gps.py`](visibility_comparison/fit_visibility_gps.py) |
| Run compact benchmark campaign | [`visibility_comparison/run_visibility_campaign.py`](visibility_comparison/run_visibility_campaign.py) |
| Compact benchmark config | [`visibility_comparison/paper_campaign_config.yaml`](visibility_comparison/paper_campaign_config.yaml) |
| Compute paper metrics | [`visibility_comparison/compute_paper_metrics.py`](visibility_comparison/compute_paper_metrics.py) |
| Generate thesis figures | [`visibility_comparison/thesis_plots/make_thesis_figures.py`](visibility_comparison/thesis_plots/make_thesis_figures.py) |
| Generate supervisor-feedback diagnostic figures | [`visibility_comparison/thesis_plots/make_supervisor_feedback_figures.py`](visibility_comparison/thesis_plots/make_supervisor_feedback_figures.py) |

## Exploratory Failure Benchmarks

| Purpose | File |
| --- | --- |
| Experiment B AWS/JdeRobot-style campaign config | [`visibility_comparison/aws_campaign_config.yaml`](visibility_comparison/aws_campaign_config.yaml) |
| Experiment B AWS/JdeRobot-style smoke config | [`visibility_comparison/aws_smoke_config.yaml`](visibility_comparison/aws_smoke_config.yaml) |
| Offline coarse route evaluator | [`visibility_comparison/coarse_route_evaluator.py`](visibility_comparison/coarse_route_evaluator.py) |

The coarse route evaluator is diagnostic only. It scores automatically generated
routes on the known driveable 2D layer with and without a GP-derived ambiguity
proxy. It must not be used as mission-waypoint evidence.

## Perception Training And Capture

The perception scripts are support tooling. They are relevant when regenerating the detector or validating a detector dataset, but they are not direct paper evidence unless the paper discusses detector training provenance. Experiment B should use a world-specific detector rather than the compact Task A detector.

Everything else in this folder should be treated as diagnostic, exploratory, or legacy until a paper section explicitly depends on it.
