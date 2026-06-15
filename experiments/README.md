# Experiments And Campaign Results

[Back to repository overview](../README.md)

This module packages the benchmark surface: tasks, conditions, seeds, metrics,
representative figures, and reproduction commands.

## Story

The representative planning behavior is tested across a locked multi-task,
matched-seed campaign. This page shows the aggregate result and keeps caveats
visible.

## Visual Demonstration

![Robustness spread](../paper_artifacts/figures/robustness_spread.png)

This figure overlays all seeded trajectories for the current four-task campaign
on the learned reliability map.

Planned media is listed in [`demos/`](demos/): outcome plots, a per-task table
image, a task-panel GIF, and a campaign montage video.

## Inputs And Outputs

| Input | Output |
| --- | --- |
| `scripts/visibility_comparison/aws_f31b1_final_config.yaml` | seeded run directories |
| local YOLO checkpoint | `experiment.csv`, `run_manifest.json`, `run_summary.json` |
| GP artifact `aws_gp_v7b` | `robustness_metrics.csv` and summary tables |
| `warehouse_aws.world.sdf` | paper figures and provenance files |

## Method

The campaign compares two planner conditions across four tasks and five seeds
per condition:

- C1: `constant_R_efe`
- C2: `visibility_aware_efe`

The comparison keeps world geometry, route seeds, driveable/no-go layer, local
tracking, noise settings, and success/collision criteria fixed across
conditions.

## Performance And Diagnostics

Aggregate outcome:

| Condition | Clean reaches | Collisions | Near-success | Invalid |
| --- | ---: | ---: | ---: | ---: |
| C1 | 12/20 | 8/20 | 0/20 | 0/20 |
| C2 | 16/20 | 2/20 | 1/20 | 1/20 |

Per-task outcome:

| Task | C1 | C2 |
| --- | --- | --- |
| `F31_b1_apron_a3_mid` | 3/5 clean, 2/5 collisions | 3/5 clean, 1/5 near-success, 1/5 collision |
| `b5_a4_apron_to_a2_mid` | 4/5 clean, 1/5 collision | 4/5 clean, 1/5 collision |
| `b2_a0_west_to_a1_upper` | 0/5 clean, 5/5 collisions | 4/5 clean, 1/5 infrastructure-invalid |
| `b6_a0_west_to_a1_low_control` | 5/5 clean | 5/5 clean |

Evidence files:

- [`../paper_artifacts/metrics/robustness_metrics.csv`](../paper_artifacts/metrics/robustness_metrics.csv)
- [`../paper_artifacts/metrics/robustness_summary.txt`](../paper_artifacts/metrics/robustness_summary.txt)
- [`../docs/experiment_registry.md`](../docs/experiment_registry.md)

Additional packaged campaign diagnostics:

- [`../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_across_tasks.png`](../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_across_tasks.png)
- [`../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_error_map.png`](../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_error_map.png)
- [`../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_recovery_contrast.png`](../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_recovery_contrast.png)
- [`../paper_artifacts/campaigns/robustness_v2_partialoccl/solve_diagnostics.png`](../paper_artifacts/campaigns/robustness_v2_partialoccl/solve_diagnostics.png)

## Reproduce

Run the locked campaign:

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/aws_f31b1_final_config.yaml \
  --log-root logs/visibility_comparison/aws_f31b1_final_v1
```

Compute metrics from a completed campaign:

```bash
python3 scripts/visibility_comparison/compute_paper_metrics.py \
  --campaign-log logs/visibility_comparison/aws_f31b1_final_v1/campaign_log.json \
  --gp-artifact paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz \
  --out logs/visibility_comparison/aws_f31b1_final_v1/paper_metrics.csv \
  --summary-out logs/visibility_comparison/aws_f31b1_final_v1/paper_summary.txt
```

## Relevant Implementation Files

| File | Role |
| --- | --- |
| [`../src/experiments/launch/warehouse_primary_comparison.launch.py`](../src/experiments/launch/warehouse_primary_comparison.launch.py) | Main comparison launch. |
| [`../src/experiments/config/tasks.yaml`](../src/experiments/config/tasks.yaml) | Benchmark task definitions. |
| [`../src/experiments/experiments/nodes/experiment_logger.py`](../src/experiments/experiments/nodes/experiment_logger.py) | Run logging and summaries. |
| [`../scripts/visibility_comparison/run_visibility_campaign.py`](../scripts/visibility_comparison/run_visibility_campaign.py) | Campaign runner. |

## Limitations

- Raw logs and model weights are local/private unless packaged under
  `paper_artifacts/` or released externally.
- One C2 run is infrastructure-invalid and should not be described as a robot
  outcome.
- New claims need a full evidence chain: world, detector, visibility data, GP,
  config, logs, metrics, figures, and wording.

See planned visual media in [`demos/`](demos/).
