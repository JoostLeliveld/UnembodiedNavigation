# Experiments Demo Media Plan

[Back to experiments module](../README.md)

Planned media paths are story slots, not tracked files yet. Existing visuals are
linked from `paper_artifacts/`.

## Existing Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/figures/robustness_spread.png`](../../paper_artifacts/figures/robustness_spread.png) | Main campaign trajectory map. |
| [`../../paper_artifacts/metrics/robustness_summary.txt`](../../paper_artifacts/metrics/robustness_summary.txt) | Human-readable metrics table. |
| [`../../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_across_tasks.png`](../../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_across_tasks.png) | Additional localization campaign diagnostic. |
| [`../../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_error_map.png`](../../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_error_map.png) | Spatial localization-error diagnostic. |
| [`../../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_recovery_contrast.png`](../../paper_artifacts/campaigns/robustness_v2_partialoccl/localization_recovery_contrast.png) | Recovery contrast diagnostic. |
| [`../../paper_artifacts/campaigns/robustness_v2_partialoccl/solve_diagnostics.png`](../../paper_artifacts/campaigns/robustness_v2_partialoccl/solve_diagnostics.png) | Additional solve diagnostic. |

## Planned Media Slots

| Planned path | Type | Story beat | Target | Source |
| --- | --- | --- | --- | --- |
| `images/outcome_counts_by_condition.png` | PNG bar plot | Aggregate clean/collision/near/invalid counts by condition. | 1400 px wide | `robustness_metrics.csv`. |
| `images/campaign_result_table.png` | PNG table | Compact visual table of aggregate and per-task outcomes. | 1600 px wide | `robustness_metrics.csv`. |
| `animations/task_panel_cycle.gif` | GIF preview | Cycle through the four task panels. | 10-15 s | `robustness_spread.png` panels. |
| `videos/campaign_montage.mp4` | MP4 clip | World -> detector -> GP -> route -> aggregate result. | 45-60 s | Gazebo clips, route overlays, metrics table. |
| `videos/representative_pair.mp4` | MP4 clip | One matched-seed C1/C2 comparison. | 30-60 s | Paired mechanism run data. |

## Capture Checklist

1. Use only runs that match the current campaign metrics CSV.
2. Label C1 and C2 consistently with the paper-facing names.
3. Mark collisions, near-success, and infrastructure-invalid runs explicitly.
4. Keep continuous localization summaries scoped to clean successes only.
