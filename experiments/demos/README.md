# Experiments Demo Media

[Back to experiments module](../README.md)

This folder holds README-facing media for the current campaign evidence.
Regenerate the static assets from the repository root with:

```bash
python3 scripts/paper_figures/make_readme_visuals.py
```

## Available README Media

| Path | Type | Story beat |
| --- | --- | --- |
| [`images/outcome_counts_by_condition.png`](images/outcome_counts_by_condition.png) | PNG bar plot | Aggregate clean/collision/near/invalid counts by condition. |
| [`images/campaign_result_table.png`](images/campaign_result_table.png) | PNG table | Compact visual table of aggregate and per-task outcomes. |

## Existing Source Assets

| Asset | Use |
| --- | --- |
| [`../../docs/paper_vs_current/current/figures/robustness_spread_current.png`](../../docs/paper_vs_current/current/figures/robustness_spread_current.png) | Current 40-run trajectory map. |
| [`../../docs/paper_vs_current/current/README.md`](../../docs/paper_vs_current/current/README.md) | Human-readable current metrics table. |
| [`../../docs/paper_vs_current/current/figures/paired_mechanism_taskA_current.gif`](../../docs/paper_vs_current/current/figures/paired_mechanism_taskA_current.gif) | Existing paired route GIF for task A. |
| [`../../docs/paper_vs_current/current/figures/paired_mechanism_west_current.gif`](../../docs/paper_vs_current/current/figures/paired_mechanism_west_current.gif) | Existing paired route GIF for the hard west route. |

## Next Media Slots

| Planned path | Type | Story beat | Source |
| --- | --- | --- | --- |
| `animations/task_panel_cycle.gif` | GIF preview | Cycle through the four task panels. | Current robustness spread panels. |
| `videos/campaign_montage.mp4` | MP4 clip | World to detector to GP to route to aggregate result. | Gazebo clips, route overlays, metrics table. |
| `videos/representative_pair.mp4` | MP4 clip | One matched-seed C1/C2 comparison. | Paired mechanism run data. |
