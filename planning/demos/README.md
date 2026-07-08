# Planning Demo Media

[Back to planning module](../README.md)

This folder holds README-facing media for the route-choice contribution.
Regenerate the static assets from the repository root with:

```bash
python3 scripts/paper_figures/make_readme_visuals.py
```

## Available README Media

| Path | Type | Story beat |
| --- | --- | --- |
| [`images/paired_route_choice.png`](images/paired_route_choice.png) | PNG still | Same west-route task/seed with C1 and C2 over the reliability field. |
| [`images/covariance_along_route.png`](images/covariance_along_route.png) | PNG plot | Constant `R` versus GP-scaled `R_plan` along the route. |

## Existing Source Assets

| Asset | Use |
| --- | --- |
| [`../../docs/paper_vs_current/current/figures/paired_mechanism_west_current.png`](../../docs/paper_vs_current/current/figures/paired_mechanism_west_current.png) | Current paired mechanism still. |
| [`../../docs/paper_vs_current/current/figures/paired_mechanism_west_current.gif`](../../docs/paper_vs_current/current/figures/paired_mechanism_west_current.gif) | Existing paired route GIF. |
| [`../../docs/paper_vs_current/current/figures/robustness_spread_current.png`](../../docs/paper_vs_current/current/figures/robustness_spread_current.png) | Current 40-run trajectory spread. |
| [`../../paper_artifacts/figures/paired_mechanism_west_current_data/`](../../paper_artifacts/figures/paired_mechanism_west_current_data/) | Source data for the README route visual. |

## Next Media Slots

| Planned path | Type | Story beat | Source |
| --- | --- | --- | --- |
| `animations/c1_rollout.gif` | GIF preview | Constant-covariance rollout on a discriminator task. | C1 run logs and planner samples. |
| `animations/c2_rollout.gif` | GIF preview | Learned-covariance rollout on the same task/seed. | C2 run logs and planner samples. |
| `videos/c1_c2_route_compare.mp4` | MP4 clip | Side-by-side Gazebo or map-space route comparison. | Representative pair run. |
