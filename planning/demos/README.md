# Planning Demo Media Plan

[Back to planning module](../README.md)

Planned media paths are story slots, not tracked files yet. Existing visuals are
linked from `paper_artifacts/`.

## Existing Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/figures/robustness_spread.png`](../../paper_artifacts/figures/robustness_spread.png) | Seeded C1/C2 campaign trajectories. |
| [`../../paper_artifacts/figures/paired_mechanism_taskA.pdf`](../../paper_artifacts/figures/paired_mechanism_taskA.pdf) | Paired mechanism figure. |
| [`../../paper_artifacts/figures/paired_mechanism_taskA_data/`](../../paper_artifacts/figures/paired_mechanism_taskA_data/) | Source data for the paired mechanism figure. |

## Planned Media Slots

| Planned path | Type | Story beat | Target | Source |
| --- | --- | --- | --- | --- |
| `images/paired_route_choice.png` | PNG still | Same task/seed, C1 and C2 side by side on reliability background. | 1600 px wide | `paired_mechanism_taskA_data`. |
| `images/covariance_along_route.png` | PNG plot | Predicted covariance/reliability along C1 vs C2 route. | 1400 px wide | `plan_samples.csv` and GP artifact. |
| `animations/c1_rollout.gif` | GIF preview | Constant-covariance route rollout on a discriminator task. | 10-15 s | C1 run logs and planner samples. |
| `animations/c2_rollout.gif` | GIF preview | Learned-covariance route rollout on the same task/seed. | 10-15 s | C2 run logs and planner samples. |
| `videos/c1_c2_route_compare.mp4` | MP4 clip | Side-by-side Gazebo or map-space route comparison. | 30-60 s | Representative pair run. |

## Capture Checklist

1. Use matched task and seed for C1 and C2.
2. Overlay trajectory, belief covariance, goal, collision status, and reliability.
3. Keep the known driveable geometry identical in both panels.
4. Export compact GIF previews plus a high-resolution MP4.
