# Estimation Demo Media Plan

[Back to estimation module](../README.md)

Planned media paths are story slots, not tracked files yet. Existing visuals are
linked from `paper_artifacts/`.

## Existing Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/figures/localization_pathway.png`](../../paper_artifacts/figures/localization_pathway.png) | Current localization pathway figure. |
| [`../../paper_artifacts/figures/inputs/loc_pathway_frame_v7b.jpg`](../../paper_artifacts/figures/inputs/loc_pathway_frame_v7b.jpg) | Source camera frame for the pathway figure. |

## Planned Media Slots

| Planned path | Type | Story beat | Target | Source |
| --- | --- | --- | --- | --- |
| `images/image_to_bev_01.png` | PNG still | Camera pixel and BEV projection side by side. | 1400 px wide | `make_localization_pathway_figure.py`. |
| `images/localization_error_trace.png` | PNG plot | Truth-belief error over one representative run with detection/miss regions shaded. | 1400 px wide | `experiment.csv` plus perception diagnostics. |
| `animations/belief_update.gif` | GIF preview | Prediction grows covariance; detection update tightens belief. | 10-15 s | Planner/state logs from a representative run. |
| `videos/state_pipeline.mp4` | MP4 clip | Topic walkthrough from `/perception/pixel_pose` to `/state/bev`. | 30-45 s | ROS bag or experiment CSV plus overlays. |

## Capture Checklist

1. Select a representative run with clear detections and at least one weak
   detection region.
2. Render image-space point, projected BEV point, truth, and belief estimate.
3. Highlight that heading is odometry-driven under `camera_xy_only`.
4. Export a small GIF and a higher-resolution MP4.
