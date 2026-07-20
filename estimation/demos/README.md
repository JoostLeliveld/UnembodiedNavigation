# Estimation Demo Media

[Back to estimation module](../README.md)

This folder holds README-facing media for the image-to-BEV and calibration
contribution. Regenerate the static assets from the repository root with:

```bash
python3 scripts/paper_figures/make_readme_visuals.py
```

## Available README Media

| Path | Type | Story beat |
| --- | --- | --- |
| [`images/image_to_bev_01.png`](images/image_to_bev_01.png) | PNG still | Camera pixel and ground-plane projection side by side. |
| [`images/affine_calibration_before_after.png`](images/affine_calibration_before_after.png) | PNG diagnostic | Raw homography residuals versus affine-corrected residuals. |

## Existing Source Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/figures/localization_pathway.png`](../../paper_artifacts/figures/localization_pathway.png) | Earlier localization pathway figure. |
| [`../../paper_artifacts/figures/inputs/loc_pathway_frame_v7b.jpg`](../../paper_artifacts/figures/inputs/loc_pathway_frame_v7b.jpg) | Source camera frame used by the README visual. |
| [`../../../midterm_presentation/assets/slide04/belief_noise_terms.png`](../../../midterm_presentation/assets/slide04/belief_noise_terms.png) | Presentation-style noise-term explanation. |

## Next Media Slots

| Planned path | Type | Story beat | Source |
| --- | --- | --- | --- |
| `images/localization_error_trace.png` | PNG plot | Truth-belief error over one representative run. | `experiment.csv` plus perception diagnostics. |
| `animations/belief_update.gif` | GIF preview | Prediction grows covariance; detection update tightens belief. | Planner/state logs from a representative run. |
| `videos/state_pipeline.mp4` | MP4 clip | Topic walkthrough from `/perception/pixel_pose` to `/state/bev`. | ROS bag or experiment CSV plus overlays. |
