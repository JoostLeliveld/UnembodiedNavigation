# GP Demo Media

[Back to GP module](../README.md)

This folder holds README-facing media for the reliability-to-covariance
contribution. Regenerate the static assets from the repository root with:

```bash
python3 scripts/paper_figures/make_readme_visuals.py
```

## Available README Media

| Path | Type | Story beat |
| --- | --- | --- |
| [`images/collection_to_covariance_story.png`](images/collection_to_covariance_story.png) | PNG explainer | Detector-score collection to GP trust to explicit `R_plan` matrix. |
| [`images/induced_covariance.png`](images/induced_covariance.png) | PNG plot | GP planner trust converted into image-space observation covariance. |
| [`images/r_plan_map_and_ellipses.png`](images/r_plan_map_and_ellipses.png) | PNG diagnostic | How `R_plan` changes across the map; glyph size shows covariance scale. |

## Existing Source Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/figures/gp_pipeline_aws.png`](../../paper_artifacts/figures/gp_pipeline_aws.png) | Earlier end-to-end GP pipeline figure. |
| [`../../paper_artifacts/figures/current_surface/gp_pipeline_current.png`](../../paper_artifacts/figures/current_surface/gp_pipeline_current.png) | Current detection-rate GP pipeline figure. |
| [`../../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_manifest.json`](../../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_manifest.json) | Artifact metadata. |

## Next Media Slots

| Planned path | Type | Story beat | Source |
| --- | --- | --- | --- |
| `images/score_samples.png` | PNG plot | Raw detector-score samples over the warehouse floor. | GP artifact training samples. |
| `images/gp_mean.png` | PNG plot | Mean reliability field. | `P_mean_map` in the GP artifact. |
| `images/gp_uncertainty.png` | PNG plot | GP predictive uncertainty. | `F_std_map` in the GP artifact. |
| `animations/covariance_mapping.gif` | GIF preview | Robot point moves through map while covariance grows/shrinks. | Planner covariance query over a path. |
| `videos/gp_field_walkthrough.mp4` | MP4 clip | Pan through reliability, uncertainty, and covariance. | Generated maps plus route overlays. |
