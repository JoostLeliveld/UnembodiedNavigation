# GP Demo Media Plan

[Back to GP module](../README.md)

Planned media paths are story slots, not tracked files yet. Existing visuals are
linked from `paper_artifacts/`.

## Existing Assets

| Asset | Use |
| --- | --- |
| [`../../paper_artifacts/figures/gp_pipeline_aws.png`](../../paper_artifacts/figures/gp_pipeline_aws.png) | Current end-to-end GP pipeline figure. |
| [`../../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_fit_summary.csv`](../../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_fit_summary.csv) | Fit summary. |
| [`../../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_manifest.json`](../../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_manifest.json) | Artifact metadata. |

## Planned Media Slots

| Planned path | Type | Story beat | Target | Source |
| --- | --- | --- | --- | --- |
| `images/score_samples.png` | PNG plot | Raw detector-score samples over the warehouse floor. | 1400 px wide | `gp_pipeline_aws.png` panel A or regenerated figure. |
| `images/gp_mean.png` | PNG plot | Mean reliability field. | 1400 px wide | `yolo_score_raw_gp.npz`. |
| `images/gp_uncertainty.png` | PNG plot | GP predictive uncertainty. | 1400 px wide | `F_std_map` in the GP artifact. |
| `images/induced_covariance.png` | PNG plot | Reliability converted to image-space covariance. | 1400 px wide | Planner covariance mapping. |
| `animations/covariance_mapping.gif` | GIF preview | Robot point moves through map while covariance grows/shrinks. | 10-15 s | Planner covariance query over a path. |
| `videos/gp_field_walkthrough.mp4` | MP4 clip | Pan through reliability, uncertainty, and covariance. | 30-45 s | Generated maps plus route overlays. |

## Capture Checklist

1. Load `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz`.
2. Render raw samples, mean, conservative reliability, and uncertainty maps with
   consistent axes.
3. Animate a representative route querying the map.
4. Export a small GIF for GitHub and a compressed MP4 for higher-resolution
   explanation.
