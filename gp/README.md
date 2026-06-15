# GP-Derived Predictive Camera Covariance

[Back to repository overview](../README.md)

This module turns empirical detector reliability into the state-dependent camera
covariance used by the visibility-aware planner.

## Story

Detector performance is spatially uneven. The GP makes that unevenness
planner-readable by turning score samples into a conservative reliability field
and then into predictive image-space covariance.

## Visual Demonstration

![GP pipeline](../paper_artifacts/figures/gp_pipeline_aws.png)

The figure shows raw YOLO-score samples, the conservative planner reliability
map, and the induced image-space covariance.

Planned media is listed in [`demos/`](demos/): separated score/mean/uncertainty
plots, a covariance map, a covariance-mapping GIF, and an optional field
walkthrough video.

## Inputs And Outputs

| Input | Output |
| --- | --- |
| Detector score targets over warehouse `(x, y)` | `paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz` |
| Locked world/camera geometry | `P_conservative_plan_map` |
| GP fit parameters | planner-facing reliability and covariance lookup grid |

## Method

1. Capture detector outcomes at sampled robot poses.
2. Aggregate raw YOLO scores by planar robot position.
3. Fit a logit-space GP over `(x, y)`.
4. Apply an uncertainty discount to create a conservative planner map.
5. Convert reliability into image-space observation covariance through the
   planner precision blend.

## Performance And Diagnostics

Fit summary:

| Method | Train points | Target mean | Target min | Target max |
| --- | ---: | ---: | ---: | ---: |
| `yolo_score_raw` | 238 | 0.550 | 0.0016 | 0.9232 |

Artifact metadata:

- [`../paper_artifacts/gp/aws_gp_v7b/gp_fit_summary.csv`](../paper_artifacts/gp/aws_gp_v7b/gp_fit_summary.csv)
- [`../paper_artifacts/gp/aws_gp_v7b/gp_manifest.json`](../paper_artifacts/gp/aws_gp_v7b/gp_manifest.json)
- [`../paper_artifacts/figures/gp_pipeline_aws_provenance.json`](../paper_artifacts/figures/gp_pipeline_aws_provenance.json)

## Reproduce

Fit a GP from local capture/target artifacts:

```bash
python3 scripts/visibility_comparison/fit_visibility_gps.py \
  --gp-targets logs/visibility_comparison/aws_targets_v7b/gp_targets_xy_aggregated.csv \
  --capture-manifest logs/visibility_comparison/aws_capture_v7/capture_manifest.json \
  --out logs/visibility_comparison/aws_gp_v7b \
  --grid-nx 220 \
  --grid-ny 200 \
  --gp-length-scale 0.90 \
  --gp-noise-var 0.05 \
  --beta 0.5
```

Regenerate the public GP pipeline figure:

```bash
python3 scripts/paper_figures/make_aws_gp_pipeline_figure.py
```

## Relevant Implementation Files

| File | Role |
| --- | --- |
| [`../scripts/visibility_comparison/capture_visibility_samples.py`](../scripts/visibility_comparison/capture_visibility_samples.py) | Visibility sample capture. |
| [`../scripts/visibility_comparison/build_gp_targets.py`](../scripts/visibility_comparison/build_gp_targets.py) | Target table construction. |
| [`../scripts/visibility_comparison/fit_visibility_gps.py`](../scripts/visibility_comparison/fit_visibility_gps.py) | GP fitting and artifact writing. |
| [`../src/planning/planning/core/visibility_gp_map.py`](../src/planning/planning/core/visibility_gp_map.py) | Planner-side artifact loader. |

## Limitations

- The fitted field is setup-specific: world, camera pose, detector, and robot
  appearance all matter.
- Detector confidence is used as an empirical proxy, not as a calibrated
  probability.
- The GP changes camera `(x, y)` measurement covariance only. It is not a direct
  route reward and it does not model heading.

See planned visual media in [`demos/`](demos/).
