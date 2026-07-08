# GP-Derived Predictive Camera Covariance

[Back to repository overview](../README.md)

This module turns empirical detector reliability into the state-dependent camera
covariance used by the visibility-aware planner.

## Contribution At A Glance

| Question | Answer |
| --- | --- |
| Problem | A fixed warehouse camera is not equally reliable everywhere, but the planner needs that unevenness in a usable form. |
| Contribution | Detector scores are fit with a spatial GP, discounted into a conservative trust field, and converted into planner-facing `R_plan`. |
| Implementation | GP fitting lives in [`../scripts/visibility_comparison/fit_visibility_gps.py`](../scripts/visibility_comparison/fit_visibility_gps.py); planner loading lives in [`../src/planning/planning/core/visibility_gp_map.py`](../src/planning/planning/core/visibility_gp_map.py). |

## Visual Demonstration

![Induced covariance](demos/images/induced_covariance.png)

The GP predicts spatial detector trust. It does not learn `R` online. The
planner maps trust to observation covariance through a precision blend between a
sharp visible-camera covariance and a broad missed-camera covariance.

![R plan map and ellipses](demos/images/r_plan_map_and_ellipses.png)

In the locked model, `R_plan` is a symmetric image-space covariance matrix:

```text
R_plan = diag(r_plan^2, r_plan^2)
```

The drawn ellipses are therefore circular glyphs whose size changes across the
map. They visualize how uncertain an expected camera measurement would be, not
the physical robot footprint.

Additional media is catalogued in [`demos/`](demos/).

## Inputs And Outputs

| Input | Output |
| --- | --- |
| Detector score targets over warehouse `(x, y)` | `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz` |
| Locked world/camera geometry | `P_conservative_plan_map` |
| GP fit parameters | planner-facing reliability and covariance lookup grid |

## Method

1. Capture detector outcomes at sampled robot poses.
2. Aggregate raw YOLO scores by planar robot position.
3. Fit a logit-space GP over `(x, y)`.
4. Apply an uncertainty discount to create a conservative planner map.
5. Convert reliability into image-space observation covariance through the
   planner precision blend.

The locked blend is:

```text
1 / R_plan = trust / R_visible + (1 - trust) / R_miss
```

`R_visible` and `R_miss` are fixed covariance settings. The GP supplies the
trust value that scales between them.

## Performance And Diagnostics

Fit summary:

| Method | Train points | Target mean | Target min | Target max |
| --- | ---: | ---: | ---: | ---: |
| `yolo_score_raw` | 238 | 0.550 | 0.0016 | 0.9232 |

Artifact metadata:

- [`../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_manifest.json`](../paper_artifacts/gp/warehouse_visibility_gp_v1/gp_manifest.json)
- [`../paper_artifacts/figures/gp_pipeline_aws_provenance.json`](../paper_artifacts/figures/gp_pipeline_aws_provenance.json)

## Reproduce

Fit a GP from local capture/target artifacts:

```bash
python3 scripts/visibility_comparison/fit_visibility_gps.py \
  --gp-targets logs/visibility_comparison/warehouse_visibility_targets_v1/gp_targets_xy_aggregated.csv \
  --capture-manifest logs/visibility_comparison/warehouse_visibility_capture_v1/capture_manifest.json \
  --out logs/visibility_comparison/warehouse_visibility_gp_v1 \
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

See available and planned media in [`demos/`](demos/).
