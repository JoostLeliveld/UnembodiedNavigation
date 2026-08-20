# Localization metrics and evidence contract

This is the canonical entry point for every localization number in this repository. It
separates measurement error, belief error, uncertainty calibration, and navigation outcome;
it also records which data were available online and which were evaluation-only.

If another document gives a localization number without the context required here, this
document wins. The machine-readable companion is
[`localization_metrics_registry.json`](localization_metrics_registry.json).

## The rule that prevents almost every mix-up

Never write “localization error” or “camera error” by itself. A quoted statistic must name:

1. **object** — camera measurement, filter belief, odometry, or executed trajectory;
2. **statistic** — signed bias, mean Euclidean error, median, p95, or RMSE;
3. **reference** — Gazebo ground truth, commanded set-pose truth, or odometry;
4. **projection/runtime** — current floor-plane IPM or a named historical version;
5. **evidence unit** — detections, update steps, or runs;
6. **dataset/run IDs** — never “all runs” or an unnamed pooled set;
7. **status** — current, locked mechanism-only, historical, or diagnostic.

Recommended compact citation:

```text
camera-measurement mean Euclidean error vs commanded GT,
current floor-IPM, balanced set-pose dataset, n=1844 detections: 66.6 mm
```

## What is known, by whom, and when

| Stage | Information available online | Not available online |
|---|---|---|
| Camera capture | RGB image, capture timestamp, camera ID, fixed intrinsics/extrinsics | Robot ground truth, localization error, future detections |
| Detector/projection | Bounding box, chosen bottom-centre pixel, camera model, projected 2-D point and propagated covariance | Ground truth, per-frame error, true robot yaw in the current yaw-blind IPM |
| Camera manager | Timestamped per-camera points/covariances and whatever other cameras reported in its association window | Ground truth and any error statistic scored against it |
| EKF update | Prior belief/covariance, odometry increment, accepted camera measurement, `R`, gate outcomes | Ground truth, NEES, coverage, belief error |
| Planner | Current belief and covariance plus frozen planning/reliability fields | Ground truth, camera measurement error, eventual run outcome |
| Offline evaluator | Ground truth, route, yaw, camera ID, dataset stratum, and all logged estimates | Nothing operational may consume these evaluation-only fields |

Therefore a camera never “has 77 mm error.” An offline evaluator measured a signed component
of a particular camera’s residual under a particular historical pipeline and sampling design.

## Four different objects that were previously called “error”

Let `p_gt(t)` be ground-truth position at the measurement/update timestamp, `z_c(t)` a
projected point from camera `c`, and `mu(t)` the filter belief mean.

| Level | Canonical quantity | Allowed statistics | Interpretation |
|---|---|---|---|
| Camera measurement | `e_cam,c = z_c - p_gt` | mean/median/p95 of `||e||`; signed radial/lateral bias; component SD | Quality of one projected camera observation before filtering |
| Filter belief | `e_belief = mu - p_gt` | RMSE, median, p95 | End-to-end estimator accuracy after odometry and camera fusion |
| Belief honesty | `e_belief` together with `P` | NEES, 50/95% ellipse coverage, stated 1σ | Whether claimed uncertainty matches belief error; not another accuracy metric |
| Navigation/run | time series of `e_belief` and physical outcome | time-aligned ATE/RMSE/p95, final error, clean goal, contact | Closed-loop outcome; the experimental unit is a run/seed, not a detection |

Additional definitions:

- **Mean Euclidean error:** `mean(||e_i||)`. This is not RMSE.
- **RMSE:** `sqrt(mean(||e_i||²))`. It weights large errors more strongly.
- **Signed bias:** a component of `mean(e_i)` in a stated frame. It may be negative and is
  not total error.
- **Radial/lateral:** components along and perpendicular to the camera-to-target ground
  bearing. They are not world `x/y`.
- **p95:** the 95th percentile of per-sample error magnitudes, not a 95% covariance ellipse.
- **95% coverage:** fraction inside the filter’s stated ellipse. “Outside” is `1-coverage`.
- **NEES:** squared belief error normalized by belief covariance. In 2-D, compare median NEES
  with the 2-D median reference 1.386, or mean NEES with mean reference 2.0; never cross those.
- **Mean stated σ in `BELIEF-V2`:** `mean(sqrt(trace(P)/2))`, an RMS per-axis 1σ. It is not
  expected to equal 2-D radial RMSE; for a calibrated isotropic zero-mean Gaussian, radial
  RMSE is `sqrt(2)·σ`. Use NEES and ellipse coverage to judge honesty.

## Canonical current camera-measurement comparison

This is the only table currently suitable for comparing Cameras A–D on localization error.
All rows use the same detector, set-pose protocol, four cardinal robot yaws, camera model,
floor-plane IPM, ground-truth reference, and scoring code. Ground truth and yaw are used only
for offline scoring/stratification.

Source: `logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json`, generated by
`experiments/pixel_ground_path/e7_ipm_zero_parameter.py`.

| Camera | Detections | Mean range | Mean `||e_cam||` | Median | p95 | Radial bias | Lateral bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 455 | 8.4 m | 64.6 mm | 71.0 mm | 98.9 mm | −25.3 mm | −6.8 mm |
| B | 476 | 8.2 m | 68.1 mm | 76.1 mm | 103.1 mm | −25.3 mm | +14.4 mm |
| C | 449 | 7.8 m | 66.6 mm | 75.0 mm | 100.4 mm | −26.3 mm | +18.8 mm |
| D | 464 | 8.3 m | 67.1 mm | 74.7 mm | 104.7 mm | −32.5 mm | −16.0 mm |
| **Pooled** | **1844** | — | **66.6 mm** | **74.2 mm** | **102.1 mm** | **−27.3 mm** | **+2.6 mm** |

Interpretation: under the current measurement path, Camera C is not unusually inaccurate.
The per-camera mean errors span only 64.6–68.1 mm. The signed biases differ, but none is the
historical 76.9 mm Camera C value.

### Current conditional-covariance audit

`RCOND-UV-CURRENT-AUDIT` reuses the exact 1844 `PG-IPM-CURRENT` detections to ask a
different measurement-level question: does bottom-centre pixel residual covariance vary
with camera-to-robot range? Six leave-one-spatial-block-out folds compare a pooled constant
covariance with non-negative `a + b r²` component variances; both share per-camera constant
bias terms. The fitted slopes are exactly zero. Both arms have mean Gaussian NLL 5.699 and
87.0% coverage at a nominal 95% ellipse. Consequently only constant conditional pixel
covariance is current; a spatial `R_cond` claim is forbidden. Spatial ground-space ellipses
may still arise by propagating this constant pixel covariance through the camera projection
Jacobian. Source:
`logs/studies/factorized_observation_successor/rcond/summary.json`.

## Historical v2 driving/belief evidence

The July driving captures are retained because they demonstrate a valid filter mechanism:
repeated, correlated residuals make a filter overconfident. They do **not** provide current
camera accuracy or a fair A–D camera ranking.

| Property | Value |
|---|---|
| Runs pooled | `smoke1_20260716`, `smoke2_20260716`, `fusion_handover_20260721` |
| Sampling | driven routes; two recorded axis-aligned headings; camera/route/region/yaw confounded |
| Projection | retired `projection_calibration_v2`, contact plane 0.05 m plus fitted along-bearing terms |
| Measurement-level sample | 1421 scored rows in the mesh-validation analysis |
| Belief-level sample | 1424 filter update steps in the belief analysis |
| Historical Camera C number | +76.9 mm **signed cross-bearing bias**, not mean total error |
| Why it appeared | −4.06 px silhouette/contact-point offset at the sampled yaw/viewpoint; CAD mesh reduces it to −0.33 px and +8.1 mm residual bias |
| What remains valid | the covariance-floor/temporal-correlation mechanism and its within-study ablations |
| What is prohibited | calling 77 mm current; calling it Camera C’s total error; ranking camera hardware; mixing it with the balanced-IPM table |

The belief study’s statistics are also a separate level. Under the historical v2 inputs:

| Filter arm | Belief RMSE | Belief p95 | Mean stated 1σ | Median NEES | Outside stated 95% ellipse |
|---|---:|---:|---:|---:|---:|
| Trust every camera | 53.2 mm | 94.3 mm | 19.2 mm | 4.22 | 41.9% |
| Per-camera correlation floor | 50.0 mm | 92.3 mm | 50.6 mm | 0.46 | 3.3% |

These are belief-level, within-study comparisons over the same 1424 update steps. They may
not be compared numerically with the per-detection IPM table as though both measured the same
object. The floor arm is conservative (median NEES 0.46 versus 1.386; 3.3% outside versus
nominal 5%), not “honest to 2%” because its 50.0 mm radial RMSE happens to be close to its
50.6 mm RMS per-axis σ.

## Run and dataset registry

| ID used here | Unit | World/protocol | Purpose | Comparison permission |
|---|---|---|---|---|
| `PG-IPM-CURRENT` | 1844 detections | four-camera set-pose grid, four cardinal yaws | Current camera measurement accuracy | Compare A–D and projection arms within this dataset |
| `RCOND-UV-CURRENT-AUDIT` | 1844 detections | same rows as `PG-IPM-CURRENT`, six spatial folds | Conditional pixel covariance commissioning | Constant versus range-conditioned covariance only; not independent new data |
| `MC-DRIVE-V2` | 1421 scored detections | three July driven captures | Historical residual/silhouette diagnosis | Compare models on identical rows; no fair A–D ranking |
| `BELIEF-V2` | 1424 update steps | same capture family, retired v2 observations | Filter honesty mechanism | Compare filter arms only within this study |
| `HONEST-CAMPAIGN-V1` | 43 closed-loop runs | original campaign and canonical GT columns | Belief/run metrics | Compare registered campaign conditions/seeds only; not A–D measurement accuracy |
| old `truth_*` logs | rows/runs | legacy odometry-as-truth | Diagnostic history | Never use as GT evidence |

## Allowed and forbidden comparisons

Allowed:

- A versus B versus C versus D inside `PG-IPM-CURRENT`.
- Raw IPM versus v2/v3/v4 on the exact same 1844 detections, explicitly labelled as a
  projection ablation.
- Filter arm versus filter arm inside `BELIEF-V2`, with its historical-input caveat.
- Campaign condition versus condition only when route set, seeds, runtime, and GT scoring are
  fixed by the campaign registry.

Forbidden:

- 76.9 mm historical signed lateral bias versus 66.6 mm current mean Euclidean error.
- Camera-measurement error versus belief RMSE.
- Mean error versus RMSE without recomputing both on the same samples.
- Pooled detections as if they were independent camera or run replicates.
- Comparing camera rows from driven logs whose routes/yaws/regions differ.
- Pooling arbitrary “available runs,” successful runs only, or whichever run directories a
  glob happens to return.
- Any metric against `truth_*`/wheel odometry presented as ground-truth localization error.

## Run-selection policy

Every analysis must declare its run manifest before reading data. The manifest must specify:

- exact run/capture IDs;
- inclusion and exclusion rules independent of outcome;
- projection/runtime version;
- timestamp join and maximum allowed delta;
- experimental unit and aggregation hierarchy;
- statistic and uncertainty interval;
- required GT fields;
- treatment of missing detections and incomplete runs.

Globbing is allowed only to resolve the members of a frozen manifest and must assert that the
resolved set equals the manifest. A script must fail on missing or extra runs.

## Reporting template

Use this block in new result summaries and figure provenance:

```yaml
metric_object: camera_measurement | filter_belief | belief_honesty | navigation_run
metric_name: mean_euclidean_error | rmse | p95 | signed_lateral_bias | median_nees | coverage_95
reference: commanded_gt | gazebo_gt
frame: world_xy | camera_bearing
projection_runtime: floor_ipm_current | v2_historical | <frozen contract id>
dataset_or_campaign: <registered id>
run_ids: [<exact ids>]
experimental_unit: detection | update_step | run
n_units: <integer>
online_inputs: [<fields available to the method>]
evaluation_only_inputs: [<fields used only to score>]
status: current | locked_mechanism_only | historical | diagnostic
```
