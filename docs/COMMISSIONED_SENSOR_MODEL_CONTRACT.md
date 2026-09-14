# Commissioned sensor-model implementation contract

This document is the code-facing version of the current paper plan. It preserves the
scientific choices that are already settled while leaving the correction and covariance
winner open for the planned comparison.

## Commissioning data unit

The independent unit is a complete reference-controlled drive. Fit, development and audit
partitions contain whole drives. Frames from one drive may not be split across model-fitting
and evaluation folds. Every scheduled camera opportunity is retained, including detector
misses and gate refusals.

Ground truth is used offline to construct correction targets and residuals. It is never
passed to the sensor gate, correction model, covariance query, fusion algorithm, estimator,
or planner.

## Deterministic sensor gate and availability

The analysis and deployment gate is `config/commissioning_sensor_gate_v2.yaml`. It applies
four checks to the fixed detector return: confidence of at least 0.25, finite box width and
height of at least 16 source-image pixels, at least five pixels between the complete box and
every image boundary, and a finite ground-plane projection of the box bottom-centre. The
gate cannot inspect belief, predicted boxes, expected robot size, association residuals,
localization error, innovation, NIS or localizer acceptance.

The completed reference-controlled drives retain their original
`commissioning_sensor_gate_v1` outcomes for provenance. Replaying v2 reads the recorded
detector and projection fields and writes new derived outcomes. It does not overwrite the
collected tables.

The target learned as `q_i(p)` is exactly:

```text
fixed detector hit AND deterministic sensor-gate pass
```

`q_i(p)` is a future-observation forecast. It is not a second runtime gate. NIS remains a
separate estimator-consistency decision after the measurement and covariance exist.

The fitted field is `q_i(p)`: camera identity and two-dimensional ground-plane position are
its only query inputs. Outcomes are pooled over the headings sampled at each position. Heading
is not an availability-model input.

## Candidate correction and matched R

Every correction starts from the ground-plane projection of the raw frozen-detector box
bottom centre. Visual-hull observations, hull-equivalent positions, belief-projected boxes,
and CAD silhouettes are prohibited as correction bases and runtime inputs.

The candidate families remain open until the development-drive comparison:

| correction | input | matched covariance candidates |
|---|---|---|
| `raw_box` | raw projected box bottom centre | freshly fitted per-camera constant or spatial field |
| `box_mlp` | raw-box and runtime camera features | freshly fitted per-camera constant or spatial field |
| `box_mlp_visibility_residual` | `box_mlp` plus the 16x16 visibility matrix | freshly fitted image-conditioned runtime model with a planner spatial marginal |

For every row, `R` is learned only from whole-drive out-of-fold residuals produced by that
row's correction. Reusing one correction's covariance for another candidate invalidates the
comparison. Fit the correction first, measure temporal and simultaneous cross-camera
dependence second, and select `R` last.

## Runtime and planning boundaries

Runtime order is:

```text
detector -> sensor gate -> projection -> correction -> matched R
         -> correlation-aware camera fusion -> NIS -> robot belief update
```

The primary estimator contains one robot filter and consumes each physical camera frame
once. Persistent per-camera or shared error is represented with an augmented joint state if
the commissioned drives justify it. A first-stage posterior is never reused by a second
filter as an independent measurement.

Both navigation conditions receive the same detector, selected correction, matched `R`,
runtime estimator, candidate routes and geometric/rollout feasibility tests. The comparison
changes only the future observation model used in belief prediction. In particular,
`q_i(p)` may change expected information and route cost but not route eligibility.

Process-noise covariance `Q_k` is frozen in the estimator and planner configuration. It is not
estimated or selected by this commissioning study.

## Table and plotting interface

`camera_opportunities.csv` is the primary commissioning table. It exposes the versioned
sensor-gate outcome for every opportunity alongside detector, box, projection, reference and
raw-measurement fields. `camera_measurements.csv` contains only admitted measurements with a
valid offline reference. Candidate scripts append model-specific corrected positions,
residuals, fold IDs and matched covariance predictions; they must not overwrite raw fields.

Paper-facing plots use the opportunity table as points first. At minimum they distinguish
detector misses, deterministic refusals, admitted raw measurements and corrected residuals.
Spatial fields, bins and covariance ellipses may be overlaid, but they must report the number
of complete drives and opportunities supporting each summary.

`experiments/reference_controlled_commissioning_v1/plot_points.py` renders the initial
opportunity and raw-residual panels directly from one or more drive tables. It deliberately
does not interpolate coloured support patches. Candidate-specific correction and covariance
plots are added only after their model IDs and whole-drive fold IDs have been written.
