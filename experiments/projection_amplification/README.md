# projection_amplification — is the spatial camera noise detector variation or geometry?

**Chapter served:** [09 — multicamera handover & fusion](../../research_story/09_multicamera_handover_fusion/)
(ICRA-2027 observation-model workstream; index in
[`modules/07_multicam_handover_fusion`](../../modules/07_multicam_handover_fusion/README.md)).

## Question

Every reliability model in this repo assumes camera accuracy is a *learnable
spatial field*. An oblique ground-plane camera, though, amplifies a **fixed** pixel
error into a ground error that grows with range and image row. A detector that is
equally good in every part of the image would therefore still produce a
position-dependent world error. That null model has to be priced before any
heteroscedastic `R_cond(x)` is fitted, or the fit will be credited with structure
that is pure projection geometry.

Three variance models, held out by region, on the same deployed-corrected
residuals:

| model | form | free parameters per camera |
|---|---|---|
| `R1-iso` | `s² I` | 1 |
| `R1-full` | one 2×2 `Σ_c` (the deployed conditional-covariance form) | 3 |
| `R2-geom` | `σ_pix² J_g J_gᵀ` — all spatial structure from the projection Jacobian | **1** |

`R2-geom` has the *same parameter count* as `R1-iso`, so a win is a statement about
where the structure comes from, not about capacity.

## Method notes

- `J_g` is the finite-difference Jacobian of the **exact deployed projection path**
  (`reliability.projection._projection_derivative`, including the
  `projection_calibration_v2` along-bearing correction) — the same derivative the
  runtime covariance propagation uses.
- Folds are **leave-region-out** (contiguous bands along the dominant spread axis),
  not random: residual frames are 0.2 s apart, so a random split leaks the answer.
- Residuals are centred on the **training-fold mean**, so this compares variance,
  not bias — bias is the sibling study
  [`external_camera_bias_model`](../external_camera_bias_model/).
- A second, clearly-labelled **oracle-centred** scoring (test-fold mean removed,
  not deployable) isolates "the bias does not transfer across regions" from "the
  variance model is wrong".
- `σ_pix` fitted here is an **effective** pixel noise: it absorbs detector
  localization noise, residual calibration error, timing error and contact-point
  error. It is not a detector property and must not be quoted as one.

## Run

```bash
# needs residuals.csv from the sibling study (already committed under logs/studies)
python3 experiments/projection_amplification/exp1_geometry_vs_detector.py
```

Outputs → [`logs/studies/projection_amplification/exp1_geometry_vs_detector/`](../../logs/studies/projection_amplification/exp1_geometry_vs_detector/)
(`summary.json`, `fig_g1`–`fig_g3`, `RESULTS.md`).

## Reuse map

| need | reused from |
|---|---|
| projection + its Jacobian | `reliability.projection` (`camera_model_from_world`, `_project_pixel_to_world`, `_projection_derivative`, `load_projection_calibration`) |
| residual dataset, world/camera/site constants, deployed calibration path | `experiments/external_camera_bias_model/residual_audit.py` (imported, never re-declared) |
| NLL / coverage | `reliability.conditional_covariance.matrix_nll`, `chi2_coverage` |
| binning | `scripts/shared/metrics.binned` |
| camera model | `unav_common.camera_model.ObliqueCameraModel` |

## Evaluation-only data

Ground truth (`true_x/true_y`, via the residual CSV) is used **only** to measure
residuals. It never enters a projection, a Jacobian, a fitted parameter or a
covariance. The oracle-centring diagnostic is the one place held-out information is
used deliberately, and it is labelled as a diagnostic everywhere it appears.
