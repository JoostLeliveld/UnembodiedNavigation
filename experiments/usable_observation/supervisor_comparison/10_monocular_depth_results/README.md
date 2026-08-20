# Monocular-depth results for supervisor discussion

This folder contains the measured monocular-depth evidence only. It is deliberately separate
from `03_depth_raycast`, whose figures explain a prospective planner mechanism using modeled
depth and are **not** outputs of the monocular models.

## Recommended figure order

1. [`figures/01_model_outputs.png`](figures/01_model_outputs.png) — the same warehouse image
   and four predictions before and after floor anchoring. Raw metric outputs and all anchored
   outputs use the Gazebo reference scale; the relative model remains unitless before
   anchoring. The highlighted floor pixels are the deployment-legal fitting input, while
   Gazebo depth is displayed only as the evaluation reference.
2. [`figures/02_floor_anchoring_accuracy.png`](figures/02_floor_anchoring_accuracy.png) — the
   principal result. A deployment-legal scale/shift fitted from camera calibration and an
   open-floor mask removes most of the raw scale error.
3. [`figures/03_compute_cost.png`](figures/03_compute_cost.png) — inference time and GPU memory.
   This supports a commissioning-time method; it does not by itself justify live inference.
4. [`figures/04_four_camera_depth_accuracy.png`](figures/04_four_camera_depth_accuracy.png) —
   the new Camera A–D test. Each camera is calibrated from the clear frame at 0.4 s and the
   same fit is reused after a pallet appears at 1.2 s.
5. [`figures/05_four_camera_visibility_raycast.png`](figures/05_four_camera_visibility_raycast.png)
   — a top-down depth-buffer raycast beside the evaluation-only geometry oracle, for every
   camera and for the logical any-camera fusion.
6. [`figures/06_camera_a_dynamic_shadow.png`](figures/06_camera_a_dynamic_shadow.png) — the
   dynamic question in isolation: does Camera A's inferred map lose visibility behind the
   newly placed pallet in the same cells as the oracle?
7. [`temporal_anchor_sequence/figures/01_temporal_sequence.png`](temporal_anchor_sequence/figures/01_temporal_sequence.png)
   — 21 repeated four-camera updates, comparing a fresh affine with the Bayesian affine.
8. [`temporal_anchor_sequence/figures/02_anchor_dropout.png`](temporal_anchor_sequence/figures/02_anchor_dropout.png)
   — the predeclared floor-anchor dropout test: fresh fits refuse, while a recent Bayesian
   affine keeps the current RGB-derived map available.

## Result table

Depth MAE is scored against co-located Gazebo optical-axis depth on 451,005 non-floor pixels.
The floor pixels used to fit the deployment-legal anchor are excluded from scoring.

| Model | Raw MAE | Floor-anchored MAE | Oracle-affine MAE | Anchored AbsRel | Within 25% |
|---|---:|---:|---:|---:|---:|
| UniDepthV2 ViT-S | 1.632 m | **0.247 m** | 0.235 m | 3.7% | 99.0% |
| Metric3D v2 ViT-S | 5.195 m | **0.327 m** | 0.305 m | 5.1% | 96.6% |
| DA-V2 relative Small | unitless | **0.337 m** | 0.276 m | 5.1% | 98.0% |
| DA-V2 metric Small | 0.621 m | **0.420 m** | 0.415 m | 6.5% | 96.5% |

## What “oracle affine” means

For one prediction `p`, oracle affine finds the best global scale and shift against Gazebo
truth on the pixels that will be scored: `z = a·p + b`. For an inverse-depth model it fits
`1/z = a·p + b` and then converts back to metres. It is deliberately non-deployable because
it uses the answers during calibration.

It is useful as a diagnostic ceiling, not as a method. If floor anchoring is close to oracle
affine, the model's depth *shape* is useful and the deployment-legal floor ruler recovered
almost all the scale available to a global affine correction. Oracle affine is not perfect
depth and is not the geometry visibility oracle; it cannot repair local shape errors.

## How to interpret “0.3 m”

Approximately 0.3 m MAE means: over the scored pixels, take the absolute optical-axis depth
error at every pixel and average it; the answer is about 30 cm. It does **not** mean 30 cm
robot-localization error, a 30 cm visibility-boundary error, or that every pixel is within
30 cm. MAE also hides the sign and the tail, so bias, RMSE and the downstream visibility map
still matter.

The dashed 0.3 m line in the four-camera plot is only a reading guide from the earlier
Camera-A study. It is not a pass threshold, and the new numbers are not directly pooled with
that study: this is a harder four-camera world, the fit is reused across a dynamic event, and
the evaluation mask is defined by simulator-visible structure rather than by the original
single-camera open-floor band.

## Four-camera result

This test uses eight 1280×720 frames: Cameras A–D at a clear timestamp and again after a
loaded pallet is placed on Camera A's sightline. The method receives RGB, fixed calibration,
the floor plane, and the planner's 2-D traversable-aisle regions. The fit from the clear frame
is reused unchanged. Co-located Gazebo depth and the geometry raycast are evaluation-only.

| Model | Raw structure MAE | Reused floor-fit MAE | Oracle-affine MAE | Visibility balanced accuracy | Visible IoU | Default scale gate |
|---|---:|---:|---:|---:|---:|---|
| DA-V2 metric S | 1.131 m | 0.984 m | 0.618 m | 92.2% | 90.2% | pass A–D |
| DA-V2 relative S | unitless | 0.935 m | 0.559 m | 91.9% | 90.2% | pass A–D |
| Metric3D v2 ViT-S | 4.737 m | 0.518 m | 0.466 m | 93.6% | 90.8% | **fail A–D** |
| UniDepthV2 ViT-S | 1.921 m | 0.637 m | 0.568 m | **95.6%** | **93.8%** | pass A–D |

The rows are medians over eight frame-level measurements. “Structure” means pixels where
Gazebo depth is at least 0.10 m in front of the analytic floor intersection, plus rays that do
not meet the floor; that evaluation-only definition prevents open floor from dominating the
error. Visibility is scored only on in-FOV visible/occluded oracle cells at a target height of
0.35 m.

Metric3D is diagnostically strong after affine alignment, but its clear-frame scale factors
are 2.11–2.93. Because it claims metric output, all four violate the operational pipeline's
default `[0.5, 2.0]` sanity band. The result is therefore plotted but explicitly gate-marked;
the operational method should refuse it rather than silently trusting that correction.

UniDepthV2 is used for the raycast figure because it has the lowest four-camera depth MAE
among models that pass the default scale gate on all cameras. At the pallet timestamp its
four-camera fusion reaches 94.6% balanced accuracy and 95.6% visible IoU. Camera A detects all
15 oracle cells in the new pallet shadow, but predicts 49 lost-visibility cells in total:
100% shadow recall and 30.6% precision. That is useful conservative behavior, but the extra
34 cells show that the dynamic map is not yet calibrated tightly enough to treat as finished.

## Bounded interpretation

The supported result is: **the networks recover useful warehouse depth structure, but their
raw scale is unreliable; a floor-derived affine anchor makes that structure metric without a
depth sensor or CAD.** For UniDepthV2, the anchored MAE is 0.247 m, only 1.2 cm above the
ground-truth-fitted affine oracle on this dataset.

The original result remains mechanism evidence from 12 Camera-A frames in one Gazebo world,
with the anchor refitted per frame. The four-camera result adds a commissioning-fit reuse
test and direct visibility scoring, but only at two timestamps in one scenario. Neither is a
definitive model ranking, a robot-localization result, or evidence of improved navigation.

## Complete 21-update temporal replay

The complete dynamic timeline has now been processed with DA-V2 relative Small: 21
synchronized updates × four cameras = 84 RGB-derived depth frames. The source captures are
replayed at the proposed 10 s update cadence, creating 21 Bayesian updates over a 200 s
operational horizon. This is a filter replay, not 200 seconds of new illumination or camera
drift.

On the untouched sequence, both arms produce 84/84 valid camera maps. A fresh affine and the
Bayesian affine have identical median visibility balanced accuracy (92.02%) and visible IoU
(90.18%). The Bayesian arm's median cycle structure-depth MAE is 0.922 m versus 0.921 m for
fresh fits: 1.6 mm worse. The primary median successive scale/shift change is exactly zero
for fresh fits, so this deterministic sequence does **not** demonstrate a natural stability
gain from filtering.

The Bayesian layer does demonstrate graceful degradation. At four predeclared synchronized
updates, an empty external floor mask creates 16 camera-frame anchor failures. Fresh fits
produce 0/16 maps; the temporal arm produces 16/16 by reusing a posterior only 10 s old.
Against the untouched temporal replay of those same frames, this changes median
structure-depth MAE by +0.7 mm, visibility balanced accuracy by +0.003 percentage points,
and visible IoU by 0.000 percentage points. Current RGB-derived depth is still used; no
obstacle map is carried forward.

This supports the stale-anchor fallback mechanism, not a claim that natural segmentation
failures occur at that rate. A real-camera duration/lighting campaign would be required for
a natural drift benefit. The next downstream test is a navigation ablation with and without
the inferred visibility field.

Single-camera machine-readable values and source paths are in [`results.json`](results.json).
Four-camera metrics are in [`four_camera/results.json`](four_camera/results.json), and the
saved map arrays are in [`four_camera/maps/`](four_camera/maps/). The frozen longitudinal
protocol and results are in
[`temporal_anchor_sequence/PROTOCOL.md`](temporal_anchor_sequence/PROTOCOL.md) and
[`temporal_anchor_sequence/RESULTS.md`](temporal_anchor_sequence/RESULTS.md). Regenerate the
original plots with:

```bash
python3 experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/make_plots.py
```

Regenerate the four-camera evaluation from the saved RGB-only predictions with:

```bash
python3 experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/four_camera_study.py --plots-only
```

Omit `--plots-only` to rerun missing model predictions on a CUDA device. The script and result
JSON record the method/evaluation boundary and the exact commissioning-fit reuse rule.

Regenerate the longitudinal report and figures from its saved predictions/arm caches with:

```bash
python3 experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/temporal_anchor_sequence.py --evaluate
```
