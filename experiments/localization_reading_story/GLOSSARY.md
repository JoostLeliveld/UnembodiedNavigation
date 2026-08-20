# Glossary — one name, one quantity

**This file is the authority for how these quantities are named in writing.** Every
confusion this project has hit had the same shape: one name carrying two quantities. The
right-hand column is the one that does the work.

Scope: the **writing layer** — figures, captions, the brief, the presentation, thesis text.
The code keeps its existing names; §4 maps between them. Do not rename anything in `src/`
while a campaign is running.

---

## 1. The three fields (never fold these together)

| Write | Exact estimand, in words | Units | Current value | It is **NOT** |
|---|---|---|---|---|
| **`R_noise(x)`** | Covariance of a reading about **the prediction at that pose**, given a usable reading arrived | cm² (or px²) | **camera frame: 0.88 cm along the ray × 0.61 across, aspect 1.45**<br>map frame pooled: 0.80 × 0.73 (aspect 1.10 — *diluted by ray rotation, do not read as round*)<br>*(keypoint retrained, both markers visible, n=369)* | the expected total error; the spread about a drive average; anything to do with a missing detection |
| **`b(x)`** | The part of the error that repeats — same sign, same size, every time at that pose | cm | **−0.30 cm along the ray, −0.02 across; near-CONSTANT** (χ²/dof 1.48 constant vs 1.23 linear; the d²/H height law is rejected at 3.96). True spatial sd over the floor 0.25 (E–W) / 0.38 cm (N–S) *(keypoint retrained, both markers visible)* | noise; something a covariance can carry; something that averages down over N readings |
| **`p_use(x)`** | Probability that, with the robot at `x` and heading uniform, this camera yields a reading that passes the acceptance gate | probability | see `availability_paper` | a confidence score; a measure of how *accurate* the reading is |

**The one sentence that prevents the whole argument:** `R_noise` says how fast the belief
sharpens with more readings; `b(x)` says where that sharpening stops. A planner can shrink
the first (look longer, add a camera) and can only *avoid* the second.

For a safety margin the two enter differently, which is why they cannot share a name:

```
margin(x)  ≈  |b(x)|  +  k · σ_noise(x) / √n
              never shrinks   shrinks with every reading
```

## 2. Quantities that must never appear bare in text

| Never write | Write instead | Why |
|---|---|---|
| `R` | `R_noise` / `R_uv` / `R_xy` | "R" has meant the variance about the mean, the second moment about zero, and a stand-in for missing data — in the same paragraph |
| `R_total` | **"second moment about zero (scatter + lean)"**, spelled out | It is a diagnostic, not a model term. Calling it `R` is exactly the bias/variance conflation the supervisor flagged. Proof it is not a usable covariance: feeding the filter a *perfect* oracle `R_total` moves RMSE from 9.70 → **9.69 cm** — 0.01 cm, because no `R` moves the mean |
| `R_spread` | `R_noise` if it is about the per-pose prediction; **"spread about the drive average"** if it is pooled | The repo's `R_spread` is pooled per drive, so on a 1.3–9.7 m sweep it absorbs the variation of the mean and reads ~3× too big |
| `r_miss` | — nothing. There is no covariance of a reading that did not arrive | A miss is *no update*, not a wide update. The correct value is ∞, i.e. zero precision |
| innovation / NIS | `innov_px` / `nis_px` **or** `innov_m` / `nis_m` | The two filter paths measure in different spaces (§3) and both call it NIS |
| a bare error figure | `0.74 cm (keypoint, retrained)` | See §5 |

### ⚠️ `R_noise` — always say WHICH FRAME

Measured 2026-08-18 on the 369 clean held-out readings. In the **camera frame** the covariance
is elongated **1.45:1 along the viewing ray**, matching the closed form aspect = slant/height
(1.42 at 4.8 m, 1.77 at 11 m). In the **map frame**, pooling over a ±35° fan of ray directions
rotates the ellipse and averages it to aspect 1.10 — which reads as "round" and is an artefact
of pooling, not a property of the reading. **Quote the camera frame; state the frame either way.**

### ⚠️ Single-marker readings are a separate failure mode, not part of `b(x)`

| readings | along-ray mean | along-ray sd | n |
|---|---:|---:|---:|
| both markers labelled visible | −0.30 cm | 0.88 cm | 369 |
| **only one marker visible** | **−1.79 cm** | **3.21 cm** | 49 |

The single-marker readings cluster at longer range and manufacture an apparent range trend in
`b(x)` — that is where the retired "−1.10 cm at 9–20 m" figure came from. They carry a runtime
flag (`front_labelled_visible` / `rear_labelled_visible`), so this is an argument for passing
the visibility flag into the filter, **not** for a bias model.

## 3. Measurement space — always say which

Two live paths, sharing one gate chain
(`src/planning/planning/core/belief_correction.py`):

| path | `z` | `h(x)` | `R` | used by |
|---|---|---|---|---|
| **pixel** (`PixelMeasurementSource`) | `(u,v)` px | nonlinear, via the camera model | `R_plan` from the visibility GP | single-camera / paper-1 |
| **metric** (`FusedMapMeasurementSource`) | `(x,y)` m | `[I₂ \| 0]` | fused covariance | multicam |

**Why this must be stated on every filter number:** in the pixel path the innovation is
`measured_px − h(belief)`, and both sides pass through the same calibration. A bias in
`h()` pushes the belief the same way, so the belief converges to whatever pose drives the
pixel residual to zero. The error becomes **a bias in the estimate, never a large
innovation** — and NIS cannot see it. Measured on the locked campaign:

| | paper-1 pixel path |
|---|---|
| pixel innovation | median 1.53 px |
| `R` applied | 2.5 px |
| NIS (gate 9.21) | median **0.29** — a healthy 2-DOF filter sits at 1.4–2, so `R` was *loose* |
| NIS rejections | 1 / 6370 = **0.016 %** |
| **belief error vs truth** | **p95 0.127 m** |

Every internal check passed while the belief sat 13 cm off. **So never quote a NIS or an
innovation as evidence of health without the metric residual beside it in the same table.**

## 4. Code name → writing name

| in the code | write | note |
|---|---|---|
| `R_plan`, `conditional_cov_uv` | `R_uv` | pixel-space covariance handed to the pixel path |
| `r_visible_uv = 2.5` | — | ~5× larger than the measured σ_px (0.47–0.51 px); it is a bias floor wearing a pixel-noise costume |
| `r_miss_uv = 40 / 120` | — | not a real quantity; `MissEndpointPolicy.require_reconciled()` is right to refuse it |
| `R_total` (from `oracle_noise`) | "second moment about zero" | never `R` |
| `R_spread` (from `oracle_noise`) | "spread about the drive average" | not `R_noise` |
| `eval_res_x/y` | metric residual vs truth | eval-only, firewalled, never trains a deployable covariance |
| `p_vis`, `tau`, `trust` | `p_use` | one name for availability |
| `floor_c` | `b_c` / bias floor | the per-camera part that never averages down |
| `sigma_px` | `σ_px` | measured **0.47 px** raw / **0.51 px** corrected |

## 5. Every number carries its reading method

The largest single source of error in this project's own writing. Box-bottom-era numbers
have been quoted as current more than once — including the anisotropy claim, the 2.5 cm
covariance floor, and "geometry is 6.4× too small."

**Rule: a number without its method is not quotable.**

| reading method | tag to use | typical miss | systematic |
|---|---|---:|---|
| bottom of the detected box → floor | `(box bottom)` | 7.78 cm | −6.34 cm, swings −2.76→−8.85 cm with heading |
| marked point, model for the *old* camera | `(keypoint, old camera)` | 1.51 cm | −0.42 / −0.28 cm |
| marked point, retrained | `(keypoint, retrained)` | **0.74 cm** | −0.02 / −0.41 cm |

Retired-input studies say so on their face: `achievable_precision_map` is historical-v2
(its 77 mm floor is box-bottom era) — **quote its model, never its numbers.**

## 6. Two worlds

| world | role |
|---|---|
| `warehouse_aws` | method development |
| `warehouse_full_4cam` | frozen-method evaluation only |

⚠️ Open: **EXP-AVAIL-SOURCE ran in `warehouse_full_4cam`**, which `research/06_world_camera_design.md`
reserves for evaluation. Needs a registered exception or a `warehouse_aws` re-run of the
geometric arms before promotion.
