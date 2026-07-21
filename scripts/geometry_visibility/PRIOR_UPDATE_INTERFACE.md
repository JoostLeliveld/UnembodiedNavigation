# Initial prior → online update: the interface contract

This is the seam between the **perfected initial prior** (offline, geometry-only,
available at t=0 for a new layout) and the **online GP update** (which refines it from
driving data). The prior gives the update a *calibrated mean* and an *honest strength*;
the update owns everything after that.

Build the prior artifact with:

```bash
python3 scripts/geometry_visibility/build_calibrated_prior.py
# -> logs/geometry_visibility_prior/calibrated_prior_v1/calibrated_prior.npz
```

## What the prior provides (`calibrated_prior.npz`)

| field | shape | meaning |
| --- | --- | --- |
| `xs`, `ys` | (nx,), (ny,) | grid axes (world metres) |
| `fov_mask` | (ny, nx) | camera can see the cell at all (hard gate) |
| `p_detect_map` | (ny, nx) | **calibrated** P(reliable detection) ∈ [0,1] |
| `prior_logit_mean_map` | (ny, nx) | `logit(p_detect)` — GP prior **mean** (logit space) |
| `prior_pseudocount_map` | (ny, nx) | prior **strength** n0(x) ∈ [1, 20]: obs-equivalent |
| `r_plan_std_map` | (ny, nx) | pixel measurement std for the EKF (2.5→40 px) |
| `min_clearance_map`, `px_per_m_min_map` | (ny, nx) | the two geometric features (debug/re-fit) |
| `detector_response` | json | `{intercept, coef, feature_names}` — the fitted link |

`p_detect = in_fov · σ(intercept + b_clr·clearance_m + b_pxm·log10(px/m))`. The link
coefficients are a **one-time detector-response characterisation** (a property of the
camera + YOLO model, not the layout); the two geometric features are recomputed per
layout, so the prior transfers zero-shot to a new warehouse.

## How the online update consumes it

The prior is deliberately shaped to drop into a **Beta–Bernoulli grid** (what the
existing `online_update_demo.py` uses). Per cell, convert the mean + strength into a
Beta prior:

```
α0(x) = n0(x) · p_detect(x)
β0(x) = n0(x) · (1 − p_detect(x))          # so α0 + β0 = n0  (prior worth n0 observations)
```

After the robot has driven and produced `k` reliable detections out of `N` attempts
whose belief was near cell `x` (weighted by an RBF kernel over neighbours, and — per the
certainty-spread finding — spread by the belief covariance rather than hard-gated):

```
posterior_mean(x) = (α0(x) + k) / (n0(x) + N)
```

The **strength** n0(x) is the whole point of the seam: it is high (→20) where geometry
is unambiguous (clear line of sight or clearly blocked, well inside the image), so a
handful of noisy drive-by detections cannot flip a confident prediction; it is low (→1)
where geometry is epistemically fragile (grazing an occluder top, at the image edge, or
where a sensed height map returned nothing), so **driving data dominates immediately**
exactly where the prior is least trustworthy — including the regional detection deficits
the geometry cannot see (e.g. the held-out block where `p_detect` over-predicts).

For a **logit-space GP** instead, use `prior_logit_mean_map` as the GP mean function and
set the prior variance from the same strength, e.g. `σ0²(x) ≈ c / n0(x)` (larger prior
variance where n0 is small → data pulls the posterior faster there).

## How the planner consumes it

`r_plan_std_map` is the per-cell pixel measurement std handed to the EKF/planner (the
existing `r_visible_uv`/`r_miss_uv` precision blend, but with **trust = calibrated
`p_detect`** rather than the old mis-calibrated score). Better `R_plan` → better
navigation is already established in the paper; this only makes the `R_plan` fed in
reflect the true per-cell reliability. NIS on 20140 real corrections
(`rplan_nis_calibration.py`) shows the runtime `R_plan` is *conservative*
(mean 0.48 ≪ χ²(2)=2), so this map is an upper bound on the noise.

## Honesty / scope

- Ground-truth positions are **never** a prior input — used only to score.
- The occluder geometry for the `clearance` feature is **complete CAD as an EVALUATION
  reference**. Deployment senses clearance from a depth/stereo height map
  (`height_map_from_points`; `depth_occlusion_prior.py` showed depth recovers CAD-level
  occlusion). The characterised link is unchanged.
- The link is validated by leave-4×4-spatial-block-out CV on 556 per-sample YOLO labels
  (whole regions held out). **Cross-*layout* transfer is untested** — it needs a second
  world; block-CV only demonstrates spatial generalisation within this layout.
