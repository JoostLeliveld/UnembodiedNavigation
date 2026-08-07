# PROPOSED — control-plane wording for EXP-PROJ-AMP

**Status: draft for approval. Not applied. `registry.yaml` and `08_figures.md` are unchanged.**

EXP-PROJ-AMP is `LOCKED`, and three things changed under it on 2026-08-07 without any new
data. The recorded wording no longer describes what the experiment shows.

## What changed

1. **Primary figure changed identity.** `fig_g2` was 3–7 range-binned medians per camera with
   a Spearman over those bins. It reported ρ = −1.00 for camera A on a relationship running
   opposite to its own prediction. It is now one point per detection over all 1,424, plus
   `fig_g4_camera_{A,B,C,D}` — footprint map and RMS-to-RMS per 0.5 m cell.
2. **A range-only baseline was added** at equal parameter count. It ties `JJᵀ` on every
   camera, which answers the standing reviewer question about whether `JJᵀ` encodes anything
   beyond distance from camera — on this data, no.
3. **Bias and variance were separated**, and that reversed the reading. See below.

## The finding, stated at its actual strength

Scoring the covariance against *total* error mixed in a deterministic per-cell offset of
20–85 mm. Removing it first:

| camera | cells | ρ total | ρ conditional | cond obs/pred | median bias |
|---|---:|---:|---:|---:|---:|
| A | 5 | −0.80 | **+0.70** | 0.18 | 80 mm |
| B | 7 | +0.18 | **+0.75** | 0.66 | 20 mm |
| C | 19 | +0.10 | **+0.70** | 0.54 | 85 mm |
| D | 22 | −0.12 | **+0.53** | 0.47 | 30 mm |

Projection geometry **ranks** the conditional spread correctly on all four cameras and
**over-predicts its magnitude** by 1.5–5×, as expected when σ_pix is fitted against
bias-contaminated residuals.

**It does not establish that `JJᵀ` beats range.** Along the sampled routes
ρ(σ_max(J), range) = 0.976/0.969/0.999/0.996. The two models are near-identical by
construction here.

## Proposed `registry.yaml` edits

```yaml
  - id: EXP-PROJ-AMP
    primary_metric: >-
      ground-error amplification ratio (model-only), and held-out rank agreement between
      predicted and measured CONDITIONAL spread per spatial cell
    promotion_gate: >-
      Separate projection geometry from detector residuals, and separate per-cell bias from
      per-cell spread before scoring any covariance model.
    next_action: >-
      Preserve as locked supporting mechanism. Do NOT cite as evidence that JJ^T beats a
      range-only model: sigma_max(J) and range are collinear at rho >= 0.97 on the sampled
      routes. That comparison needs the 2-D footprint commissioning capture.
```

## Proposed `08_figures.md` F01 edits

Replace the claim-limit cell with:

> C1 only, and narrower than previously recorded. Structured, repeated projection residuals
> exist and model-only amplification varies 4.1× across a footprint. Once the per-cell bias
> (20–85 mm) is separated, geometry ranks the remaining spread correctly on all four cameras
> (ρ = +0.53 to +0.75) while over-predicting its magnitude by 1.5–5×. **No claim that the
> full Jacobian beats a range-only baseline** — they are collinear at ρ ≥ 0.97 on the sampled
> routes, and the two models tie. Decomposition into robot-geometry and camera-specific terms
> remains unresolved (E6).

Add to the generator cell:

> `experiments/projection_amplification/exp1_geometry_vs_detector.py` now emits `fig_g1`
> (model-only), `fig_g2` (per-detection raw-vs-corrected ablation) and
> `fig_g4_camera_{A,B,C,D}` (map + RMS-to-RMS, deployed-corrected only).

## Wording to avoid

Two phrasings I used and am retracting, recorded so they do not reappear:

- ~~"geometry explains essentially none of the spatial variation"~~ — an artefact of scoring
  a variance model against a bias.
- ~~"a null with a caveat"~~ — a null asserts a valid test found no effect. This experiment
  never varied geometry independently of range. The supportable statement is
  **insufficient design**, not a null.

## Related

- `PROPOSED_commissioning_capture.md` — the capture that would make the range-vs-`JJᵀ`
  comparison possible, and which also closes the E6 confound.
- RQ01 ("Why not use distance only?") is currently `OPEN`. This evidence is directly relevant
  and should be linked, but it **cannot answer it** on the present data.
