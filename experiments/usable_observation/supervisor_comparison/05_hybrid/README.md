# Hybrid geometry plus operational updates

## What this method asks

Can sensed geometry provide the discontinuous structure of occlusion shadows while a learned
residual corrects detector- and deployment-specific errors? The hybrid is valuable only if
it improves on both the depth-only and GP-only arms under the same data budget.

A clean candidate form is:

`logit(p_use) = logit(p_depth) + residual_GP`,

followed by calibration, epistemic conservatism and the same fallback contract as the other
methods. The exact equation remains a design decision and must be frozen before comparison.

## Begin state

At cold start, the field is the depth/raycast prior, including sharp shadows and unknown-cell
handling. The residual GP begins at zero with high epistemic uncertainty. The begin-state
figure should therefore have three aligned maps: geometric prior, zero/uncertain residual,
and combined reliability.

## Map used in planning

The planner consumes only the calibrated combined `p_use` map. For explanation, always show
the components beside it:

1. depth-derived prior;
2. learned residual and support;
3. final conservative hybrid field.

This decomposition prevents a visually attractive combined map from hiding whether geometry
or observations caused the change.

## Updates

Detector opportunities update the residual model, not the sensed height map. A rescan updates
the geometric prior and must define what happens to the old residual:

- retain only where the coordinate/layout correspondence remains valid;
- reset in changed regions; or
- decay by map age.

The supervisor update panel should show two independent clocks: geometry-map age and
operational-sample support.

## Expected plans

- R1: should avoid the shadow from the first scan, before driving through it.
- R2: should preserve sharp north/south occlusion differences while adapting systematic
  over- or under-prediction.
- R3: can correct camera-specific handover reliability beyond line-of-sight geometry.
- R6: the residual should remain near zero and must not create a spurious detour.

## Failure mode to make visible

If the geometric prior is confidently stale, the residual may need many misses to overturn
it. Show the chosen prior strength and reset/decay policy. Without that panel, “hybrid” is not
a reproducible method.
