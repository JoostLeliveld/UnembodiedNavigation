# Frozen analysis contract

Frozen on 2026-08-18 before examining the new B+C fused camera configuration.

## Roles and split

- Development configuration: cameras A+B.  Earlier work already exposed this
  pairing, so it cannot provide a confirmatory result.
- Configuration-level holdout: cameras B+C.  This pair was absent from the E3
  subset sweep (`A+B` and `A+C` were the two-camera pairs).
- Tasks: the four already declared warehouse tasks; no task is selected after
  observing an outcome.
- Availability: frozen depth-prior-plus-GP per-camera fields.  No CAD map enters
  the operational planner.
- Conditional measurement model: current `PG-IPM-CURRENT` detections only.
  Spatial folds group all yaw replicates at a commanded `(x,y)` location.
- Ground truth: commissioning and evaluation only.

The B+C result is a camera-configuration holdout, not a new-data holdout: its
individual camera fields and detections existed during commissioning.  Claims
must use exactly that wording.

## Conditional-covariance model

The detector observation is bounding-box bottom centre in pixels.  Both models
fit a per-camera constant pixel bias on training data.

- Baseline: one pooled constant 2-D residual covariance.
- Geometry candidate: pooled `var_u=a_u+b_u r^2` and
  `var_v=a_v+b_v r^2`, with non-negative coefficients and one pooled correlation.

Six leave-one-spatial-block-out folds are used.  The geometry model is accepted
only when its pooled out-of-fold Gaussian NLL improves by at least 0.001 nats,
its absolute 95% ellipse-coverage error is no more than 0.01 worse, and its fitted
variance changes by at least 1% over the commissioned range.  The last condition
prevents the nested model from earning a spatial claim by collapsing exactly to
the constant baseline.  Failure selects the constant model and forbids a claim
of spatial `R_cond`.  This non-degeneracy correction was added after the first
fit exposed zero slopes, and before inspecting the B+C configuration.

## DS-Route decision rule

1. Use one common, deduplicated route library for every arm.
2. Keep routes no longer than 1.05 times the shortest candidate.
3. Minimize the exact expected longest missed-update run.
4. Break numerical ties by expected conditional information
   `sum_c p_use / trace(R_cond,ground)`; then by length and a stable route ID.

The 5% budget and tie order are fixed here and are not tunable weights.

## Fail-closed gates

Development A+B passes only if all conditions hold across the four tasks:

- the selected route changes by at least 0.25 m maximum pointwise separation on
  at least two tasks;
- median expected-longest-miss reduction is at least 15%;
- every selected route respects the 5% length budget;
- fused `p_use` has at least 0.10 dynamic range on driveable cells.

The B+C holdout passes only if:

- every route respects the length budget;
- median expected-longest-miss reduction versus shortest is positive;
- median regret under the CAD evaluation reference is no worse than the shortest
  route; and
- tail calibration is reported (never silently omitted), with at least 20 nearby
  empirical pose events per task for route-local summaries.

Closed-loop execution is authorized only if the covariance gate, development
gate, and B+C gate all pass.  If any gate fails, generate the audit and stop.
