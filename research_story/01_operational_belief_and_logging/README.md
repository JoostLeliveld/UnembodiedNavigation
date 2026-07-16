# 01 — Do ordinary drives produce usable uncertain-input training records?

**Question.** When the robot just drives its tasks, do we get trust-training records whose
position uncertainty is (a) real, (b) quantified honestly by the filter, and (c) large
enough that ignoring it should hurt?

**Status: PARTIAL.** World: original warehouse. This chapter is infrastructure + a go/no-go
measurement for Contribution 1 — not a contribution itself.

## What a "yes" looks like

A per-camera-frame record, with the **prior** belief captured before that frame corrects
the pose:

`(t_k, μ_k⁻, P_k⁻, opportunity_k, hit/miss m_k, confidence c_k, u_k, v_k, Δt_k)`

```text
odometry prior → record μ⁻,P⁻ → detector outcome → save trust observation → optional camera update
```

and evidence that P⁻ means something: covariance grows during odometry-only stretches,
shrinks on accepted corrections, and tracks the (evaluation-only) actual error.

## The results we're aiming for

- **Fig 01A** — a commissioning route with covariance ellipses, hits/misses, and
  camera-unobserved sections: visual proof the uncertain-input problem *occurs*.
- **Fig 01B** — aligned traces: time-since-camera-update, tr(P_xy), actual error. Aim:
  visibly correlated growth.
- **Fig 01C** — 50/90/95% ellipse coverage vs nominal, after the chosen smoothing/inflation.
  Aim: coverage near nominal (raw filter is known-overconfident, see below).
- **V01** — data-collection video (ellipse expands/contracts live, samples appear).

## Implemented now

| Item | Tag | Note |
|---|---|---|
| Belief-stamped GP events (`build_belief_gp_events.py` → `logs/visibility_comparison/belief_gp_events`) | measured_in_sim | pairing (prior vs posterior belief) **unaudited** — must verify before ch.03 consumes it |
| NEES honesty study (exp5): raw filter NEES 16.8 → 2.8 after smoothing | measured_in_sim | raw covariance is overconfident; the inflate-vs-replace rule from Option-A applies |
| Belief/uncertainty docs (`docs/uncertainty_propagation.md`, `runtime_dataflow.md`) | established | |
| Campaign CSVs with belief + GT columns | established | ALWAYS load via `campaign_metrics.load_run` (six overlapping position fields) |

## Gap → next experiment

1. **Audit** `belief_gp_events` pairing; if posterior, build the prior-belief logger as a new
   study `experiments/belief_prior_logger/`.
2. Choose and freeze the covariance honesty treatment (smoothing window / inflation) using
   exp5's method; then produce Figs 01A–01C from `honest_campaign_v1` runs.

## Gate (GO/NO-GO for chapter 03)

> Do not start uncertain-input fitting until covariance is at least directionally related to
> actual error. Poor covariance calibration is a stop condition.

## Caveats

GT appears only as an evaluation overlay. Wheel-odom-as-truth is a solved historical trap
(GT bridge since 2026-07-01) — never reintroduce.
