# Confidence critique response (§10) — deliverable D

**Evidence status:** EVALUATION ONLY. This analysis deliberately uses the Gazebo ground-truth
localization residual as the *target* confidence is tested against (sanctioned by §10B
priority-3). The residual is **never** an observability feature or label; it lives entirely
outside the firewalled observability dataset. Confidence, bbox, and image position are
operational (PIXEL); range is operational (belief). **Date:** 2026-07-24 · **Git:** `9cf9664`.

Code: `scripts/reliability/analyze_confidence.py` + pure stats
`reliability.confidence_analysis_stats`. Data: 7885 real honest_campaign_v1 detections.
Reproduce:

```
python3 scripts/reliability/analyze_confidence.py \
    --campaign logs/visibility_comparison/honest_campaign_v1 \
    --output   logs/studies/usable_observation/confidence_v1
```

This document reports what the experiments support. The conclusion below was **computed**, not
pre-written.

## The question

The prior work was criticised for an unsupported assumption linking YOLO confidence to
measurement noise ("high confidence proves low localization error"; "the GP learns R"). §10
tests that assumption directly and separately from the planning model.

## 10A — confidence as an indicator of operational quality

On this single-camera corpus the binary `quality_label` is saturated (0.997 among detections;
the only quality failures are the 23 confidence-gate rejections, which are tautological w.r.t.
confidence). There is therefore **no independent binary operational-quality variation** for
confidence to predict here. The meaningful, non-tautological quality signal is the continuous
localization residual, analysed in 10B. (When p_qual varies — the multicam world — 10A becomes
answerable on its own terms.)

## 10B — confidence vs conditional geometric error (the substantive test)

All correlations use the GT-calibrated residual; 95% CIs bootstrap whole **runs**.

| quantity | value |
|---|---|
| confidence spread among detections (p05–p95) | 0.787 – 0.921 (IQR 0.064) — **nearly constant** |
| Spearman(confidence, residual) raw | **+0.296** [95% CI 0.241, 0.356] |
| Spearman(confidence, residual) calibrated | **+0.609** [95% CI 0.542, 0.662] |
| partial Spearman(confidence, residual \| range, bbox, u, v) | **+0.589** |
| Spearman(confidence, range) | −0.597 (closer ⇒ more confident) |
| Spearman(range, residual) | −0.332 (closer ⇒ larger error) |
| held-out (leave-one-route-out) large-error AUPRC, geometry only | 0.511 |
| held-out large-error AUPRC, geometry **+ confidence** | 0.511 (Δ ≈ +0.0005) |

Three findings, all pointing the same way:

1. **Confidence carries little information.** Among real detections it is nearly constant
   (IQR 0.064); there is almost nothing to map to a covariance.
2. **The sign is reversed, and the shape is non-monotonic.** Confidence correlates
   *positively* with localization error (raw +0.30, calibrated +0.61; CIs exclude 0). The
   decile plot (`residual_vs_confidence_decile.png`) is U-shaped: error is lowest at *mid*
   confidence (~0.8, ~0.04 m) and rises at both extremes — among the bulk of detections
   (0.8–0.92) error *increases* with confidence to ~0.28 m. The detections the naive assumption
   would trust most are **not** the most accurate.
3. **It is not merely a geometry proxy, yet adds no deployable value.** The positive association
   survives controlling for range/bbox/image-position (partial Spearman +0.59, barely below the
   raw +0.61). But out-of-route it adds essentially nothing for flagging large errors beyond
   geometry (AUPRC 0.511 → 0.511). Part of the raw link is the geometry confound (closer ⇒ both
   more confident and higher error), and the residual within-route signal does not transfer
   across routes.

## Conclusion (computed)

**C-reversed.** YOLO confidence is *positively* associated with localization error (partial
Spearman +0.59 after geometry controls), so mapping confidence to inverse measurement covariance
would be **backwards**; and it adds **no** out-of-route predictive value for conditional error
beyond geometry. Both the sign and the null-added-value independently forbid treating confidence
as measurement covariance.

This is the empirical refutation of the criticised assumption, on real data. It directly
supports the P4 design decision: the planner input is **p_use** (a usable-observation
probability), never raw or expected confidence, and never a confidence→covariance map. The main
planning behaviour is valid under this outcome (the task requires validity under A/C/D).

## What this does and does not claim

- **Does** claim: on this corpus, confidence neither indicates low localization error nor adds
  predictive value beyond geometry; its usable role is as the detection signal (p_det), not as a
  reliability/covariance surrogate.
- **Does not** claim a universal law about YOLO confidence; a different detector, calibration, or
  camera could differ. It also does not use this residual as any observability label (firewall).

## Answers to final-report questions

- **Q2 (confidence predictive of operational quality?)** — Not assessable as a binary here
  (quality saturated); against the continuous residual, no — see Q3.
- **Q3 (confidence predictive of conditional geometric error?)** — Yes in association but with the
  **wrong sign** (higher confidence ⇄ higher error), and with **no** out-of-route predictive value
  beyond geometry. Confidence must not be used as inverse covariance.

## Artifacts

`confidence_analysis.json`, `confidence_hist.png`, `residual_vs_confidence_decile.png`,
`residual_vs_range_by_confidence.png`.
