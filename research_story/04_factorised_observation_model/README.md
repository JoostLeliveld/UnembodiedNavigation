# 04 — Factorised observation model (candidate second contribution A)

**Question.** Are detection availability and localisation accuracy *given* detection
different spatial phenomena — and does modelling them separately produce a better-calibrated
filter than any scalar trust value?

```text
p_det(s)  = P(usable detection | s)
R_cond(s) = Cov(e_camera | usable detection, s)
R_plan(s) = g(p_det(s), R_cond(s))
```

**Status: PLANNED — opens only if ch.02's Fig 02D shows the scalar collapse loses
information.** World: original warehouse for the proof; `warehouse_full_4cam` later to check
whether the four cameras have *different* availability/noise structures (they should — the
layout gives each camera different occluders and ranges).

## What the contribution looks like

> *A factorised external-camera observation model separating measurement availability from
> conditional localisation noise.*

**Binding promotion rule (all four required):** both components learned from operational
data; both independently validated; the factorisation beats the best scalar trust on filter
calibration; and it changes R_plan or navigation measurably. Anything less → this folds back
into ch.02 as a target ablation.

## The results we're aiming for

- **Fig 04A ("two maps, not one")** — availability map next to conditional-error map with
  visibly different spatial structure. Likely the strongest single figure of the option.
- **Fig 04B** — the four-region taxonomy with concrete examples: frequent+accurate,
  frequent+inaccurate, rare+accurate, rare+inaccurate. If region 2 and 3 are non-empty, the
  scalar collapse is provably lossy.
- **Fig 04C (decision figure)** — NIS histograms vs χ² for constant-R / confidence-mapped /
  spatial-conditional / factorised. **Aim: factorised closest to nominal χ² coverage.**
- **Fig 04D** — ablation table (availability NLL, innovation NLL, NIS coverage, navigation).
- **V04** — two detections, same confidence, different measurement consequence.

## Implemented now

Nothing — by design. Reusable pieces when it opens: ch.03's frozen GP machinery fits both
fields; ch.02's evaluation-only BEV-error labels provide R_cond training data; the N5
condition slot in ch.06 is reserved for the navigation test.

## Gate

Opens on ch.02's decision; promotes on the four-part rule above; N5 exists in ch.06 only if
promoted.
