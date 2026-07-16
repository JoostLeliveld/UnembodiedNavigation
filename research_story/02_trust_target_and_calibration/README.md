# 02 — What should "camera trust" actually be?

**Question.** Which operational signal should the spatial GP learn — raw confidence,
calibrated confidence, hit/miss availability, NIS acceptance, innovation magnitude, or a
combination? And does that signal predict what actually matters: localisation quality?

**Status: PARTIAL.** World: original warehouse (4-cam only as a later transfer check).
Supporting investigation inside Contribution 1 — its output is a **frozen target choice**.

## What a contribution-grade answer looks like

Not a new metric — a defensible decision with evidence: "the GP learns X because X is the
best held-out predictor of usable, accurate camera measurements; here is the table." The
expected (but unproven) headline that would *justify chapter 04*:

> Confidence predicts availability, not localisation accuracy — so availability and
> conditional error deserve separate models.

## The results we're aiming for

- **Fig 02A** — confidence vs projected bottom-centre BEV error, stratified by distance /
  image-edge / occlusion / pose uncertainty. This must be the **confound-controlled redo**
  of exp0 (see below). Decision input #1.
- **Fig 02B** — reliability diagram: predicted confidence vs empirical usable-detection rate.
- **Fig 02C** — spatial miss-rate map P(usable detection | s).
- **Fig 02D** — the decision table: per candidate target → held-out NLL, Brier, correlation
  with BEV error, ability to predict NIS rejection. **The row that wins gets frozen.**
- **Fig 02E / V02** — failure panel + reel (high-conf wrong, low-conf right, miss-in-FOV).

Evaluation-only labels (never deployment inputs): projected BEV error, camera localisation
error, estimation improvement. The detector is judged through its projected bottom-centre
point, not mAP.

## Implemented now

| Item | Tag | Note |
|---|---|---|
| exp0 confidence audit, 4 figures | measured_in_sim | **Simpson-confounded** — a marginal confidence-error trend was explained by distance/gate strata; motivation only until redone stratified |
| Detection-rate switch precedent (raw conf → detection RATE, calibration-invariance; field corr 0.84 with v7b) | established | this history is a disclosure owed in the writing, and detection rate is the incumbent target |
| Clean detector v1 (periphery flat 0.027 m) + `warehouse_visibility_{capture,targets}_v1` | established | the labelled dataset for 02A–02D |
| `scripts/shared/metrics.py` | established | the only allowed scorer |

## Gap → next experiment

One study (`experiments/trust_target_comparison/`): build the target-candidate table on
route-disjoint splits of the existing capture + campaign logs; produce 02A–02E. No new
simulation required.

## Gate

Fig 02D decides: **scalar thesis** (one trust field, ch.04 stays closed) vs **factorised
thesis** (ch.04 opens as candidate second contribution). Do not presume the factorised
outcome before 02A/02D exist.
