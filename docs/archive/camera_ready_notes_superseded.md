# Superseded Camera-Ready Notes

This historical note is retained for traceability only. Detector training and
dataset capture should use `scripts/perception/YOLO_DATASET_PIPELINE.md`.

# Camera-ready / rebuttal notes — improvements after submission

These are post-submission improvements to disclose in the camera-ready or arXiv
version. The submitted PDF's numbers are frozen; these change the *method
description* and the *result numbers* (mechanical drop-in once the v3 campaign
finishes).

## 1. Detector retrain (current-camera, in-distribution)

**What the submitted paper has:** the YOLO detector (`aws_yolo_simseg_v2`,
trained 2026-05-13) was trained before the external camera was moved
(z≈4.5→4.8). At the 2026-06-12 runtime it ran out-of-distribution and produced
confidently *mislocated* boxes at the grazing periphery (~0.5–0.7 m off).

**Fix:** retrained on the current camera via an analytic-label pipeline
(`aws_yolo_simseg_v3`): label each captured RGB frame by projecting the robot's
true pose through the runtime camera model (proper 3D pinhole `K·R·(p−c)`, since
`world_to_pixel` uses a z=0 homography that ignores elevation). Result: Box
mAP50 0.62→0.888; periphery localization 0.177→0.037 m; pixel bottom-centre
error median 8.7 px with **no periphery blow-up** (periphery median 9.0 px,
p90 17.3 px). Validated case-level fixes: b5/C2 stuck→goal, b2/C2 collision→safe.

## 2. Reliability signal: raw confidence → accurate-detection rate

**What the submitted paper has:** the GP reliability field is fit on the raw
detector confidence (`yolo_score_raw`).

**Why it needed changing:** the v3 detector is *more reliable but
lower-confidence-calibrated* (analytic rectangular-polygon labels). Near the
camera it detects 98% of the time and localizes to 8.7 px, yet its median raw
confidence is only ~0.30 (vs ~0.83 for the old detector). The runtime blends the
camera precision **linearly** in the reliability `p`
(`prec = p·visible_prec + (1−p)·miss_prec`), so a confidence-scale offset would
wrongly make a clearly-visible aisle look ~73% "miss".

Key identity: both detectors' score fields ≈ `detection_rate × (calibration
constant)` — old ≈0.85, v3 ≈0.30. Raw confidence conflates calibration with
reliability.

**Fix (principled, calibration-invariant):** fit the GP on the **empirical
accurate-detection rate** — the fraction of headings at each `(x, y)` where the
detector returns a box within τ = 30 px of ground truth. This removes the
detector-specific calibration constant and additionally counts gross
mislocations (the v2 failure mode) as misses. The new field
(`paper_artifacts/gp/aws_gp_v3`) is spatially correlated **0.84** with the
submitted field (same physical structure — high in open aisles / camera-facing
band, low behind racks and in the deep north), with near-camera reliability
restored to 0.987.

Suggested one-line method-text edit: *"the reliability field is the GP posterior
mean of the detector's empirical accurate-detection rate (a detection within τ of
ground truth) over the workspace"* — a strict generalization of the
score-as-soft-detection proxy used before, robust to detector confidence
calibration.

## 3. Results

To be dropped in from the v3 campaign (`aws_f31b1_v3_campaign`, 4 tasks × C1/C2 ×
5 seeds): aggregate outcomes, collision counts, belief error. Pending campaign
completion.
