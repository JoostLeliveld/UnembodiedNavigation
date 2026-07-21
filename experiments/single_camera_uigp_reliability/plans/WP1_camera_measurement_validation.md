# P1-WP1 — camera measurement pipeline validation

**Claim.** The selected image point + projection model produce a measurable
warehouse-position observation with documented error, bias and failure regions —
*before* any reliability learning depends on it.

**Serves:** `research_story/02_trust_target_and_calibration` (the measurement
contract). Consolidates previously-scattered validation (clean detector retrain
periphery 0.027 m; projection/affine fix) into one frozen ch.02 deliverable.

## Assumptions / non-assumptions
- Assumes: one robot, one external camera, `ObliqueCameraModel` calibration.
- Non-assumes: no GT in the pipeline; GT scores the residual map only (eval-only).

## Tasks
1. **Detection association** — class filter, min confidence, nearest-predicted-box
   / association gate, duplicate + no-detection + stale handling. Log the reason:
   `accepted | no_box | ambiguous | outside_gate | stale | invalid_projection`.
2. **Pixel-source comparison** — box centre vs **bottom centre** vs (optional)
   mask contact point vs keypoint. Per source, eval-only image + ground-plane error.
   Justify the chosen point.
3. **Projection validation** — `z_xy = g(z_uv; θ)` via `ObliqueCameraModel`: frame
   convention, axis orientation, units, homography domain, distortion, affine
   correction, singularities, extrapolation beyond calibrated support (reject it).
   Round-trip check (project → unproject).
4. **Timestamp audit** — image-acquisition, detector-output, state, fusion-update
   times; end-to-end latency; Δt_camera / Δt_detector / Δt_transport distributions.
5. **Spatial error map** (GT eval-only): RMSE, MAE, median, p90/p95/p99, max, x/y
   bias, covariance; error by range / image position / heading / near image edge.
6. **Failure gallery** — manually reviewed: far, edge, partially-occluded,
   low-confidence-correct, high-confidence-poor-contact, misses, false positives,
   association failures.

## Deliverables
`camera_observation_contract.yaml`, `projection_validation.csv`,
`pixel_source_comparison.pdf`, `latency_report.md`, spatial residual maps, failure
gallery, frozen calibration model.
Outputs → `logs/studies/single_camera_uigp_reliability/wp1_measurement_validation/RESULTS.md`.

## Gate G1
Bottom-centre (or replacement) point justified; p95 projection error known; bias
corrected or explicitly modelled; timestamp age logged and below the association
tolerance; unsupported projection regions rejected; failure frames documented.

## Reuse (no new runtime code)
`unav_common.camera_model.ObliqueCameraModel`; existing projection/affine +
clean-detector artifacts; `scripts/shared/metrics.py`. Data source = the WP2
commissioning drive (`DATA_SOURCE_commissioning_drive.md`) — same log carries the
opportunity + association fields WP1 audits and WP2/WP3 learn from.
