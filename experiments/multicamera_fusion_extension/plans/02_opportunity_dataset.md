# Plan 02 — Expected-observation-opportunity dataset + LOO labels

## Purpose
Training data for the reliability models WITHOUT operational ground truth (§6).
A miss only counts when the camera had an opportunity; usability labels must
not be circular (same camera producing measurement and label).

## Opportunity definition (§6.2)
Camera i has an opportunity at t iff:
1. belief mean projects inside the validated image region;
2. enough of the belief ellipse projects inside (threshold pre-registered);
3. predicted robot scale ≥ minimum;
4. stream live & not stale;
5. timestamp inside association window.

Labels:
- `A` (availability): detection received & associated | opportunity.
- `G` (usability): |LOO residual| ≤ δ & association valid | A=1.

## Label sources (in priority order, §6.3)
1. **Leave-one-camera-out reference**: reference state from odom + all cameras
   except i → `e_LOO = z_i − h_i(x̂^(−i))`. Runs offline over replay exports.
2. Short-horizon odometry reference in single-camera regions (store odom
   covariance; down-weight as it grows).
3. Small manually annotated subset — bias check on LOO labels only.
GT (`gt_*`) never labels training data; it may score the labels in an
evaluation-only audit (firewall-tested).

## What exists / reuse
- `reliability.export` split exports (`operational/` vs `evaluation_only/`) — the firewall pattern.
- `reliability.projection` + calibration JSONs for predicted (u,v)/scale.
- `reliability.fusion.sequential_kalman_update_2d` for the (−i) reference filter.
- Belief-stamped event precedent: `build_belief_gp_events.py` (ch.01).

## New code
- `experiments/multicamera_fusion_extension/tools/build_opportunity_dataset.py`
  — reads a multicamera export, writes one CSV per camera with the §6.2 schema
  (camera_id, stamp, belief_xy, belief_cov_xy, predicted_uv, inside_valid_region,
  stream_healthy, detection_received, association_valid, raw_confidence,
  bbox_geometry, measured_uv, innovation, innovation_cov, label_source).
- `tools/build_loo_labels.py` — runs the (−i) reference filter per camera and
  appends `e_loo_uv`, `e_loo_xy`, `usable` (G) columns + reference-uncertainty column.
- Small additions to `contracts.CameraObservation` if fields are missing
  (`calibration_hash`, `detector_hash` provenance).

## Gates
- Unit tests with synthetic geometry: opportunity logic truth table; a robot
  genuinely outside support yields NO opportunity rows (not failures).
- Leakage: extend `test_leakage_firewall.py` — builder must hard-fail if fed
  `evaluation_only/` records.
- Data audit page per capture: opportunities/camera, A-rate, G-rate, LOO
  reference uncertainty distribution, label_source counts.
- Blocked for real data on commissioning M1/M2 (detector retrain + projection v3).
