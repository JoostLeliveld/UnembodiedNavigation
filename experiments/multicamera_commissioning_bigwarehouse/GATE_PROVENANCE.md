# Gate parameter provenance — audit of 2026-07-16

Every commissioning gate value in `config/study.yaml` was traced to its origin
before the paper campaign. Summary: **all eight values are copied library
defaults, none is derived from data, and none originates in the
`warehouse_aws` method-development world.** They were carried unchanged from
the retired two-camera `warehouse_big` pilot. Two (`min_overlap_pairs: 30`,
`max_overlap_outlier_rate: 0.10`) are pre-registered in
`research_story/09_multicamera_handover_fusion/evidence.yaml`; the rest are
asserted. `config/paper_protocol.yaml` is now the frozen single source of
truth for the paper campaign — module defaults are implementation details, not
protocol.

| parameter | frozen value | origin | data derivation | known mismatch |
|---|---|---|---|---|
| `min_spatial_trust` | 0.45 | `CameraManagerConfig` default | none — asserted | pilot pair-trust peaked ≈0.41; release cliff at 0.40–0.425 (see sweep below) |
| `max_cross_camera_disagreement_m` | 0.30 | `CameraManagerConfig` / `overlap.py` default | none — asserted | pilot C↔D mean disagreement 0.247 m is a near-pure +y **bias**, leaving ~5 cm noise margin |
| `min_overlap_pairs` | 30 | ch.09 pre-registration | pre-registered, not derived | `overlap.py` module default is **1** — code paths not using the protocol config are effectively ungated |
| `max_overlap_outlier_rate` | 0.10 | ch.09 pre-registration | pre-registered, not derived | `overlap.py` module default is **0.05** (stricter) |
| `max_measurement_age_s` | 0.15 | `CameraManagerConfig` default | none — asserted | equals `age_decay_s` default; two distinct roles, one constant |
| `max_overlap_time_delta_s` | 0.05 | `CameraManagerConfig` default | none — asserted | — |
| `candidate_score_margin` | 0.08 | `CameraManagerConfig` default | none — asserted | unit tests exercise 0.05 |
| `required_consecutive_better_frames` | 3 | `CameraManagerConfig` default | none — asserted | — |

## Disclosure artifacts

- **Threshold sensitivity**: `tools/threshold_sensitivity_sweep.py` replays the
  pilot M8 pipeline across `min_spatial_trust` ∈ [0.05, 0.70]. Results in
  `logs/studies/multicamera_commissioning_bigwarehouse/threshold_sensitivity_v1/`.
  Headline: the pilot's "0 corrections released" holds for every threshold
  ≥ 0.425; the first release appears at 0.40; below 0.25 the policy releases
  freely (43–54 updates) and the NIS gate starts doing the rejection work in
  the 0.225–0.30 band. The sweep is a disclosure, **not** a tuning source.
- **Bias attribution (real GT, 2026-07-16)**: `tools/record_evaluation_truth.py`
  + `tools/attach_evaluation_truth.py` measured per-camera projection error
  against `/ground_truth_tf` on a live handover pass
  (`logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716/`).
  Each camera pulls the robot toward its own wall — a near-edge box-bottom
  projection bias, not one bad calibration: C −0.176 m (south), D +0.092 m
  (north), B +0.050 m; C+D sum 0.268 m reproduces the pilot's 0.247 m C↔D
  disagreement. Camera C dominates. (An earlier odom-referenced audit that
  implicated the north cameras was contaminated by wheel-odom drift and is
  superseded.)
- **Bias FIXED (2026-07-16)**: the pull is distance-dependent, so
  `tools/fit_projection_calibration.py` fits
  `correction = intercept + slope·ground_distance` per camera against
  simulation truth (two runs, 930+ samples; residual std 0.013–0.048 m);
  frozen constants live in
  `logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v2/`.
  Applied at record time (recorder `--projection-calibration`) and in the live
  manager node (`projection_calibration` param), C↔D disagreement drops
  **0.247 → 0.078–0.107 m**. Remaining known issue: camera C carries a
  ~0.11 m cross-bearing residual near the central pillar (occlusion-clipped
  boxes) — a perception-quality effect for the trust/association gates, not
  projection. See `gt_validation_smoke2_20260716/RESULTS.md`. The 4-detector
  GPU OOM is likewise fixed (`yolo_device_camera_a` defaults to `cpu`).

## Rules going forward

1. The frozen values stay frozen for the paper campaign; any change must be
   justified from `warehouse_aws`-era evidence and re-registered in the
   ch.09/10 manifests *before* the affected data is collected.
2. Quote gates from `config/paper_protocol.yaml` (or `study.yaml`), never from
   `CameraManagerConfig`/`overlap.py` defaults — the module defaults differ
   for `min_pair_count` (1 vs 30) and `max_outlier_rate` (0.05 vs 0.10).
3. `min_spatial_trust = 0.45` sits 0.025–0.05 above the pilot's release cliff;
   report the sensitivity curve alongside any release/no-release claim.

## Demo

The projection-bias fix (before/after, C↔D verdict vs the 0.30 m gate) is
rendered as a four-panel demo — see `DEMOS_D4_CAMERA_METHOD.md`.
