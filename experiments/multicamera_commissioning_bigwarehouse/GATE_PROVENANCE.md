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
- **Bias attribution**: `tools/attach_evaluation_truth.py` +
  `tools/record_evaluation_truth.py` attach simulation truth to future runs so
  cross-camera disagreement can be attributed to a specific calibration.
  A preliminary odom-referenced audit of pilot run 01 shows the +y projection
  bias concentrated on the north-wall cameras (B ≈ +0.33 m, D ≈ +0.27 m vs
  A ≈ −0.11 m, C ≈ −0.06 m), implicating camera D in the C↔D 0.247 m offset —
  to be confirmed against real ground truth before any correction.

## Rules going forward

1. The frozen values stay frozen for the paper campaign; any change must be
   justified from `warehouse_aws`-era evidence and re-registered in the
   ch.09/10 manifests *before* the affected data is collected.
2. Quote gates from `config/paper_protocol.yaml` (or `study.yaml`), never from
   `CameraManagerConfig`/`overlap.py` defaults — the module defaults differ
   for `min_pair_count` (1 vs 30) and `max_outlier_rate` (0.05 vs 0.10).
3. `min_spatial_trust = 0.45` sits 0.025–0.05 above the pilot's release cliff;
   report the sensitivity curve alongside any release/no-release claim.
