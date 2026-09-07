# 05 — Optional pixel-to-BEV covariance publication

Implemented 2026-09-07 after coordinator assigned `pixel_to_bev_state_node.py` to 05. This closes the optional full-XY-covariance publication gap recorded in the earlier helper handoff. Manager integration remains owned by 07 and artifact-loader binding by 12; their files were not edited here.

## Change

`src/state/state/nodes/pixel_to_bev_state_node.py::_pixel_callback` now consumes `pixel_covariance_to_metric` and writes the full XY block to ROS covariance indices 0, 1, 6 and 7. The existing affine XY mean map's linear part A is applied to covariance as `A R Aᵀ`. Translation does not change covariance. The previous constant Y offset retains the same covariance.

The callback maps transform-only uncertainty even with zero pixel noise. Missing, nonfinite, asymmetric or indefinite helper covariance returns before publication and before updating held position/heading. The declared zero-noise case remains semidefinite rather than receiving an invented positive variance floor. No trust-blend endpoints, Q/R configuration, detector/model settings, camera geometry or heading policy were tuned.

Preexisting dirty affine-parser changes were retained. This optional single-camera path does not replace the active manager's frozen metric reference covariance. The code uses the helper's first-order covariance about nominal configured geometry; it does not claim empirical residual calibration or account for all heading/position dependence.

## Verification

Dedicated `tests/state/test_pixel_covariance_publication_05.py` executes the actual callback body with ROS message/publisher substitutes and the real transformer. Independent finite differences establish the expected world covariance; rotated and sheared affine matrices establish congruence direction and cross terms. Cases also verify stamp/frame retention, unchanged yaw variance, invalid-covariance refusal, transform-only uncertainty and the unchanged precision-blend formula.

- Before repair: **8 failed**; [retained output](05_repair_evidence/node_before.txt).
- After repair, including geometry/elevated-plane tests and a new precision-blend regression: **53 passed**; [retained output](05_repair_evidence/node_tests.txt).
- Source whitespace verification passed. No ROS simulator campaign was launched.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q \
  tests/state/test_pixel_covariance_publication_05.py \
  tests/reliability/test_observation_geometry_05.py \
  tests/state/test_pixel_to_bev_at_z.py
```

The edited node and dedicated test hashes are recorded in [node_identity.json](05_repair_evidence/node_identity.json). No commit or push was made; other threads' dirty work remains intact.
