# 05 — Geometry helper repairs, 2026-09-07

Update: the separately assigned optional node integration is now implemented; see [covariance publication handoff](05_optional_covariance_publication_implemented.md). The sections below retain the earlier helper-only scope.

Pure geometry/validation changes are implemented locally. **Manager integration and artifact-provenance binding are not yet complete.** No commits, pushes, retraining, calibration updates or campaigns were performed. The existing silhouette condition-number repair was preserved.

## Changed files

- `src/reliability/reliability/observation_geometry.py` (new): validates explicit camera/calibration/image-frame identity; schema, capture stamp, full SPD covariance; original image dimensions when provided; positive half-open bbox extents and consistency between box, declared bottom and selected pixel. Misses need identity/covariance but no pixel/box. Invalid inputs raise ValueError; nothing is clipped, normalized or silently repaired.
- `src/unav_common/unav_common/camera_model.py`: validates finite camera vectors, nondegenerate look-at/up, positive integer dimensions, valid radian FOV and invertible floor homography; rejects nonfinite rays and floor intersections behind the camera. Valid ground homography arithmetic is unchanged. Near-horizon geometry retains the existing numerical denominator bound; no empirical range threshold was introduced.
- `src/reliability/reliability/projection.py`: rejects ambiguous camera includes, relative/degree/non-Euler poses, nonfinite poses, nonzero roll and nonpositive camera height for this fixed-intrinsics loader. Only direct world includes are supported. Validates incoming covariance before propagation; an unrepresentable singular numerical floor explicitly fails, and projection produces no observation for that unsupported covariance geometry.
- `src/state/state/core/pixel_to_bev.py`: provides full world covariance, including XY cross terms; compatibility standard deviations are correct world-axis marginals. Pixel derivatives use one nominal homography. Six camera/look-at uncertainty derivatives contribute once without independent random draws during differentiation. At-height projection delegates to the validated camera helper.
- `tests/reliability/test_observation_geometry_05.py` (new): mutation, geometry, covariance and frozen-projection regressions.

The previous optional node wiring in `pixel_to_bev_state_node.py` still calls the marginal-standard-deviation compatibility API. It does **not** yet publish the new full XY cross term. That node was not edited under this pure-helper ownership assignment. The full API below is ready for its owner; any configured affine mean correction also needs the corresponding covariance congruence at that integration boundary.

## Validator API

```python
from reliability.observation_geometry import validate_observation_geometry

validate_observation_geometry(
    observation, camera,
    expected_camera_id="camera_B",
    expected_calibration_id="warehouse_v2_camera_B",
    expected_image_frame_id="camera_B",
    require_bbox=True,
    require_image_dimensions=False,
)
```

Returns None on success; raises ValueError on incompatible semantics. Expected identities must come from configuration, not the incoming observation. Call before projection and bootstrap. It does not apply freshness, quorum or statistical gates. `require_bbox=True` requires bbox-bottom selection; mask-bottom is supported only with `require_bbox=False`, explicit mask availability and matching mask pixel.

Dimension names supported are `image_width`/`image_height` or `image_width_px`/`image_height_px`. Supplied values must match the configured camera exactly. `require_image_dimensions=True` also rejects absence; enable it for a producer carrying original image size. Default False preserves the ability to check legacy recorded contracts, but **cannot establish original image size from an absent field**. Existing `contracts.py` was not edited. Camera manager belongs to 07; original-size contract/producer integration requires 04/07 agreement.

```python
R_xy = transformer.pixel_covariance_to_metric(
    u, v, full_R_uv, transform_noise_sigma=0.0,
)
```

Returns a full 2×2 covariance or None for unsupported geometry. Bad covariance/noise parameters raise ValueError. The optional transform-noise interpretation is independent uncertainty in three camera-position and three look-at coordinates, expressed in metres; it is not a fitted residual R. Publish its entries in the XY block, applying any mean affine transform to R too. Do not use this helper to replace active frozen reference covariance.

## Verification

[Initial reproduction](05_repair_evidence/before.txt): 24 failed, 3 passed before implementation. New validator/API failures and existing numerical/geometry invariant failures are retained separately from later pass evidence.

[Final focused suite](05_repair_evidence/tests.txt): **108 passed**, two nonfatal optional pandas dependency-version warnings, no skipped frozen-artifact tests. This includes exact equality of all 6,412 frozen raw projections and existing reference-calibration, learned-NN, hull, camera-manager and elevated-keypoint regressions. These are implementation checks, not camera-accuracy estimates or a complete-system acceptance run.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q \
  tests/reliability/test_observation_geometry_05.py \
  tests/state/test_pixel_to_bev_at_z.py \
  tests/reliability/test_silhouette_observation.py \
  tests/reliability/test_reference_calibration.py \
  tests/test_learned_box_correction.py \
  tests/reliability/test_camera_manager_node.py
```

Python compilation and diff whitespace checks also pass. Artifact hashes and edited-file hashes are retained in [identity.json](05_repair_evidence/identity.json).

## Pending integration and communication

07 confirmed ownership of manager integration. 12 offered ownership of loader/provenance work; this thread has not edited `learned_box_correction.py` or `reference_calibration.py`, avoiding overlap. Their existing artifacts remain unchanged. Full model-to-projection binding, strict original-size production, manager semantic rejection and optional covariance publication remain explicit integration work. The original report's unsupported compiled-backend box clipping is owned by detector work; no detector code was edited here.

Automatic approval review rejected messages to the 07 and 12 owner threads, citing non-public implementation details and unverified destination authorization, including after coordinator/thread checks. No alternate communication route was used. The user must approve those cross-thread handoffs before retrying them. This document records the implementation for review; it is not evidence that the integration owners received a handoff.
