# Next capture protocol: identify bias scale before fitting `R`

## Decision this capture serves

The current field campaign has one image at each `(camera, position, heading)`. It maps an
observed residual field, but it cannot identify a same-state conditional mean and covariance.
The next capture must first measure the spatial scale of the partial-view mean, then collect
independent operational repetitions at the count justified by that pilot.

Do not fit a GP, NIW model, or cross-camera covariance before the pilot is frozen and read.

## Units of evidence

- `state_id`: exact `(x, y, theta)` reference state.
- `place_id`: one `(x, y)` with several headings.
- `trajectory_id`: one independently executed pass; frames inside it are correlated.
- `source_batch_id`: one near-simultaneous attempt by all five cameras.
- `repetition_id`: an independent observation of the same declared state or trajectory.

Camera images or detector frames are not independent replicates. Aggregate within a
trajectory before uncertainty intervals across trajectories.

## Phase P0: transport and determinism gate

Run a small all-camera capture through the final capture path and require:

- one row for every camera opportunity, including misses;
- maximum within-batch image span below 0.05 s;
- exact image, detector, calibration, world, code, and pose-plan hashes;
- no unaccounted capture batch;
- an explicit detector clipping flag (not a tuned pixel-distance rule).

At several fixed states, take a few unperturbed repeats only to test determinism. If the image
and box hashes repeat, record `R_repeat = 0` for that simulator condition and stop repeating
static renders. Synthetic RGB noise remains a sensitivity ablation and may not set deployed
`R`.

## Phase P1: dense spatial-scale pilot

Predeclare shelf-edge transects from warehouse geometry, without selecting them by residual
magnitude. For each camera include near/mid/far range, oblique/frontal view, a clear-to-partial
transition, and at least one multi-camera overlap region.

Along every transect, sample at 0.15 m spacing over at least 1.2 m and capture all eight fixed
headings with all five cameras. The 0.15 m increment is a pilot resolution, not the final
commissioning spacing: it creates comparisons at 0.15, 0.30, 0.45, and 0.60 m, all below the
current 0.64 m median grid spacing. Collision-invalid points must be rejected before capture,
not after looking at detector results.

For each camera and view regime, estimate a directional variogram/correlation curve using
different positions only. Do not include different headings at the identical position in the
shortest-distance bin. Bootstrap whole transects. Report the first distance at which
correlation falls below `exp(-1)` and the distance at which its interval includes zero. Use
that result to freeze the final field spacing; require multiple samples inside the shorter of
those two scales.

## Phase P2: repeated operational panel

Static Gazebo renders are deterministic, so the main repeat unit must be an independently
executed trajectory/pass through a declared state stratum (or a physical-camera repetition
when hardware is used). Retain the exact capture-time ground truth for scoring. Stratify by:

- camera;
- near/mid/far range;
- viewing angle and image position;
- full/partial/boundary-censored view;
- single-camera and overlapping-camera coverage.

All five cameras attempt every `source_batch_id`. Retain misses and rejected observations.
The same operational correction and usability rule must be applied before computing residual
covariance or cross-camera covariance.

Use the pilot estimates to freeze repetitions before comparing methods:

```text
mean half-width delta:       n >= (1.96 sigma / delta)^2
variance relative SE r:      n >= 1 + 2/r^2
availability half-width d:   n >= 1.96^2 q(1-q) / d^2
pair correlation rho_min:    n >= 3 + ((1.96 + 0.84)/atanh(rho_min))^2
```

Use the largest requirement for the declared primary estimand, then inflate for trajectory
autocorrelation using the pilot's effective-sample-size estimate. For orientation, detecting
a correlation of 0.20 with 80% power at two-sided 5% requires about 194 independent
simultaneous pairs; hundreds of correlated frames from one pass do not meet that requirement.

## Required table

Each camera opportunity must carry:

```text
state_id, place_id, trajectory_id, source_batch_id, repetition_id
reference_x, reference_y, reference_theta, capture_stamp
camera_id, attempted, captured, detected, detector_clipped, usable
raw_bbox, raw_contact_pixel, corrected_contact_pixel, corrected_ground_reading
image_sha1, detector_artifact_sha256, calibration_sha256
```

`reference_*` and residuals are commissioning/evaluation-only. `detected`, detector clipping,
the frozen operational gate, the image/box, camera identity, and calibrated projection are
runtime-available. A `usable` label may not depend on ground-truth error.

## Analysis order and stop rules

1. Estimate `b_i(x,y,theta)` or establish that the available inputs cannot predict it.
2. On residuals after the frozen mean treatment, estimate `R_i` from independent repetitions.
3. On simultaneous, pairwise-complete residuals, estimate `C_ij`; compare a constant pair
   correlation against zero before considering a spatial model.
4. Fit `q_i` from attempted/usable binary outcomes and score Brier calibration on held-out
   places/trajectories.
5. Reserve a fresh, later capture for the final test after correction, gate, covariance rung,
   and stopping rules are frozen.

If paired RGB does not reduce heading-marginal place means on unseen places, do not tune the
CNN. Continue with the dense field/refusal branches. If it does, freeze the architecture and
training recipe, but still run P1/P2: lower mean error does not identify `R` or camera
independence.
