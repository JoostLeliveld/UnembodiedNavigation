# Reference-controlled commissioning v1

This prospective campaign collects the evidence needed to compare corrections of the
frozen-YOLO camera reading and fit a covariance model matched to each correction. It is not
a navigation study and does not preselect the winning correction or `R` family.
The robot follows native-pose-controlled lane-centre transects while the operational
camera pipeline receives no ground truth.

## Locked operating point

The nominal straight-line cruise is 1.0 m/s. The controller may slow while correcting
heading or approaching the endpoint. The validator rejects any other nominal cruise and
explicitly rejects the retired 0.22 m/s operating point. Five cameras publish at 5 Hz and
YOLO runs on device `0` at 960 px.

The four transects are the west spine, central spine, south cross-aisle and north
cross-aisle. Each is driven in both directions. Their coordinates come only from the
declared traversable geometry. No detector output, localization error, belief or learned
availability field selected a route.

At the start of each drive, the robot performs a stationary heading sequence at
-90°, -45°, 0°, 45°, 90° and back to 0° relative to the travel direction. Each heading
is held for 0.60 s. The validator checks the full rotated body at every anchor. These
anchors separate static observation geometry from motion and timing effects. Bidirectional
motion at the same locations then makes latency error change sign with velocity while
geometric bias remains approximately fixed.

## Allocation

`campaign_v2.yaml` fixes 18 complete drives before fitting:

- six fit drives: three repetitions of two closed survey laps;
- six development drives: three repetitions of two different closed survey laps;
- six sealed-audit drives: three repetitions of two further closed survey laps.

The acquisition order interleaves the three roles so simulator drift or machine state is
not identical to partition. Every drive has a unique actuation-noise seed. All synchronized
views from a drive stay in that drive's partition. Video frames are never randomly split.

## Runtime boundary

`reference_pose_controller.py` is the only online component allowed to subscribe to
`/ground_truth_tf`. It publishes velocity commands to `/cmd_vel_raw`; the existing
actuation-noise node and simulator provide the realized motion. The controller logs its
native pose, command, heading error and terminal status. The detector, support rule,
correction model, estimator and planner must not receive this reference.

The existing launch runs with `enable_mission=false`, so it publishes no goal and the
planner emits no command. The detector still records every five-camera opportunity. One
fused robot estimator runs in shadow only to consume source-batch-identified correction
envelopes and close the runtime integrity ledger. It cannot affect the commanded path:
`reference_pose_controller.py` owns `/cmd_vel_raw`. Belief-dependent manager admission is
disabled, and later learning starts from the pre-NIS detector/manager journals and selected
RGB crops. Consequently, shadow-estimator NIS decisions are diagnostics, not labels for
bias, availability or covariance learning.

`raw_frame_identity_recorder.py` hashes the complete ROS image contract and payload for
every source frame. This gives every detector opportunity a camera, integer timestamp,
source-frame identity and image identity without duplicating all raw images. Selected
detections retain their RGB crop and deployed box features in the detector's NPZ output.

## Learning order

First fit each candidate mean correction on admitted fit-drive detections:

```text
target = native_xy_at_capture - raw_projected_xy
z_corr = raw_projected_xy + candidate_correction(runtime_features)
```

Estimate signed residual bias by camera, along-camera direction, across-camera direction,
range, heading and speed. Freeze the feature contract and deterministic support rule using
only fit and development drives.

Next estimate availability from every attempted camera opportunity, pooling the
surveyed headings at each ground-plane position:

```text
q_i(p) = P(YOLO hit passes deterministic support | camera i, position p)
```

Misses and deterministic refusals remain explicit zeros. Heading is not an input to this
model. NIS is excluded because it depends on the estimator belief.

Estimate `R_hit` last, after producing whole-drive out-of-fold residuals for a correction
candidate. Every candidate receives a fresh matched `R`; covariance may not be reused across
corrections. Compare the predeclared rungs—global isotropic, per-camera isotropic,
per-camera full 2×2, then a minimal runtime-observable stratification—and select the
correction/`R` pair on development-drive proper score, containment and sharpness. An
image-conditioned runtime covariance must also expose a spatial marginal that the planner
can query before future images exist. Open the audit drives once for the final paired raw
versus corrected, availability and covariance report.

The supervisor's double-cascaded Gaussian idea remains a later online, truth-free `R`
diagnostic comparator, not the primary method. The primary estimator is one robot filter
that consumes every corrected current-frame observation once. If temporal persistence
matters, first test a measured effective update interval, then an augmented joint filter
with persistent per-camera bias states. The accumulated posterior of a preliminary camera
filter must never be passed as a fresh independent measurement, because that would reuse
the same camera evidence.

## Execution gates

Run the immutable preflight first:

```bash
python3 experiments/reference_controlled_commissioning_v1/validate_protocol_v2.py --json
```

Inspect one command without creating output:

```bash
python3 experiments/reference_controlled_commissioning_v1/run_drive.py \
  --protocol experiments/reference_controlled_commissioning_v1/campaign_v2.yaml \
  --drive-id fit_settle_lap1_r1 --dry-run
```

Then run one excluded pilot into a separate output root. Do not change the campaign status to
`pilot_passed_frozen_before_collection` until the pilot proves camera/frame identity matching,
reference cadence, endpoint accuracy, full detector accounting and clean terminal shutdown.
After freezing, execute the 24 drive IDs in their declared order. Collect audit drives but do
not inspect their images, outcomes or summary metrics until the complete sensor model is frozen.

Abort a drive on collision, reference timeout, start-pose mismatch, missing controller status,
missing run summary, missing frame identities, runner wall timeout, an invalid run summary,
an unverified terminal stop or an invalid correction ledger. Never overwrite a failed drive
directory. Diagnose it and repeat under a new campaign version or an explicitly declared
replacement ID.

All controller trace stamps use ROS simulation time. Wall time is used only by the outer
watchdog because five-camera rendering can run substantially below real time. The active v2
campaign sets this watchdog to 900 s through `runner_wall_timeout_s`; the original v1
campaign retained 720 s. Camera, detector, controller and native-reference alignment must
never mix these clocks.
