# State estimation, stochastic process, updates and logging — module review

Reviewed 2026-09-06 against the working tree. This is a software review requested for
the first module in the module-by-module inspection. Production code and experimental
settings were not changed. Source hashes and synthetic reproductions are retained in
[reviewed_results.json](../experiments/estimator_consistency/module_review_20260906/reviewed_results.json)
and [probe.py](../experiments/estimator_consistency/module_review_20260906/probe.py).

The focused existing suite passes **138 tests, with 3 archived pixel-trace tests skipped**.
Eleven additional synthetic probes reproduce the issues below. These are constructed
inputs with known behavior, not measurements of camera accuracy or findings about any
drive. No run results were selected, rescored or promoted.

The strongest remaining defects concern motion support, event identity and time. The
tested linear XY correction agrees with an independent Joseph covariance calculation.
Passing that calculation does not establish the correctness of callback wiring or the
statistical calibration of Q and R.

## Scope and the actual recursion

The active configuration used to distinguish deployed behavior from optional paths is
[network_navigation_runtime_pilot.yaml](../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml):
fused metric EKF, mandatory identified envelope, noisy odometry prediction, coupled
heading, 0.25 s planner step, 1.5 s correction replay cap, and no hard reanchoring.
The optional coherent-drift flag is off by default. This configuration is a scope
reference, not a selected experimental result.

The robot belief lives in `UnicyclePlannerNode`, not in the pixel-to-BEV conversion node.
Its state is `(x, y, theta)` with a joint 3x3 covariance. The deployed camera manager
produces a position measurement and 2x2 R. The estimator uses:

```text
timestamped noisy odometry (v, omega)
  -> motion replay to camera capture time
  -> m_minus = f(m, u), P_minus = F P F' + Q
  -> innovation = z - H m_minus, H = [I2 | 0]
  -> innovation covariance = H P_minus H' + R
  -> gain, NIS gate, accepted update or explicit refusal/recovery
  -> committed belief at the measurement time
  -> read-only prediction to the controller/publication time
```

In coupled mode, position updates can change heading through cross-covariance. That is
indirect information from motion and position, not a direct camera heading measurement.
The review covers this recursion, its optional correction modes, process-noise helpers,
the simulated encoder/actuation boundaries, and event logging/alignment. It is not a full
review of camera projection/fusion, planner objectives or offline trajectory smoothing.
The localization contract and the earlier runtime audit were read before this review.

## Findings affecting the active path or its evaluation

**F01 — P1: motion support is checked only in the long-camera-gap handler.**

`_replay_cmd_log_interval` accepts any sample preceding the interval as a valid held
input, regardless of its age. `_predict_belief_to_now` does the same. Neither checks
bounded input gaps. `covers_interval` is called only by `_advance_belief_over_outage`.
A recent camera correction therefore does not imply that its motion prediction has
recent odometry.

The `stale_motion` probe gives the filter an odometry sample at 1.0 s, a stopped command
at 9.8 s, and an update interval of 9.9–10.0 s. It predicts 2 cm of motion using the
old odometry, accepts a matching measurement, and reports odometry replay with no fallback,
although the existing coverage checker rejects that history. This can continue while
fresh camera messages keep the total correction interval below the replay cap.

Repair direction: use the same explicit support check for correction, monitoring and
control prediction. Validate ordering, finite values, start support and intervening/tail
gaps. Return support and fallback metadata with the prediction. A motion-source failure
needs an explicit recovery/control policy; it must not be called measured motion.

Locations: `unicycle_planner_node.py:1274–1318, 1849–1893, 2320`.

**F02 — P1: the event scorer can label a pre-update belief as post-update.**

`aligned.belief_at_fusion_events` checks that the batch was eventually accepted, then
selects the first logged belief whose state stamp exceeds `fused_stamp`. It ignores
`apply_stamp` and has no belief revision tying the selected belief to that batch.
Read-only predictions can exceed the capture time before the delayed correction arrives.
The 0.30 s selection window can also omit updates that arrive later.

The `premature_event_belief` probe has capture time 1.0 s, application time 1.5 s and a
prediction at 1.1 s made before the update. The loader selects that prediction and labels
it an accepted event. Its synthetic 100 cm error is the deliberately wrong prior, not
the corrected belief. This contradicts the event-selection contract; this function must
not supply post-correction accuracy claims until repaired.

Repair direction: write the committed mean, full covariance, state time and revision in
the batch's terminal update record, then score that exact record at its own state time.
Checking application time alone is insufficient to reconstruct an immediate posterior
if further updates happen before the next periodic belief sample.

Locations: `aligned.py:418–460`; `unicycle_planner_node.py:2353–2368`.

**F03 — P1: planning-time bootstrap bypasses the mandatory correction envelope.**

When the belief is absent, `_resolve_state_belief_ekf` calls `_apply_state_correction`
on the compatibility `/state/bev` message, even when `require_state_correction_envelope`
is true. That bypasses envelope validation and applies the measurement without its
source-batch identity. Publishing the envelope first does not enforce delivery order
across separate ROS subscriptions.

The `untracked_bootstrap` probe delivers the compatibility pose, runs the planning
resolver, then delivers its valid envelope. The pose has already initialized the belief
without an assimilation record; the identified event is subsequently logged as
`dropped/not_newer_than_belief`. Thus the ledger says a consumed measurement was refused.

Repair direction: in mandatory-envelope mode, bootstrap exclusively through that
validated event. The resolver should report no belief until it arrives.

Locations: `unicycle_planner_node.py:2655–2664`; contrast `805–824, 827–876`.

**F04 — P2: distinct fusion events can disappear at the same receipt-clock time.**

`ExperimentLogger._fusion_decision_cb` discards a message if its receipt time is equal
to the previous callback's receipt time. This is not detector-event identity. Several
queued callbacks may run between simulated clock updates.

The `same_clock_batches` probe delivers two different batch IDs and capture timestamps
at one receipt-clock value. Only one row is written. Downstream accounting may catch the
missing evidence, but the logger has already lost it.

Repair direction: identify/deduplicate using the physical batch ID and the camera event,
and retain receipt time as a timing field. Preserve separate events sharing a clock tick.

Location: `experiment_logger.py:1699–1703`.

**F05 — P1: unsupported motion is paid for in position uncertainty but not heading.**

The existing unsupported-history fallback replays only a suffix and moves the belief
stamp across the whole gap. It adds an XY reach term for the omitted time, but no heading
uncertainty for a possible omitted turn. The earlier audit describes this fallback as a
heuristic; the missing heading consequence remains important with coupled updates.

The `unsupported_heading` probe advances ten seconds with only the last portion of the
motion history available. The 1.5 s replay cap leaves 8.5 s unsupported. Heading variance
grows only by the ordinary Q for 1.5 s, while position receives a large reach inflation.
A turn during the omitted interval is therefore absent from both heading mean and its
additional uncertainty. Later XY updates use that joint belief.

Repair direction: preserve uncertainty about the full missing SE(2) motion, or enter an
explicit invalid/relocalization state with an appropriate control response. Do not imply
that a position-only reach term establishes reliable heading after missing motion.

Location: `unicycle_planner_node.py:2321–2339`.

**F06 — P2: latest odometry heading is stamped as an older camera-time heading.**

Bootstrap/reanchoring uses `_map_frame_heading()`, which reads the latest received
odometry yaw, but assigns the camera's capture stamp to the resulting belief. In
`camera_xy_only` mode the accepted-update path also makes this substitution on every
correction. Predicting from that capture time then integrates some rotation again.

The `backdated_heading` probe supplies scripted odometry with yaw 0.8 rad at capture
9.8 s and 1.0 rad at receipt 10.0 s. Bootstrap stores 1.0 rad at 9.8 s. Replaying to
10.0 s produces 1.2 rad. In the active coupled configuration this affects a moving
bootstrap; recurring hard reanchors are disabled there.

Repair direction: obtain heading at the declared belief time from timestamped odometry,
with checked support, and use uncertainty for that same time. Alternatively require a
documented stationary initialization rather than implicitly assuming it.

Locations: `unicycle_planner_node.py:1900–1905, 2106–2119, 2580–2585`.

**F07 — P2: logged Q is computed from a different formula than applied Q.**

The prediction diagnostic calls `planner.process_noise(age)` without heading or speed.
That invokes the diagonal fallback, whose heading variance is `sigma_theta² * age/dt`.
Actual `predict` passes heading and speed and uses `sigma_theta² * age`.

The `q_diagnostic` probe uses the active 0.25 s planner step. Over 0.1 s, actual heading
process variance is 0.00004 rad² but `_latest_Q_theta_theta` is 0.00016 rad²: four times
larger. This is a reporting defect, not evidence that the filter applied the larger Q.
The yaw-increment diagnostics also use a different interval convention from the replay:
they assign the new sample to the preceding interval and omit start-held input.

Repair direction: accumulate and return diagnostics from the actual replay transaction,
instead of computing them through separate formulas and loops.

Locations: `unicycle_planner_node.py:1828–1847, 1864–1868`; `base_planner.py:397–410`;
`dynamics.py:138–144`.

## Findings in optional modes and encoder metadata

**F08 — P2: enabling coherent drift does not implement its cumulative-distance model.**

`coherent_drift_increment` requires the distance before and after a step. The runtime
wrapper never passes accumulated distance, so every prediction uses the default zero.
The `coherent_wrapper` probe executes fifty 0.1 s steps with the real planner wrapper.
Its added cross-track variance is 50 times smaller than the declared total-distance
block. The flag is off in the active configuration.

Also, the test describing vanishing white cross-track uncertainty sums individual Q
matrices without `F P F'`. That is not the filter recursion. The probe verifies that the
ordinary white model composes correctly across the same straight-motion partition when
the state Jacobian is retained. The lateral Q term depends on v, not on turning rate w;
setting w to zero does not make it vanish. This does not prove the white model is
calibrated for the simulated encoder.

Repair direction: define the persistent-error state and its update semantics first,
then carry them through filtering and forecasting. A cumulative-distance counter that
arbitrarily resets on each camera update is not by itself a persistent-bias model.
The CasADi forecast also currently has no coherent-drift term, so enabling the runtime
flag would leave the forecast using different dynamics uncertainty.

Locations: `base_planner.py:397–410`; `dynamics.py:80–141`;
`tests/planning/test_coherent_drift.py:test_fixes_the_anisotropy_that_gate0_found`;
`casadi_efe.py:118–144`.

**F09 — P1 when enabled: per-camera quorum recovery acts before freshness checks.**

`_apply_map_observations` runs `_reanchor_on_camera_quorum` before individual observation
age/order gates. It compares readings against the old committed position rather than a
prediction at their timestamps. A group of stale readings can therefore reanchor the
belief and move its timestamp backward, even though every subsequent update is dropped.

The `stale_quorum` probe starts with a belief stamped 10 s, wall/simulation time 12 s,
and two agreeing observations stamped 9 s. Recovery moves the belief to their distant
position and backdates it to 9 s before freshness rejects them. The active fused
configuration with `state_reanchor_m: 0` does not take this path.

Repair direction: validate freshness and causal order before considering quorum recovery;
compare observations and the predicted prior at a common supported time, then log one
explicit recovery event with its committed state.

Locations: `unicycle_planner_node.py:2202–2214, 2243–2275`.

**F10 — P2: the legacy pixel path can reuse or backdate a measurement.**

Unlike the metric path, the pixel path does not reject a timestamp at/before the
committed belief. Nonpositive dt becomes the nominal dt. With callback mode's default
zero throttle, repeated delivery contracts covariance again; an older but fresh message
can move the belief clock backward. A positive throttle rejects equal timestamps but
does not reject negative timestamp differences.

The `duplicate_pixel` probe applies one 9.95 s measurement twice, reducing XY variance
from 0.02222 to 0.01429 without another observation. Delivering 9.90 s then backdates the
belief. `use_pixel_correction` is false in the active configuration.

Repair direction: make event uniqueness and causal ordering shared prerequisites for
both correction sources, independent of the update-rate throttle.

Locations: `unicycle_planner_node.py:1225–1238, 1321–1331, 1670–1728`.

**F11 — P2: encoder covariance keeps adding noise when its generator adds none.**

The encoder mean branch injects no noise at a hard stop or when disabled, but covariance
propagation always uses the configured additive/slip variances. At a hard stop its
Jacobian is identity, yet additive covariance continues to accumulate. Disabled moving
operation also retains the noise/scale-uncertainty model.

The `encoder_covariance_at_stop` probe shows increasing position and yaw covariance
during ten seconds of stationary input. This concerns published encoder pose/twist
metadata and any commissioning method that consumes it; the active robot EKF reads
velocity and its own Q, not this pose covariance.

Repair direction: share the enabled/motion-activity conditions between the actual
generator and its declared covariance. Distinguish instantaneous twist variance from
the long-window AR covariance approximation as a separate modeling decision.

Locations: `encoder_noise_node.py:200–207, 285–305`.

## Stochastic-model decisions still open

The existing [runtime integrity audit](runtime_integrity_audit.md) already identifies
these limits; this review does not count them as newly discovered repairs:

- Actuation disturbances advance per incoming command, including immediate plan
  publication as well as periodic publication. AR correlation and random-draw position
  consequently depend on callback count. Equal seeds do not identify equal disturbances
  at equal simulated times across execution schedules.
- The encoder includes a nonzero speed-scale mean, AR(1) slip and per-message white
  perturbations. The EKF uses a fixed zero-mean white process model. The continuous-rate
  Q parameters and per-message generator standard deviations need a declared mapping;
  equal numeric values are not that mapping. A systematic mean error is not removed
  by increasing covariance.
- A returning camera is still refused solely because total time since the last
  committed correction exceeds 1.5 s, even when motion support is complete. Repeated
  rejection inflation is also retained. These are baseline policies, not evidence that
  sensor R is correct or incorrect.
- Runtime robust fusion and the planner's independent precision forecast remain
  different observation models. Pixel updates also still borrow a planning visibility
  covariance in the legacy path. Delivered-measurement noise and future availability
  need explicit separate semantics.

## Recommended repair order and verification

First repair F01–F04: one supported prediction, one validated identified update and one
durable event record. Then repair heading/recovery timing (F05–F06) and make diagnostics
describe the actual transaction (F07). Address optional-mode bugs before enabling those
modes. Change stochastic/recovery policies in separately identified configurations after
the deterministic wiring is reliable.

A useful update record contains batch/camera identity, measurement stamp, application
time, input/output belief revision, motion-support result, source of replay, prior mean
and full covariance, measurement and full R, innovation and innovation covariance, gain,
decision/reason, committed mean and full covariance, and committed state time. Periodic
belief publication should carry that revision so later analysis can distinguish a
prediction from a posterior belonging to a particular update.

From the repository root, the synthetic probes run without creating a ROS graph:

```bash
source install/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/estimator_consistency/module_review_20260906/probe.py
```

The probe asserts the reviewed defective behavior so that the baseline reproduces.
Its successful exit means reproduction succeeded, not that the estimator passed the
desired invariants. After repairs, convert the relevant cases into regressions asserting
the corrected behavior. Source hashes distinguish the reviewed tree from later changes.

Existing-suite command used during review:

```bash
source install/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/planning/test_belief_correction.py tests/planning/test_planner_node_state_correction.py tests/planning/test_planner_node_correction_wiring.py tests/planning/test_planner_node_per_camera_correction.py tests/planning/test_outage_motion_replay.py tests/planning/test_runtime_transactions.py tests/planning/test_coherent_drift.py tests/planning/test_commissioning_joseph.py tests/experiments/test_logger_schema.py tests/experiments/test_logger_time_alignment.py tests/experiments/test_fusion_study_alignment.py tests/sim/test_command_watchdog.py
```
