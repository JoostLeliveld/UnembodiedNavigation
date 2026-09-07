# 07 — Multi-camera fusion and covariance semantics

Reviewed 2026-09-06. **The active equal-capture-time aggregation implements its documented robust covariance policy and reproduced one registered fused event exactly. Confirmed defects remain in common-time motion support and publication/event accounting. Optional GP, direct-camera and bias-floor paths have additional contract defects.** Keeping one camera's covariance for identical equal-quality cameras is deliberate; it is not a missing division by N.

This is a report-first investigation. No runtime source, detector/model artifact, configuration, other audit, or experiment output was edited. Only `docs/module_audits/07_*` evidence was created. Running experiments were neither launched nor stopped. Shared manager edits remain unclaimed. Investigation 08 was given the actual fusion equations and comparison boundaries; 01 owns robot recursion, 04 detector batching, 05 geometry, and 06 gate/recovery policy.

The six requested documents, workspace/repository instructions, current ICRA status, investigation map and existing module audits were read. Preserved historical comments do not override current configuration. In particular, comments describing the manager as a median-seeded identity-prior KF are obsolete.

## Evidence and active path

- [Independent numerical probe](07_fusion_probe.py) and [results](07_fusion_probe_results.json): closed-form Gaussian/Joseph oracles, full correlated 2×2 matrices, independent convex Huber optimization, permutations, duplicates, time support, floors and malformed inputs.
- [Actual-node probe](07_fusion_node_probe.py) and [results](07_fusion_node_probe_results.json): unchanged manager/filter methods with real message types, recording/failing publishers and the existing node fixtures. No ROS graph; motion is explicitly controlled to isolate fusion.
- [Registered-event probe](07_registered_event_probe.py) and [results](07_registered_event_probe_results.json): the earliest multi-camera published event in the explicitly selected P0 run, through `aligned.py`, with required input hashes checked.
- [Existing regression transcript](07_existing_regressions.txt): **150 passed**. These cover the four requested suites plus bias floors, direct-camera wiring, runtime transactions and investigation 04's acquisition probes. Some acquisition tests deliberately assert current defects; passing them is not proof those defects are repaired.

At probe time HEAD was `cc394cb3a5156d5ace480f926b1b06f647265b48`, with extensive pre-existing staged/unstaged/untracked changes. Another workstream subsequently preserved this source in checkpoint `3c4ddeae4a7427bf374514dad6a1d65dc728a91c`. Exact source hashes are in `07_fusion_probe_results.json/source_identity`. The probed `fusion.py`, `camera_manager_node.py`, `reference_calibration.py`, `unicycle_planner_node.py` and campaign YAML match the registered corrected-runtime protocol. The protocol does not hash every dependency; the probe records those separately. Installed reliability/planning egg links point through `build/` symlinks into the inspected `src/` files.

**Concurrent-repair boundary:** during final QA, investigation 04 changed the shared manager's batch buffer and decision claim. Its new code must not be confused with the snapshot above. Numbered manager references and baseline reproductions below refer to that snapshot; source links open the evolving working file. The new [SourceBatchBuffer](../../src/reliability/reliability/source_batch_buffer.py) adds bounded pending/closed sets, expiry/outcome events and per-camera timestamp watermarks; conflicting duplicate content now aborts the pending transaction. `_decide` claims the batch before publication, closing the baseline manager retry window in F02. These changes address the pending-callback part of F07, while raw fusion helpers remain unchanged. They do not resolve robot commit/terminal publication, common-time motion support or the aggregation equations. Investigation 04 completed its repair with **101 passing tests**, including manager/fusion regressions and the publication-error/conflict cases: [repair addendum](04_camera_acquisition_and_batching.md), [test transcript](04_repair_test_results.txt), [source hashes](04_repair_source_sha256.txt). The final manager/buffer hashes were checked against those files. That validation is 04's; live restart, DDS delivery and joined assimilation-ledger validation remain outstanding. `decision_completed` only means the manager routine returned, not that a correction was published or assimilated. The [snapshot reproducer](07_reproduce_snapshot.py) independently reran the original **150 regressions and all three probes successfully**, with exact matches to saved JSON results, in a temporary extracted checkout; [verification output](07_snapshot_reproduction_results.json). It changed no shared source or experiments.

The reference configuration is [network_navigation_runtime_pilot.yaml](../../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml), registered under `network_navigation_runtime_pilot`. Launch resolution was checked against `visibility_launch_common.py` and the actual launch evaluation retained by [investigation 01](01_state_estimation.md).

| Selected boundary | Effective setting |
|---|---|
| Detector/manager cameras | A–E, strict source batch IDs, maximum timestamp spread 0.05 s; manager decisions at 5 Hz |
| Manager | Active authority, fusion enabled, `joint_network`; robot consumes mandatory fused envelope |
| Camera mean and R | Frozen bbox-feature NN, then subtract per-camera residual offset; `commissioned_reference_r`, full constant map-frame R for that corrected reference |
| Admission | Bbox plausibility against nearest operational belief; bootstrap quorum 2, pairwise bootstrap tolerance 0.91 m; post-alignment median-distance gate 0.6 m |
| Eligibility | Launch defaults: max age and decay 1.25 s, minimum trust 0.15, minimum association 0.30 |
| GP providers in manager | Empty artifact template; `require_gp_artifacts=false`. The planner's three score fields do not become manager GP providers |
| Time alignment | To newest eligible capture time, always in fusion mode; nearest-pose tolerance 0.35 s, drift parameter 0.05 m/s |
| Additional propagation to decision time | Disabled; this does not disable cross-camera common-time alignment |
| Covariance additions | Both bias-floor slopes zero; common-mode standard deviation zero |
| Robot update | Metric fused EKF, coupled heading, NIS 9.21, Q parameters 0.01/0.02; direct-camera mode off |
| Optional/other code | `covariance_mapping.py`, Student-t reweighting, information-best selection, health/dynamic-occlusion replay are not the selected runtime fusion rule |

The registered aggregate selection became available during this audit. Its use below is limited to one source-bound algebra/identity trace. No recorded-run accuracy, RMSE, NEES, coverage, field ranking or navigation effect is reported. The historical fixed-route campaign remains invalidated. Investigation 01's identified `aligned.belief_at_fusion_events` defect also precludes using that helper to claim an immediate posterior until repaired.

## Ranked findings

P1 indicates a causal measurement/evidence defect to resolve before relying on the affected path. P2 indicates a narrower conditional defect; P3 a smaller numerical/diagnostic problem. “Confirmed” means the trigger was reproduced in the pinned audit source, not that it occurred in a recorded drive. The concurrent-repair note qualifies F02/F07 and batch-buffer rows; it does not invalidate the remaining reproduced mechanisms.

| Rank | Finding | Reachability | Classification |
|---|---|---|---|
| F01 / P1 | Nearest poses can relabel an uncompensated observation, use post-target motion or absorb belief jumps | Active when captures differ; additional propagation optional | Confirmed current |
| F02 / P1 | A correction's publication, state commit and terminal record are not one recoverable transaction | Active, startup ordering or publication failure | Confirmed; manager retry window changed by concurrent 04 repair, robot gaps remain |
| F03 / P1 if enabled | Direct-camera receiver ignores the supplied frame and loses physical batch identity | Optional direct-camera path | Confirmed current |
| F04 / P1 if enabled | GP quality lookup replaces full commissioned camera R with a different isotropic R | Optional manager GP artifacts | Confirmed current |
| F05 / P1 if relied upon | Runtime “belief floor” only floors the fused measurement; repetition still contracts robot P | Optional nonzero floor | Confirmed implementation/semantic mismatch |
| F06 / P2 | Three-camera floor combination and equal-trace selection depend on order | Optional floors / `best_single` | Confirmed current |
| F07 / P2 | Duplicate IDs corrupt raw fusion helpers; conflicting duplicate deliveries choose a payload by callback order | API boundary; malformed strict delivery | Confirmed baseline; concurrent 04 repair aborts conflicting pending duplicates, raw helpers unchanged |
| F08 / P2 | Absolute determinant cutoff calls valid well-conditioned covariance singular | Alternative scales/profiles; no selected-R trigger | Confirmed numerical defect |
| F09 / P3 | IRLS can hit its iteration cap without reporting non-convergence | Active covariance values, constructed admitted means | Confirmed current |
| F10 / P2 | Provenance validation still permits inconsistent member/common-time fields; optional flooring repairs invalid input | Malformed envelope / optional helper | Confirmed current, separate from repaired SPD envelope validation |

### F01 — Common-time compensation does not establish supported motion

**Locations:** [manager:225](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:225), [337](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:337), [1224](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1224), [1654](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1654), [1762](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1762).

**Trigger:** staggered captures within the allowed 0.05 s; sparse/delayed/incorrectly ordered pose history, or fewer than three odometry poses. `_nearest_state_pose` independently picks the closest pose within 0.35 s at each endpoint. It does not interpolate, verify support between endpoints, reject future-only support, preserve frame/covariance/revision, or ensure the two selected poses describe different instants. The caller chooses odometry solely by `len(history)>2`, otherwise belief history.

**Expected:** translate each mean by a verified displacement between its capture time and the declared common time, or reject that operand explicitly. **Observed:**

- Cameras at 10.000 and 10.040 s, synthetic straight motion 0.22 m/s: exact endpoints correctly move the older reading by **0.88 cm**. One pose at 9.900 s is accepted for both endpoints instead: displacement zero, no rejection, yet its stamp becomes 10.040 s. A lone pose at 10.300 s also passes both lookups. The added covariance is still only `(0.05×0.04)² I = 0.000004 I m²`; it contains no uncertainty for missing temporal support.
- Equal-distance endpoint poses delivered in reversed callback order produce older-camera shifts **0.34375 cm versus 1.03125 cm**. The physical sample set and target are identical. This is nearest-neighbor tie/order dependence, separate from floating-point summation in fusion.
- The actual manager ignores two exact stationary odometry endpoint poses in favor of a belief history that jumps 0.40 m from a correction. It moves the older camera mean from `(1,2)` to `(1.4,2)` and labels it common-time. A changing belief correction is not physical travel; a constant position offset canceling in a difference does not justify this fallback.
- Empty history correctly rejects the older operand; equal-time observations correctly need no displacement. Those positive cases do not cover stale-but-near history.

**Frame consequence:** `_odom_callback` discards `header.frame_id` and adds raw odometry XY differences to `map_bev` readings. Encoder output retains `odom` ([encoder:315](/home/joostleliveld/Thesis/UnembodiedNavigation/src/sim/sim/encoder_noise_node.py:315)); launch defines its map rotation from spawn yaw ([launch:934](/home/joostleliveld/Thesis/UnembodiedNavigation/src/experiments/experiments/core/visibility_launch_common.py:934)). At a 90° rotation, `(0.0088,0)` in odom must become `(0,0.0088)` in map. The helper instead adds the former. Selected traverse spawn yaw is zero (`tasks.yaml:457`), so this frame bug has no demonstrated selected-config rotation trigger. Odometry heading error can still spoil a short displacement even with zero fixed frame rotation; absolute translation cancellation does not remove heading error.

**Smallest repair:** define a displacement query over one immutable, finite, ordered, frame-labelled motion snapshot. Evaluate endpoints at their stated times with a declared interpolation/replay support rule and maximum gaps; return unsupported status rather than borrowing an arbitrary nearby pose. Transform displacement to map coordinates. Remove the corrected-belief-difference fallback, or replace it with motion-only replay whose revision and support are explicit. Retain original capture observations and record endpoint samples, displacement, source and uncertainty separately. Apply the same rule to optional propagation-to-now. Coordinate the support API with 01/02; this does not require changing detector cadence, geometry or admission thresholds.

### F02 — A published correction need not have a truthful terminal record

**Locations:** [manager:1523](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1523), [1827](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1827); [robot:867](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:867), [2341](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:2341), [2508](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:2508), [2655](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:2655).

**Trigger/results:** actual-method failure injection produced these distinct cases:

1. The manager publishes the fused envelope, then the compatibility publisher fails. There is **one envelope, zero decision messages, and no `_last_decided_source_batch_id`**. Retrying the decision publishes a second envelope with the same ID. In normal runtime an uncaught manager exception may end the process; the retry is a controlled recovery probe, not a claim of automatic live retry. The receiver's duplicate guard prevents a second ordinary assimilation by treating duplication as fatal.
2. The filter commits the state, then diagnostic publication fails before terminal assimilation publication and before its seen-ID set is updated. The state advances to 9.950 s, **zero terminal rows** exist, and the ID is absent from the seen set. In a controlled retry, it reports `dropped/not_newer_than_belief` even though the first attempt consumed the measurement. The actual fatal-stop behavior is intentional; it does not retroactively make the ledger complete.
3. Reverified [01's startup bypass](01_state_estimation.md): compatibility pose arrives before its envelope; the planning resolver bootstraps from it without identity even with mandatory-envelope mode enabled. The subsequent envelope records `dropped/not_newer_than_belief`. This can occur without a publisher exception. Publishing the envelope first is not a delivery-order guarantee across topics.

**Documented later repair:** 04's `_decide_once` wrapper preclaims the batch ID and emits decision-error/completion outcomes. Its completed regression suite verifies case 1's same-process manager no-retry behavior. Cases 2 and 3 are unchanged; a manager outcome is not a robot assimilation acknowledgment.

**Expected/consequence:** every published fused correction needs exactly one truthful, reconstructable terminal outcome. Partial publications can leave missing or misleading evidence. Restart loses manager/receiver in-memory dedup state; 04 also establishes that source IDs lack a producer epoch. Reliable topics and serialization locks do not provide durable exactly-once processing across a crash.

The logger observes the later decision topic rather than the fused envelope (`experiment_logger.py:1000,1684`). Reverified 01/10's equal-receipt-clock defect: two distinct batches delivered at one logger clock value yield one logged row (`1699–1703`). These are separate failure modes, not double counting one finding.

**Smallest repairs, with owners:** 01 should make mandatory-envelope startup exclusive and commit state plus the immutable terminal outcome/seen ID in one transaction. Publication should send/retry that stored outcome, never recompute it after commit. 07/10 should preconstruct and retain an immutable fusion event and publication state; compatibility/diagnostic failure must not create a second measurement. 10 should log identity-bearing correction publication itself and deduplicate by event identity, with state revision/full posterior in terminal records. 04/13 should define session/epoch identity and restart reconciliation. Durability or explicit run-invalidating crash recovery is needed for cross-restart guarantees. Reordering publishes alone cannot solve this.

### F03 — Direct-camera transport is not an equivalent validated interface

**Locations:** [fusion:14](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:14), [67](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:67); [manager:1529](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1529); [robot:879](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:879), [2182](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:2182), [2214](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:2214).

**Trigger:** enable `state_correction_mode=per_camera`, deliver a valid-shape batch in `camera_optical`. **Expected:** reject before updating a map-frame belief. **Observed:** `_map_observations_cb` stores the parsed frame as `_frame_id` and ignores it; the actual robot posterior changes exactly as if those values were map coordinates, with no error.

`MapObservation` has no physical batch/observation ID, frame, capture/common-time distinction or model version. Its constructor validates mean and R but not finite timestamp or nonempty camera ID. NaN timestamp and empty ID are accepted by the numerical contract; JSON emitted by the proper serializer blocks NaN, while permissive input parsing can still construct it. These are validation omissions, not proof every malformed value survives later conversion. The direct receiver deduplicates by camera/timestamp, has no allowed-camera registry at this seam, and calls `_apply_metric_correction` without `source_batch_id`. **Three accepted same-time cameras produce zero `correction_assimilations` records** in the probe.

The manager publishes this batch before eligibility, common-time support, bootstrap quorum and metric disagreement checks. It has already applied projection/mean construction and bbox admission, but it is not the same admitted event stream as the fused path. Thus the current direct switch changes more than aggregation and cannot satisfy the current source-batch assimilation contract unchanged.

**Smallest repair:** extend the per-camera wire contract with batch epoch/ID, unique observation IDs, explicit frame/reference/model and capture stamps; validate those before processing. Preserve each camera R and capture mean. Record one batch outcome containing per-camera terminal outcomes and posterior revision. For a controlled comparison feed both methods the same predeclared admitted events, while preserving camera times; gate-policy differences need a separate arm. Coordinate receiver changes with 01/06/10.

### F04 — GP lookup silently overwrites commissioned full R

**Locations:** [manager:815](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:815), [1518](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1518); [replay:572](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/replay.py:572), [585](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/replay.py:585).

**Trigger:** configure a manager quality provider alongside a commissioned covariance profile. `_with_provider_quality` calls the replay helper that replaces covariance with `v I`, where `v=p×0.04²+(1−p)×0.40²`. **Expected under the selected observation contract:** provider quality may affect eligibility; an actual delivered reading retains the commissioned conditional camera R unless a different noise model is explicitly selected.

**Observed:** input `[[.016,.0006],[.0006,.009]] m²`, provider p=.9, becomes `.01744 I m²`; anisotropy and correlation disappear and the source label becomes the provider's. The mean is unchanged. Provider query position is nearest belief XY (measurement XY at bootstrap), and the query timestamp is decision time, so this alternative R also depends on belief/availability rather than only the delivered camera model.

**Reachability:** disabled in the selected pilot because manager providers are empty. The existence of planner GP fields does not trigger it. The replay mode's designed mapping is intentional in replay; using it to decorate commissioned runtime observations is the interface error.

**Smallest repair:** a quality-only replacement in the manager, preserving `xy_m`, `covariance_m2` and model identity. Make any score-to-R alternative an explicit incompatible covariance profile with its own provenance and tests. Keep current empty-provider configuration and artifacts unchanged.

### F05 — A floor on measurement R cannot enforce a floor on recursive P

**Locations:** [bias_floor:29](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/bias_floor.py:29), [106](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/bias_floor.py:106); [manager:506](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:506), [558](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:558).

**Trigger:** enable nonzero floor slopes and repeatedly observe a stationary state. The library explicitly describes a posterior floor preventing repeated looks from shrinking persistent error. The manager calls that operation on the fused **measurement** covariance, then sends only the resulting R to the robot. No physical floor is transported/applied to the robot posterior.

**Observed independent limiting case:** a `.01 I m²` floor raises the camera/fused R to `.01 I`; with initial robot P=I, zero Q and 100 repeated corrections, planar P becomes `1/(1+100/.01) I = .000099990001 I m²`. Its standard deviation is about **1 cm**, below the supposed **10 cm** floor. The library's direct posterior-floor helper does hold the posterior bound; runtime placement changes its meaning.

**Consequence:** a user enabling this flag to model persistence receives only a per-event conservative covariance. Ordinary process noise may create a steady-state variance, but does not make that variance equal to the declared floor.

**Smallest repair:** decide and type the intended quantity. If this is a fused-report floor, name/document it accordingly and make no persistence claim. If it is a posterior bound, send an explicit bound to the single filter owner and apply a justified full-state covariance rule there; do not paste a 2×2 block into a 3×3 P without preserving cross-covariance/SPD. A persistent bias state is a separate model experiment. Do not add the same bound at both places by accident. Active slopes are zero, so this is not an explanation of the selected pilot.

### F06 — Order invariance has two uncovered exceptions

**Locations:** [bias_floor:171](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/bias_floor.py:171), [205](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/bias_floor.py:205); [fusion:336](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:336).

**Trigger/results:**

- Three anisotropic floors at range 20 m, bearings 0, .7, 1.4 rad, using default library slopes: orders (0,1,2) and (0,2,1) yield respectively `[[.001504988,.000531205],[.000531205,.001090628]]` and `[[.001049588,.000155657],[.000155657,.000995894]] m²`. Each dominates every input in the Loewner sense; the conservative bound property passes. Sequential flooring is not associative, so the two-camera order test misses the three-camera failure. The maximum x marginal standard deviation differs by about **0.64 cm**.
- Equal-trace `best_single` cameras at x=0 and x=.2 select the first item. Reversing input selects a different mean by **20 cm**, with the same R. The strict active callback assembly normally uses configured camera order, so ordinary callback permutations do not trigger this tie. Reordering configured cameras or calling the library does.

**Expected/consequence:** camera-order invariance unless a tie priority is part of the named policy. The first floor result is not invalid covariance; it is an undocumented order-dependent bound. The three-camera claim in `sequential_kalman_update_2d` is stronger than the implementation.

**Smallest repair:** deterministic camera-ID ordering makes the existing floor fold reproducible with a declared order; it does not make the mathematical operation symmetric. A symmetric joint bound requires its own chosen rule and validation. Break best-single trace ties by a stable camera ID and expose that rule. Test all three-camera permutations, including noncommuting covariance axes. Active `joint_network` with floors off passed every tested permutation to floating-point precision.

### F07 — Duplicate identity is not enforced by fusion helpers

**Locations:** [fusion:300](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:300), [375](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:375), [412](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:412), [452](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:452); [manager:1191](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/nodes/camera_manager_node.py:1191); [eligibility:288](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/camera_manager.py:288).

**Trigger:** two copies of camera A at `(1,2)`, R=.01I, supplied directly to `_gated_fusion` or its raw helpers. **Observed:** independent fusion halves R to .005I; `distance_angle` returns **mean `(2,4)` and R=.02I** because weights are keyed by camera ID, normalized once, then used once per list entry. Joint identical-duplicate covariance happens to remain .01I, but `used` contains A twice; duplicating only one member of a disagreeing set also changes the robust objective and median gate. “No 1/N contraction” is not deduplication.

**Active boundary:** `CameraManager._eligible_by_camera` and strict `_latest` produce one map operand per ID; the identical-duplicate probe returns one eligible camera. Ordinary complete redelivery and extra timer ticks are guarded, independently reverified by 04's tests. Do not report the raw helper defect as ordinary active double fusion.

**Baseline active malformed-delivery case:** before batch completion, two different camera-A payloads claiming the same source batch overwrite the same slot. Delivering pixels 100 then 200 versus 200 then 100 makes the completed batch retain 200 versus 100. No conflict is reported. In strict normal delivery this should be retransmission of identical content; last-writer choice hides a producer/transport integrity violation. **Documented later repair:** 04's new buffer closes such a pending batch with `conflicting_duplicate`, verified by its completed regression suite; that repair is separate from this retained baseline reproduction and does not validate unique IDs passed directly to fusion helpers.

**Smallest repair:** validate unique observation/camera identities at the fusion seam; identical redelivery is idempotent, conflicting payloads fail with a reason. Retain per-batch content digests/tombstones at receipt. Do not deduplicate by means, since different cameras can legitimately report identical positions. Batch epoch/reset and timestamp-window membership remain 04's responsibility.

### F08 — The 2×2 inverse uses an absolute determinant singularity test

**Location:** [fusion:880](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:880).

**Trigger:** a valid covariance whose determinant is at most 10⁻¹², or a valid large covariance whose precision determinant crosses that bound. **Observed:** a one-camera `R=10⁻⁶ I m²` is accepted as SPD by `MapObservation` but independent fusion raises `matrix is singular`; condition number is exactly one. `10⁻⁷ I` and `10⁶ I` also fail. A single-camera joint estimate bypasses the inverse and returns R, so limiting behavior differs by rule.

**Expected:** a well-conditioned invertible covariance remains usable across scales. **Consequence:** valid precise observations can crash an alternative fusion/profile; the rule can fail for numerical units/scale rather than information content. The selected commissioned camera R matrices do not approach this cutoff.

**Smallest repair:** use a stable SPD solve/factorization, with scale-aware conditioning and finite-result validation. Keep invalid matrices rejected. Do not inflate camera noise to work around a dimensionful determinant cutoff.

### F09 — A non-converged robust estimate is returned without status

**Location:** [fusion:480](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/fusion.py:480).

**Trigger:** the five fixed camera R matrices with the synthetic means in `IRLS_iteration_cap`; all five pass the actual 0.6 m gate. **Observed:** default 25 iterations returns `(-.035711794,.124418108) m`, while independent convex optimization returns `(-.035456087,.125132435) m`. Difference **0.07587 cm**, runtime gradient norm .022813 versus 4.19×10⁻¹⁵ at the oracle. A longer IRLS solve agrees with the oracle. The requested change tolerance is 10⁻⁹ m, but hitting the cap produces no convergence flag and covariance code refers to the “converged” point.

**Consequence:** this probe establishes a small optimization/diagnostic defect, not a large live localization failure or an invalid Huber objective. **Smallest repair:** return/log iteration count, terminal step and stationarity/convergence status; define a bounded fallback/refusal for non-convergence. Any tighter accuracy/latency target is a policy choice. Keep the covariance formula explicit for the returned estimate.

### F10 — Validation closes covariance shape/SPD, but not every semantic invariant

**Locations:** [robot:845](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/unicycle_planner_node.py:845); [contracts:479](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/contracts.py:479); [bias_floor:106](/home/joostleliveld/Thesis/UnembodiedNavigation/src/reliability/reliability/bias_floor.py:106).

The repaired fused callback rejects unsupported schema 99, foreign frame, indefinite and asymmetric R before filtering; all reverified. `MapObservation` also rejects finite indefinite/singular/asymmetric and nonfinite R. **No active path was found that silently regularizes a malformed fused-envelope R into a good measurement.** Numerical posterior regularization is not the same operation.

Remaining reproduced gaps: the envelope accepts duplicate/empty `accepted_camera_ids`, `common_capture_stamp` later than the correction stamp, and boolean `schema_version=true` through Python equality with 1. These fields do not change the update directly; they undermine provenance validation. The current manager generates ordinary valid values, so no nominal producer fault is asserted. `CameraObservation` validates pixel covariance/schema and topic camera agreement but not calibration/image-frame/pixel-vs-box semantics; [05 G01/G02](05_observation_geometry_and_calibration.md) owns those confirmed upstream defects.

Optional `apply_belief_floor` checks finite shape/symmetry but not that its input P is positive semidefinite: `diag(-1,.001)` with floor .01I silently becomes .01I. This demonstrates missing validation in that helper, not a demonstrated route around the active fused envelope. Projection's numerical floor has a separate extreme-scale limitation already reported by 05 G07.

**Smallest repair:** a shared typed envelope validator should enforce exact schema type, finite ordered times, nonempty unique allowed members, reference/frame/model identity and source epoch semantics. Validate PSD/SPD of input covariance before optional flooring; regularization must have a bounded roundoff-only contract or a reasoned refusal. Keep these integrity checks separate from statistical gate policy.

## Capture-to-correction quantity contract

Short file names below refer to `src/reliability/reliability/`, except robot and logger, which name their node files. Timestamps are seconds on the simulation/ROS clock; wall inference durations are milliseconds and are not state timestamps.

| Quantity / owner | Camera and physical event identity | Instant, frame and units | Covariance meaning and robot-belief dependence |
|---|---|---|---|
| `CameraObservation` / detector → manager callback | Topic camera must equal payload camera; outer `source_batch_id`; source currently encodes ordered camera@nanosecond capture list, not a restart epoch | `timestamp_s` is capture; image pixel uv/xyxy, image-frame/calibration labels; separate receipt/inference/publish fields | `conditional_cov_uv` is px²; availability/association are dimensionless. No robot prior in detector contract. Image/frame semantics still need 05 validation |
| Pending/ready batch / manager 1177–1220 | Dict keyed by batch then camera; all configured cameras, including misses, needed to complete | Max member capture stamp is the high-water mark. No manager receipt timestamp retained here | No covariance combination. Latest complete batch can supersede another before a timer tick; 04 owns missing/superseded outcomes |
| Provisional IPM / manager 1319–1327, projection 92–138 | Same camera model selected by ID; outer batch identity remains in manager | Mean at original capture, map ground XY metres; Jacobian J in m/px | Replace incoming pixel R by commissioned pixel R; compute `J R_uv Jᵀ`, tiny numerical floor. No belief. Active reference profile later replaces this provisional R |
| Corrected camera reading / manager 1427–1509 | NN/camera residual offset/R keyed by same camera; frozen reference calibration binds NN hash | Capture-time robot ground-reference XY in `map_bev`, metres | Active `z=NN(raw_xy,bbox,score,camera)−b_i`; full constant corrected-residual R_i in m². No belief enters active mean/R. Optional hull uses nearest belief XY/yaw and transformed R; 05 owns that model |
| `MapObservation.quality` / 1508–1519 | `quality.camera_id` must match observation ID; batch is absent from this dataclass | Pixel-quality metadata is not map R. Query position at nearest belief, query time `now_s` | Active empty providers leave camera mean/R untouched. Bbox admission depends on belief; optional provider also overwrites R (F04). Quality fields are not Huber weights |
| Eligibility / camera_manager 283–319 | One candidate per camera | Age `now−capture`; future captures explicitly rejected; score `p_available×association×exp(−epistemic)×exp(−age/decay)` | Scalar eligibility, not precision or covariance. Active fusion does not weight by this score after admission |
| Common-time operands / manager 337–389,1651–1661 | Same camera ID, outer source batch; `original_by_camera` keeps immutable originals | `t*=max eligible capture`; `z_i*=z_i+d_i` in map metres | `R_i*=R_i+(s_d Δt_i)²I`, m². Translation with deterministic displacement leaves covariance axes unchanged; uncertainty added is a heuristic. No endpoint uncertainty, cross-camera motion covariance or prior cross-covariance represented. Belief fallback introduces prior-derived displacement |
| Bootstrap/metric gate / 1666–1728,492–555 | Candidate IDs retained; bootstrap largest agreeing subset, then used/rejected lists | All operands now labelled `t*`; componentwise median centre in metres; Euclidean residual in metres | No prior in median gate. Prior existence and per-camera bbox admission determine candidate membership. `nis_by_camera` here actually stores metres, not NIS. An empty-set identity covariance is a sentinel and is not published |
| Fusion normal equations / fusion 425–525 | Ordered paired `(observation,precision)` list; IDs accompany whole objects | Same common state, map metres. Residual e in m, R in m², precision in m⁻² | Huber distance/weights and n_eff dimensionless; A in m⁻², information vector in m⁻¹, B in m⁻². No robot P in these equations. Covariance is heuristic network report uncertainty |
| Bias/common terms / manager 1588–1616,558–563,1802 | Bias bounds from used camera ranges/bearings; shared term one per report | Range/bearing built from aligned target XY; ray floor rotated to map m². Optional propagation-to-now changes state instant again | Generalized floor, then optional motion-to-now covariance, then `sigma_common² I` once. Both physical terms disabled in selected configuration. R calibration/bounds need a non-overlapping statistical interpretation |
| Envelope and compatibility pose / 1794–1829 | Envelope carries source batch + used cameras. Pose topic cannot carry physical ID | Envelope `common_capture_stamp`, `correction_stamp`, `xy`, map frame; pose stamped at correction time | Same full 2×2 report R in both. Pose identity yaw with variance π² is noninformative metadata, not heading evidence. JSON serialization is not a transaction |
| Robot correction / robot 827–876,2373–2529 | Serialized seen-batch guard; envelope-only normal callback; optional per-camera stamps instead | Predict committed belief to correction instant; H=[I₂ 0], state m/rad | Innovation uses P and report R; Joseph covariance with unit active gain. This is the one recursive filter. Time-compensation/admission dependence is not represented as prior-measurement cross-covariance |
| Decision/terminal log / manager 1857–1885, logger 1684–1759, robot 2341 | Batch ID, camera rows and terminal status/reason | Capture `obs_stamp/xy/cov` separately from `common_capture_stamp/aligned_xy/aligned_cov`, fused values at fused stamp; receipt/apply stamps separate | Terminal records lack full posterior/revision. Final fusion rows omit some earlier-rejected candidates; camera opportunities are needed to reconstruct delivered inputs. 01/10 own exact assimilation-state association |

The two copies of per-camera values in the current success payload are correctly paired by camera ID. The actual-node probe verifies original `(1,2)` at 9.950 s and aligned `(1.0088,2)` at 9.990 s with distinct R values; fused time remains 9.990 s when propagation-to-now is off. No in-place overwrite of the original dataclass occurred. This preservation does not cure F01's unsupported displacement.

## What each configured rule actually computes

Let the admitted common-time observations be `z_i,R_i`, with SPD full 2×2 R in the same map frame. They are **measurements**, not robot posteriors. The manager first computes a componentwise median c and retains `||z_i−c||≤0.6 m`. This gate is not an all-pairs 0.6 m gate: two cameras 1 m apart each sit .5 m from their midpoint and pass. Bootstrap separately requires a mutually agreeing group. Gate semantics belong to 06.

| Rule | Mean and covariance, before optional bounds/additions | Zero/one/equal-quality limit |
|---|---|---|
| `best_single` | k minimizes `tr(R_k)`; report `(z_k,R_k)`; first item wins a trace tie | Empty standalone selector returns None; manager refuses empty fusion. One camera unchanged. Equal-quality tie is F06 |
| `distance_angle` | `g_i=abs(height_i)/sqrt(rho_i²+height_i²)/(rho_i²+1e−6)`; `w_i=g_i/sum g`; `z=Σw_i z_i`, `R=Σw_i²R_i` | One unchanged; all zero geometric scores fall back to equal weights. Equal weights/R give R/N only under independent errors. Geometry uses the reported common-time position; weights are data-dependent |
| `independent` | `J=ΣR_i⁻¹`; `R=J⁻¹`; `z=J⁻¹ΣR_i⁻¹z_i` | Empty raises; one algebraically unchanged subject to F08; equal R gives average mean and R/N. Off-diagonal XY covariance is retained |
| `joint_network` | Huber IRLS mean, followed by the A/B/n_eff covariance below | Empty raises; explicit one-camera fast path unchanged. Identical means and equal R give R, not R/N |

For `joint_network`, δ=2.5, maximum 25 iterations, change tolerance 10⁻⁹ m. Start at coordinatewise median. At each iterate:

```text
e_i = z_i − z
d_i = sqrt(e_iᵀ R_i⁻¹ e_i)
w_i = 1                    if d_i <= δ
      δ / d_i              otherwise
A   = Σ w_i R_i⁻¹
z   = A⁻¹ Σ w_i R_i⁻¹ z_i
```

The mean targets the convex objective `Σ ρδ(d_i)`, with `ρδ(d)=d²/2` below δ and `δ(d−δ/2)` above. Re-evaluate weights at the returned mean, then:

```text
B       = Σ w_i² R_i⁻¹ (R_i + e_i e_iᵀ) R_i⁻¹
n_eff   = (Σ w_i)² / Σ w_i²
R_joint = n_eff A⁻¹ B A⁻¹
```

This is the implemented conservative covariance approximation. It is not ordinary independent Gaussian precision addition, nor a statistically exact Huber sampling covariance: A is the reweighted normal matrix, not the full derivative of the radial Huber score, and n_eff is an explicit extra multiplier. Calibration must be earned separately.

Derivable limits, reproduced by the probe:

1. **Identical means, arbitrary R:** all weights are one, residual scatter is zero, so `R_joint=N(ΣR_i⁻¹)⁻¹`. Equal R reduces to R. This intentionally removes ordinary independent contraction for identical equal-quality observations.
2. **All residuals below Huber threshold:** mean equals independent GLS. `R_joint=N R_ind + N R_ind[Σ R_i⁻¹ e_i e_iᵀ R_i⁻¹]R_ind`, generally larger than `N R_ind` by a PSD scatter term. For equal R it is `R + (1/N)Σe_i e_iᵀ`.
3. **Unequal information:** an agreeing camera with `R→∞` still has w=1. Adding it can widen the report despite contributing negligible precision: `.01I` from one good view becomes approximately `.02I` with a second `10⁵I` view. n_eff counts robust scalar weights, not independent information content. This is a policy limitation to state/test, not evidence of an accidental normalization mismatch in the code.
4. **Extreme outlier:** with two zero means, one x=2 m, all R=.01I, raw Huber mean is x=.125 m versus independent x=2/3 m. Its influence is bounded, not zero. With the manager's 0.6 m median-distance gate, that outlier is removed before either rule. Raw-estimator robustness and the full manager gate must not be conflated. Large standardized disagreement saturates Huber score contributions; report covariance need not grow without bound with outlier distance.
5. **Robustness is conditional on R:** small assumed R increases a camera's score/precision even under Huber weighting. A median initialization does not guarantee an overconfident camera cannot dominate the optimum. Neither Huber nor a large ellipse detects a bias shared by every camera.
6. **Empty/rejected:** no eligible set emits a reasoned no-measurement decision, not a fused correction. `_gated_fusion` with all cameras rejected returns an identity-R sentinel plus empty used IDs; `_decide_fused` checks those IDs and never publishes it. Existing belief plus one accepted camera is legal; cold bootstrap still requires quorum. `bootstrap_min_cameras=1` is accepted at initialization but `_largest_agreeing_group` never returns a singleton, so it cannot actually enable one-camera bootstrap (06 handoff).

Optional floor construction is `F_i=T(beta_i) diag((a rho_i)²,(b rho_i)²) Tᵀ`, map-frame m². `apply_belief_floor(P,F)` whitens by F, raises eigenvalues below one, and maps back; it is a Loewner bound, not an elementwise maximum or addition. `combine_floors` folds this operation across used-camera bounds. `_fusion_report_covariance` finally adds `sigma_common² I` once. The active reference R **replaces**, rather than adds to, provisional projected pixel R, and no nonzero common/floor terms are enabled.

Adding a common-mode matrix is principled only if the remaining R terms exclude that shared component, or the resulting intentional conservatism is declared. For the independent model `z_i=x+b+v_i`, independent `v_i` with covariance D_i and common `Cov(b)=C`, the correctly constructed measurement has covariance `(ΣD_i⁻¹)⁻¹+C`. Feeding marginal `D_i+C` into fusion and adding C again includes some shared variance twice. Current residual-calibration R is not labelled as such a decomposed D_i. Joint scatter/n_eff also cannot be assumed to estimate C. No particular nonzero common-mode value is justified by these tests.

`covariance_mapping.py` exposes a different, optional pixel model: bounded interpolation `R_good+(1−tau)^gamma(R_bad−R_good)` or scalar precision blend `[tau/r_visible²+(1−tau)/r_miss²]⁻¹I`, with probability clipping in the latter helper. Its 40/120 px endpoint decision and trust/health gate are not active commissioned metric-R inconsistencies. Likewise, Student-t reweighting, information-gain selection and dynamic-occlusion replay in the requested tests are separately named algorithms; none calls the active joint estimator.

## Independence, prior reuse and the planner mismatch

Full off-diagonal **within-camera XY correlation** is handled correctly by the precision matrices and the independent NumPy checks. **Cross-camera error covariance** is absent from the input type. For fixed linear fusion gains L_i, the true covariance is `Σ_iΣ_j L_i Cov(e_i,e_j)L_jᵀ`; the usual `Σ L_i R_i L_iᵀ` drops the i≠j terms.

Synthetic equal-camera example: marginal R=.04I, common cross-camera C=.03I, N=3. True mean covariance is `.033333I`; independent fusion reports `.013333I`; joint identical observations report `.04I`. The last is conservative for this chosen model, but this example does not calibrate its general formula. Repeating a biased camera over time is another dependence dimension; keeping one-camera R within a batch does not prevent recursive contraction across batches.

If common-time compensation uses shared uncertain displacement U, aligned cameras share U and the robot prediction may share it too. With R=.04I, U=.0025I and three shifted observations, true average covariance is `R/3+U=.015833I`; treating each `R+U` independently gives `.014167I`. A nonzero prior/measurement cross-covariance also changes innovation and gain. With `C=Cov(prior_error,measurement_error)` under estimate-minus-true error conventions, `S=HPHᵀ+R−HC−CᵀHᵀ` and `K=(PHᵀ−C)S⁻¹`. Current interfaces carry neither those blocks nor temporal persistence. The drift inflation and robust n_eff factor do not encode them.

Active NN mean/R and `_gated_fusion` contain no robot prior, so there is **no ordinary two-KF posterior cascade** on this selected path. Dependence still enters through belief-conditioned bbox admission, optional belief-displacement fallback, shared measured motion, and persistent camera calibration/interpretation error. Optional hull linearization has a local prior-dependent equivalent measurement; its exact assumptions are documented by 05. Optional per-camera quorum reanchoring reuses evidence in the robot recursion, confirmed by 01 R11 and left to that owner.

The current planner has two deliberately distinct interfaces ([camera_network.py:120](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/core/camera_network.py:120), [139](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/core/camera_network.py:139)):

- IWAI score proxy: `R_proxy=[Σ(s_i R_i⁻¹+(1−s_i)R_miss,i⁻¹)]⁻¹`, then projected into one fixed cost chart. The s_i are expected detector scores, not usable-hit probabilities. The finite miss endpoint contributes precision even at score zero. Belief sigma points average scores; this is a designed cost proxy, not an actual observation.
- Separate q/R forecast: independent hit/miss branch averaging, or expected information `P_info=[P⁻¹+Σ q_i HᵀR_i⁻¹H]⁻¹`. It omits runtime median gating, Huber weights, n_eff/scatter, motion-support refusal and shared errors. The existing IWAI horizon is not a recursive simulation of realized camera corrections; 08 owns that forecasting distinction.

Even at all scores/probabilities one, equal cameras and agreeing means, the planner's precision sum gives R/N while the runtime gives R. With unequal cameras it gives `R_ind` versus at least `N R_ind` when Huber weights stay one. Availability, event cadence, common-time motion and gates introduce further differences. **Do not “repair” this by inserting a blanket division by N into runtime or multiplying every forecast covariance by N.** First choose the measurement model and intended planner claim; residual-dependent robust covariance cannot generally be forecast from q and R alone. This is a documented model limitation with confirmed equations, not a discovered accidental use of robot P inside fusion.

## Matched robust-versus-direct checks

Mathematical equivalence is expected for the same admitted simultaneous observations with common linear H, fixed R_i, conditionally independent errors independent of the robot prior, unit Kalman gain, no relinearization/gates/recovery/bounds between cameras, and Q applied only once before that instant. The information identities are:

```text
P+⁻¹ = P−⁻¹ + Hᵀ(ΣR_i⁻¹)H
P+⁻¹ m+ = P−⁻¹ m− + HᵀΣR_i⁻¹z_i
```

The numerical probe uses a full 3×3 P with position-heading cross terms, a common initial mean, three unequal correlated 2×2 camera R matrices, the same linear XY model and a fixed Q. The actual robot callbacks produce a maximum absolute numerical covariance difference **5.20×10⁻¹⁸** between direct cameras and one independently fused correction. Replay intervals are `.05,0,0` versus `.05`: same-time B and C add information without another motion/Q step. Repeated direct delivery changes neither mean nor covariance. The standalone Joseph/info oracle also agrees.

For exactly those events, robust fusion yields a different posterior. Planar covariance entries are:

| Mode | P_xx (m²) | P_yy (m²) | Final XY (m) |
|---|---:|---:|---|
| Direct / independent fusion | .004311139 | .003409364 | (.014248975, .011591028) |
| Joint robust report | .010988507 | .008644363 | (.015369766, .006866101) |

These are deterministic filter outputs, not localization accuracy. Even if both aggregators produce the same measurement mean, their different R changes the robot gain and thus its posterior mean.

Equivalence is inappropriate when:

- Individual NIS gates are applied sequentially. Synthetic x=±.25 m, R=.01I and prior P=.05I: the first camera passes (NIS 1.0417), the second fails (11.4583); reversing order flips the accepted mean. The node sorts camera IDs deterministically, making arrival permutations reproducible but not making a sequential gate equivalent to a joint gate. The robust manager's metric gate uses different information.
- Captures are staggered with process noise. Scalar example P0=.1, R=.04, interval Q=.01: update old camera, predict, update new camera gives final variance .01963636; naively moving old R to R+Q, predicting the prior with Q and treating common-time observations as independent gives .01848739. Shared process uncertainty must be retained for an exact common-time construction. Deterministic known motion with Q=0 can restore equivalence after correct alignment.
- Mean/R depend on the current posterior or data-dependent robust/geometric weights; shared camera errors or temporal persistence exist; reanchoring, heading replacement, inflation or floors occur between updates; membership or timing differs. Bootstrap is also a distinct case because it may set rather than update the prior.

A fair next experiment must freeze exact observation events/IDs, capture times and arrival order, initial belief, mean model, camera R, Q and common motion history. First reproduce the live baseline decisions on those events. Then separately compare aggregation with common gates versus the intentionally different per-camera gate policy. Capture-time replay is an idealization unless it reproduces processing delay and refusal policy. The present direct wire path needs F03's identity/frame/terminal-record repairs before it can support an evidence-grade live comparison.

## Registered event and regression status

The trace is the earliest multi-camera published event in registered `fusion_network_traverse__P0__seed210`, run `experiment_20260906_213015`. The selection hash is `66b928fbaf8f470332acff23922003822fcb58113d4bd085180ade5c54c0cfd0`. The probe fails on changed required-file hashes and does not infer a run by directory age. `aligned.observations` and `aligned.assimilations` supplied the records.

Source batch is `strict:camera_A@3400000000,camera_B@3400000000,camera_C@3400000000,camera_D@3400000000,camera_E@3400000000`. The final candidate/used list contains B and C; A/D/E's absence here does not classify all of their earlier opportunities.

| Stage | B | C |
|---|---|---|
| Capture/common stamp | 3.400 s / 3.400 s | 3.400 s / 3.400 s |
| Camera XY, map metres | (−7.886615475, −8.633010424) | (−7.919193896, −8.618793378) |
| Full camera R, m² | [[.033313104,.012617475],[.012617475,.279090819]] | [[.137744246,.040883670],[.040883670,.059743370]] |
| Common-time values | Exactly the capture mean/R | Exactly the capture mean/R |

Reconstructed `joint_network` output is `z=(−7.892793496,−8.614564967) m`, `R=[[.053725481,.016534396],[.016534396,.086262108]] m²` at 3.400 s. Mean and R exactly match the logged fused values. Decision receipt and terminal apply time are 3.802 s; exactly one terminal record for this event says `accepted_bootstrap`, belief stamp 3.400 s. This is a software/interface trace without a ground-truth comparison or an inferred immediate posterior.

| Recent defect | Reverification in this audit | Remaining boundary |
|---|---|---|
| Adjacent 0.20 s capture rounds fused as simultaneous | 04's strict batching regression passes; manager 0.05 s window excludes the older round | Timestamp tolerance is not a physical round/epoch identity; 04's conditional mixing/reset cases remain |
| One invocation reused on manager ticks | 04's repeated-delivery/60-tick actual-method test passes | Unidentified compatibility mode, conflicting duplicates, partial publication and restart differ |
| Distinct second camera at same time discarded | Actual-node comparison and existing tests accept all distinct same-time cameras once | 01 R13 reports a remaining 0.5 ms strictly-later rejection; physical batch identity is absent in direct mode |
| Singular one-axis bias floor | Existing tests and numerical probe verify rejection; manager now checks both slopes positive or both zero at startup (935–945) | Optional three-floor order, runtime placement, invalid-P flooring and extreme numerical scale are separate issues |
| Shape-valid bad envelope | Actual callback rejects bad frame/schema 99/asymmetric/indefinite R before filtering | F03/F10 and 05 cover distinct missing semantic checks; 01's bootstrap bypass remains |

Reproduce the pinned audit from repository root, including all 150 regressions and exact JSON comparison for the three probes:

```bash
python3 docs/module_audits/07_reproduce_snapshot.py
```

The original commands below apply to the recorded source snapshot. On later source, changed fixtures or repaired-defect assertions can legitimately fail; do not overwrite baseline evidence with a mixed-source result.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/reliability/test_fusion_v2.py tests/reliability/test_fusion_arms.py tests/reliability/test_dynamic_occlusion_fusion.py tests/reliability/test_camera_manager_node.py tests/reliability/test_bias_floor.py tests/planning/test_planner_node_per_camera_correction.py tests/planning/test_runtime_transactions.py tests/perception/test_camera_acquisition_audit_04.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/07_fusion_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/07_fusion_node_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/07_registered_event_probe.py
```

All three probes completed with their independent assertions passing. They intentionally retain observed-defect outputs. The registered probe reads the selected run; other probes use synthetic events. Real DDS loss/backpressure, process restart/durability, simulated clock rollback, altered-spawn-frame integration and live staggered-capture frequency were not exercised here. Cross-camera/persistence covariance and the robustness model's statistical calibration remain unmeasured by this audit.

## Handoffs and repair order

1. **01/02/10:** close anonymous bootstrap and state/outcome commit gaps; use explicit revision/identity to score the terminal state. Coordinate with the already reported logger equal-clock/event-scorer defects.
2. **07 with 01/02/05:** make common-time displacement causal, supported and frame-correct; retain capture evidence and the displacement model. Keep shared uncertainty visible as a model limitation until a justified representation exists.
3. **04/13:** retain ownership of producer batch membership, conflicts, epochs, bounded pending sets and shutdown/restart semantics. Fusion must consume an immutable validated batch, not infer physical identity from numeric closeness.
4. **07/06/01:** repair direct-camera frame/identity/evidence, optional GP-R overwrite and floor semantics before enabling those branches. Keep statistical admission/recovery changes distinct from transport fixes. Hand the nonfunctional min-one bootstrap configuration and per-camera refusal differences to 06.
5. **07/08:** preserve the current robust baseline with its actual equations. Decide which matched estimator/forecast model is being claimed, then evaluate independent precision, robust aggregation and shared/persistent-error alternatives on identical events. No new planner tuning or covariance scaling follows from a unit-test pass.

No runtime repair is claimed by this report. The ranked reproductions make the next changes concrete while preserving the frozen experimental baseline.
