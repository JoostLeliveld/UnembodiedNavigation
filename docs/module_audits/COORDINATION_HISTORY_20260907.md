# Audit completion and implementation queue

Coordinator: **Investigate issue-prone modules**, thread
`01a07849-8bf2-72c1-8c4f-66e3de7b1533`.

Initial coordination snapshot: 2026-09-06, 21:04–21:09 UTC (23:04–23:09 Amsterdam).
Latest monitoring update: 2026-09-06, 21:53 UTC (23:53 Amsterdam).
This is a changing work queue. Audit findings describe the source snapshots recorded in
their reports; recheck the current source before changing a reported defect.

## User priority and resumed work

02's [current-source handoff](02_current_source_handoff.md) is now complete, with
all callback/timer and seven recursive writer sites mapped. It records a final repeat
of 19 behavior probes and 90 focused tests on stable unicycle `b3e78d48b56b84f8` and
EFE `c2546f81863703e1`. Packages A/B remain verified in their stated scopes. The next
runtime sequence is command/goal atomic ingestion and invalid-age handling, followed
by coherent belief epoch/revision/frame/target/identity commit and publication ordering.
01 owns validation of held predictions before rejected-outcome commits and remaining
motion-support/capture-time heading work. Shared-file ownership still applies; the
handoff is a verified repair plan, not implementation of these remaining changes.

08 has now delivered its final audit and scoped repair handoff. It reports 139 focused
current tests passing; final source/evidence identities are in
`docs/module_audits/08_final_verification.json`, with baseline and current snapshots
kept separate. LOCAL/direct stop, age and missing-context checks and Q-cache identity
are verified repaired. The CasADi diagonal-solve reload failure has a minimal reproduction;
the current replacement preserves value/gradient and reload behavior. Prior uncertain
cache-failure status below is superseded by this handoff, not by an assumed transient pass.
Remaining planner priorities are GLOBAL validity/completion/appended-connection admission,
coherent goal/belief/config/epoch request identity, and complete finite-result validation.
The expiry-side-effect concern is API/inspection evidence only under the live single
planning callback; do not present it as a confirmed overlapping-solve failure.

02's current-source timing handoff reports 90 focused A/B/runtime/encoder regressions
passing and 19 deterministic probes reproducing remaining defects. Source was stable
during those probes: unicycle `e6566d3ef0956f58`, EFE `f3750f4a4dcb0969`, motion
`ac5ea439b956f342`. Evidence is in
`experiments/runtime_integrity/timing_callbacks_revalidation_20260907/`.
Priority remaining mechanisms are older-after-newer belief publication; equal-time
corrections ignored by the manager; mixed state/anchor/frame snapshots; incomplete epoch
reset with stale in-flight publication; command receipt-history reordering; and goal
message/progress-signature mismatch. A diagnostic publisher exception also commits a
correction without retaining its batch ID/terminal record, despite invoking fatal stop.
This demonstrates an accounting/transaction defect, not proven double assimilation.
Coordinator forwarded the handoff to 01/10/15. 02 is finishing the ownership map and
sequenced repair plan; no production repair of these newly verified cases is claimed.

14 has explained the profile mismatch and saved its
[final simulator report](14_simulator_world_and_physical_outcomes.md): frozen hash
`3559f0a78b` matches git HEAD; live `742cb32c69` adds only a separate low-CPU profile.
Its comparison found the consumed original camera/world mappings unchanged. The failure
is whole-registry provenance drift, not demonstrated historical geometry corruption.
Historical loaders still need a sound repair: verify/use the exact frozen configuration
bytes, preserve its manifest hash, and report live-source drift separately. This remains
a proposed repair, forwarded to 13 for coordination with 12/14; do not bypass validation.
The low-CPU world's changed physics/contact rates are a separate performance treatment,
not an equivalent baseline merely because camera geometry is unchanged.

Refactor's subsequent full-suite handoff reports **1066 passed, 4 skipped, 2 failed**,
and a passing five-package colcon build. Besides the existing machine-local-path failure,
`tests/test_icra_geometry.py` now detects a mismatch between current
`src/experiments/config/world_profiles.yaml` and the hash locked in the frozen
`warehouse_v2_bbox_characterization_20260831/capture_manifest.json`. Refactor reports
changing neither provenance file. Coordinator requested joint 13/14 investigation of
the actual diff, current loaded geometry and correct reconciliation. Preserve the frozen
capture hash; do not update it to silence the test. Cause and geometry impact remain
unverified. The suite/build counts are owner-reported and do not close this blocker.

08 current-source handoff after resumption identifies a priority runtime gap: on
`f9e1a312` plus the concurrent tree, GLOBAL route installation bypasses the new tape
installation gate. Controlled checks still admit partial/invalid/nonfinite or obsolete
routes after stop/clock changes. The direct path's ordinary-stop, age and missing-context
repairs pass their checks; changed-belief/goal/config revalidation remains open there.
Evidence: [08 current checks](08_current_result_checks.json), with recorded source hashes
and no drift during that probe. Its 19 behavior groups include reproductions of defects,
not 19 acceptance passes. The coordinator forwarded this priority to 03 and 15; 08 is
finishing its report and 02 owns a newly noticed expiry-side-effect window. Queue F1
therefore includes stop/clock request validity at GLOBAL installation, not only geometry.

After the usage-limit snapshot below, the user confirmed new capacity and authorized
reviving the interrupted chats. **Correctness of the entire unicycle runtime and state
update chain is now the first priority.** The coordinator resumed 02, 08 and 09, and
requested a current-source verification handoff from 01. Simulator/integration,
performance and refactor chats were already active on inspection; logging and campaign
chats had returned to idle. Earlier usage-limit rows below are historical until their
next completion handoffs, not a claim that those chats remain unavailable.

01 and 02 must reconcile every callback and recursive m/P/time/frame write, initialization,
motion support, delayed/reordered corrections, rejection/drop outcomes, clock resets,
publication ordering and planner snapshots. Their immediate scope is verification and
sequenced remaining repair packages; shared mutations require coordination with the
active refactor owner. Integration 15 and the refactor owner received this priority.

Refactor ownership is now explicit: reliability fusion/contracts/bias-floor/silhouette
and manager delegation; belief-correction input validation; typed tracker safety and
EFE controller dispatch; unicycle configuration parsing and per-camera frame/error
handling; launch/campaign boolean validation. The owner excludes timing, epoch,
publication and clock ordering but requests findings before any overlapping production
edit. This handoff was forwarded to 01/02. Its attempted shared-parser patch failed
atomically; no common/state/logger changes from that patch should be assumed present.

Runtime **source** is already changed by packages A/B. The current uncommitted unicycle
diff additionally centralizes local-controller configuration validation. None of those
facts proves that an already running ROS process loaded the changes, or that the entire
combined runtime has passed acceptance. Do not equate a commit, an audit or a test count
with deployment or complete update correctness.

The user's current request is to identify implementable work and monitor audit completion.
This coordinator has requested bounded repair plans and ownership information, not a
repository-wide rewrite. Runtime edits underway elsewhere keep their existing authorization.

## Scope, checkpoint and operating constraints

- GitHub checkpoint: `3c4ddeae4a7427bf374514dad6a1d65dc728a91c` on
  `cleanup/remove-dead-filter-interventions`. It contains the code and reports available
  at checkpoint time. Later changes and ignored raw logs/models are outside that commit.
- Later local commits are now present: `5e844a89` (state-estimation package A) and
  `f9e1a312` (command package B). This monitor verified their existence and read their
  implementation notes; it did not create them or verify that they were pushed.
- **Advance ICRA Localization Paper** confirms its last live experiment completed and
  was cleaned up. Reporting is also complete; it has released all source/document
  ownership. Frozen experiment identities remain unchanged by subsequent repairs.
- Cameras continue observing during turns. Artificial missing observations in a test
  are fault-injection fixtures, never an operational camera-disable rule.
- Preserve the detector, NN, mean correction, R, Q, fusion model and experiment settings
  while fixing a stated implementation invariant. Any changed timeout, recovery, noise
  cadence or availability decision needs its actual semantics and evidence recorded.
- One owner edits a shared production file at a time. Reports/probes can proceed in
  parallel under their numbered filenames. Thread activity is not audit completion:
  a completed audit can have an active planning or repair follow-up.
- No broad new navigation campaign is scheduled by this coordination request. Component
  regressions and complete-chain acceptance must describe their actual source identity.

## Audit status

At this snapshot, **13 of 15 main reports are available: 01–13**. Newly saved reports
09 and 13 contain ranked findings and recorded verification. Their chat turns, and the
final follow-ups for 08 and 10, stopped at an account usage limit before a final completion
handoff. Reports 14 and 15 remain absent; saved probes and the draft integration matrix
do not establish final acceptance. Completed earlier audits keep their completed status.

The 21:49–21:53 UTC pass found the same usage-limit error in 02's follow-up, 08, 09, 10,
13, 14, 15, **Speed Up Slow Runs**, and **Refactor loose pivot code**. No retry messages
were sent into that known limit. Existing requests remain pending; this is an execution
blocker, not a finding that the underlying audits or repairs failed their tests.

| ID | Exact thread title | Thread ID | Audit / follow-up status | Report |
|---|---|---|---|---|
| 01 | Audit Belief Correction Pipeline | `01a0785f-350d-7bc0-a398-e6767f5cf232` | Audit complete; package A now implemented in local commit 5e844a89 | [01](01_state_estimation.md), [implementation](01_repair_package_a_implemented.md) |
| 02 | Audit timing and belief ownership | `01a07860-74a5-7882-a07a-c7e039575911` | Original audit and current-source handoff complete; remaining repair sequence delivered | [02](02_timing_and_callbacks.md), [current handoff](02_current_source_handoff.md) |
| 03 | Trace command lifecycle regressions | `01a07862-d60a-7711-bed0-786172f4f299` | Audit complete; command package now implemented in local commit f9e1a312 | [03](03_command_execution.md), [implementation](03_command_execution_repair_implemented.md) |
| 04 | Audit camera batch identity flow | `01a07864-74ac-71d3-98bb-b3203ccc6636` | Audit and acquisition repair complete; final component QA delivered, shared files released | [04](04_camera_acquisition_and_batching.md) |
| 05 | Audit camera calibration pipeline | `01a07865-4cd1-7381-9304-3f0668af55be` | Audit complete; 04 manager handoff available, future manager edits must be sequenced | [05](05_observation_geometry_and_calibration.md) |
| 06 | Audit observation admission policy | `01a0786c-1500-7c80-a666-015ad4a8957b` | Audit complete; correctness overlaps deduplicated, policy proposals held separately | [06](06_admission_bootstrap_recovery.md) |
| 07 | Investigate multi-camera fusion | `01a0786d-25a9-7662-992d-82745b1e3895` | Report available; latest follow-up turn completed | [07](07_multicamera_fusion.md) |
| 08 | Audit planner future model | `01a07871-a04d-7442-92cb-db37b1b1427f` | Final audit complete; current repairs verified, scoped remaining repairs delivered | [08](08_planner_and_future_camera_model.md) |
| 09 | Audit route tracking regressions | `01a07872-4398-7733-a7ff-87d1712068c2` | Report saved with verification; completion handoff interrupted by usage limit | [09](09_tracking_routes_termination.md) |
| 10 | Audit runtime logging pipeline | `01a07872-f2d5-73b1-b076-23fbbeaea2db` | Report and repair outline delivered; final QA/handoff interrupted by usage limit | [10](10_logging_and_event_accounting.md) |
| 11 | Audit offline alignment pipeline | `01a07873-9e1b-7b52-8335-a7642d26a613` | Audit complete; shared-validator plan delivered, no production edits | [11](11_alignment_replay_and_reporting.md) |
| 12 | Audit capture integrity pipeline | `01a07874-6b80-7eb1-ba2e-48b72726c579` | Audit complete; final source boundary recorded, no production/model changes | [12](12_capture_commissioning_and_export.md) |
| 13 | Investigate visibility campaign flow | `01a07875-10cf-73c0-bbea-437f88f1319a` | Report saved with verification; completion handoff interrupted by usage limit | [13](13_configuration_provenance_campaigns.md) |
| 14 | Audit simulator integrity | `01a07875-e61a-7b41-a651-553b41b0c935` | Interrupted by usage limit; simulator/transport/asset evidence saved | Final report pending |
| 15 | Run final integration audit | `01a07876-c011-7ab0-a5a9-805cfd242bbe` | Interrupted by usage limit; draft matrix and source snapshots saved; no final verdict | Final report pending |

## First implementation queue

“Candidate” means a concrete invariant and reproduction exist; it does not mean the patch
has been implemented, reviewed or tested on today's final combined source.

| Queue | Proposed change | Existing evidence / duplicate findings | Owner and dependencies | State |
|---|---|---|---|---|
| A | Complete camera batch receipt, validate result counts, record failures and prevent replay after partial publication | 04 F01/F02/F05; overlaps 07 publication identity | 04 handoff complete; shared production files released | Implemented by 04; 101 focused tests passed, combined acceptance pending |
| B1 | Replay the exact immutable motion snapshot that passed the coverage check | 01 R02, 02 T02-05, 06 D02 | Implemented package A; 02 publication work must build on its motion API | Local commit 5e844a89; combined integration pending |
| B2 | Permit mandatory-envelope initialization only through the identified correction callback | 01 R03, 02 T02-02, 06 D01, 07 F02 | Implemented package A; 10 still owns durable terminal event accounting | Local commit 5e844a89; optional startup branches remain outside this repair |
| B3 | Make odometry sample ingestion chronological/atomic and reject old encoder input before changing its integrated clock | 01 R01/R06, 02 T02-01 | Implemented package A; 02 review of remaining interleavings/support | Local commit 5e844a89; missing-motion validity remains open |
| C1 | Retain distinct fusion decisions received in the same clock tick, using physical identity; persist raw envelopes and 04 batch outcomes | 01 R08, 06 D11, 10 L10-01/L10-04/L10-05 | 10 proposed logger owner; 07 producer identity and 04 outcome schema | First package outlined; final source QA pending |
| C2 | Stop/drain before final summary, attempt every evidence-file closure, and mark completion only after durable summary write | 10 L10-02; 15 preliminary harness | 10 proposed owner; coordinate producer drain and terminal event accounting | First package outlined; final source QA pending |
| C3 | Use one raw correction-ledger validator before truth filtering or scoring | 11 A11-02; 10 L10-01; 15 probes | 11 proposed owner; 10 schema coordination; 13 run-validity reuse | Repair-ready API plan in final 11 report |
| C4 | Separate public-belief scoring from committed-posterior scoring; report posterior unavailable when its terminal evidence is absent | 01 R04/R07, 10 L10-06, 11 A11-01 | 11 consumer repair; 01/02/07/10 coordinate future terminal m/P/frame/time/revision schema | Consumer repair proposed; complete posterior ledger is a separate shared-schema task |
| D1 | Cancel an in-flight command result when an ordinary stop occurred after computation started; enforce existing age checks at both install paths | 03 C03-01/C03-05; 15 reproduction | Implemented command package; 08/09/15 must revalidate their baseline installer probes | Local commit f9e1a312; physical stop delivery and belief/goal revision policy remain open |
| D2 | Prevent older belief predictions overtaking newer publications; carry coherent anchor/frame/revision metadata | 02 T02-03/T02-06/T02-07/T02-08 | 02 after B; 10/11 agree revision event contract; manager consumer after 04 | Candidate; sequence cross-file schema work |
| E1 | Validate bbox/pixel consistency, image/calibration identity and artifact/projection agreement before producing a measurement | 05 G01/G02, 06 D10; some 04 member-stamp overlap | 05 proposed owner; 04 handoff available; 12/13 metadata coordination | Concrete candidate; no new empirical admission threshold |
| E2 | Compute common-time displacement from supported, frame-labelled motion rather than nearest corrected belief endpoints | 07 F01, 06 P05, 01/02 motion-support overlap | 07 after B motion API; 04 handoff available, serialize manager changes with 05 | Report available; scope/unsupported-path semantics need explicit agreement |
| F1 | Validate global solver success and route segments before accepting a goal connection | 08 P08-01, 09 T09-05 | 08/09 after command package; performance/refactor source handoff remains pending | Candidate; current global route boundary needs revalidation after f9e1a312 |
| F2 | Apply capture preflight to the actual resume/writer path, and verify source/artifact hashes at export | 12 C12-01/C12-03; 05 artifact identity and 13 provenance overlap | 12 proposed owner; 13 resolves shared provenance contract | Final report delivered; repair candidate, not yet implemented |
| F3 | Bind each campaign attempt to its own summary; validate complete evidence before resume or successful ledger finalization | 13 C13-01/C13-02/C13-06; 10/11 ledger checks | 13 after performance runner handoff; reuse C3 validator rather than another validity definition | Concrete candidate from saved report; final audit handoff interrupted |
| F4 | Use one resolved configuration for launch, manifest and resume; validate supplied keys and exact loaded artifact identities | 13 C13-04/C13-05/C13-07/C13-08; 05/12 artifact identity | 13 primary; 05/12 loaders and 10 manifest coordination | Candidate; preserve declared values and record previously ineffective overrides |
| F5 | Require usable operational belief before waypoint progress or goal success; invalidate stale readiness/holds | 09 T09-02, 02 T02-08 | 02/09 shared validity contract; 10 logger and mission consumers | Candidate; coordinate time/frame/revision semantics, no truth-based online decisions |
| F6 | Prevent a clearance recovery tolerance accumulating anew on every replan; reject nonfinite geometry results | 09 T09-01; 08 P08-08 | 09 with current tracker/refactor owner | Baseline reproduction; revalidate against ControlSafetyResult edits before assigning a patch |
| F7 | Stamp mission goals in ROS time and preserve zero-centred corridor seeds | 09 T09-09/T09-10 | 09; mission clock repair coordinated with 02, lane seed case independent | Smaller candidates; zero-centred route defect is optional, not active pilot diagnosis |
| F8 | Preserve campaign cleanup scope and ledger history across startup failure, malformed history and concurrent writers | 13 C13-03/C13-09/C13-11 | 13 primary; 04 launch shutdown and 10 finalization coordination | Candidate; no permission to terminate unrelated processes |
| F9 | Simulator infrastructure and final chain acceptance | 14/15 saved evidence | Their existing owners after usage recovery and component repair handoffs | Final reports pending; do not infer acceptance from saved probe counts |

### Why B1, B2, B3 and C1 are suitable early candidates

The coordinator directly re-read the source during the initial 21:04–21:17 pass.
The first three mechanisms below are now addressed by package A; this preserves the
reason for the repair and must not be read as a claim that the old code remains current:

- `_advance_belief_over_outage` copies and validates history, then calls a predictor
  that takes another history snapshot. Using one snapshot repairs the demonstrated race
  without increasing the replay limit or altering the stochastic model.
- `_resolve_state_belief_ekf` can still initialize from `state_msg` before its identified
  envelope. Removing that bypass in mandatory-envelope mode enforces the existing contract.
- `_odom_cb` publishes parts of a sample outside the history lock and appends input in
  callback order; the encoder overwrites `_last_stamp` before rejecting nonpositive dt.
  The input clock and accepted sample must change together. Skipped-gap recovery is a
  separate choice and should not be silently redesigned in that patch.
- `_fusion_decision_cb` suppresses rows on `receipt_stamp <= last_receipt_stamp`. Two
  distinct source batches at one simulation tick therefore lose evidence. Receipt time
  belongs in a field; event identity controls duplicate handling.

These are code/reproduction findings, not new measurements of a drive's accuracy.
Owners must rerun the independent reproductions against their proposed patch.

### Repair packages and implementation handoffs

- [01 package A plan](01_repair_package_a_plan.md) specified four narrow changes and
  independent regressions. The [implementation handoff](01_repair_package_a_implemented.md)
  and local commit `5e844a89` now record those changes. An immutable snapshot includes
  both odometry and command fallback histories.
  Refusing late odometry must not erase an accepted sample or move its watermark. The
  encoder's existing positive-large-gap rebase must continue to permit later valid input;
  this chronology repair does **not** establish complete motion support across that gap.
- [03 command package](03_command_execution_repair_package.md), now implemented in
  `f9e1a312` with an [implementation handoff](03_command_execution_repair_implemented.md), uses an ordinary-stop
  generation and final locked installation check for both execution paths. A cancelled
  old result must not erase a newer replacement. A fresh request after an ordinary stop
  must still work. Existing plan-age limits are enforced without implicitly enabling
  latency skipping. Proposed production scope is `efe_agent_node.py` with a separate
  command-installation regression file. Later work must preserve its shared
  `_install_control_tape` transaction and use current source, not the older plan hashes.
- [11 final report](11_alignment_replay_and_reporting.md) includes the proposed
  `validate_correction_ledger(publications, outcomes)` API and raw-ledger checks before
  reference filtering. Owner reports 59 synthetic cases and 17 existing regressions,
  with 21 inspected source files unchanged during final verification. This is an audit
  and a repair plan, not an implemented validator or a navigation-accuracy result.

Package A's note reports 1015 passing tests and 4 skipped; package B's note reports
1036 passing and 4 skipped, including 21 new installation regressions. Both acknowledge
a pre-existing `test_no_machine_local_paths.py` failure. Package B also records mutation
checks for cancellation and expiry. These are implementation-note claims, not tests
rerun by this coordinator or a final source-homogeneous integration verdict.

At 21:53 UTC, the current package A source hashes matched its recorded post-repair
prefixes: planner node `8aae15c44861`, motion history `ac5ea439b956`, encoder
`f48cbb9dee41`. The current EFE node hash was `48b27fca909d`. Shared refactor/performance
changes exist alongside these commits, so recheck identity when resuming verification.

### Next implementation ordering after A and B

**User-priority override:** the current 01/02 runtime handoffs take precedence over the
earlier broad ordering below. First close atomic command/goal ingestion, invalid-time
handling and invalid held-prediction commits; then coordinate coherent belief snapshots,
epoch/revision identity and publication ordering with durable correction accounting.
GLOBAL route admission remains a parallel planning priority under its own file owner.
No arbitrary camera suppression, new recovery thresholds or automatic cancellation on
every valid camera update is implied by that sequencing.

1. Complete logger event identity/finalization and the shared raw-ledger validator (C1–C4).
2. Reconcile the shared refactor/performance changes, then validate global route admission
   and operational-belief consumers (F1/F5). Preserve the new stop/age transaction.
3. Make campaign attempt identity and resume validation use that same evidence contract
   (F3/F4/F8); finish capture writer/export integrity (F2).
4. Continue coherent belief publication, supported common-time motion, and the remaining
   simulator/integration work with their stated API and ownership dependencies.

This orders the remaining candidates; it does not assign new runtime edits under the
monitoring request or silently adopt the separate policy/model proposals below.

## Already owned changes and shared files

### Audit 04 — completed patch and released ownership

Owner reports **101 focused tests passing**, final QA complete, audit addendum delivered
and all shared production files released. The coordinator read the passing test transcript
at [04_repair_test_results.txt](04_repair_test_results.txt); source/test identities are in
[04_repair_source_sha256.txt](04_repair_source_sha256.txt). This is component evidence,
not a combined-system acceptance verdict.

Released production files:

- `src/perception/perception/core/four_camera_batch.py`
- `src/perception/perception/nodes/batched_four_camera_yolo_node.py`
- `src/reliability/reliability/source_batch_buffer.py` (new)
- `src/reliability/reliability/nodes/camera_manager_node.py` (receipt/decision boundary)
- `src/experiments/experiments/core/visibility_launch_common.py` (node-exit handling)

Reported changes: per-chunk result validation and physical-call IDs; detector session/cycle
IDs and outcomes; bounded buckets with explicit eviction, steady expiry and missing-camera
liveness; bounded manager receipt with conflict/timeout abort and replay prevention;
preclaimed decision identity; explicit backwards-clock stop requiring coordinated restart.

The patch also introduces explicit overflow/transaction policy: 32 detector buckets,
64 pending manager batches and a 2-second manager receipt timeout. Review these recorded
semantics in integration, including delivery/slow-inference cases. The owner
states 5 Hz/all-five scheduling, weights, selection, R/Q and fusion mathematics remain
unchanged. Durable CSV accounting for the new outcomes remains with 10; currently they
are published as topics and process logs. Optional detector paths, real producer capture
round IDs and coordinated hot reset remain outside this patch. The new
`source_batch_buffer.py` and `test_source_batch_buffer.py` were untracked at handoff and
must be included when this repair is committed.

### Other active work

| Exact thread title | Thread ID | Current role / edit overlap |
|---|---|---|
| Advance ICRA Localization Paper | `01a06ff2-1a5a-7fd2-b387-0e59f08cce46` | Live run, cleanup and reporting finished; all ownership released |
| Speed Up Slow Runs | `01a0787c-ebd2-7572-930b-b75d27a57abd` | Authorized cache/JIT, duplicate traffic and runner edits; turn interrupted by usage limit, final ownership/test handoff pending |
| Refactor loose pivot code | `01a07886-2432-7422-89ca-4a698d07c8c2` | User subsequently authorized fixes; shared fusion/filter/geometry/tracker edits exist, turn interrupted by usage limit, final handoff pending |

`unicycle_planner_node.py`: package A is present; sequence 02's publication work and
coordinated 10 event payload changes on top. `efe_agent_node.py`: command package B is
present; resolve the refactor/performance handoff before 08/09 changes.
`camera_manager_node.py`: 04 released its patch, but the later refactor now overlaps;
resolve that handoff before 05/07 modifications. Research-status document/registry ownership
has also been released, while the frozen run/metric contract still applies.

08 detected concurrent cache/JIT and expanded Q/H/R cache keys plus status/runner/launch
edits. The experiment owner explicitly did not make these changes. 08 has since
acknowledged performance ownership and contacted **Speed Up Slow Runs**. Exact shared-file
release is still pending. Expanded Q/H/R keys fix the baseline omitted-noise cache identity
bug. The initial combined suite had **53 passed and 2 failed** on cache reload.
The coordinator read [08_followup_regressions.txt](08_followup_regressions.txt): both
`FunctionCache.load` regressions return no loaded function after CasADi deserialization
failure. Later [isolated tests](08_cache_isolated_regressions.txt) passed all 4 cache
tests, and the saved [combined repeat](08_cache_combined_repeat.txt) records **59 passed**.
The failure is therefore not reproduced in that later transcript; its cause and exact
source/test transition still need the interrupted owners' final handoff. Do not call the
cache broken solely from the old output, or call the full runtime accepted from the repeat.

The refactor chat's latest visible work extracts fusion admission to
`measurement_fusion.py`, removes fabricated measurements on rejected batches, separates
distance residuals from NIS, hardens malformed filter inputs and per-camera frames, and
uses `ControlSafetyResult` rather than diagnostic wording for rotation recovery. It reports
701 combined planning/reliability tests passing before the interruption. These changes
overlap 05, 06, 07 and 09 baseline findings. They are not final handoffs and must not be
duplicated or marked wholly verified by this monitor. Relevant tracked changes include
`belief_correction.py`, `tracker_guard.py`, `bias_floor.py`, `contracts.py`, `fusion.py`,
`camera_manager_node.py` and `silhouette_observation.py`, plus new regression files.

## Changes requiring their own design or experiment

Do not package these as incidental correctness fixes:

- Accepting the first camera return based on motion support rather than total camera gap.
- Replacing repeated rejection inflation, reanchoring or bootstrap policy.
- Changing belief-dependent admission or observation-availability labels.
- Replacing robust fusion with independent precision addition or direct-camera recursion.
- Changing process Q, camera R, persistent-bias models or disturbance cadence.
- Introducing stamped command transport, a downstream watchdog or a new reset protocol.
- Rejecting every plan after every camera update without an explicit validity/revalidation
  design; frequent valid updates must not make the robot unable to execute.
- Changing waypoint spacing, clearance, controller recovery or planner objective/tuning.

The user explicitly wants to avoid arbitrary rules that damage the thesis method.
Any design proposal must explain its trigger, positive recovery/continuation cases,
changed behaviour, evidence and consequences for comparison with the preserved baseline.

## Monitoring procedure

Native thread heartbeat **Track module audits** (`track-module-audits`) is active every
10 minutes for this coordinator thread. It reads this queue, checks new handoffs and
updates material changes. It pauses itself once all 15 final audits are delivered and
the implementation queue is consolidated; implementation completion is tracked separately.
The completion condition is not met on this pass. While the same usage-limit state
persists, check saved state without retrying or repeatedly notifying unchanged failures.
Resume pending handoff requests only after new progress or an external availability change.

1. Inspect the listed existing threads and reports. Use thread reads/status plus the
   owner's completion message; a Markdown file or passing probe alone is insufficient.
2. Record audit completion separately from subsequent plan, implementation and validation.
   Completed review status does not regress when a repair-planning turn starts.
3. Read only new/changed reports and owner responses. Deduplicate by mechanism and shared
   source, preserving report finding IDs and evidence links.
4. Update this file with readiness, dependencies, source drift, claimed/released files and
   concrete blockers. Ask an existing owner for a missing plan/status when useful; do not
   repeatedly ping a chat already working on that same request.
5. Report material changes to the user: newly finished audits, repair handoffs, validation
   results, conflicts or important newly confirmed defects. Avoid repeated unchanged updates.
6. Under the present monitoring request, prepare plans and coordinate existing authorized
   work. Do not launch broad experiments, introduce new operating policies, or automatically
   commit every concurrent file. The pushed checkpoint remains available.
7. Once all 15 audits have delivered their final reports, consolidate the final queue and
   stop the recurring audit-completion monitor. If implementation is then requested, manage
   it as the next explicit work phase with owners and checkpoints.

## Coordination requests already sent

- Runtime owner: live-source freeze and ownership; **answered, all ownership released**.
- 04: current patch ownership and tests; **answered, final QA complete and shared files released**.
- Performance owner: shared-file ownership/handoff; **interrupted by usage limit; request remains pending**.
- 01: small repair plan for B1/B2 and chronological-input parts of B3; **plan and current-source reproductions delivered**.
- 03: revalidated ordinary-stop/age-gate repair plan; **plan delivered**.
- 08: cache/JIT changes belong to another owner; **baseline report and later passing repeat available, final follow-up interrupted**.
- 10: final report plus event-ID and shutdown repair package; **report/outline delivered, final source QA interrupted**.
- 11: final report plus shared-validator/posterior-evidence plan; **final audit and plan delivered**.
- 09: supplied 10/14 thread IDs and requested completion handoff; **report saved, final completion interrupted**.
- 12: supplied 13 thread ID and requested completion handoff; **final audit delivered**.
- 13: **new ranked report saved; final completion interrupted**.
- 14/15: **final reports pending; usage limit interrupted work, evidence remains on disk**.

No new messages were sent to blocked chats on the 21:49–21:53 pass. Saved simulator
evidence remains in `experiments/runtime_integrity/simulator_world_20260906/`; integration
drafts, `matrix.tsv` and source snapshots remain in
`experiments/runtime_integrity/end_to_end_20260906/`. Their old-source failing cases must
be reconciled with packages A/B and the shared refactor before final acceptance.

This coordinator wrote only this queue during the initial pass; it did not implement a
runtime repair or run a new simulation. Tests reported above belong to their stated owners.

