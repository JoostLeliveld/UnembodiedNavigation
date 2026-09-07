# 02 — Current-source timing verification and repair handoff

2026-09-07. **Repair packages A and B hold in the focused regressions. Publication ordering, complete belief provenance and reset validity remain open.** This supersedes the *current-status claims* in the [September 6 baseline](02_timing_and_callbacks.md), while retaining that report and its original evidence as history.

The current run reproduced **19 deterministic cases** and passed **90 focused regression tests**. The probes assert observed behavior, including defects; their success is not a clean-runtime certificate. This investigation performed no ROS graph, simulator, live reset, campaign, scoring or production-edit operation, and changed no camera-during-turn, Q/R, rejection-gate or recovery policy.

## Source, configuration and ownership boundary

The required repository guidance, plan, localization contracts/registry, open questions and runtime audit were read for the original investigation. This continuation inspected the current working files, [package A](01_repair_package_a_implemented.md), [package B](03_command_execution_repair_implemented.md), [audit 01](01_state_estimation.md), and the completed [logging audit 10](10_logging_and_event_accounting.md). This is a software investigation, not a run comparison or localization-accuracy result.

HEAD is `f9e1a312c9990c29223f91f487eeb6c5aa2e71d3`, including package A `5e844a89` and package B `f9e1a312`. Uncommitted refactoring was already present. [The manifest](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_manifest.json) and [results](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/results.json) establish that all 16 retained input files matched their snapshots before execution and remained unchanged through it. Imported classes resolve to the repository's `src/` tree. This does not establish which source bytes a running or past experiment imported.

| Probe source | SHA-256 prefix |
|---|---|
| `unicycle_planner_node.py` | `e6566d3ef0956f58` |
| `efe_agent_node.py` | `f3750f4a4dcb0969` |
| `motion_history.py` | `ac5ea439b956f342` |
| `belief_correction.py` | `0f5f0b32f561ad13` |
| `camera_manager_node.py` | `8788520eb3a19476` |
| `goal_mission_node.py` | `deedad91515c4bd1` |

All code-line links below target retained source copies, so concurrent edits cannot silently move the evidence. [Supplemental source hashes](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/supplemental_source_manifest.json) cover detector, logger, goal marker and repair notes.

**Final concurrent-edit recheck:** later owner changes updated waypoint defaults, launch footprint/margin defaults, and the tracker penetration guard. [The source comparison](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/final_source_boundary.json) and `post_refactor_snapshot/` retain that boundary. A second execution on unicycle `b3e78d48b56b84f8` / EFE `c2546f81863703e1` produced [the same 19 behavioral results](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/results_post_refactor.json) with unchanged files during execution, and [90 tests passed again](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/focused_tests_post_refactor.txt). These are repeated checks, not 38 distinct cases or 180 distinct tests. The original 19-case results/source copies were preserved. Timing methods in the cited snapshot did not change in that diff; line numbers after the tracker edit may move in the live file.

The unchanged [scope configuration][scope] is fused metric EKF with a mandatory identified envelope, coupled heading, `/odom_noisy` prediction, hierarchical EFE route plus local `turn_then_go`, `dt=0.25`, a 1.5 s correction-gap cap, no hard reanchor, no pixel correction and `reset_world: false`. Launch defaults provide 10 Hz belief/command publication and 4 Hz LOCAL planning, with ROS simulation time. Camera publication-to-now compensation is disabled; common-capture alignment remains enabled. Other declared modes are explicitly marked conditional below. This audit is not a claim that this YAML was redeployed after A/B.

[Audit 01's current launch evaluation](../../experiments/estimator_consistency/unicycle_current_20260907/wiring.json) independently resolves the prediction speed cap to **0.22 m/s** and correction freshness to **0.5 s**. A raw YAML/default value is not necessarily the final Node parameter. Its [current numerical report](01_unicycle_current_verification.md) verifies the estimator at the same original source hashes; its overlapping test counts must not be added to ours.

Coordination: **Audit Belief Correction Pipeline** owns correction mathematics, motion support and capture-time heading; **Audit planner future model** owns GLOBAL/direct solve handoff and model validity; **Refactor loose pivot code** owns startup/configuration validation, per-camera parsing and tracker dispatch/safety. This investigation owns scheduling, synchronization, event/epoch provenance and publication boundaries, and writes only its reports and dedicated evidence. The refactor owner fixed the newly noticed undefined `belief_frame_id` reference in the optional camera callback before this snapshot; the callback now uses `_resolve_plan_frame_id()`. That repair does not bind the resolver to the committed belief's frame.

## Verified repairs: do not implement them again

| Prior finding | Current result and limit |
|---|---|
| T02-01 odometry completion order/torn yaw and velocity | **Closed by A for odometry.** `[yaw, velocity, origin, history, integer-nanosecond watermark]` commits under one data lock; invalid, late and equal-time odometry are refused. The first equal-time event wins. Command receipt chronology is a separate remaining case below. |
| T02-02 anonymous `/state/bev` startup in mandatory-envelope mode | **Closed by A for the configured mode.** Planner waits for the envelope. Identified bootstrap and either cross-topic delivery order pass. Explicit legacy paths still have initialization writers. |
| T02-05 outage coverage/replay race | **Closed by A for the outage helper.** Anchor, both motion buffers and source selection are captured together; support checking and replay receive the same frozen `MotionHistorySnapshot`. A trim during replay cannot erase the checked turn. This is not universal motion-support validation. |
| Ordinary stop followed by installation of an older solve | **Closed by B at LOCAL/direct tape installation.** Request generation is captured before planning input/prediction; installation checks it inside the tape/publication lock. |
| Computation-age gate missing on installation paths | **Closed by B at LOCAL/direct tape installation.** Current-generation requests older than the actual safe-prefix duration, negative-age requests and missing context are rejected. The strict `>` boundary is preserved. |
| Three original transaction defects | **Existing repairs hold:** one prediction target supplies age/header; concurrent corrections serialize; a prediction whose mean/covariance anchor objects were superseded is suppressed, including a real same-stamp correction. |
| Fatal stop, exhausted/pre-reset tape and idle diagnostics | Existing regression coverage remains passing. These guards do not establish complete filter/goal epochs or solve provenance. |

Evidence: [90-test output](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/focused_tests.txt), including runtime transactions, outage replay, state correction, command installation and encoder chronology. The supported blind-turn test still integrates the recorded turn. No camera-during-turn exclusion was introduced or proposed.

## Ranked remaining findings

Priority reflects consequence and reachable conditions, not measured frequency. The descriptions below are confirmed at actual callback/helper boundaries unless marked otherwise.

| Rank | ID / priority | Current reachability | Expected versus reproduced behavior; smallest repair |
|---|---|---|---|
| 1 | **T02-03 / P1** | Active reentrant belief timer, two executor workers | Slow target 10.0 publishes after target 10.2: **`[10.2, 10.0]`**. Prevent older public state time within an epoch; retain superseded-anchor validation and add an atomic publication watermark. |
| 2 | **T02-04 / P1 conditional** | Backward `/clock` jump while processes remain alive; intentional reset disabled in scope | At ROS 1, anchor 100 remains; planning reports age **0**. New odometry 1 is refused, new correction .95 is refused, and an in-flight target 100.1 can publish after rewind. Fail closed on invalid epoch before planning/publication; use coordinated restart or a complete epoch reset transaction. |
| 3 | **T02-01c / P2** | Active command subscription; replay consequence when odometry absent or command prediction selected | Reentrant receipts append **`[11, 10]`**, restore old `last_cmd`, and replay **0.2 m instead of 0.3 m** in the controlled example. Make command chronology/held value one ingestion transaction with a declared late/equal-time policy. |
| 4 | **T02-06 / P2** | Active `/planner_belief` → manager admission/geometry feedback; unchanged ROS time between publications | At time 10, corrected x changes **0 → .0625**, but manager retains **0**. Carry revision independently of state time; update equal-time consumers coherently. |
| 5 | **T02-07 / P2** | Active correction during planning prediction | Returned mean is from anchor **9.9**, while returned `belief_stamp` is **9.95**. Return provenance with the original immutable snapshot; coordinate result revalidation with audit 08. |
| 6 | **T02-08 / P2 conditional** | Active compatibility subscription, unexpected frame-changing producer; normal manager uses fixed `map_bev` | Unchanged map belief publishes as **`camera_optical`** after unrelated compatibility input. Bind frame to belief/configuration and validate incoming frames against it. |
| 7 | **T02-09 / P2 mechanism** | Active correction entry points in reentrant I/O group | One slow correction and one lock waiter occupy both workers while data lock is free; odometry cannot start. Put correction entry points in a dedicated mutually exclusive callback group or a single worker queue without consuming executor workers as lock waiters. Keep numerical work out of the data lock. |
| 8 | **T02-12 / P2 failure path** | Active envelope path when post-commit diagnostic/terminal publication raises | Posterior/stamp commit, but source ID and terminal record are absent; fatal stop is called. Commit identity and an immutable outcome record with the posterior before fallible reporting. |
| 9 | **T02-13 / P2 conditional** | Two distinct goals delivered while earlier goal callback is delayed; current default suppresses unchanged repeats | Goal **x=2** and progress signature **x=1** survive together. Commit goal, signature, progress invalidation and revision in one data-lock section. |
| 10 | **T02-10 / P2 optional** | Mission initial-belief gate or multi-waypoint mode | Old/wrong-frame belief remains usable; invalid covariance does not clear readiness; waypoint advances after reset. Store/validate one complete belief event and invalidate readiness on unusable input. |
| 11 | **T02-11 / P2 optional** | Pixel correction with positive timer interval; disabled in scope | Timer selects **9.9**, then applies pixel value from **9.95** under 9.9. Select and pass an immutable pixel event inside the correction transaction. |

The complete unicycle handoff also depends on [audit 01's current P1 findings](01_unicycle_current_verification.md): an invalid held prediction can be committed after an update is refused; ordinary prediction/correction paths disagree about unsupported motion; and moving bootstrap stores latest yaw at an earlier capture time, then integrates that turn again. That owner numerically reproduced the last case as expected current heading **1.3 rad**, returned **1.5 rad**. These are coordinated correction/support findings, not new probes or production repairs claimed by audit 02. [Audit 08's current result checks](08_current_result_checks.json) separately verify GLOBAL/direct handoff gaps, including goal changes and stop generation; configuration mutation is an API case because startup parameters are static on the inspected active node.

### T02-03 — Public order is independent of anchor supersession

Locations: [callback groups/timer][u-groups], [publisher][u-publish], [two-worker executor][e-main]. Probe: `publication_overtakes` in [the current results][results]. An event barrier suspends prediction A after its target is selected; B publishes at 10.2; A then publishes 10.0. Each synthetic mean, full planar covariance, frame and timestamp is internally coherent. The final published message is nevertheless older. Both pass the pointer guard because no recursive update occurred.

The installed rclpy timer/ReentrantCallbackGroup behavior inspected in the baseline permits another invocation while the first callback is running. This method-level probe establishes the required interleaving; it does not measure live DDS latency or occurrence frequency. Logger and mission keep the last arrival, and `/planner_belief` is retained at depth one, so the final retained message can move backward.

Smallest repair: inside the existing final publication lock, require the same epoch and anchor revision and reject a target older than the last published target. Permit a higher belief revision at an equal target. Do not restamp old calculations with completion time. A dedicated mutually exclusive publication group also prevents overlapping expensive predictions, but should not replace the explicit ordering invariant.

### T02-04 — The node still lacks complete epoch validity

Locations: [ordered odometry][u-odom], [future-age check][u-age], [EKF resolver][u-ekf], [publisher][u-publish], [request capture/install][e-install]. `reset_retains_anchor_and_new_request_can_install` advances 100.1 → 1. Package A now leaves history **`[100]`** and increments the old-input refusal counter, rather than corrupting history to `[100,1]`. The filter anchor and seen IDs survive. Publication initiated after reset returns without a message. Planning converts invalid future age to zero and returns the old state. A new-epoch correction is dropped as `not_newer_than_belief`.

The probe separately feeds a request captured after the reset through the real package-B installer: it installs. This is a composition of the bad-belief resolver and fresh-request installation boundaries, **not a full solver/executor demonstration**. B correctly rejects a request that started before the rewind; it has no belief-epoch token to reject a new request built from retained old state.

`reset_during_public_prediction` adds the complementary interleaving: a timer chooses 100.1, pauses in prediction, ROS time changes to 1, and the timer still publishes the pre-reset target. The final anchor-object comparison cannot detect a clock epoch change.

Current upstream qualifications matter: the [detector clock guard][d-clock] and [manager guard][m-clock] now detect backward movement beyond .005 s and demand coordinated restart. They are documented existing repairs, not missing hooks. The raw `FourCameraBatcher` probe still refuses reset frames because its watermark persists; that does not mean the current detector silently continues. Those upstream fatal paths reduce exposure but do not make the planner's invalid-state return or the interval before shutdown correct. No in-process full-stack reset was exercised; the campaign prohibits intentional live reset.

Smallest complete policy: fail closed locally on invalid clock/epoch and require a supervised restart of the affected runtime. If in-process reset is required, invalidate in-flight predictions/solves before clearing anchor, odometry/command histories and watermarks, correction identities, retained public belief, camera query history and mission readiness/progress. Clearing only the tape or odometry list is insufficient. Remove the resolver's `None → 0` conversion even if restart remains the operational policy.

### T02-01c — Command ingestion retains the pre-A chronology mechanism

Locations: [command callback][u-cmd], [ordinary correction replay][u-replay]. Probe `command_receipt_completion_order` stops A after obtaining receipt time 10, lets B commit time 11, then resumes A. Stored order is `(11,.2),(10,.1)` and the held command becomes .1. Real correction replay with no odometry produces x=.2 over [9,12]; an independently ordered evaluation produces x=.3. Sorting is used only as the oracle, not as a proposed repair.

The configured path prefers odometry; the replay error requires missing odometry for that interval or explicit command prediction. The callback and its `last_cmd` mutation are always active. Twist is unstamped: this node cannot reconstruct command issue or actuator-application time from callback receipt, even after append order is repaired. Direct publication also writes `last_cmd`, so delayed receipt can overwrite the publisher's more recent held value.

Smallest repair: validate finite command values, associate receipt time and local sequence with the event, and atomically reject stale events before changing history or held value. Declare equal-time event order instead of using timestamp alone as identity. Moving the receipt read into the commit lock can define a consistent *commit-time* command history, but must not be described as source/application chronology. Audit 03 owns any stronger stamped command protocol. Do not retroactively sort a history already used by accepted corrections without rewind/replay semantics.

### T02-06, T02-07 and T02-08 — State time, revision and frame need separate ownership

Locations: [planning resolver][u-resolve], [frame resolver][u-frame], [publication][u-publish], [manager consumer][m-belief]. Three independent real-method probes establish the gaps:

* `equal_time_revision_lost_by_consumer`: a correction at 9.95 lands between two publications targeting ROS 10. Both headers are 10, but means are 0 and .0625. The manager's `timestamp != last_timestamp` condition keeps the first. This can occur with quantized or paused simulation time; repeated state time does not mean repeated posterior.
* `planner_metadata_from_superseding_update`: correction during prediction leaves returned m/P from 9.9 and metadata from 9.95. Copies of m/P are coherent; the metadata was reread afterward. A solve/diagnostic can therefore claim an anchor it never used.
* `frame_from_different_event`: the mandatory-envelope compatibility callback stores an unrelated `camera_optical` pose but cannot update the recursive filter. `_resolve_plan_frame_id()` reads that compatibility message while building the publication, relabeling the old map coordinates. The same mutable resolver also defines which frame a later envelope is allowed to carry. No coordinate transform occurs.

Smallest repair: an immutable belief snapshot must carry `(epoch, revision, anchor_ns, frame, mean, full covariance)`; a prediction result must add its exact target, chosen motion snapshot and diagnostics. Every accepted/rejected posterior mutation increments revision even when anchor time is unchanged. Frame comes from the validated configuration/bootstrap event, not a compatibility display. Return metadata with the prediction, never by rereading shared state. A compatible public pose may remain, but consumers requiring event identity need a versioned coherent envelope or an equivalently explicit identity contract.

Ordinary fixed-frame full-covariance publication passes. The actual same-stamp correction test also confirms the existing object-identity guard suppresses a superseded prediction. No current in-place recursive-array writer was found in the inspected node; the pointer guard's success depends on replacement semantics. This does not fix downstream equal-time identity.

Result invalidation needs a declared policy. Do not reject every routine camera update during a slow solve: continue assimilating observations during turns and planning, then revalidate the candidate against a fresh coherent belief/goal and its motion/age support. Epoch changes, explicit stops and changed goals require invalidation. Audit 08 owns the GLOBAL route acceptance path, which bypasses the LOCAL/direct tape installer; package B does not certify it.

### T02-09 — Serialization protects the filter but can consume both workers

Locations: [correction decorator][u-decorator], [I/O assignments][u-groups], [executor][e-main]. `correction_worker_starvation` uses an observed lock-contention event and a two-worker pool. A is computing replay under `_correction_lock`; B is blocked acquiring that lock; queued odometry cannot run despite `_data_lock` being free. Releasing A lets all finish. This establishes worker exhaustion, not a permanent deadlock or a measured latency/drop rate in ROS.

Normal acquisition order is correction lock → data lock. Nested correction calls and nested publication/stop calls use reentrant locks. No reverse data-lock → correction-lock cycle was found in the audited production call paths. Pixel selection and legacy resolver writers still require explicit ownership; a lock-order claim is not a proof that every mode is transactional.

Smallest repair: a dedicated mutually exclusive correction callback group prevents the executor from spending a worker waiting on a Python correction mutex. Keep the mutex for non-executor callers/nested helpers as needed. All correction entry points must follow the same ownership policy. A bounded input queue must emit reasoned terminal outcomes for identified events it refuses or evicts; silently coalescing identified corrections would break event accounting. ROS publication currently occurs while holding the data lock to close stale-output races; a slow publisher can block input. Its actual blocking latency was not measured. An ordered outbox is a later option, provided it preserves revision/epoch validation at publication.

### T02-12 — A successful state commit can precede a failed identity commit

Locations: [envelope callback][u-envelope], [metric correction][u-metric], [assimilation publisher][u-assim]. `committed_correction_without_identity_or_terminal_record` injects an exception in the numerical diagnostic publisher after real correction mathematics and posterior commit. Anchor becomes 9.95 and x=.0625, but the seen-ID set is empty and there is no terminal assimilation row. The envelope catches the exception and invokes fatal stop.

For diagnosis only, the test stubs shutdown and redelivers the envelope. It is then recorded as `dropped/not_newer_than_belief`, although the earlier attempt changed the filter. **No double assimilation was observed; normal production is expected to shut down after the first failure.** The defect is state/identity/accounting atomicity on the failure path. Concurrent duplicate delivery without failure remains safe: the existing correction lock serializes duplicate checking, calculation and eventual set insertion, producing one assimilation and a duplicate fatal error. The apparent separation into two data-lock sections is not itself a race while the encompassing correction lock is held.

Smallest repair: create the immutable terminal outcome and commit `(posterior, revision, source identity, outcome record)` in one internal transaction before optional diagnostics. Publish from that record; retain/report failure and make retries idempotent by identity. Preserve the fatal integrity policy. Exactly-once internal assimilation does not imply exactly-once DDS delivery; ledger consumers need a duplicate policy. Coordinate terminal schema/finalization with audit 10 rather than making a second logging design.

### T02-13 — Goal and progress metadata are separate commits

Locations: [goal callback][u-goal], [progress-origin helper][u-progress]. `goal_and_progress_signature_are_different_events` uses actual callbacks and a barrier before A's progress update. B completes both updates, then A writes its signature: retained goal x=2, signature x=1. This does not demonstrate a solver steering directly toward x=1; the consequence is stale progress ownership, misleading change detection and an unreliable foundation for goal-revision validation. The current default mission no longer repeats unchanged goals, reducing routine exposure. Distinct waypoint/external goal changes remain reachable.

Smallest repair: compute validated coordinates/frame locally and commit goal reference, signature, revision and progress invalidation together. A result must retain the goal revision it used. Do not use a goal's system-time header as its revision or compare that header with belief simulation time.

### T02-10 and T02-11 — Optional consumers must also preserve event identity

Locations: [mission belief callback][g-belief], [mission send][g-send], [pixel callback/timer][u-pixel], [pixel snapshot][u-pixel-snapshot]. `mission_clock_and_readiness` accepts a stale, wrong-frame belief; an invalid-covariance message does not clear the earlier readiness flag; a waypoint advances after a reset with no valid new belief. `_belief_xy` is replaced before covariance validity is known. Validate finite pose/full planar covariance, expected frame, state age and epoch as one record before readiness/arrival use, and explicitly clear readiness on invalidation.

`pixel_timer_reads_new_measurement_with_old_stamp` retains the baseline 9.9 stamp/9.95 measurement mismatch. Acquire/select the immutable event inside the serialized correction transaction; passing only its old timestamp while rereading current pixels is insufficient. These modes are disabled or optional in the scope configuration; stronger pixel-model and heading changes remain with audit 01.

## Complete node ownership map

[The static inventory](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/ownership_map.json) lists all **87 base-node methods, 28 EFE methods and four mission methods**, their self-attribute assignments, locks, calls and clock reads from retained source. It is an AST index, not an inferred whole-program call graph. The following table classifies every external callback/timer entry point and the recursive writers relevant to this audit. `C` means correction RLock, `D` data RLock, `I/O` reentrant group; EFE runs two workers. Mission uses its default mutually exclusive group and `rclpy.spin`.

| Entry point / owner | Serialization and writes | Freshness / remaining boundary |
|---|---|---|
| `_state_cb` | I/O, C; D replaces compatibility `state_msg`; calls correction only for fused non-envelope mode | Mandatory mode is read-only with respect to recursive belief, but stored frame is still consumed as authoritative. |
| `_state_correction_envelope_cb` | I/O, C for parse/dedup/calculate/commit/report; D for posterior and ID set | Schema/finite/SPD/frame validation, source-ID duplicate failure, capture-age and anchor-order gates. Post-commit reporting failure precedes ID insertion. |
| `_map_observations_cb` → `_apply_map_observations` | Optional I/O, C; per-camera seen stamps, quorum policy, sequential metric commits | Sorted batch and per-camera timestamp rules; optional same-anchor corrections. Frame resolver is still mutable. Audit 01 owns quorum/heading/recovery semantics. |
| `_pixel_cb` → `_apply_pixel_correction` | I/O, C; D replaces pixel value/stamp; may initialize/commit belief | Capture-age/throttle gates. Immediate and timer paths must select the same complete event. |
| `_pixel_correction_timer_cb` | Optional I/O; D reads stamp, releases it, then enters C update | The selected value/diagnostics are not frozen with the selected stamp. |
| `_goal_cb` | I/O; two D sections via helper | Goal and signature/progress can disagree; header is not checked for age. |
| `_cmd_cb` | I/O; ROS time read before D; D replaces held command/appends/trims | No input chronology/finite gate; receipt is not issuance or application. |
| `_odom_cb` | I/O; locals parsed before D; D commits all accepted odometry fields | Integer-nanosecond strict-newer watermark, no retroactive insertion. Order does not prove support or clock epoch. |
| `_diagnostic_odom_cb` | Optional I/O; latest TF lookup then D stores pose/stamp | Uses latest transform with source odometry stamp; diagnostic resolver reports age zero. Disabled for comparisons; no timed-transform integration test here. |
| `_detection_diag_cb` | I/O; D replaces diagnostic object | Pixel matching compares stamps; no independent event revision. |
| `_belief_publish_tick` | I/O; D snapshot, prediction outside D/C, D supersession-check/publish | Recursive m/P/stamp read-only; frame, output-order and epoch gaps remain. Shared prediction diagnostics can change. |
| EFE `_plan_once` → base/direct or hierarchical body | Mutually exclusive plan group; D captures request; expensive resolution/solve outside D/C | B stop-generation/ROS-age token. Goal and belief snapshots separate; GLOBAL route state is plan-owned but lacks equivalent commit gate. |
| EFE `_publish_active_plan_command` timer | I/O; D covers tape read/expiry/publication, nested reentrant D calls | Negative/exhausted tape stops; fatal latch prevents nonzero. Timer may read pending-plan `_current_*` diagnostics with an older installed tape. |
| EFE `_run_dir_cb` | Default callback group; updates artifact directory/pending artifact, may write files | Artifact persistence only; not a belief writer. Do not move it into input-critical locks. |
| Safe/fatal stop helpers | Called from multiple owners; tape/command mutation under D; fatal latch retained | Increment stop generation, clear tape, publish zero/idle. Do not acquire C while holding D. |
| Mission `_belief_cb` | Default group, stores XY/sigma/readiness | No state-time/frame/epoch validity gate. |
| Mission `_send_goal` → `_maybe_advance` | System-clock timer/default group; writes waypoint/send state | Wall startup/unchanged-goal policy; arrival consumes held belief with no freshness check. |

The **seven post-construction recursive assignment sites** are:

| Writer | State changed and owner |
|---|---|
| `_init_belief_from_state` [1196][u-init] | Mean, covariance, anchor from compatibility state under D. Used by optional pixel/legacy initialization, including planning; not protected by C when called from those resolvers. |
| `_apply_pixel_correction` [1724][u-pixel-apply] | Accepted mean/P/anchor/last correction under C then D. |
| `_resolve_state_belief_for_planning` [2090][u-legacy] | Legacy non-EKF mean/P/anchor under D; also rereads recursive fields without D in heading handling. This is an explicit planning writer outside the configured EKF path. |
| `_reanchor_belief_to_xy` [2166][u-reanchor] | Mean/P/anchor/last correction under D, reached from correction/recovery callers. Latest yaw is not capture-time yaw. |
| `_inflate_belief_after_rejection` [2341][u-inflate] | Replaces covariance under D without changing anchor. Must count as a new belief revision. |
| `_advance_belief_over_outage` [2357][u-outage] | Mean/P/anchor after A's immutable motion snapshot; called by C-owned correction path. No internal standalone C decorator. |
| `_commit_metric_correction_outcome` [2615][u-commit] | Accepted/rejected predicted posterior and anchor, or reanchor; called under C. Some rejections move anchor without accepting a measurement, and covariance-only changes remain revisions. |

Normal planning and monitoring of an initialized coupled metric belief do not advance the recursive state. `_predict_belief_to_now` and heading/planning helpers do write shared `_latest_*` diagnostics, however. Their scalar fields can describe different concurrent predictions; they should be returned with the prediction record. Audit 01 owns motion-support and heading correctness, including why ordered odometry alone does not establish valid prefix/interior/tail replay or capture-time yaw.

## Timestamp and clock map for the current source

ROS means the node clock driven by `/clock` in this launch. System means civil/epoch time. Steady means `monotonic`/`perf_counter` elapsed time. ROS header seconds/nanoseconds contain no clock-domain or reset-epoch tag. A consumer must know that contract; numeric similarity does not establish common time.

| Value and described instant | Clock / permitted writer | How freshness or identity is determined |
|---|---|---|
| Image header / `PendingFrame.stamp_ns`: camera capture | Camera producer, copied by [detector callback][d-receipt] | Exact per-camera capture key, strict-newer/duplicate/skew checks. Producer epoch and clock-reset guard are separate from raw batcher watermarks. |
| `receive_stamp_s`: callback entry after transport/executor delay | Detector ROS `_clock_s()` | Capture-to-receipt latency uses ROS; not physical wire arrival. Immutable pending frame retains it. |
| `receive_wall_s`: that callback's steady reading | Detector `perf_counter` | Pending expiration/liveness compares steady time only. Intentionally independent of simulation progress. |
| YOLO start/finish stamps; inference duration | Detector ROS boundary reads plus separate `perf_counter` duration [1055][d-inference] | ROS interval may be zero during pause; steady milliseconds measure execution. Do not subtract one domain from the other. |
| YOLO output/publication stamp and frame age | Detector ROS output construction | Capture-age/future checks; publication time is not capture time. |
| `source_batch_id`, `producer_epoch` | Detector derives physical capture identity and publishes producer epoch | Identity is distinct from float capture/receipt time. Planner schema currently carries source ID without a coordinated belief epoch. |
| Per-camera `timestamp_s`, `common_capture_stamp`, fused `correction_stamp` | Manager preserves captures, aligns to common capture, and writes correction envelope/pose [1543][m-fusion] | Scope stamp remains common capture. Optional publication-to-now mode propagates both state and covariance before changing the state time. Admission uses separately read decision `now_s`. |
| Odom header sample time, `_odom_accepted_stamp_ns`, `_odom_origin_stamp_s` | Odom producer; planner's single D ingestion transaction | Strictly newer integer key required. Origin is first accepted sample. No accepted receipt/epoch token is stored with yaw/velocity; latest yaw is not a queried historical yaw. |
| Command log time and held `last_cmd` | Callback ROS receipt plus a second `last_cmd` writer at direct publication | No source stamp or actuation acknowledgement; current append/held-value order can differ from receipt order. |
| `belief_stamp`: recursive posterior anchor | Seven writer sites above | Correction capture age bounded; fused times at/below anchor tolerance refused; optional distinct same-stamp cameras allowed. Timestamp cannot identify covariance-only or same-time revisions. |
| `_last_correction_stamp` | Accepted/bootstrap/reanchor writers | Pixel throttle compares measurement stamps. It need not equal anchor after rejection/outage advancement. |
| Diagnostic `apply_stamp`, assimilation `apply_stamp` | Separate ROS reads after commit [2410][u-assim] | Reporting instants, not one atomic commit or callback-receipt instant. Mean/P/revision are absent from terminal envelope. Equal apply times may represent different events; audit 10 owns downstream accounting. |
| Prediction target `now_msg` and age | Planning resolver/publication chooses one ROS target; predictor uses target minus age | Publisher age/target/header share one reading. Invalid future anchor drops publication but becomes age zero in EKF planning. Outage check/replay shares one motion snapshot after A. |
| Public belief header, mean and full planar covariance | Publisher chooses projection target and copies under D | Anchor-object validation before publication; no target watermark or explicit revision/epoch. Frame is independently resolved from compatibility state. |
| Planner goal/state references, m/P and returned anchor metadata | Goal/state refs under one D read; belief copied separately; anchor metadata reread later | No complete snapshot ID or freshness token; candidate may outlive goal/belief changes. |
| Request `started_at`, `stop_generation` | EFE captures ROS time/generation under D before input/prediction | B checks generation and computation age at LOCAL/direct install. Generation changes on explicit stop, not every correction. |
| Solve start/completion/duration | `perf_counter` and solver `solve_time_s`; `monotonic` for log throttles | Computation durations only; not state timestamps. No single retained ROS solve-completion field establishes pose freshness. |
| `_pending_plan_started_at`, active remaining duration | LOCAL ROS read before controller generation | Used for optional latency handoff; separate from earlier request age. Legacy/direct route has separate pending-context concerns owned by audit 08. |
| `_active_plan_started_at`, tape elapsed/index | ROS time at installation or compensated earlier origin | Command timer uses elapsed/`dt`, stops on negative or exhausted age. The pre-install computation-age gate is a separate check. |
| `/plan` and `/plan_preview` header, nested pose headers | ROS read when completed plan artifact is built [2918][u-plan-publish] | Publication time of a route/rollout artifact, not capture time or each future state's time. Logger stores this as plan time. No state-time freshness inference should be made from it; carry input target/solve provenance separately. |
| Planner diagnostics/text time | Diagnostics read some live execution/prediction fields; text payload uses ROS now | Reports can mix input/result/live-tape provenance; unstamped arrays do not identify a coherent solve. |
| Mission `start_time`, timer, delay | Explicit SYSTEM_TIME clock [74][g-clock] | System elapsed startup policy. Civil jumps affect elapsed delay; it is not a simulation timer. |
| Goal header | Mission SYSTEM_TIME at publication | Planner/logger use goal frame and coordinates, not header-to-state age. Goal marker forwards the header to visualization. |
| Mission belief age/readiness | Belief header supplied by planner, but unused for readiness | Sigma check only; no age/order/epoch validation currently. |
| Infrastructure timeouts | Campaign civil/wall elapsed; node throttles steady | Independent of simulation elapsed by policy. This continuation does not inject campaign clock changes or modify a running timeout. |

The mission header verdict is unchanged: [planner goal handling][u-goal], hierarchical goal use and [logger callback][l-goal] do **not** compare system-stamped goals with simulation-stamped state. `mission_clock_and_readiness` verifies a system header 1,780,000,000 at ROS .1 is accepted as a static goal. [Goal marker][g-marker] forwards the header; no live TF/RViz failure was established. The wall startup timer is not itself a demonstrated bug. Current `repeat_unchanged_goal=false` gives send counts `[0,1,1,1]` for wall times `[99,103,90,104]`, unlike the older repeat-enabled baseline. A steady startup duration would be more robust to civil adjustments if that is the chosen requirement; goal message clock policy can be decided separately.

Pause is also different from reset: ROS timers/age ordinarily stop advancing during a pause, while an already-running callback and steady expiration timers can continue. A long steady solve with no simulation progress is not automatically stale in physical-state time. Pending camera liveness policy may deliberately fail during a long pause. The equal-target revision probe exercises the relevant unchanged-clock case; it is not a live paused-simulator test.

Two smaller boundaries are retained without promoting them to active P1 bugs. `clock_domain_error_fails_open` deliberately supplies SYSTEM_TIME to `_stamp_age_s`, whose exception handler returns zero and calls the measurement fresh; configured ROS clocks do not normally take that path. Return invalid rather than fresh on conversion/domain failure. `bounded_unpredicted_publication_interval` confirms the publisher can stamp 10.0005 on the unpropagated anchor at 10 because of its .001 s cutoff. Replay also skips substeps at/below .0001 s. These are bounded numerical approximations, not the former unbounded solve-latency restamping error; exact-instant contracts should remove or explicitly declare them.

## Hypotheses and limits

* The installer checks current generation/expiry under D, then releases D before warning and safe-stop side effects. A helper-level fresh replacement inserted in that interval could be cleared. However, **another normal planning callback cannot install a fresh tape while the old callback runs**, because the plan group is mutually exclusive. Audit 08 independently agrees with this reachability limit. This is an inspected API/refactoring hazard, not a newly confirmed live overlapping-planner defect. Keep any repair small: revalidate generation inside the stop transaction.
* Shared `_latest_*` prediction and `_current_*` waypoint diagnostics can be overwritten by a new calculation while another result/tape is reported. Writer/read boundaries are inspected; this continuation did not add a separate numerical mixed-diagnostic probe. Exact posterior event logging is already limited independently by audit 10.
* No permanent deadlock, live callback-starvation duration, DDS drop count, physical actuator age or complete reset recovery has been measured. Method barriers show code behavior without forcing an active experiment into these states.
* Missing-motion support, capture-time heading and optional recovery semantics remain with audit 01. Do not infer those are fixed from ordered ingestion or change Q/R to mask timing failures.

## Sequenced repair handoff

1. **Small independent transaction repairs:** timing owner with refactor coordination: atomic goal/progress update, command ingestion chronology/finite checks, future/invalid-age fail-closed return. State owner additionally validates the held prediction before any rejected-outcome commit (audit 01 U01); input validation alone does not protect every route to an invalid posterior. Reuse current barrier examples as desired-invariant regressions. Preserve A/B and turn-time camera updates. No routine calculation should hold D for its whole runtime.
2. **One internal belief record:** state owner defines correction semantics; timing owner supplies ownership. Route the seven writers through a central commit of full mean/P, frame, integer anchor, epoch and revision, with correction identity/outcome where relevant. Capture checked motion history with the prediction inputs; audit 01 supplies the support contract. Return prediction diagnostics/provenance instead of shared scratch fields. Keep covariance/noise/gate values unchanged.
3. **Publication and consumer boundary:** add the final epoch/revision/target ordering check; use one serialized publication owner or ordered outbox. Update manager/mission/logger consumers so equal state time is not treated as duplicate revision. Preserve full planar cross-covariance and the real prediction target. Introduce any message schema change jointly with the logging owner.
4. **Planning acceptance:** audits 03/08 extend request provenance to coherent goal/belief snapshots and a GLOBAL route commit gate. Stops/resets/goal changes invalidate; ordinary newer camera observations trigger a defined revalidation/reprojection rule rather than blanket cancellation. Installation and diagnostic publication must describe the accepted request. Retain B's strict age comparison and safe-prefix duration.
5. **Scheduling and event accounting:** move all correction entry points to a single executor scheduling owner; select complete pixel events there. Commit terminal identity/outcome before fallible diagnostics, then publish/retry with explicit delivery accounting. Keep C → D order. Do not silently drop an identified correction to relieve queue pressure.
6. **Operational epoch policy and isolated acceptance:** align the planner/mission with the detector/manager's coordinated-restart policy, or implement all-node epoch clearing before reinitialization. After code freezes, use an isolated non-scoring trial for pause/resume, backward jump, delayed input, slow prediction, and correction during both LOCAL and GLOBAL work. Verify public order, state/P/frame/target coherence and command cancellation; do not reset existing studies.

This is a proposed handoff, not authorization to replace another owner's shared files. The coordinator has received the verified findings and source hashes. Production repairs remain unimplemented by this investigation.

## Reproduce the current evidence

[Probe](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/probe.py), [results][results], [output](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/probe_output.txt), [ownership generator](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/ownership_map.py), and [focused tests](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/focused_tests.txt) are retained separately from the September 6 evidence. Assertions that describe defects must be inverted/rewritten when a repair lands; a failing baseline probe after a repair is not automatically a regression.

```bash
source install/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/runtime_integrity/timing_callbacks_revalidation_20260907/probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/planning/test_runtime_transactions.py tests/planning/test_outage_motion_replay.py tests/planning/test_planner_node_state_correction.py tests/planning/test_command_installation.py tests/sim/test_encoder_input_chronology.py
```

[results]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/results.json
[scope]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/experiments/icra_commissioning/network_navigation_runtime_pilot.yaml
[u-decorator]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:50
[u-groups]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:612
[u-envelope]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:838
[u-progress]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:908
[u-goal]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1019
[u-cmd]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1032
[u-odom]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1049
[u-init]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1196
[u-pixel]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1225
[u-replay]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1294
[u-pixel-snapshot]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1397
[u-pixel-apply]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1724
[u-age]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:1822
[u-legacy]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2090
[u-reanchor]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2166
[u-inflate]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2341
[u-outage]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2357
[u-assim]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2410
[u-metric]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2443
[u-commit]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2615
[u-ekf]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2711
[u-resolve]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2782
[u-frame]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2809
[u-publish]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2854
[u-plan-publish]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2918
[e-install]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/efe_agent_node.py:617
[e-main]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/planning/planning/nodes/efe_agent_node.py:1541
[m-clock]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1089
[m-belief]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1154
[m-fusion]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1543
[d-clock]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/perception/perception/nodes/batched_four_camera_yolo_node.py:615
[d-receipt]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/perception/perception/nodes/batched_four_camera_yolo_node.py:798
[d-inference]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/perception/perception/nodes/batched_four_camera_yolo_node.py:1055
[g-clock]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/experiments/experiments/nodes/goal_mission_node.py:74
[g-belief]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/experiments/experiments/nodes/goal_mission_node.py:88
[g-send]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/experiments/experiments/nodes/goal_mission_node.py:118
[g-marker]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/experiments/experiments/nodes/goal_marker_node.py:53
[l-goal]: ../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1776
