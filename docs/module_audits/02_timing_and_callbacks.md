# 02 — Timing, callback ownership and belief publication

2026-09-06. **The recent single-clock and correction-writer repairs hold, but publication order, input-history order and reset ownership remain incomplete.** A slow belief prediction can replace a newer publication without any correction occurring. Delayed odometry callbacks can corrupt replay order. An in-place simulation reset can leave planning using the previous epoch's belief while reporting age zero.

This is a software audit with deterministic synthetic inputs, not a navigation or localization-accuracy result. Priority ranks consequence and reachability, not measured frequency. No experiment was started, stopped, reset or rescored. This investigation changed only this report and its dedicated reproduction directory; no production, filter mathematics, launch configuration or frozen experiment source was edited.

## Ranked report

| Rank | ID | Classification | Finding and reachability |
|---|---|---|---|
| 1 | T02-01 | **P1 confirmed current bug** | Reentrant odometry callbacks append in completion order, and replay trusts that order. Active `/odom_noisy` path. |
| 2 | T02-02 | **P1 confirmed current bug; previously identified** | Planning can bootstrap through `/state/bev` despite the mandatory identified envelope. Active startup delivery ordering; independently reproduces the estimation review. |
| 3 | T02-03 | **P1 confirmed current bug** | Two predictions of the same anchor can publish newest first, oldest last. Active reentrant belief timer with two executor workers. |
| 4 | T02-04 | **P1 confirmed reset failure** | Backward simulation time retains recursive state, input history and identities; planning turns a future-anchor error into age zero. Conditional on resetting live processes; scope campaign has `reset_world: false`. |
| 5 | T02-05 | **P2 confirmed current bug** | Motion coverage is checked against one buffer and replayed from another. Active long-gap handler near history retention boundary. |
| 6 | T02-06 | **P2 confirmed current bug** | Equal public timestamps hide distinct revisions; camera manager retains the earlier belief. Active belief→camera-admission feedback. |
| 7 | T02-07 | **P2 confirmed snapshot defect / documented execution gap** | Planner metadata rereads a newer anchor after predicting the old one. Results lack belief/goal revision checks; command consequences are independently reproduced in audit 03. |
| 8 | T02-08 | **P2 confirmed conditional frame defect** | Publication takes the frame from a compatibility message unrelated to the committed belief. Active receiver, triggered by a frame-changing producer; configured manager normally emits fixed `map_bev`. |
| 9 | T02-09 | **P2 confirmed scheduling mechanism; latency unmeasured** | One slow correction plus another waiting for its lock occupies both executor workers and prevents odometry/command callbacks from running. Active callback layout. No permanent deadlock established. |
| 10 | T02-10 | **P2 confirmed mission validity defect** | Readiness/waypoint advancement ignore belief age, epoch and frame; invalid covariance does not clear prior readiness. Readiness gate and multi-goal variants are optional in the scope campaign. |
| 11 | T02-11 | **P2 confirmed optional-path bug** | Pixel timer selects a timestamp, then applies a newer pixel value under that old timestamp. Requires `use_pixel_correction` and positive timer interval; both disabled here. |

The report below distinguishes these from verified repairs, retained filtering policies, and hypotheses requiring a separate integration trial.

## Evidence and current-path boundary

Read first: `AGENTS.md`, `PLAN.md`, `docs/localization_metrics.md`, `docs/localization_metrics_registry.json`, `docs/open_questions.md`, and `docs/runtime_integrity_audit.md`. Also inspected [the existing estimation review](../state_estimation_module_review.md), [investigation map](../module_issue_investigations.md), the active configuration and launch wiring, and the named nodes/tests. `docs/module_audits/01_state_estimation.md` was not yet present at the last coordinated inspection; the state-estimation investigation owns that report and all filter-mathematics proposals. Its earlier F01/F03/F05/F06/F07/F09/F10 findings remain relevant; they are not claimed as new discoveries here.

The scope reference is [network_navigation_runtime_pilot.yaml](../../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml): fused metric EKF, mandatory source-batch envelope, coupled heading, `/odom_noisy` replay, hierarchical global EFE route plus local `turn_then_go`, 0.25 s control step, 1.5 s correction-gap cap, no hard reanchoring, no pixel correction, `reset_world: false`, isolated campaign cleanup. [Launch defaults](../../src/experiments/experiments/core/visibility_launch_common.py:28) supply 0.5 s pixel/correction age tolerance, 10 Hz belief/command publication and 4 Hz local planning. [Launch construction](../../src/experiments/experiments/core/visibility_launch_common.py:1919) sets `use_sim_time: true`. Camera-manager publication-to-now compensation is disabled; alignment of different camera capture instants to one common capture instant remains enabled.

Source identity matters because the checkout was already extensively modified. [source_manifest.json](../../experiments/runtime_integrity/timing_callbacks_20260906/source_manifest.json) records audit-start hashes. [results.json](../../experiments/runtime_integrity/timing_callbacks_20260906/results.json) records hashes at probe execution and the actual resolved imported source paths. Core files inspected here remained unchanged:

| File | SHA-256 prefix |
|---|---|
| `unicycle_planner_node.py` | `56100b37b8e9351` |
| `motion_history.py` | `4e2f71eb96d2a0` |
| `goal_mission_node.py` | `f2b208e86c3320` |
| `camera_manager_node.py` | `5e11bd202de065` |

Coordinated with **Audit Belief Correction Pipeline**, the command-execution investigation, and **Advance ICRA Localization Paper**. The paper investigation preserved the running frozen pilot, then changed `efe_agent_node.py` and its transaction tests during this audit. Initial EFE hash `33e7c4d36e239a13` became `a575ba53e74a835e`: fatal publication latch, negative tape-age stop, idle diagnostics and opt-in rotation recovery. Initial focused verification had **21 passed / 1 failed** (fatal-stop handoff). The final run has **24 passed**, retained in [existing_tests.txt](../../experiments/runtime_integrity/timing_callbacks_20260906/existing_tests.txt). Those command repairs are credited to that investigation, not to this report. Audit 03 retains its earlier source boundary; do not read its baseline fatal/reset findings as a claim that the later specific tests still fail.

## T02-01 — Input completion order silently becomes motion history

**Location:** [UnicyclePlannerNode 1015–1048](../../src/planning/planning/nodes/unicycle_planner_node.py:1015), [replay 1274–1319](../../src/planning/planning/nodes/unicycle_planner_node.py:1274), [prediction 1849–1898](../../src/planning/planning/nodes/unicycle_planner_node.py:1849). `_odom_cb` sets yaw before taking `_data_lock`; it appends the sample and sets velocity later. Neither path rejects an older sample or validates the resulting order. `_cmd_cb` similarly reads receipt time before acquiring the append lock. The command variant has the same mechanism but was not separately barrier-tested here.

**Trigger and expected behavior:** callback for odometry time 10 s is delayed before locking; callback for 11 s completes; 10 s resumes. Motion history must preserve an explicit temporal order or explicitly refuse unsupported late input. Latest yaw, velocity and origin time must belong to a coherent sample. A late sample must not silently rewrite motion already consumed by the filter.

**Observed reproduction:** `odometry_callback_finishes_out_of_order` uses an event barrier in timestamp conversion. History becomes `[11, 10]`; velocity belongs to 10 s, yaw to 11 s. Real replay over `[9,12]` returns synthetic straight displacement **2 m**, versus **3 m** for the same samples in timestamp order, while reporting no fallback. These unit-speed inputs isolate interval accounting; they are not a deployed-speed or trajectory-error measurement. The discrepancy scales with the scripted velocities. For some start/end partitions the result happens to agree; the probe deliberately puts both samples inside the replay loop.

**Consequence:** correction prediction, control prediction and public belief can integrate the wrong intervals. The bounded-gap checker can reject such a history, but ordinary replay does not call it. A late sample at/before the anchor can also replace the held start velocity through `previous[-1]`.

**Smallest repair:** ingest `(sample_stamp, receipt_stamp, yaw, velocity)` under one ownership boundary; define duplicate/late-input policy and log refusals. Supply a finite, ordered immutable history to every replay. Rejecting late input is the small causal baseline; inserting it retroactively requires explicit rewind/replay, owned with investigation 01. Do not fix this merely by sorting the mutable buffer after some updates already used it.

## T02-02 — Planning initializes the filter outside the envelope contract

**Location:** [state callback 805–824](../../src/planning/planning/nodes/unicycle_planner_node.py:805), [envelope 827–876](../../src/planning/planning/nodes/unicycle_planner_node.py:827), [planning resolver 2655–2664](../../src/planning/planning/nodes/unicycle_planner_node.py:2655).

**Trigger:** compatibility pose arrives before its envelope and a planning callback runs before the identified update. Publishing the envelope first on a separate topic does not impose cross-subscription execution order.

**Expected:** mandatory-envelope mode remains uninitialized until a validated identified correction commits; planning is a read-only consumer.

**Observed:** `planning_bootstrap_bypasses_envelope` invokes the actual callbacks and resolver. The pose initializes the belief at 9.95 s without an assimilation record. The envelope then records `dropped / not_newer_than_belief`. A consumed measurement is described by the ledger as refused. This independently confirms F03 of the earlier estimation review.

**Consequence:** identity, frame/covariance validation and event accounting are bypassed at startup. The planning resolver is a recursive writer despite its read-only description. The analogous per-camera bootstrap can consume the fused compatibility pose before a direct-camera update; that variant was inspected, not separately reproduced here.

**Smallest repair:** remove the resolver bootstrap whenever a correction-owned EKF is configured; return unavailable until its callback initializes the belief. At minimum, guard it by `not require_state_correction_envelope` and restrict compatibility bootstrap to a specifically declared legacy mode. Coordinate the stronger correction-only ownership with investigation 01.

## T02-03 — An old public prediction overtakes a newer prediction

**Location:** [callback groups 601–604](../../src/planning/planning/nodes/unicycle_planner_node.py:601), [belief timer 739–745](../../src/planning/planning/nodes/unicycle_planner_node.py:739), [publisher 2777–2828](../../src/planning/planning/nodes/unicycle_planner_node.py:2777); EFE uses two executor threads ([current main](../../src/planning/planning/nodes/efe_agent_node.py:1382)).

**Trigger:** prediction A targets 10.0 s and blocks; prediction B targets 10.2 s and publishes; A finishes. No correction or anchor replacement occurs.

**Expected:** the public stream must not replace a newer state-time projection with an older one from the same revision. Each message must still use its actual prediction target, never completion time.

**Observed:** `publication_overtakes` uses an event barrier and produces stamps **[10.2, 10.0]**. Each message's synthetic mean, full covariance, frame and timestamp is internally coherent. The final message is nevertheless older. The pointer guard accepts both because their anchor objects are identical. Logger and mission callbacks retain the last arrival without an order gate.

**Reachability:** the belief timer is in `ReentrantCallbackGroup`. Installed ROS Humble `rclpy/executors.py:385–396,466–488` takes a timer and clears `_executor_event` before awaiting its callback; `ReentrantCallbackGroup` allows the next ready invocation to run. Therefore the two-worker layout permits the tested overlap when an invocation exceeds its period. The probe tests real callback methods, not a live DDS/executor schedule.

**Smallest repair:** preserve the existing superseded-anchor check and add a publication watermark `(epoch, target_time, belief_revision)` under the final publication lock; never publish a smaller target within an epoch. Optionally put the timer in its own mutually exclusive callback group to avoid duplicate expensive replay. A monotonic watermark also protects future independent publication triggers.

## T02-04 — Simulation reset leaves contradictory validity decisions

**Location:** [history/state initialization 703–741](../../src/planning/planning/nodes/unicycle_planner_node.py:703), [future-age check 1768–1781](../../src/planning/planning/nodes/unicycle_planner_node.py:1768), [EKF resolver 2669–2671](../../src/planning/planning/nodes/unicycle_planner_node.py:2669), [publisher 2798–2802](../../src/planning/planning/nodes/unicycle_planner_node.py:2798), [batch identity gate 239–253](../../src/perception/perception/core/four_camera_batch.py:239). No application clock-jump reset hook was found in these nodes.

**Expected:** a backward jump invalidates the current epoch, pending work, readiness and old identities. No fresh-looking control state may be constructed from the previous epoch.

**Observed:** `backward_reset_keeps_epoch_state` changes ROS time from 100.1 to 1 s. The old anchor remains at 100 s; odometry history becomes `[100,1]`; source IDs remain. Publication refuses negative age. Planning instead converts the future-age failure to **0.0 s** and returns the old state. A valid new-epoch correction at 0.95 s is dropped as not newer. `camera_batch_reset` independently confirms the strict batcher rejects the first reset frame as `out_of_order`; its per-camera watermark persists. Manager ready-batch/history watermarks and mission readiness also lack an epoch reset by inspection.

**Consequence:** some components stop emitting while planning can install new work from the wrong epoch. Waiting until the simulation catches the old stamp is not a reset policy. Clearing only the command tape is insufficient: a subsequent planning cycle can rebuild one from the stale belief. The paper investigation's new negative tape-age stop repairs the already-installed tape case, not these retained filter/mission states.

**Smallest repair:** treat a negative jump as an explicit invalid epoch and stop until coordinated reinitialization. A supervisor restart of all affected nodes is a smaller complete policy than scattered buffer clears. An in-process reset must clear histories, source IDs, camera watermarks, belief/readiness/goal-progress state, and invalidate in-flight results using an epoch token. Correct the resolver's `None -> 0.0` conversion immediately. The current campaign avoids intentional live reset; no running simulator was reset for this audit.

## T02-05 — The history check does not certify the history actually replayed

**Location:** [outage helper 2309–2323](../../src/planning/planning/nodes/unicycle_planner_node.py:2309), [second history copy 1824–1826](../../src/planning/planning/nodes/unicycle_planner_node.py:1824), [coverage helper](../../src/planning/planning/core/motion_history.py:11).

**Trigger:** odometry arrives and trims the 60 s buffer after `covers_interval(entries, ...)` but before `_predict_belief_to_now` copies it again. `_correction_lock` deliberately permits odometry during a correction.

**Expected:** support validation and integration use the exact same immutable input sequence and target.

**Observed:** `coverage_check_and_replay_use_different_buffers` starts at 1 s with dense 0.1 s samples through 60.9 s. Coverage returns true. A sample at 61.05 s trims the start sample before replay; replay silently assumes zero motion until the first remaining sample. Synthetic unit-speed displacement is **59.8 m instead of 59.9 m**, yet the committed stamp is 60.9 s and the full 59.9 s interval is treated as supported. This is a deterministic retention-boundary example, not a recorded drive.

**Consequence:** even the repaired long-gap coverage gate can authorize a prediction it did not validate. A forward clock/input jump can make the changed interval much larger. Ordinary unsupported replay remains the separate known F01 issue.

**Smallest repair:** pass the copied entries into a shared replay function; return support, integrated duration and diagnostics from that call. Do not hold the I/O lock for the calculation. Coordinate unsupported-motion covariance/recovery policy with investigation 01.

## T02-06 — State time is used as belief identity

**Location:** [belief publisher](../../src/planning/planning/nodes/unicycle_planner_node.py:2777), [camera-manager consumer 1239–1270](../../src/reliability/reliability/nodes/camera_manager_node.py:1239), [capture-matched admission 1365–1368](../../src/reliability/reliability/nodes/camera_manager_node.py:1365). Assimilation records contain `belief_stamp_after` but no revision or posterior state ([2341–2371](../../src/planning/planning/nodes/unicycle_planner_node.py:2341)).

**Trigger:** publish a prediction at ROS time 10 s; apply a delayed capture at 9.95 s; publish again before `/clock` advances. Reentrant work delayed into one clock tick or work finishing while the simulation pauses can produce this. Distinct direct-camera corrections at one capture instant are another legitimate same-time revision source.

**Expected:** a newer belief revision at equal state time must be distinguishable; a consumer must not discard the correction just because time is equal.

**Observed:** `equal_time_revision_lost_by_consumer` publishes x values **0 and 0.0625 m**, both stamped 10 s. The real manager callback retains only the first, x=0. The producer's superseded-anchor pointer guard is correct for an in-flight same-time correction (`same_stamp_supersession_guard`); the downstream timestamp-only identity rule is not.

**Consequence:** camera admission can use the old belief after a corrected one was delivered. Logger retains latest arrival, manager keeps first equal-time arrival, and offline event reconstruction has no shared revision identity. The earlier estimation review's post-correction event-scoring finding is the analysis counterpart; no run scores were computed here.

**Smallest repair:** carry `(epoch, revision, state_time)` in an immutable belief envelope and terminal correction record. Within the existing fixed single-publisher topic, replacing the last equal-time manager sample with the latest arrival is a small mitigation, but requires T02-03's producer ordering repair and does not create an event ledger. Keep state time and revision as separate quantities.

## T02-07 — Planner result provenance is not one input snapshot

**Location:** [input snapshot 2972–2978](../../src/planning/planning/nodes/unicycle_planner_node.py:2972), [belief resolver 2705–2730](../../src/planning/planning/nodes/unicycle_planner_node.py:2705), [base solve/publish 3091–3120](../../src/planning/planning/nodes/unicycle_planner_node.py:3091), [active LOCAL installation](../../src/planning/planning/nodes/efe_agent_node.py:954).

**Trigger:** a correction commits while read-only planning prediction runs, or the goal changes while a solver/controller is computing.

**Expected:** the returned mean, covariance, anchor revision/time, prediction target and frame describe one immutable input. Result acceptance must explicitly assess whether later belief/goal changes invalidate execution.

**Observed here:** `planner_metadata_from_superseding_update` predicts the 9.9 s anchor, then accepts a correction at 9.95 s before returning. Returned mean remains x=0; metadata says `belief_stamp=9.95`, whose actual committed mean is x=0.0625. Mean/covariance were copied together; the metadata was reread later. No prediction target or revision is returned. This metadata field is not itself a logged exact LOCAL snapshot today, so do not infer a specific numeric log corruption from it.

**Execution evidence:** [audit 03, C03-04/C03-05](03_command_execution.md) and its [31-case results](../../experiments/command_execution_audit_20260906/results.json) independently reproduce LOCAL and direct-solver installation after belief/goal changes and ineffective age-gate plumbing. `_plan_group` prevents two simultaneous solves, so a newer solve overtaking an older solve is **not** the demonstrated path. The demonstrated superseding events are corrections, new goals and stops. The global solver creates a frozen route; its controls are not the active actuator tape.

**Smallest repair:** return one planning input record with epoch, belief revision, anchor time, target time, goal revision and frame. Recheck tokens at installation. For a changed belief, revalidate the candidate against a fresh supported prediction or discard/replan according to a declared policy; rejecting every harmless correction without a policy can starve planning. Keep that policy and command-generation ownership coordinated with audit 03. The fatal publication latch is already repaired separately; it does not implement belief/goal revalidation.

## T02-08 — The public frame can come from a different event

**Location:** [compatibility callback 805–807](../../src/planning/planning/nodes/unicycle_planner_node.py:805), [frame lookup 2732–2738](../../src/planning/planning/nodes/unicycle_planner_node.py:2732), [message construction 2819–2828](../../src/planning/planning/nodes/unicycle_planner_node.py:2819). Envelope frame validation also consults this mutable compatibility state at line 848.

**Trigger:** compatibility `/state/bev` changes frame while a belief prediction runs. Mandatory-envelope mode stores the pose without applying it, but publication independently reads its frame.

**Expected:** belief frame belongs to the committed state, covariance and epoch. A changed compatibility frame must not relabel those coordinates.

**Observed:** `frame_from_different_event` delivers `camera_optical` compatibility state during replay. The old `map_bev` belief is published labeled `camera_optical`; pointer validation passes because no belief update occurred. The probe uses the real compatibility callback. The scope manager uses fixed `map_bev`, so this needs a changed/malformed publisher or reconfiguration, not merely normal message delay.

**Consequence:** coherent numeric arrays acquire incorrect coordinate meaning; the frame check can also be influenced by an unrelated compatibility message. Hierarchical planning does not call the base `_validate_plan_frames` method, another inspected difference relevant to this boundary.

**Smallest repair:** own an authoritative belief frame independent of `/state/bev`; validate all inputs against it and copy it with the anchor. Include frame/epoch in publication and installation validation.

## T02-09 — Serializing writers can consume all callback workers

**Location:** [correction decorator 49–55](../../src/planning/planning/nodes/unicycle_planner_node.py:49), [reentrant subscriptions 601–663](../../src/planning/planning/nodes/unicycle_planner_node.py:601), [two-worker executor](../../src/planning/planning/nodes/efe_agent_node.py:1385).

**Trigger:** one correction holds `_correction_lock` during replay; another correction or compatibility callback starts and blocks waiting for it. Both workers are occupied. The corrected single-writer property does not reserve an I/O worker.

**Observed:** `correction_worker_starvation` runs the real correction methods in a two-worker pool. A contention-aware RLock and events prove one worker is calculating and the other waiting. An odometry task cannot execute until release, even though `_data_lock` is free. No sleep threshold or CPU-speed assumption establishes the ordering.

**Consequence:** telemetry, belief publication and periodic command dispatch can miss service while queued corrections occupy the workers; depth-one queues can drop intermediate inputs. One long planner callback plus one long prediction/correction is a second inspected capacity limit. Actual scheduling delay, watchdog response and DDS drops were not measured.

**Smallest repair:** use a mutually exclusive correction callback group for all correction entry points, separate from odometry/command I/O, so a waiting writer is not dispatched into a worker. Keep `_correction_lock` for direct/non-executor entry until ownership is consolidated. Consider a dedicated correction event worker if replay still monopolizes I/O. More threads alone do not establish latency bounds.

**Lock audit:** nested production acquisitions inspected here are correction lock→data lock; data-lock calls into publication/command methods reacquire the same RLock. No current data lock→correction lock inversion or reproducible permanent deadlock was found. Publication occurs while holding `_data_lock`; a slow/blocking publisher would block other data users. That latter middleware behavior remains a hypothesis requiring an isolated transport trial, not a reproduced deadlock.

## T02-10 — Mission readiness is not a freshness decision

**Location:** [mission initialization 63–75](../../src/experiments/experiments/nodes/goal_mission_node.py:63), [belief callback 81–96](../../src/experiments/experiments/nodes/goal_mission_node.py:81), [advance/send 98–145](../../src/experiments/experiments/nodes/goal_mission_node.py:98).

**Trigger:** an old/wrong-frame belief, a pause/reset, or invalid covariance after a valid ready belief. `_belief_cb` overwrites XY immediately, checks only two variances, and leaves the previous readiness flag when covariance is invalid. `_maybe_advance` uses XY without readiness or timestamp checks.

**Observed:** `mission_clock_and_readiness` supplies an old belief in an unrelated frame, rewinds simulation time, then supplies NaN covariance. Readiness remains true and the wall-timer callback advances an intermediate waypoint using that unvalidated position. The configured runtime pilot is a single-goal mission with the initial readiness gate defaulting false; the demonstrated release/advance consequences require enabling the relevant mission options. The consumer's unchecked cached state is present in either mode.

**Expected / smallest repair:** cache one validated `(pose, covariance, frame, state_time, receipt_time, epoch)` record. Reject nonfinite pose/covariance, explicitly clear readiness on invalid input, and require same-frame/same-epoch fresh belief for initial release and waypoint progression. Reset mission progress only under an explicit mission/epoch policy. This does not require replacing the wall startup timer.

## T02-11 — Pixel timer pairs an old stamp with new pixels

**Location:** [timer 1204–1211](../../src/planning/planning/nodes/unicycle_planner_node.py:1204), [pixel receive 1189–1202](../../src/planning/planning/nodes/unicycle_planner_node.py:1189), [correction snapshot 1343–1362](../../src/planning/planning/nodes/unicycle_planner_node.py:1343).

**Trigger:** timer copies `pixel_stamp`; a pixel callback replaces `pixel_meas/pixel_stamp`; timer enters the serialized update using its old stamp. Serialization begins after event selection, and snapshot code rereads the current measurement.

**Observed:** `pixel_timer_reads_new_measurement_with_old_stamp` selects 9.9 s, delivers a pixel at 9.95 s before entering the update, then observes that new value attached to measurement stamp **9.9 s** in the actual correction snapshot.

**Consequence:** replay interval, innovation, duplicate/throttle decision and logging belong to the wrong physical observation. The earlier review's separate duplicate/backdating bug also remains in this optional path.

**Smallest repair:** select an immutable pixel event including value, stamp, diagnostics and identity inside the correction transaction, then pass that event through the update. Merely locking the eventual write is insufficient. Coordinate measurement-model changes with investigation 01.

## Timestamp and clock ownership map

`ROS` below is the node ROS clock, driven by `/clock` in the scope launch. `system` is civil/epoch time; `steady` is `monotonic`/`perf_counter`, suitable for elapsed execution durations. ROS message stamps carry seconds/nanoseconds, **not** an explicit clock-domain or epoch identifier.

| Value / instant described | Domain and owner allowed to set it | Consumer's freshness/ordering rule |
|---|---|---|
| Image `header.stamp`, `PendingFrame.stamp_ns` — camera capture | Simulator/camera producer; detector copies exact integer key. [detector 739–748](../../src/perception/perception/nodes/batched_four_camera_yolo_node.py:739). | Batcher per-camera strict newer watermark, bounded pending age/skew; repeated/older frames refused. Epoch absent (T02-04). |
| `receive_stamp_s` / `yolo_receive_stamp` — detector callback entry, after transport/executor wait | Detector ROS clock; fixed in immutable pending frame. Same lines. | Diagnostic receipt/capture difference; it is not capture or wire-arrival time. |
| `receive_wall_s` — same callback's steady reading | Detector `perf_counter`. | Pending expiration uses elapsed steady time; intentionally expires processing queues even with slow/paused simulation. [batcher 192–220](../../src/perception/perception/core/four_camera_batch.py:192). |
| `yolo_start_stamp`, `yolo_finish_stamp` — inference boundary readings | Detector ROS clock; `_process_frames` [982–1006](../../src/perception/perception/nodes/batched_four_camera_yolo_node.py:982). | Simulation elapsed interval; separate from computation duration. Clock may stay constant during work. |
| `yolo_inference_ms`, callback ms | Detector `perf_counter` subtraction. | Steady elapsed computation/receipt-to-output duration, no subtraction from capture stamp. |
| `yolo_publish_stamp`, frame age — output construction time | Detector ROS clock [1152–1170](../../src/perception/perception/nodes/batched_four_camera_yolo_node.py:1152). | Capture-to-publication age rejects material future capture (>0.005 s); tiny negative tolerance is explicit [frame_age helper](../../src/perception/perception/core/four_camera_batch.py:76). |
| `source_batch_id` — one physical inference batch | Detector derives camera IDs + exact capture nanoseconds [947–949](../../src/perception/perception/nodes/batched_four_camera_yolo_node.py:947). | Manager completeness/ready watermark; planner seen-ID set inside correction transaction. Identity must be scoped by epoch on reset. |
| Per-camera `timestamp_s` / logged `obs_stamp` | Detector contract capture; manager copies it for raw opportunities. | Camera age/admission at manager decision time; capture spread ≤0.05 s in scope configuration. |
| `common_capture_stamp`, `correction_stamp` / pose header / `fused_stamp` | Manager aligns views to newest capture using operational motion, sets one common state instant [343–381](../../src/reliability/reliability/nodes/camera_manager_node.py:343), [1754–1827](../../src/reliability/reliability/nodes/camera_manager_node.py:1754). | In scope, publication-to-now compensation is off, so correction stamp remains common capture time. Optional compensation uses a separately propagated mean/covariance and changes state time to manager ROS `now`; not merely a restamp. |
| Manager decision `now_s` | Manager ROS clock at `_decide` [1524–1532](../../src/reliability/reliability/nodes/camera_manager_node.py:1524). | Measurement age/selection evaluated at decision instant. It is not the fused state's timestamp. |
| Odom header/sample stamp, `_odom_origin_stamp_s`, `_latest_odom_yaw`, `odom_vel` | Encoder/odom producer supplies sample time; planner callback stores it, or falls back to receipt on malformed header. Origin is first callback observed. [1032–1048](../../src/planning/planning/nodes/unicycle_planner_node.py:1032). | No input order/age gate in ingestion; timestamp supports replay, origin supports heading-variance elapsed time. Latest yaw lacks its own attached sample stamp (T02-01; earlier F06). |
| `_cmd_log` timestamps; `last_cmd` | Planner command subscription uses ROS callback receipt time [1015–1024](../../src/planning/planning/nodes/unicycle_planner_node.py:1015); direct command publication also changes `last_cmd`. | Unstamped Twist cannot distinguish issue, transport or application time. Ring trims by received time; no explicit input revision. Audit 03 owns command ledger. |
| `belief_stamp` — recursive anchor instant | Metric/pixel commit, bootstrap, rejection/outage recovery and legacy planning paths. [2374–2640](../../src/planning/planning/nodes/unicycle_planner_node.py:2374). | Correction checks age ∈[−max(timeout,0.25), timeout]; fused times ≤anchor+0.001 s refused. Direct-camera path permits distinct same-time updates after dedup. Reject/recovery may advance anchor without assimilating a measurement. |
| `_last_correction_stamp` — last accepted/bootstrap correction time | Accepted pixel/metric update or reanchor. | Pixel timer throttle compares capture differences; this is not the current recursive anchor after rejection, nor an epoch/revision ID. |
| Correction `apply_stamp` / diagnostic apply time | ROS reads when correction diagnostics/terminal record are built [2341–2371](../../src/planning/planning/nodes/unicycle_planner_node.py:2341). | Processing/report time, after commit; not an atomic exact commit timestamp or source-receipt timestamp. Queue wait before callback is folded into capture-to-apply latency. Ledger does not carry posterior/revision. |
| Prediction target `now_msg`, age and inferred `t_start=target−age` | Planning resolver or belief publisher chooses ROS target; predictor reads it [1803–1826](../../src/planning/planning/nodes/unicycle_planner_node.py:1803). | Periodic publisher uses one target and rejects negative age. Planning handles future anchor inconsistently (T02-04). Motion-support checking is not universal and can use another buffer (T02-05). |
| `/planner_belief.header.stamp`, mean, full planar covariance | Periodic publisher's chosen prediction target, not completion time [2777–2838](../../src/planning/planning/nodes/unicycle_planner_node.py:2777). | Anchor-object check before publishing; no public revision/target watermark. Logger/mission keep last arrival; manager keeps first equal-time sample. Frame is read separately (T02-03/06/08). |
| Planner input snapshot / `belief_meta['belief_stamp']` | Goal/state references copied under lock; mean/P resolved separately; anchor metadata reread after prediction. | No single frozen record, target stamp, goal revision or acceptance token (T02-07). |
| Solve start / solve completion / `solve_time_s`, cycle milliseconds | `perf_counter` for duration; `monotonic` for logging throttle. [base call](../../src/planning/planning/nodes/unicycle_planner_node.py:2996), [global call](../../src/planning/planning/nodes/efe_agent_node.py:827). | Durations describe computation, not robot time. Wall-duration warnings do not establish simulated-time control freshness. |
| `/plan.header.stamp`, path pose headers, metrics emission | Fresh ROS read at solve-result publication [2841–2850](../../src/planning/planning/nodes/unicycle_planner_node.py:2841); all path poses share it. | Publication time for a geometric/predicted path; not each rollout pose's arrival time or exact planner-input time. No separate event/revision key; controls are validated/installed later. |
| `_pending_plan_started_at`, `_active_plan_started_at`, execution age/index/remaining time | EFE ROS readings at local computation and tape installation [918](../../src/planning/planning/nodes/efe_agent_node.py:918), [954–965](../../src/planning/planning/nodes/efe_agent_node.py:954). | Timer indexes simulated elapsed time; current timer rejects negative age. Pending start occurs after belief prediction, so it is not the belief target; LOCAL bypasses solver age gate. Audit 03 covers full timing. |
| Adapter `_last_input_stamp_s`, output diagnostic stamp | Adapter ROS receipt and separate post-output read [114](../../src/sim/sim/actuation_noise_node.py:114), [172](../../src/sim/sim/actuation_noise_node.py:172). | Watchdog uses 0.5 simulated seconds since receipt; pauses with robot simulation. No stamped command age or applied-wheel acknowledgement. |
| Mission `wall_clock`, `start_time`, goal header | Explicit `ClockType.SYSTEM_TIME`; wall timer and elapsed startup delay [69–75](../../src/experiments/experiments/nodes/goal_mission_node.py:69), goal header [135](../../src/experiments/experiments/nodes/goal_mission_node.py:135). | Consumers do not compare goal header with belief time. Readiness uses no timestamp. System jumps affect delay/repetition; see clock-policy verdict below. |
| Logger tick `stamp`, held-message age; goal/stuck/timeout clocks | Logger ROS clock and message state stamps. [goal holds](../../src/experiments/experiments/nodes/experiment_logger.py:1492), [belief extraction/age](../../src/experiments/experiments/nodes/experiment_logger.py:2489). | Goal holds/run timeout use simulation elapsed time; future belief ages are clamped nonnegative in logging. These are separate from source capture and callback receipt. |
| Campaign startup/run timeout; log throttles | Campaign `time.time()` civil elapsed [runner 1405–1428](../../scripts/visibility_comparison/run_visibility_campaign.py:1405); node diagnostic throttles `monotonic`; node solve cost `perf_counter`. | Infrastructure timeout is wall elapsed, intentionally independent of slow simulation. Civil-clock jumps can disturb runner timeout; inspected only, no campaign clock injection. Never compare these directly with ROS stamps. |

## Mission system-time headers: verdict and intentional policies

The mission's wall startup/repeat timer is **not itself a demonstrated bug**. It can release a static goal while simulation time is paused or not yet advancing. The header is currently system time even when the node has `use_sim_time=true`.

The concrete consumers do **not** subtract that header from simulation-time state: [planner `_goal_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1002) retains the goal and computes a signature from frame/XY; [frame validation](../../src/planning/planning/nodes/unicycle_planner_node.py:2980) compares frames; [hierarchical planning](../../src/planning/planning/nodes/efe_agent_node.py:606) uses goal coordinates; [logger `_goal_cb`](../../src/experiments/experiments/nodes/experiment_logger.py:1784) stores the goal and its completion checks use logger tick time. `mission_clock_and_readiness` verifies the actual planner accepts a system-stamped goal while its ROS clock is 0.1 s, without a cross-clock subtraction.

[GoalMarkerNode 53–76](../../src/experiments/experiments/nodes/goal_marker_node.py:53) forwards the header to a visualization marker. A time-dependent TF consumer could care about an epoch header; no such failure was demonstrated in the active configuration, which disables RViz. This is a compatibility risk, not evidence that the mission compares system goals with simulation state.

`mission_system_clock_jumps` runs actual send logic at system times `[99,103,90,104]`, with start=100 and delay=3. Send counts are `[0,1,1,2]`: a backward system adjustment can postpone repeats. If robust elapsed startup timing is required, use a steady timer/duration and stamp goals explicitly with the chosen message clock. This is separate from T02-10's missing belief validity.

Other distinctions that must remain explicit:

- The recent detector `use_sim_time` repair is present in [launch 1215–1221](../../src/experiments/experiments/core/visibility_launch_common.py:1215). No active system-vs-simulation subtraction was established for the core EKF clock in this scope.
- `clock_domain_error_fails_open` supplies a deliberately different `SYSTEM_TIME` clock to `_stamp_age_s`: subtraction raises, the handler returns **0**, and freshness passes. This is confirmed defensive-path behavior, not a claim that normal ROS-clock nodes take that exception. Return invalid/raise on clock-domain failure instead of calling it fresh.
- Prediction helpers omit substeps ≤0.0001 s; the belief publisher skips prediction for age ≤0.001 s but stamps the target. This is a bounded numerical-time approximation by inspection, not the former unbounded computation-latency restamping defect. Exact-instant consumers should know the tolerance; it was not promoted to a separate high-priority finding.
- Planning and periodic prediction of an already initialized coupled metric belief normally leave its recursive arrays/stamp unchanged. Returned diagnostics are shared `_latest_*` fields, however; planning and publication can overwrite them. Earlier estimation-review F07 owns returning diagnostics from the actual replay rather than recomputing/mixing them. Optional legacy state/pixel initialization and recovery still write from resolver code.
- Fused older/not-newer corrections are explicitly refused and recorded. Sorting observations within a direct-camera batch and permitting distinct same-time cameras are deliberate policies. The node is not an out-of-sequence smoother. Preserved first-return gap refusal, rejection inflation, unsupported-motion recovery and per-command-message noise are documented policies, not new repairs here.

## Reproductions and verified repair boundary

[probe.py](../../experiments/runtime_integrity/timing_callbacks_20260906/probe.py) runs **17 deterministic probes** against real node methods with controlled ROS/system clocks, immutable synthetic messages, synchronization events, an observed lock-contention barrier, fake publishers and simple checkable dynamics. It does not initialize ROS or connect to Gazebo. Assertions deliberately pin the observed baseline, **including defects**. Success means reproduction succeeded. [results.json](../../experiments/runtime_integrity/timing_callbacks_20260906/results.json) and [probe_output.txt](../../experiments/runtime_integrity/timing_callbacks_20260906/probe_output.txt) retain results and source identity.

```bash
source install/setup.bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/runtime_integrity/timing_callbacks_20260906/probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/planning/test_runtime_transactions.py tests/planning/test_outage_motion_replay.py
```

| Recent defect / asserted property | Verification and remaining boundary |
|---|---|
| Separate age/target/header clock reads | Existing regression passes after advancing the clock during prediction. Header remains the chosen target. This does not imply monotonic publication order (T02-03). |
| Concurrent corrections compute from one prior | Existing two-thread regression passes. Added simultaneous identical-envelope probe obtains exactly one assimilation; duplicate fails closed after waiting for the transaction. |
| Prediction completes after anchor superseded | Existing changed-stamp regression passes. Added same-stamp real correction also suppresses the old prediction through object identity. Explicit revisions are still absent downstream. |
| Mean, covariance, frame and stamp copied coherently | Full planar 3×3 covariance, including XY–heading cross terms, round-trips correctly in the ordinary fixed-frame case. Barrier publication probe confirms each individual message is coherent. T02-08 proves frame ownership can break that snapshot. |
| Full motion replay across supported long outage | Existing outage regressions pass. They do not cover input mutation between support check and replay (T02-05), or universal ordinary-interval support (earlier estimation F01). |
| Old command timer resurrects stopped/replaced tape | Existing transaction regressions pass; their repaired scope is retained. |
| In-flight command after fatal stop / pre-reset tape | Initial fatal-stop test failed. After the other investigation's source change, the final focused suite passes 24 tests including fatal latch, negative tape age and cleared idle diagnostics. Filter-epoch invalidation and belief/goal revalidation remain open. |

Small repair sequence: establish ordered input ingestion and one checked motion snapshot; remove planning bootstrap; attach frame/epoch/revision/target to each belief and planning record; enforce publication order; repair consumer identity and correction callback scheduling. Coordinate the command installation checks with audit 03 and all filter/recovery semantics with investigation 01. Convert each defective-baseline probe to a desired-invariant regression when its repair lands.

No live callback latency, dropped DDS samples, physical stop distance, paused-simulator behavior or full process-reset recovery was measured. Those remaining integration gates need a separately frozen isolated trial after coordination with experiment owners. The method-level reproductions establish the stated code behavior without perturbing the running study.
