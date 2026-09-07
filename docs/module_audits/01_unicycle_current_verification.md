# 01 — Current unicycle state-update verification

2026-09-07, Europe/Amsterdam. Read-only follow-up to [audit 01](01_state_estimation.md) after package A (`5e844a89`) and command package B (`f9e1a312`). **Package A's stated estimator repairs pass on the current source. The complete runtime update chain is not yet correct.** Missing-motion semantics, capture-time heading, rejected invalid predictions, publication provenance and commit/accounting transactions remain open. This report proposes repairs; it implements none.

## Evidence boundary and ownership

The executed checkout is HEAD `f9e1a312c9990c29223f91f487eeb6c5aa2e71d3` plus the explicitly retained refactor/performance changes. The seven estimator/agent source hashes were identical before and after each probe run. [Initial source identity](../../experiments/estimator_consistency/unicycle_current_20260907/source_identity.json) and the [30-file source/config/test/contract snapshot](../../experiments/estimator_consistency/unicycle_current_20260907/source_snapshot.tar.gz) preserve the reviewed code even if the live files subsequently move. Source line numbers below refer to that initial snapshot. Important initial hashes:

| File | SHA-256 prefix |
|---|---|
| `unicycle_planner_node.py` | `e6566d3ef0956f58` |
| `belief_correction.py` | `0f5f0b32f561ad132` |
| `motion_history.py` | `ac5ea439b956f342` |
| `dynamics.py` | `7a6bde9ddfbc2b10` |
| `base_planner.py` | `135901442b985455` |
| `encoder_noise_node.py` | `f48cbb9dee41b0a4` |
| `efe_agent_node.py` | `f3750f4a4dcb0969` |

Package A's motion-history and encoder files match the implementation note. The unicycle changes after A are configuration parsing and per-camera frame/error handling; the correction-core changes validate shapes, finite moments, positive measurement/innovation covariance and finite NIS. The refactor owner confirmed it has no pending correction-transaction or heading-initialization change. Its shared files were left untouched.

**Final recheck at 00:21 Amsterdam.** Other owners changed navigation defaults/config validation and the local controller's penetration allowance while the report was written. The [recorded diff](../../experiments/estimator_consistency/unicycle_current_20260907/source_drift_after_verification.json) is preserved. A second [28-case execution](../../experiments/estimator_consistency/unicycle_current_20260907/final_results.json), [130-pass/3-skip focused suite](../../experiments/estimator_consistency/unicycle_current_20260907/final_focused_tests.txt), and [actual launch evaluation](../../experiments/estimator_consistency/unicycle_current_20260907/final_wiring.json) succeeded; resolved estimator parameters were unchanged. Final executed unicycle/EFE hashes are `b3e78d48b56b84f` / `c2546f81863703e1`. The [final identity and AST comparison](../../experiments/estimator_consistency/unicycle_current_20260907/final_source_identity.json) and [exact seven-file archive](../../experiments/estimator_consistency/unicycle_current_20260907/final_source_snapshot.tar.gz) show that only unicycle `__init__` and EFE `_simple_plan_safe_to_execute` changed between executions. Afterwards, base-planner constructor validation also changed; its propagation methods remained identical. That later constructor change is recorded, not claimed covered by an earlier test run. No later controller/default change is attributed to audit 01 or certified as a navigation-policy acceptance result.

[The current probe](../../experiments/estimator_consistency/unicycle_current_20260907/probe.py) executes **28 deterministic cases**; [results](../../experiments/estimator_consistency/unicycle_current_20260907/results.json) contain the independent matrices, event ordering, import paths and an AST inventory of callback registrations and writes. All case assertions succeeded. Some deliberately reproduce defects, so this is not 28 correctness passes. The [focused suite](../../experiments/estimator_consistency/unicycle_current_20260907/focused_tests.txt) reports **130 passed, 3 skipped**. The skips are unavailable locked historical pixel traces. The suite covers package A, encoder chronology, fused/per-camera/pixel correction wiring, runtime transactions, Joseph covariance and current input validation.

Timing audit 02 independently ran 19 probes and 90 focused regressions against the same four shared estimator/agent hashes. Its [current handoff and ownership map](02_current_source_handoff.md) and [current results](../../experiments/runtime_integrity/timing_callbacks_revalidation_20260907/results.json) supply the scheduling/publication/reset/accounting evidence below. Those tests overlap this suite; the counts must not be added. Audit 01 owns this report and numerical probes; 02 owns scheduling, publication/revision/frame, planner metadata, reset and callback-starvation analysis. No shared production/test file was edited, and no experiment was started, stopped or modified.

The requested six contract documents, package-A implementation note and coordination queue were reviewed. No drive accuracy was rescored. Synthetic coordinates below are numerical expectations, not camera-performance results.

## Actual reachable configuration

[Current launch evaluation](../../experiments/estimator_consistency/unicycle_current_20260907/wiring.json) reconstructs the named `network_navigation_runtime_pilot.yaml` campaign command, applies the real launch argument defaults and evaluates the encoder and agent Node parameters. It does not execute a launch action or create a ROS graph. Installed Python imports resolve to the reviewed source through the workspace build links.

The selected runtime is `EfeAgentNode`, which inherits the unicycle callbacks and runs a [two-worker executor](../../src/planning/planning/nodes/efe_agent_node.py:1544). It uses metric fused EKF corrections, mandatory identified envelopes, coupled heading, `/odom_noisy`, `use_odom_for_predict=true`, `dt=0.25 s`, correction freshness `0.5 s`, replay-gap cap `1.5 s`, XY/yaw process parameters `0.01/0.02`, rejection inflation `0.05 m²`, and disabled hard reanchoring/pixel correction/coherent-drift option. The launch resolves the prediction speed cap to `0.22 m/s`; the YAML's `0.0` is not the resulting node value. Encoder noise is enabled on `/odom -> /odom_noisy`.

The fast configuration and separate recovery pilot retain this estimator selection; controller/performance differences do not establish that a running process loaded the current code. This is a current-source and named-configuration verification, not deployment certification or complete live-route acceptance.

## Callback and recursive-write map

The state is `m=[x_map,y_map,theta_map]` and a full symmetric `3×3 P`. Motion is forward body speed `v` and yaw rate `omega`; world XY translation rotates `v` by the state heading. `belief_stamp` is the recursive anchor time. **There is no committed belief frame or revision field.** The frame is looked up from mutable compatibility `state_msg`, falling back to `map_bev`.

The [correction decorator](../../src/planning/planning/nodes/unicycle_planner_node.py:50) holds a reentrant correction lock across callback validation, prediction, commit and diagnostic publication. Short data-lock regions protect individual snapshots/writes. Metric and pixel update entry points also take the correction lock, allowing nested calls. The following map covers every registered callback and timer in the base and inherited agent; the raw registration/assignment inventory is in the results JSON.

| Entry point | Active behavior and ownership | Recursive state effect |
|---|---|---|
| [`_state_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:816), `/state/bev` | Correction lock; stores the compatibility message under data lock. Mandatory-envelope mode does not assimilate it. | No active m/P/time write. It **can change the inferred frame**. Legacy non-envelope fused mode invokes metric correction. |
| [`_state_correction_envelope_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:838) | Active fused subscription. Validates schema, frame, finite stamp/XY, positive symmetric R and batch identity, then invokes metric correction. Seen-ID insertion occurs after update/diagnostic work. | Active bootstrap, accepted, held-prediction or outage commit through helpers below. |
| [`_map_observations_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:890) | Subscription only in optional per-camera mode. Parser/frame validation, then serialized sorted-camera application. | Same metric helpers; optional quorum and per-batch inflation also write. |
| [`_odom_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1049) | Validates into locals; data lock atomically accepts strictly newer nanosecond stamp, yaw, velocity, origin and history/trim. | Does not advance recursive m/P/time. Supplies replay and heading initialization. |
| [`_cmd_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1032) | Reentrant callback; reads receipt clock before data lock, then writes last command and appends/trims history. | Does not advance anchor. Command fallback retains ordering/finite-input defects. |
| [`_pixel_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1225) | Correction lock even when correction disabled; stores pixels/stamp. Optional immediate correction or timer mode. | No active metric-anchor write. Optional accepted pixel update writes m/P/time. |
| [`_pixel_correction_timer_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1258) | Created only for enabled/throttled pixel correction. Reads stamp before entering update lock. | Optional pixel update; old stamp/new pixel race remains with 02. |
| [`_detection_diag_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1127) | Stores matching-detection diagnostics under data lock. | None. |
| [`_goal_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1019) | Stores goal, then separately updates progress origin/signature. | None; 02 reproduced torn goal/progress metadata. |
| [`_diagnostic_odom_cb`](../../src/planning/planning/nodes/unicycle_planner_node.py:1104) | Optional diagnostic localization, absent from selected runtime. Stores transformed pose/time. | Bypasses operational localization for planner reads; no recursive-anchor write. |
| [`_plan_once` / `_resolve_belief_for_planning`](../../src/planning/planning/nodes/unicycle_planner_node.py:2782) | Mutually exclusive planning group; EFE override uses the inherited resolver. Active EKF resolver copies anchor and predicts to one target. | Read-only in selected mode after A. Legacy hard-reset and pixel initialization remain explicit exceptions. |
| [`_belief_publish_tick`](../../src/planning/planning/nodes/unicycle_planner_node.py:2854) | Reentrant high-rate timer. Copies m/P, predicts to captured target, drops result if anchor array identities changed. Publishes full planar P. | Read-only anchor. No same-anchor target watermark, epoch or immutable frame yet. |
| [`_publish_active_plan_command`](../../src/planning/planning/nodes/efe_agent_node.py:1414) | EFE command timer, same reentrant I/O group. Package B protects tape ownership/stop/age. `_publish_command` also writes `last_cmd` under data lock. | No m/P/time write; command history is populated by the subsequent command subscription. |
| [`_run_dir_cb`](../../src/planning/planning/nodes/efe_agent_node.py:477) | Agent subscription records artifact destination. | None. |

Every direct recursive writer is accounted for below. All are in `unicycle_planner_node.py`; EFE has no additional m/P/time assignment.

| Writer | m / P / time writes | Reachability and transaction boundary |
|---|---|---|
| [`__init__`](../../src/planning/planning/nodes/unicycle_planner_node.py:732) | All `None` | Startup, before execution. |
| [`_init_belief_from_state`](../../src/planning/planning/nodes/unicycle_planner_node.py:1196) | All from compatibility pose | Optional pixel initialization/reset; data lock, not intrinsically correction-serialized when called by planning. |
| [`_apply_pixel_correction`](../../src/planning/planning/nodes/unicycle_planner_node.py:1777) | Accepted posterior and measurement stamp; last accepted stamp | Optional; holds correction lock, data-lock commit. Rejection still leaves old anchor. |
| [`_resolve_state_belief_for_planning`](../../src/planning/planning/nodes/unicycle_planner_node.py:2145) | Resolved state and pose stamp | Legacy non-EKF mode; planner intentionally writes the recursive fields. Not the active read-only EKF resolver. |
| [`_reanchor_belief_to_xy`](../../src/planning/planning/nodes/unicycle_planner_node.py:2166) | All plus last accepted stamp | Active bootstrap; optional configured divergence/quorum recovery. Called inside metric transactions; heading comes from latest odometry before data-lock commit. |
| [`_inflate_belief_after_rejection`](../../src/planning/planning/nodes/unicycle_planner_node.py:2341) | P only | Active replay-gap branch or optional batch wrapper; correction transaction owns its invocation. |
| [`_advance_belief_over_outage`](../../src/planning/planning/nodes/unicycle_planner_node.py:2357) | Predicted m/P and target stamp | Active refusal of long interval, or early helper refusal. A freezes anchor and both histories together. |
| [`_commit_metric_correction_outcome`](../../src/planning/planning/nodes/unicycle_planner_node.py:2615) | Accepted m/P/time plus last accepted stamp; otherwise held prediction/P inflation/time, or delegates recovery | Active, under caller's correction lock and a data-lock write. Invalid held predictions are not checked. Accepted arrays alias the returned outcome. |

Frame ownership currently consists of the `state_msg` write at line 818 and read in [`_resolve_plan_frame_id`](../../src/planning/planning/nodes/unicycle_planner_node.py:2809). It is not part of any row's commit. Arrays are copied for planner/publication snapshots; stamp/message objects can remain shared. No current production post-commit array mutator was found, so the demonstrated outcome alias is an API hazard, not a claimed spontaneous corruption.

## Verified/fixed map and event-order evidence

| Contract / prior finding | Current result | Evidence |
|---|---|---|
| Outage preserves available odometry including an earlier turn; R02 | **Fixed by A**, earlier full-replay repair preserved | Coverage approves 0–59.9 s. Interfering real odometry callback at 60.2 trims live history to 0.3 s. Immutable replay still yields yaw **0.2 rad**, yaw variance `0.05 + 0.0004×59.9`, and anchor 59.9. Measurement remains explicitly dropped for existing replay-gap policy. |
| Anonymous compatibility initialization; active R03 | **Fixed by A** | Pose→planner→envelope and envelope→pose→planner produce identical m/P/time and exactly one `accepted_bootstrap` event. Compatibility-only planner read leaves belief absent. |
| Out-of-order/equal odometry; chronological part of R01 | **Fixed by A** | Callback order 9.4,9.7,9.5 plus conflicting duplicate 9.7 retains first/newest samples, refuses one old/one duplicate, gives prior x **0.12 m**, yaw Q **0.00024**, not old x 0.16/Q 0.00032. Refusing old samples does not repair gaps already consumed. |
| Encoder old/duplicate input clock; chronological part of R06 | **Fixed by A** | `[0,0.2,0.1,0.3]` at 1 rad/s gives **0.3 rad**, not old 0.4. Added regressions compare pose/P/slip/Jacobian/clock/publication and RNG consequences against clean input streams. |
| Concurrent corrections overwrite one prior | **Previously fixed, preserved** | Lock contention is observed using a nonblocking acquisition/barrier, not inferred from sleeps. Two actual envelope callbacks read anchor times **9.8 then 10.0**. Sequential independent F/Q/Joseph oracle matches final x **0.0953879270895**, Pxx **0.0115470962850**, anchor **10.2**. Both IDs have accepted terminal events at common apply time 10.3. |
| Distinct second camera at same timestamp | **Previously fixed in per-camera mode, preserved** | Real JSON callback applies A and B at 9.95 once each; no motion/Q for the second. Pxx is **0.0115387277902**, matching precision sum of the predicted prior and two R=0.03 readings. Whole-batch redelivery changes no m/P/time or inflation. No per-camera terminal batch ledger is emitted. |
| Rejection freezes belief clock | **Previously fixed for metric path, preserved for finite supported prediction** | At prior time 10.0, fresh 10.1 outlier is rejected by NIS; motion adds 0.02 m and covariance becomes exact predicted P plus `diag(.05,.05,0)`. Stamp becomes 10.1. Old 9.9 and stale 9.0 events leave the already newer anchor unchanged. Optional pixel rejection still freezes its anchor. |
| Gain scaling and Joseph covariance | **Previously fixed, preserved** | Scales 0,0.1,0.5,1 match `(I-KH)P(I-KH)' + KRK'` using scaled K throughout. Active metric gain is 1. Finite-difference state Jacobian error is at most **8.23e-11**; mean/Jacobian use the same body-to-world convention. |
| Prediction/publication modifies recursive estimate | **Verified read-only in active mode** | Actual publish tick and active planner resolver after a complete correction leave m/P/time bitwise unchanged. Same-time correction also invalidates old public snapshot by changed array identity (02). Metadata/publication ordering remains open below. |
| Accumulated drift state | **Mixed, explicitly separated** | Encoder scale Jacobian persists over 50 intervals and reaches `[1.1,0,0]`, producing the declared **0.000484 m²** coherent term. Optional planner `coherent_drift=true` still resets cumulative distance implicitly: 50-step covariance is 50 times below its declared one-window term. Option is off in selected runtime. |

An identified accepted event, identified late drop, identified stale drop and identified NIS rejection were delivered at the **same apply clock** in audit 01. Four corresponding terminal messages were captured with the expected state stamps. Redelivering the rejected batch triggered the existing fatal duplicate-identity path before changing m/P/time or adding inflation. That duplicate produces no additional terminal assimilation message. This verifies producer event identity; it does not close the separate CSV logger's same-clock loss defect.

## One complete active prediction/correction

`complete_trace` uses real odometry callbacks, actual identified-envelope parsing, real `UnicyclePlannerBase.predict`, the correction helper, commit and diagnostic publishers. The independent oracle constructs the continuous frozen-heading linear SDE covariance by a block matrix exponential; it does not call runtime Q. Each selected segment is straight translation or stationary rotation, so its deterministic mean also has an exact elementary solution.

Initial anchor at 9.4 s in `map_bev`:

```text
m = [1, -2, 0.4]
P = [[0.04, 0.005,  0.002],
     [0.005, 0.06, -0.003],
     [0.002,-0.003, 0.025]]
```

Accepted odometry `(stamp,v,omega)` is `(9.35,.2,0)`, `(9.6,0,.8)`, `(9.75,.15,0)`, `(10,0,0)`. Replay uses the preceding sample over each interval: translate during 9.4–9.6, turn 0.12 rad during 9.6–9.75, translate with heading 0.52 during 9.75–10. Full prediction:

```text
m- = [1.069385658998, -1.965790261139, 0.52]
P- = [[0.039941028690,  0.005205224542,  0.001140593188],
      [0.005205224542,  0.059715840852, -0.001257701616],
      [0.001140593188, -0.001257701616,  0.025240000000]]
```

The capture is 10.0, application clock 10.2, and batch ID is `audit:camera_A+camera_B@10000000000`. Measurement is `z=m-[:2]+[.02,-.01]`, `R=[[.015,.003],[.003,.025]]`, `H=[I2|0]`. NIS is **0.009300646714**. The coupled update commits:

```text
m+ = [1.084043447112, -1.973063287815, 0.520637338444]
P+ = [[0.010897785213,  0.001957503397,  0.000298408754],
      [0.001957503397,  0.017616212208, -0.000357663004],
      [0.000298408754, -0.000357663004,  0.025191895229]]
```

Maximum full covariance disagreement with the independent Joseph result is **6.94e-18**. Diagnostic prior/posterior mean entries agree with the calculation; accepted/reason codes are 1/0. The identified terminal event reports `accepted`, capture/anchor 10.0 and apply 10.2. The anchor remains at 10.0 through later publication and planning reads. The terminal wire record does **not** carry these full matrices, frame, prior revision or actual motion support; they are captured here by instrumentation, not recovered from the production ledger.

## Ranked remaining work

P1 indicates causal-state or evidence integrity; P2 is a narrower optional/diagnostic path. “Confirmed” means a deterministic current-source reproduction, not an assertion that the trigger occurred during a drive.

### P1 — U01: a refused invalid update commits a nonfinite prediction

**Confirmed current; additional finding.** [`_cmd_cb`:1032](../../src/planning/planning/nodes/unicycle_planner_node.py:1032) accepts a NaN command. With no odometry, [`_replay_cmd_log_interval`:1325](../../src/planning/planning/nodes/unicycle_planner_node.py:1325) uses that command history. Current [`compute_update`:440](../../src/planning/planning/core/belief_correction.py:440) correctly refuses nonfinite moments, but [`_commit_metric_correction_outcome`:2697](../../src/planning/planning/nodes/unicycle_planner_node.py:2697) copies the invalid `m_pred/S_pred` into the recursive anchor.

Actual command callback followed by a valid identified envelope turns finite state/P at 9.9 into nonfinite state/P at 10.0, while logging `rejected/update_failed`. Expected: invalid numerical input cannot become a committed estimate; refusing a sensor update must not poison the prior. Consequence: the recursive estimate is corrupted despite the new helper validation. Reachability is the active command fallback under a nonfinite-message fault; no such command was established in a running experiment. Smallest repair: reject nonfinite commands before any input mutation and validate the proposed **held prediction** as well as accepted posterior before commit. Use the existing integrity-failure handling for invalid numerical state; do not fabricate a current stamp, tune a gate or snap to the measurement.

### P1 — R01 remainder / R06 remainder: motion-support claims exceed integrated evidence

**Confirmed current.** Regular replay at [1294](../../src/planning/planning/nodes/unicycle_planner_node.py:1294) and planner/publication prediction at [1857](../../src/planning/planning/nodes/unicycle_planner_node.py:1857) do not apply `covers_interval` or report missing prefix/interior/tail support. Package A repairs order and checked-snapshot identity only.

- A sample at 1.0 s is held over 9.9–10.0 despite a newer stop command; correction prior moves **0.02 m**, reports odometry/no fallback and commits time 10.0, although support check is false.
- With first odometry at 9.7 for interval 9.4–10, known scripted 0.2 m/s travel should be **0.12 m**. Replay assumes a stationary prefix, yields **0.06 m**, does not use the available command prefix, and reports full replay duration/no fallback.
- With neither history but nonzero held last command, read-only prediction is **0 m** while the correction prior is **0.02 m** for the same 0.1 s interval. The two entry points implement different missing-input assumptions.
- Encoder input `[0,.1,1.1,1.2]` at 1 rad/s deliberately resets its interval baseline at the long gap, resumes at **0.2 rad** and publishes time 1.2. Scripted integrated heading is **1.2 rad**. Its pose covariance does not identify the omitted 1 rad. The planner consumes encoder twist, not that covariance, but bootstrap uses its pose yaw.
- Unsupported outage fallback over 10 s replays only 1.5 s, stamps 10 and adds yaw variance only **0.0006**, while the other 8.5 s has no heading-support representation. XY reach inflation is an explicitly retained recovery heuristic.

Smallest model-preserving package: return one immutable replay result with selected source, actual segments, support gaps and integrated/assumed duration, shared by correction and read-only prediction. Carry that support with the anchor and terminal event, and preserve the predecessor input required at a retained interval boundary. Do not silently label the existing unsupported fallback as measured motion. Deciding how to estimate/control through unsupported time, or allowing the first camera return after 1.5 s, is a separate recovery-policy decision; instrumentation alone does not close that behavioral limit.

Audit 02 additionally confirms `_cmd_cb` can finish in order `[11,10]` while preserving the old command. Its sorted-history oracle is **0.3 m**, observed replay **0.2 m**. Smallest ordering repair is to capture receipt time inside the same accepted-command append transaction, with reset handled by 02's epoch contract. Command ordering was in the planning discussion but was **not** part of committed A.

### P1 — R05: moving bootstrap assigns current yaw to past capture

**Confirmed current active bootstrap**, also affecting optional reanchoring and `camera_xy_only`. [`_reanchor_belief_to_xy`:2170](../../src/planning/planning/nodes/unicycle_planner_node.py:2170) calls unstamped latest-yaw lookup [1964](../../src/planning/planning/nodes/unicycle_planner_node.py:1964), but stores camera capture time and evaluates yaw variance at that earlier time.

Real odometry callbacks describe raw yaw 0.8 at capture 9.8 and 1.0 at receipt 10.0, with map offset 0.3. Expected capture heading is **1.1 rad**, current-time heading **1.3**. Bootstrap stores **1.3 at 9.8**; subsequent supported replay adds the same 0.2 turn again and returns **1.5 at 10.0**. The covariance's capture-time yaw variance is 0.00032 even though the mean comes from a later pose.

Smallest repair: retain timestamped yaw with accepted odometry, query/propagate a heading to the correction's capture time using the same immutable supported motion snapshot, and evaluate its declared covariance at that time. Cover delayed turn, nonzero offset, angle wrap, bootstrap and configured recovery. Do not replace the heading with a camera-derived angle or silently substitute latest yaw when capture-time support is missing. The latter case belongs to the explicit support contract.

### P1 — 02 transaction/publication findings: lock protection does not establish complete event identity

**Confirmed by current audit 02 at matching source hashes.** Its evidence shows:

- Two predictions of the same anchor publish stamps **10.2 then 10.0**. Each message is internally coherent; the public stream regresses because array identity alone is not a target-time watermark.
- A true same-time correction changes public x from 0 to **0.0625**, but the manager retains the first 10.0-stamped value. State time is not a belief revision.
- Planner result mean comes from anchor 9.9 while its returned metadata reports later committed anchor 9.95. A compatibility frame change can relabel `map_bev` coordinates as `camera_optical` without a transform.
- Reset retains old anchor 100 and old batch IDs; new-epoch odometry is now correctly refused as older by A, while the planning age becomes 0 and the epoch-invalid estimate remains usable. An in-flight public prediction is not invalidated merely by reset.
- A diagnostic publisher exception after a real metric commit leaves posterior x **0.0625** at 9.95, **no seen ID and no terminal assimilation record**. The real fatal handler is invoked. This is a failed commit/accounting transaction, not demonstrated continued normal operation or double assimilation.

Relevant boundaries are [compatibility/envelope handling](../../src/planning/planning/nodes/unicycle_planner_node.py:816), [metric commit then publication](../../src/planning/planning/nodes/unicycle_planner_node.py:2571), [planning metadata read](../../src/planning/planning/nodes/unicycle_planner_node.py:2801), and [public snapshot check](../../src/planning/planning/nodes/unicycle_planner_node.py:2900). Repair ownership remains with 02 and the coordinator: immutable m/P/time/frame/revision snapshot, epoch-aware eligibility, monotonic publication, and commit/identity/terminal-record ownership before fallible diagnostic work. Durable storage and consumer revision handling require 10/11/manager coordination. Do not infer that locking already solves these tasks.

### P2 — R07: diagnostics disagree with motion or final heading, and terminal state is incomplete

**Confirmed current.** Prediction diagnostics call `process_noise(age)` without theta/v at [1928](../../src/planning/planning/nodes/unicycle_planner_node.py:1928), entering its diagonal `dt/base_dt` fallback. At dt 0.1 and base step 0.25, yaw Q is reported as **0.00016** while the applied value is **0.00004**. The delta-yaw diagnostic at [1892](../../src/planning/planning/nodes/unicycle_planner_node.py:1892) assigns the new sample's angular velocity to the preceding interval: the tested actual turn is **0.12 rad**, reported delta **0.16**.

In optional `camera_xy_only`, the commit writes back final m/P, but `outcome.yaw_info` still describes the discarded coupled heading update. The fixture's actual heading change is **0.6 rad**, reported change **0.003749602384**. Its final covariance has zero XY/yaw cross terms and remains PSD; the diagnostic, rather than that tested matrix transformation, is wrong.

Smallest repair: carry actual accumulated replay diagnostics and regenerate heading summary after final commit transformation. Emit committed prior/posterior full P, frame, anchor/target times, revision and support from the same transaction; don't rebuild them from mutable node fields later. R04's offline posterior selection and R08's same-clock CSV loss were established in the original audit; they were **not re-executed in this bounded pass** and remain with audits 10/11, not newly verified or closed here.

### P2 — Optional paths: confirmed current defects remain distinct from active fused operation

| Path / original ID | Trigger, expected vs observed | Smallest repair boundary |
|---|---|---|
| Per-camera quorum; R11 | Stale agreeing cameras at 9 with belief 10/now 12 reanchor backwards to 9 before per-reading freshness/order checks. Two fresh R=.03 readings are used to seed R=.03 and then both updated again, producing **.01 instead of .015** variance (three information terms from two observations). | Validate eligible causal readings before recovery and avoid assimilating the initialization evidence again. Preserve configured thresholds; do not add a camera exclusion rule. |
| Per-camera outcome handling; R12 | Single-camera bootstrap returns `None`; wrapper declares no acceptance and turns **R=.03 into P=.08**. A stale-only batch adds .05 despite no update; one long-gap batch adds **.10 instead of configured once-per-batch .05**. | Explicit terminal enum/result for bootstrap/drop/reject/recovery; exactly-once batch inflation decision. Long-gap helper must respect its `inflate_on_reject` contract. Keep the .05 value unchanged. |
| Near-simultaneous per-camera stamps; R13 | A at 9.95 accepted; distinct B at 9.9505 dropped as not newer, despite positive dt. Exact-same-time is repaired, positive intervals ≤1 ms are still excluded at [2506](../../src/planning/planning/nodes/unicycle_planner_node.py:2506). | Separate integer timestamp ordering from any numerical integration cutoff; handle a new physical event once with its actual time interval. |
| Legacy pixel identity/time; R14 | Repeated 9.95 measurement reduces Pxx **.022222→.014286**, older 9.9 then backdates anchor; rejection at 9.95 leaves stamp 9.9. Timer also can mix old stamp/new pixels (02). | Immutable identified pixel event, causal/dedup handling and explicit commit outcomes before enabling this path for new comparisons. Preserve historical fixtures as historical. |
| Optional coherent planner drift; R10 | 50×0.1 s predictions reset the distance default each call, yielding 50× too little declared coherent variance. Encoder's separate scale state is verified persistent. | Carry cumulative coherent state through prediction, not a new Q amplitude. Do not enable the option as part of a correctness patch. |
| Encoder stop/disabled metadata; R09 | Noise generator suppresses perturbations at stop; covariance helper still grows yaw variance **.0001→.0005** over 10 s. | Make metadata describe the existing generator branch. Planner EKF does not consume encoder P; statistical AR/white-model identification remains separate. |

## Intentional limits and untested hypotheses

The first fresh return after an interval beyond 1.5 s is still refused even when A proves full odometry support. Existing 0.05 m² rejection inflation, 60 s history horizon, unsupported XY-reach fallback, finite age/future tolerance, and strict same-time **fused** event rule are retained policy/compatibility boundaries. This report does not change them or recommend a blanket camera restriction. Applying a reordered event older than an already committed posterior would require an explicit rewind/replay/smoothing design; current causal dropping is intentional. The positive ≤1 ms exclusion of a newer event is separately reproducible above.

The default dynamics use Euler mean propagation and a frozen-heading continuous-noise covariance. At v=.2, omega=1, T=1, one step gives XY **(.2,0)**; ten steps **(.1727509,.0834482)**; the exact continuous arc is **(.1682942,.0919395)**. Mean Jacobian and Q are internally consistent with each declared step; turning results depend on segmentation as an approximation property. Straight white-noise propagation composes to machine precision (one 5 s step versus fifty 0.1 s steps). This is not evidence to retune Q. Changing integration or the encoder's per-message AR/random-sampling model needs separate model/cadence validation.

Not established here: live trigger rates; multi-process DDS delivery/failure behavior; long-run linearization/calibration; a covariance-correct unknown-motion/relocalization model; a current production mutation of aliased outcome arrays; validity of arbitrary externally supplied odometry frames/quaternion conventions. The configured producer uses planar body twist and the expected odom frame, but the planner does not independently validate that full external contract. Numerical fault injection establishes a commit invariant failure without asserting that the current simulator produced such a message.

## Smallest remaining repair packages and acceptance

1. **Numerical commit integrity (U01).** Coordinate with the refactor's validation owner and 02. Validate finite command input before mutation and the candidate mean/full P before each accepted/held/outage commit; keep the last valid anchor intact on integrity failure. Regressions must use actual command/odom/envelope callbacks and assert prior/clock/IDs/outcome, not only helper return values. No Q/R/gate/recovery change.
2. **Complete 02's immutable belief transaction.** One owned record for m/P/time/frame/revision/epoch; identified disposition constructed with the commit; diagnostic failure must not determine whether identity is retained. Add publication target/revision watermark, epoch invalidation and matching manager/planner metadata. Test same-time correction, reordered publications, reset mid-compute, and publisher exception. Coordinate terminal schema with 10/11 before shared edits.
3. **Motion provenance, then capture-time heading.** Share immutable replay segments/support across reads and commits; finish command chronology. Retain timestamped odometry heading and query the capture instant with the same support contract. Test leading/interior/tail gaps, empty histories, trim race, yaw wrap and delayed turning bootstrap. Recording existing fallback honestly is an immediate correctness/provenance repair; changing how the vehicle continues through unavailable motion is a separately specified policy package.
4. **Optional per-camera transaction semantics.** Return explicit bootstrap/drop/reject/accepted/reanchored results, consume evidence once, make inflation once per eligible batch and compare true timestamp ordering. Preserve current thresholds, R and camera availability. Complete per-camera identity/terminal accounting before a controlled fusion comparison.
5. **Diagnostics and inactive model paths.** Use applied Q/yaw increments and final heading state for diagnostics; retain full committed covariance in the transaction record. Repair the optional coherent accumulator and pixel identity path separately, without enabling either in the active baseline.

The original report at the requested `docs/module_audits/01_state_estimation.md` remains the frozen baseline and now points here. Source changes after this snapshot must be revalidated rather than inheriting these test results. Read-only ownership coordination succeeded, and audit 02 read and cross-linked this report. Automatic approval review blocked the detailed U01 message to the refactor chat and the final report/evidence handoff to the coordinator, citing unverified destination ownership and nonpublic audit content. Both concrete handoffs remain local pending explicit authorization; no alternative messaging route was used.
