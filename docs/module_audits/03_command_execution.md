# 03 — Command execution and actuation

2026-09-06. **The frozen runtime baseline has incomplete stop precedence: an in-flight computation can install motion after a stop, including a fatal stop, and clock reset can replay an old tape. During this audit, the coordinated paper chat repaired the fatal publication boundary, negative tape age and held stop diagnostics; the follow-up focused suite passes 17 tests. Ordinary stop cancellation, stale-plan admission, adapter protection gaps and command-event logging remain open.** The adapter watchdog protects against planner receipt silence while the adapter, its clock and downstream delivery remain healthy; it does not establish command age or protect against its own death.

This is a ranked software audit, not a navigation result. No experiment was started, stopped, rescored or modified. No production code or configuration was edited. Filter-Q modelling belongs to investigation 01; route choice, waypoint geometry and controller performance belong to investigation 09.

## Ranked findings

Priority ranks consequences and reachability, not observed frequency in experiments. This table and detailed reproductions describe the **frozen baseline hashes below**, before the concurrent follow-up. “Confirmed” means reproduced against those methods or directly established by configuration; physical consequences remain unmeasured where stated. See the follow-up status for the newer working tree.

| Rank | ID | Classification | Finding and active-path reachability |
|---|---|---|---|
| 1 | C03-01 | **P1 confirmed current bug** | In-flight computation can override stop/fatal-stop. Active hierarchical LOCAL installer and optional direct solver handoff both reproduce it. |
| 2 | C03-02 | **P1 confirmed current bug** | Backward clock jump resets tape age to zero; either ordering of adapter receipt/watchdog can finish with nonzero output. Active on clock reset, not normal monotonic operation. |
| 3 | C03-03 | **P1 confirmed protection gap; physical persistence hypothesis** | Adapter restart starts disarmed without publishing zero; adapter death/shutdown removes the only configured watchdog. Active failure boundary; no live death/restart trial performed. |
| 4 | C03-04 | **P1 confirmed behaviour; documented policy/design gap** | Belief or goal can change during computation and the old result still installs. Active LOCAL path; solver path also reproduced. Revision/revalidation policy is a separate design change. |
| 5 | C03-05 | **P2 confirmed current bug** | Plan-age acceptance is bypassed: LOCAL never calls the age gate; direct solver never initializes its pending-start time. Active LOCAL, optional direct solver. |
| 6 | C03-06 | **P2 confirmed current logging bugs** | Logger mixes command events; execution diagnostics remain nonzero after expiry; no complete requested→received→applied event ledger. Active logging path. |
| 7 | C03-07 | **P2 documented intentional policy, reproduced** | Noise advances per message, so duplicate and extra immediate publications change time-indexed disturbances. Active with command noise enabled. |
| 8 | C03-08 | **P2 timing/transport policy limits, reproduced** | Tape validity ends between timer ticks; delayed unstamped messages can restart output after watchdog zero. Active; strict physical expiry is not promised by current transport. |
| 9 | C03-09 | **P3 boundary hardening / producer-dependent hypothesis** | Result acceptance omits explicit command bounds and permits absent validity fields. Synthetic malformed producer result passes; active simple tracker currently supplies bounded finite controls. |

## Evidence, configuration and coordination

Read first: `AGENTS.md`, `PLAN.md`, `docs/localization_metrics.md`, `docs/localization_metrics_registry.json`, `docs/open_questions.md`, and `docs/runtime_integrity_audit.md`. Also checked `docs/state_estimation_module_review.md`, `docs/module_issue_investigations.md`, and `experiments/icra_commissioning/planner_implementation_plan.md`. No `docs/module_audits` directory existed at initial inspection.

The scope configuration is [network_navigation_runtime_pilot.yaml](../../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml): hierarchical global EFE plus LOCAL `turn_then_go`, `dt=0.25 s`, `v_max=0.22 m/s`, command and encoder noise enabled, fused metric correction and odometry-based prediction. Launch defaults give LOCAL horizon 12, LOCAL planning 4 Hz and publication 10 Hz; handoff latency compensation defaults **false**. Adapter limits are `[0,0.22] m/s`, `[-1,1] rad/s`, default timeout 0.5 simulated seconds. See [launch defaults](../../src/experiments/experiments/core/visibility_launch_common.py:46), [local defaults](../../src/experiments/experiments/core/visibility_launch_common.py:122), [adapter construction](../../src/experiments/experiments/core/visibility_launch_common.py:1160), and [command routing](../../src/experiments/experiments/core/visibility_launch_common.py:1920).

This selection is a configuration/source reference, not a selection of run measurements. The [corrected protocol](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json) matches all four overlapping checked files exactly:

| File | SHA-256 prefix |
|---|---|
| `efe_agent_node.py` | `33e7c4d36e239a13` |
| `unicycle_planner_node.py` | `56100b37b8e9351` |
| `actuation_noise_node.py` | `ce98aba1300bafc` |
| scope campaign YAML | `72d7318718c9510` |

Full hashes, including launch, URDF and logger files not named in that protocol, are in [results.json](../../experiments/command_execution_audit_20260906/results.json). Line references describe these bytes. The checkout has extensive pre-existing changes; HEAD alone does not identify it.

Coordinated with **Audit timing and belief ownership**: investigation 02 owns recursive state/publication ordering; this report owns command invalidation, stop/fatal handoff, command diagnostics and actuator timing. **Advance ICRA Localization Paper** confirmed that the fatal-stop test was newly added and failing and P2 collection retained the frozen source. It subsequently implemented the follow-up below. This audit did not edit shared runtime files. Recheck hashes before applying a finding to changed code.

### Follow-up status at final verification

The paper chat's newer `efe_agent_node.py` SHA-256 is `a575ba53e74a835ee955625b14e5718c09939fc36d206ef8b35b4a0c405818ae`. Its diff against the frozen source adds a fatal check at `_publish_command`, clears any newly installed tape when fatal, refuses negative/nonfinite tape age, and emits idle execution diagnostics on stop/expiry. It also connects opt-in rotation recovery, which belongs to investigation 09.

Independent rerun of the two focused test files on this follow-up: **17 passed**, retained in [followup_tests.txt](../../experiments/command_execution_audit_20260906/followup_tests.txt). Thus C03-01's fatal flag case, C03-02's negative-age timer case and C03-06's held execution diagnostic are repaired for the tested conditions. Ordinary stop-generation cancellation, in-flight computations spanning reset, adapter reset/receipt ambiguity, logger event mixing and the remaining design boundaries are not resolved by those checks. No physical stop guarantee or navigation improvement is inferred. Frozen-baseline probes remain reproducible with `--frozen-baseline`; they intentionally assert the old behaviour.

## Complete command lifecycle and observability

The active path has two different kinds of “plan”: a global solver produces route states, while a geometric LOCAL tracker produces the executed control tape. The global solver's controls are **not** sent directly to the actuator.

| Stage | Actual path and boundary | What existing logs establish |
|---|---|---|
| Originating goal/belief | Goal reference snapshot and separately resolved belief at [EfeAgentNode 605](../../src/planning/planning/nodes/efe_agent_node.py:605); goal callbacks can replace `goal_msg` under the I/O lock at [1002](../../src/planning/planning/nodes/unicycle_planner_node.py:1002). Belief resolution predicts from the filter/odometry; no command revision is captured. | Published belief stream is not an immutable record of each LOCAL computation's exact starting belief. No goal/belief revision linked to tape. |
| Global solver | `global_planner.plan(m0,S0,final_goal)` at [826](../../src/planning/planning/nodes/efe_agent_node.py:826); route states become waypoints. Path/metrics/artifacts are published at [882](../../src/planning/planning/nodes/efe_agent_node.py:882). | Global route, costs and solver information. This is requested/predicted motion, not executed motion. |
| LOCAL requested controls | Snapshot-derived `m_track,target` enters controller at [938](../../src/planning/planning/nodes/efe_agent_node.py:938); finite controls and geometric safe prefix checked at [1133](../../src/planning/planning/nodes/efe_agent_node.py:1133). | Waypoint/control diagnostics are partial and held. No complete LOCAL request/result/acceptance event record. |
| Acceptance and installation | LOCAL copies safe prefix and sets a fresh installation time at [950](../../src/planning/planning/nodes/efe_agent_node.py:950). Optional direct solver uses `_result_safe_to_execute` and `_after_plan_result` at [1189](../../src/planning/planning/nodes/efe_agent_node.py:1189), [1294](../../src/planning/planning/nodes/efe_agent_node.py:1294). | Rejection warnings may exist. No explicit accepted tape ID or originating revision. Direct solver path/metrics publish before command acceptance at [3029](../../src/planning/planning/nodes/unicycle_planner_node.py:3029). |
| Latency compensation | LOCAL installs at “now” with zero skip. Optional hook truncates whole steps and backdates by the fractional remainder at [1307](../../src/planning/planning/nodes/efe_agent_node.py:1307). | Skip fields describe intended tape handling, not measured executed motion. |
| Planner publication | Immediate first control on install plus 10 Hz timer, both under `_data_lock`; timer selects `floor(age/dt)` and expires tape at [1240](../../src/planning/planning/nodes/efe_agent_node.py:1240). `_publish_command` emits unstamped `Twist` on `/cmd_vel_raw` at [1181](../../src/planning/planning/nodes/efe_agent_node.py:1181). | Logger receives raw topic; timestamps are its own receipt times. Execution diagnostic is emitted before periodic command publication, with no source timestamp/ID. Immediate publications and stops lack corresponding execution diagnostic records. |
| ROS adapter receipt | Adapter depth-one subscription at [75](../../src/sim/sim/actuation_noise_node.py:75); callback records its local simulated receipt time at [114](../../src/sim/sim/actuation_noise_node.py:114). | Internal receipt time is not in an event ledger. Noise diagnostic timestamp is a second clock read after output publication. Logger receipt is a third boundary. |
| Noise, stop deadband, clipping | Finite input check; joint near-zero stop test; correlated slip plus additive noise; clamp at [106](../../src/sim/sim/actuation_noise_node.py:106). | Normal adapter diagnostic contains paired input/output values and noise components. Watchdog and nonfinite stops have no such diagnostic. Logger holds the latest diagnostic rather than retaining every event. |
| Adapter output | `/cmd_vel` publication at [169](../../src/sim/sim/actuation_noise_node.py:169), or watchdog zero at [200](../../src/sim/sim/actuation_noise_node.py:200). | `cmd_v/w` are commands observed by the logger on the adapter-output topic. They are not measurements of wheel or body velocity. |
| Bridge/Gazebo receipt | ROS→Gazebo bridge at [bringup 398](../../src/sim/launch/bringup_sim.launch.py:398), remapped at [418](../../src/sim/launch/bringup_sim.launch.py:418), delivers to `/model/turtlebot3/cmd_vel`. | No per-command bridge or Gazebo receipt acknowledgement is recorded. |
| Actuator and physical motion | [Robot description launch](../../src/sim/launch/robot_description.launch.py:48) loads warehouse URDF. [DiffDrive block](../../src/sim/robot_description/urdf/warehouse_amr.urdf.xacro:269) specifies wheel radius 0.10 m and separation 0.44 m, 50 Hz odometry, but no timeout/velocity/acceleration limits. | Wheel odometry and ground-truth pose can describe outcomes at their own stamps; neither identifies which individual command was physically applied. No applied-command ledger or wheel-command acknowledgement. |

Upstream [Gazebo Sim 8 DiffDrive source](https://raw.githubusercontent.com/gazebosim/gz-sim/gz-sim8/src/systems/diff_drive/DiffDrive.cc) stores the received target, updates velocity limiters, converts it to right/left wheel targets `(v ± omega*0.44/2)/0.10`, then writes `JointVelocityCmd` during unpaused updates. The inspected source has no receipt timeout. This supports the downstream persistence concern; the installed binary was not source-verified or exercised here. Wheel targets are still not physical body velocity: contact and physics intervene.

## Detailed findings and smallest proposed repairs

### C03-01 — In-flight installation defeats stop precedence

**Reachability:** active LOCAL and optional direct solver. Two-thread executor and reentrant I/O permit a correction/fatal event while planning runs; the planning callback group only serializes planners with each other. References: [groups/locks](../../src/planning/planning/nodes/unicycle_planner_node.py:601), [fatal handler](../../src/planning/planning/nodes/unicycle_planner_node.py:773), [LOCAL installer](../../src/planning/planning/nodes/efe_agent_node.py:950), [solver installer](../../src/planning/planning/nodes/efe_agent_node.py:1326), [stop](../../src/planning/planning/nodes/efe_agent_node.py:1336).

**Trigger:** start computing a tape; publish stop and optionally set `_fatal_stop_triggered`; return the old computation. **Expected:** that computation cannot authorize further nonzero publication. A later, deliberately fresh ordinary plan may resume after a nonfatal stop; fatal stop must stay latched. **Observed:** LOCAL publishes `(0,0)` followed by `(0.2,0.1)`; solver hook publishes `(0,0)` then `(0.2,0.3)`, even with fatal flag true. Neither installer nor publication boundary inspects the flag. Shutdown is not a command-ownership barrier; publication may race before context teardown. A failed publish after teardown does not disprove the pre-teardown interleaving.

**Reproduction:** `local_stop`, `local_fatal`, `solver_after_stop`, `solver_after_fatal`; the existing `test_inflight_plan_cannot_publish_motion_after_a_fatal_integrity_stop` fails. Fatal probes reproduce the flag/zero side effects, not ROS shutdown transport.

**Consequence:** a stop is followed by a fresh-looking nonzero command, extending motion until another stop, expiry or receipt watchdog. **Smallest repair:** under the command lock, reject/zero all nonzero publications when fatal is latched and clear any tape installed afterward. Increment a stop generation on every stop, capture it before computation, and reject installation if changed. Apply to both installation paths. Test fatal entry before acceptance, between acceptance/install and at publication; retain the existing timer regressions. This cancellation generation is correctness work, distinct from a broader belief/goal revision design.

### C03-02 — Reset can revive a pre-reset tape and defeat reset detection

**Reachability:** active on backward `/clock` movement. References: [remaining time](../../src/planning/planning/nodes/efe_agent_node.py:596), [negative-age clamp](../../src/planning/planning/nodes/efe_agent_node.py:1255), [adapter receipt overwrite](../../src/sim/sim/actuation_noise_node.py:114), [watchdog](../../src/sim/sim/actuation_noise_node.py:190).

**Trigger:** installed at 9.9 s, clock becomes 9.0 s. **Expected:** invalidate the old epoch's commands. **Observed:** planner publishes `(0.2,0.1)` with age zero. If this receipt precedes watchdog, the old receipt time is overwritten and negative age disappears. If watchdog runs first, it publishes zero, but the subsequent old-tape delivery resumes nonzero. Both explicit orderings are reproduced (`planner_clock_reset`, `reset_receipt_first`, `reset_watchdog_first`).

**Consequence:** motion can continue while the old absolute timestamp lies in the future, then execute the tape again. The adapter's isolated reset test is insufficient for this composition. **Smallest repair:** recognize backward clock changes as command invalidation at the planner and adapter, clear tapes, and require a post-reset computation before motion. Detect receipt-time reversal before overwriting the prior timestamp. That alone cannot distinguish stale queued messages from fresh ones; explicit command epochs/stamps are the later transport design.

**Cross-audit follow-up:** [investigation 02, T02-04](02_timing_and_callbacks.md#t02-04--simulation-reset-leaves-contradictory-validity-decisions) reproduces a further reset interaction: after time changes from 100.1 to 1 s, the planning resolver retains the belief anchored at 100 s and converts failed future-age validation to age zero. Thus even the repaired tape timer can be followed by a new tape computed from old-epoch state. This is investigation 02's `backward_reset_keeps_epoch_state` result, not an additional command-probe case. Epoch validity must reach planning/installation; merely starting a computation after reset is insufficient. The scope campaign sets `reset_world: false`.

### C03-03 — Adapter restart, death and exception boundaries lack end-to-end stopping

**Reachability:** active adapter failure, restart or teardown; also `use_command_noise=false` removes the adapter entirely through [launch 1160](../../src/experiments/experiments/core/visibility_launch_common.py:1160). References: [initial watchdog state](../../src/sim/sim/actuation_noise_node.py:68), [early return](../../src/sim/sim/actuation_noise_node.py:196), [shutdown](../../src/sim/sim/actuation_noise_node.py:205), [planner teardown](../../src/planning/planning/nodes/efe_agent_node.py:1346), [URDF](../../src/sim/robot_description/urdf/warehouse_amr.urdf.xacro:269).

**Trigger:** downstream previously held nonzero; replace adapter with fresh instance and give it no new input. **Expected for an actuator safety boundary:** explicitly reach stopped state even without another planner message. **Observed:** constructor state is `_last_input_stamp_s=None`, `_watchdog_stopped=True`; ticks at 100 s emit nothing (`adapter_restart`). Normal teardown destroys the node without a final zero. An uncaught callback exception that unwinds spin has the same teardown boundary. If the adapter itself is dead, it cannot run its watchdog.

**Consequence:** there is no demonstrated bound on downstream hold after adapter/bridge failure. Restart silence is confirmed; continued physical motion is a source-supported hypothesis, not a measured live failure. **Smallest local mitigation:** start in an unconfirmed-stop state, send/retry zero after publisher matching, and best-effort zero before orderly teardown. A one-shot constructor publish is not sufficient evidence of delivery. **Complete boundary repair:** a timeout at the final actuator owner, independent of this adapter and bridge. A new downstream gate/watchdog needs isolated integration verification; do not invent an unsupported DiffDrive timeout parameter.

### C03-04 — Originating belief and goal are not revalidated

**Reachability:** active LOCAL, global route return, optional direct solver; no goal revision or belief identity in the command tape. References: [snapshot/resolve](../../src/planning/planning/nodes/efe_agent_node.py:609), [LOCAL snapshot-derived tracking](../../src/planning/planning/nodes/efe_agent_node.py:921), [goal callback](../../src/planning/planning/nodes/unicycle_planner_node.py:1002), [direct solve→publish](../../src/planning/planning/nodes/unicycle_planner_node.py:3091).

**Trigger:** controller/solver begins with belief `(0,0,0)` and goal `(1,0)`; before return, commit belief `(0,2,1)` or receive goal `(-1,0)`. **Expected under a revision-aware design:** reject or revalidate before publication. **Observed:** old `(0.2,0.1)` installs in both the full LOCAL callback and the full direct `_plan_once` call. `local_correction` models the committed effect of a correction without rerunning filter mathematics; `full_solver_correction` changes the same owned state during the fake solver. `local_goal` and `full_solver_goal` invoke the real goal callback. Existing tapes also continue across corrections; the timer has no revision check.

**Consequence:** a geometric safety check made at an obsolete pose can cease to apply, or motion can target an old goal. A later planner tick normally notices a goal change; it does not make the current installation valid. The registered pilot has one mission goal, reducing goal-change exposure there; camera corrections remain active.

**Smallest proposed design:** record stop generation, goal revision and committed belief revision with the request; atomically compare at installation. Define which corrections invalidate the currently running tape and require bounded revalidation or zero while replacing it. Do not indiscriminately cancel on every odometry update: that could starve planning. Revision tracking and correction policy are documented architecture changes, not an already-promised current feature. Investigation 02 owns coherent belief snapshots.

### C03-05 — Plan latency guard is bypassed on both execution routes

**Reachability:** active LOCAL; optional nonhierarchical EFE agent. References: [LOCAL pending start](../../src/planning/planning/nodes/efe_agent_node.py:917), [LOCAL direct install](../../src/planning/planning/nodes/efe_agent_node.py:950), [conditional age gate](../../src/planning/planning/nodes/efe_agent_node.py:1227), [base solve](../../src/planning/planning/nodes/unicycle_planner_node.py:3112).

**Trigger:** a one-step 0.25 s result is delayed. **Expected:** an age guard, where supplied, actually applies before installation. **Observed:** LOCAL delayed by 1 s installs at 11 s without any age test (`local_delayed`). Full direct solve delayed by 10 s installs at 20 s: `_pending_plan_started_at` remains `None`, so age validation and enabled latency compensation silently do nothing (`full_solver_delay`). The pending start is only assigned in the LOCAL path, which does not call the solver handoff.

**Consequence:** arbitrarily stale inputs can produce a newly dated tape; the generic age check creates a misleading sense of coverage. **Smallest repair:** capture computation start for each real execution path, validate age at the atomic installation boundary, and log the decision. Define the allowed age consistently with tape support; normal fast LOCAL computations are not evidence that the slow case is safe.

**Compensation arithmetic versus assumption:** manually exercising the hook with 0.60 s latency, `dt=.25`, one second old-tape remaining yields two skipped controls and a 0.10 s fractional backdate. The next control begins at 10.75 s as expected (`latency_fractional_boundary`); no arithmetic defect was found in this example. However `latency_after_stop` skips the same two controls after an explicit zero: it uses a snapshot of old-tape remaining duration, not executed motion. Old controls need not equal new controls even without a stop. Fix age plumbing separately; do not activate this compensation and describe it as verified physical-motion compensation. Revalidation against current supported state is a separate handoff design.

### C03-06 — Diagnostics mix stages and can retain obsolete motion

**Reachability:** active logger and publisher. References: [logger command callbacks](../../src/experiments/experiments/nodes/experiment_logger.py:1772), [command calculation](../../src/experiments/experiments/nodes/experiment_logger.py:2736), [execution diagnostic hold](../../src/experiments/experiments/nodes/experiment_logger.py:2937), [expiry early return](../../src/planning/planning/nodes/efe_agent_node.py:1260), [normal diagnostic publish](../../src/planning/planning/nodes/efe_agent_node.py:1291).

**Trigger/observed:** raw/output zero callbacks at 10.1 s arrive before their new noise diagnostic. Production logger calculation overwrites raw velocity with the old diagnostic's 0.2 m/s, keeps the fresh raw stamp 10.1 s and computes `cmd_noise_v_error=-0.2`. The probe executes the exact source calculation (`logger_mixed_command_events`). This can be callback skew, not a disturbance. Separately, timer expiry emits zero but leaves the last execution diagnostic `(0.2,0.1)` unchanged (`expiry_leaves_nonzero_execution_diagnostic`). Watchdog/nonfinite zeros also omit normal noise diagnostics (`adapter_silence`, `adapter_nonfinite`).

**Expected:** fields in a paired noise calculation refer to one event; stop/expiry evidence describes stopped state. **Consequence:** `exec_cmd_*`, `cmd_noise_*` and ages can disagree without explaining why. Planner `_cmd_log` is a self-subscription receipt log of intended pre-noise commands at [1015](../../src/planning/planning/nodes/unicycle_planner_node.py:1015); it is not applied-command history. `last_cmd` is also written immediately on publish, then can be overwritten by an older self-received message; this affects diagnostic/fallback state, not tape ownership. No claim of a measured occurrence in an existing drive is made here.

**Smallest repair:** compute diagnostic input/output error from the same diagnostic pair and label its own stamp/age; keep raw-topic values and stamps together. Emit explicit stop/expiry/nonfinite/watchdog diagnostics, clear held execution state, and distinguish publisher intent from adapter output in field names/docs. A full event ledger with command IDs and received/applied timestamps is a separate instrumentation change. Never relabel `cmd_v/w` or `exec_cmd_v/w` as physical velocity.

### C03-07 — Message cadence is part of the stochastic experiment

**Reachability:** active, command noise enabled. References: [immediate LOCAL publish](../../src/planning/planning/nodes/efe_agent_node.py:957), [10 Hz timer](../../src/planning/planning/nodes/efe_agent_node.py:332), [per-message correlated update](../../src/sim/sim/actuation_noise_node.py:97), [stop decay](../../src/sim/sim/actuation_noise_node.py:121).

**Trigger:** identical intended `(v,omega)=(0.2,0)` held for one second; seed 210; same active default noise parameters; vary delivery schedule. **Observed:** exact zero-order-hold integrals of adapter output differ:

| Delivery schedule over `[0,1)` s | Messages | Integral of output v, m | Integral of output omega, rad |
|---|---:|---:|---:|
| 10 Hz | 10 | 0.192413294 | 0.003784997 |
| 20 Hz | 20 | 0.193584986 | -0.002032524 |
| 10 Hz, every message duplicated at the same instant | 20 | 0.196758924 | 0.000518729 |
| 10 Hz plus immediate publications at 4 Hz | 14 | 0.192245051 | 0.014883434 |

These are integrals of synthetic adapter output, **not displacement/heading measured from Gazebo**, not stochastic distribution estimates, and not camera/localization metrics. Every sampled output is retained in `results.json`. Same-time duplicates consume innovations and replace the output held until the next event. More zero messages also decay correlated states more times, altering the next motion segment without consuming those same moving-command innovations.

**Expected:** equal seeds reproduce the same message sequence; they do not promise a shared time-indexed process. **Consequence:** scheduler/solver timing can change perturbations even when intended motion is identical. The recent runtime audit explicitly preserves this policy. **Smallest current repair:** qualify paired-seed claims and retain delivery timing/count evidence. **Separate experiment-policy change:** one periodic command publisher and a disturbance process advanced on a fixed simulated-time grid, with duplicate-invariance, pause/reset and sampling tests. Neither was changed here. Filter-Q matching remains investigation 01.

### C03-08 — Expiry and watchdog enforce callback-time policies, not exact physical deadlines

**Reachability:** active. References: [timer selection/expiry](../../src/planning/planning/nodes/efe_agent_node.py:1260), [watchdog timer creation](../../src/sim/sim/actuation_noise_node.py:77), [receipt timeout](../../src/sim/sim/actuation_noise_node.py:198).

**Trigger:** one 0.25 s control installed at 10.0, periodic callbacks at 10.0/10.1/10.2/10.3. **Observed:** nonzero at the first three; zero only at 10.3 (`fractional_expiry`). Expected exact end would be 10.25, so a downstream hold lasts 0.05 s longer in this ideal callback schedule. Other phase alignments approach one 0.1 s timer period; executor delay adds more. This is timer quantization, not evidence of a wrong whole/fractional skip calculation.

**Transport trigger:** watchdog zero at 10.6 after last receipt at 10.0; stale source command delivered at 10.61. **Observed:** it is accepted and resumes nonzero (`adapter_backlog`). The probe labels it old externally; `Twist` contains nothing with which the receiver could reject it. Depth one limits local queued values but does not make a delayed latest value fresh; bridge/downstream queues and other publishers are outside that adapter queue.

**Watchdog guarantee:** with continuing simulated time, responsive callback execution and successful downstream delivery, zero is requested on the first watchdog tick for which receipt age is **greater than** 0.5 s. Timer period is 0.1 s, so a normally serviced timer can reach roughly 0.6 s, plus scheduling/transport/actuator delay. This is a conditional receipt-silence bound, not a hard real-time 0.5 s physical stop. At exactly 0.5 s it does not stop. A paused simulated clock does not age the command (`adapter_pause`); if physics is also paused, no physical travel accrues. A broken ROS clock while physics advances is a different unsupported failure, not tested here.

**Smallest proposed repairs/policies:** explicitly state quantization and timeout semantics; use a tape-end timer if a precise publisher expiry is required, preserving ownership checks. Stamped commands with validity, sequence/epoch and an actuator-side age check are needed for source age as well as receipt silence. An unstamped Twist cannot establish both. Adding wall-clock supervision changes pause behaviour and must be specified/tested separately.

### C03-09 — Acceptance trusts producer bounds; acceleration/deadband contract is partial

**Reachability:** synthetic/optional solver boundary; no confirmed active simple-tracker bounds violation remains. References: [acceptance defaults](../../src/planning/planning/nodes/efe_agent_node.py:1196), [tracker clipping](../../src/planning/planning/nodes/efe_agent_node.py:985), [node bounds](../../src/planning/planning/nodes/unicycle_planner_node.py:92), [producer validity fields](../../src/planning/planning/planners/base_planner.py:1499), [adapter clipping](../../src/sim/sim/actuation_noise_node.py:155).

**Trigger:** result contains only finite `controls=[[2,3]]`. **Observed:** acceptance publishes it; absent rollout validity defaults true and absent clearance is NaN, so no rejection. Adapter subsequently clips to `(0.22,1)` with noise disabled (`acceptance_missing_bounds_and_rollout_fields`, `adapter_clip`). Expected a fully enforced command envelope would reject it earlier. Current BasePlanner explicitly emits validity/clearance fields and the simple tracker emits bounded controls: this is boundary hardening, not proof its active producer returns the synthetic malformed result.

**Limits audit:** active tracker and adapter now agree on linear maximum and angular ±1.0 bounds. The joint stop deadband is 1e-4 on each component at the adapter; tracker has no matching whole-command deadband. Intentional straight motion can acquire yaw noise; stops remain exact zero. Neither active tape prediction nor adapter imposes an acceleration/jerk envelope, and the URDF config supplies none. Calling missing acceleration limits a current mismatch would be incorrect; adding them changes prediction/execution dynamics and needs joint validation. Optional planner `w_min/w_max` overrides are not automatically propagated to the launch's hard-coded adapter ±1 bounds.

**Smallest repair:** validate finite, ordered configured limits; check every accepted control against the shared envelope; require validation fields for solver result types that promise them. Maintain a single declared bound source across planner/adapter/actuator. Adding acceleration or changing deadband/noise semantics is separate modelling/policy work.

## Ownership, exceptions and regression verdicts

There is one intended planner command publisher in the selected launch and one adapter-output publisher. The planner timer and immediate-install trigger compete in time but now share a lock. `_plan_group` prevents concurrent planner callbacks; correction/goal/command I/O may overlap planning. Adapter `rclpy.spin(node)` and its default callback group serialize its callback/watchdog in normal use; the reset problem needs only different callback order, not simultaneous execution.

There is **no command arbiter or publisher-identity enforcement** on either Twist topic. A second publisher's nonzero message after another publisher's stop is accepted like any other input (the `adapter_backlog` ordering also demonstrates this boundary). A direct `/cmd_vel` publisher bypasses adapter limits/watchdog. No second publisher was established in the selected launch and no live ROS graph was queried; treat actual competing publishers as an integration hypothesis. If external command sources are supported, route them through one explicit command owner and test priority/lease rules.

| Regression/failure path | Result and limits |
|---|---|
| Old timer snapshot after stop | Existing deterministic test passes: no nonzero publication. Lock plus identity check preserves ownership. |
| Obsolete tape expiry clearing replacement | Existing deterministic test passes: replacement identity remains installed; obsolete callback publishes nothing. |
| Tracker angular bounds in both directions | Both existing tests pass at ±1.0 rad/s. Documented repair independently verified. |
| In-flight solver after fatal flag | Existing test fails; C03-01. This is not one of the previously verified repaired timer properties. |
| Planner silence with live adapter | Existing silence test passes; extended probe outputs one zero. Bound is conditional on clock/scheduling/delivery. |
| Isolated adapter negative clock age | Existing test passes; combined reset orderings still resume motion (C03-02). |
| Non-finite planner controls | Both acceptance paths reject nonfinite arrays by inspection; adapter probe confirms nonfinite input produces zero. Adapter does not latch fatal state; later finite input may resume by current policy. |
| Intentional stop through noise | Stop branch preserves zero; noise cannot itself turn exact stop into drift. Duplicate stops do affect later correlated state. |
| Solver exception | Base `_call_planner` routes exceptions to fatal stop; first global solve does likewise. In-flight other callbacks remain C03-01. Multi-goal global-replan fallback is route policy, left to 09. |
| LOCAL controller / diagnostic / publisher exception | No general LOCAL try/finally stop surrounds dispatch/gate/publication. Exception may leave tape/last output until teardown or watchdog. No callback-exception/ROS teardown integration trial conducted. Use fail-closed wrapper and best-effort zero; adapter death still needs downstream protection. |
| Planner orderly shutdown | Main shuts executor/destroys node without explicit final zero; a healthy adapter eventually covers silence. No guaranteed immediate stop or delivery acknowledgement. |

## Reproduction and repair separation

[probe.py](../../experiments/command_execution_audit_20260906/probe.py) runs **31 deterministic cases** using controlled clocks, real node methods, fake publishers, a fake solver, committed-state substitutions and explicit callback interleavings. It invokes no ROS initialization/graph, Gazebo, campaign runner, process cleanup or experiment data loader. Its assertions pin **observed behaviour**, including defects; successful completion is not a claim that those behaviours are correct. [results.json](../../experiments/command_execution_audit_20260906/results.json) contains all cases and source hashes; [probe_output.txt](../../experiments/command_execution_audit_20260906/probe_output.txt) is the readable transcript.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/command_execution_audit_20260906/probe.py --frozen-baseline
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/planning/test_runtime_transactions.py tests/sim/test_command_watchdog.py
```

The initial focused suite was **14 passed, 1 failed**; retained [existing_tests.txt](../../experiments/command_execution_audit_20260906/existing_tests.txt). Subsequent shared tests and the coordinated repair yield the separate **17 passed** follow-up above. No broad suite is claimed. Fake time exercises method behaviour, not ROS timer scheduling, DDS transport or Gazebo physical application.

**Correctness repair tranche:** fatal publication latch and stop-generation cancellation; backward-clock tape invalidation; real age-gate plumbing; coherent diagnostic pairs and explicit stop diagnostics; configured bounds validation; orderly fail-closed exception paths. Preserve controller, Q and noise cadence while testing these.

**Separate design/policy tranche:** belief/goal revision and correction revalidation rules; stamped valid-until commands with epochs and receipt/applied event identity; downstream actuator watchdog independent of adapter/bridge; fixed-simulated-time disturbance grid and single periodic publisher; acceleration/deadband model changes. Each needs a new frozen protocol where it changes execution. A fatal latch alone does not settle ordinary stop cancellation, reset, command age or disturbance matching.

**Remaining isolated integration gates, after coordinating with experiment owners:** exercise planner death, adapter death/restart, transport delay/duplicate/competing publishers, clock pause/reset and bridge failure while recording raw publication, adapter receipt/output and wheel-command application. Establish stop-to-physical-response timing separately from the software-method reproductions. Do not run those failure injections against the current campaign.
