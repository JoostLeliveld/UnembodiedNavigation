# Module investigations and chat prompts

Prepared 2026-09-06 for separate investigation chats. Repository:
`/home/joostleliveld/Thesis/UnembodiedNavigation`.

This is an investigation map, not a claim that every area has been audited. It is based
on the working tree, recent repair history, the runtime audit, and the ongoing research
thread. The checkout contains substantial staged, unstaged and untracked work; HEAD
`cc394cb3` alone does not identify the current runtime. Recheck ownership and source
identity when starting a chat.

The existing chat **Review estimation and update logic** already covers investigation 01
and parts of 10. **Advance ICRA Localization Paper** has been repairing and running the
runtime. This document neither starts chats nor changes those runs.

## Recent examples to use as regression seeds

“Documented repair” means the cited audit/history records a repair, often with a focused
regression. This mapping pass has not independently rerun those tests or established that
the complete system is now correct. Historical run statistics are deliberately not copied.

| Example | What went wrong | Status/source | Investigations |
|---|---|---|---|
| Lost motion during camera outage | Recovery replayed only the tail of available odometry, discarded a turn, and labelled the old state current. | Documented repair: [runtime audit](runtime_integrity_audit.md); `tests/planning/test_outage_motion_replay.py`. | 01, 02, 06 |
| Pose/header disagreement | Belief age, prediction target and publication stamp came from separate clock reads. | Documented repair: runtime audit; `tests/planning/test_runtime_transactions.py`. | 02, 10, 11 |
| Lost correction or resurrected command | Concurrent callbacks used the same prior; an old timer could publish after a stop or clear a replacement command sequence. | Documented transaction repairs: runtime audit and runtime transaction tests. | 01, 02, 03 |
| Predicted motion differed from executed motion | The tracker permitted turns beyond the actuator's declared angular bounds. | Documented repair: runtime audit; tracker checks. | 03, 08, 09, 14 |
| Equal timestamp confused with duplicate | The optional direct-camera path rejected a distinct second camera at the same capture instant. | Documented repair; repeated delivery must still add no information. | 01, 04, 07 |
| Malformed measurement accepted | A correctly shaped envelope could still contain an unsupported frame/schema or a non-SPD covariance. | Documented receiver validation repair in runtime audit. | 05, 07, 13 |
| Mixed capture rounds and repeated assimilation | Adjacent camera rounds could mix; manager ticks could reuse one detector invocation. | Repair history `4207b8e1`; current registry retains the invalidated campaign boundary. | 04, 07, 10 |
| Refusal became an absorbing state | A refused measurement could leave the belief clock frozen; conflicting timeout limits were not all represented in configuration. | Historical repair `50038696`; later outage audit shows why advancing the stamp alone was insufficient. | 01, 06, 13 |
| Logging and scoring changed the question | Held observations were counted repeatedly; wrong-time references and wheel odometry labelled as truth distorted monitoring. | Metrics contract, `0545aabe`, alignment tests. | 10, 11, 15 |
| Completed evidence disappeared or was rejected | Summary parsing returned no summary; reasoned refusals invalidated runs in multiple validators; campaign ledger persistence lost completed entries. | `50038696`, `6628c636`; current ledger tests cover atomic replacement. | 10, 11, 13 |
| Partial update had inconsistent covariance | Gain scaling needed a matching Joseph covariance update. | Listed as correctness work in `ICRA_STATUS.md`; independent-formula regression in `tests/planning/test_commissioning_joseph.py`. | 01, 11 |
| Waypoint handoff broke navigation | Spacing and arrival tolerance made the tracker cut the corner, reproduced with ideal motion. | Documented tracker remedy pilot in `experiments/icra_commissioning/planner_implementation_plan.md`. | 09, 14 |
| Wrong analysis selected silently | One CLI spelling left the default route selected, while writing apparently valid figures. | Historical repair `6628c636`. | 11, 13, 15 |

Several consequential behaviours remain explicit policy/model questions in
[runtime_integrity_audit.md](runtime_integrity_audit.md): first-return refusal after a
camera gap, repeated rejection inflation, admission against a single uncertain pose,
robust fusion versus the planner's independent precision model, temporal error
persistence, missing command revision/transport evidence, and noise advanced per command
message. Changing these can change an experiment and must be identified separately from
repairing an implementation defect.

## Additional leads from this mapping pass

These are code observations worth testing, not newly established live failure diagnoses.

| Lead | Code to inspect | Smallest useful investigation |
|---|---|---|
| Goal messages use system-time stamps while state uses simulation time. | `src/experiments/experiments/nodes/goal_mission_node.py`, `wall_clock`, `_send_goal`. | Trace whether any consumer compares these stamps; pause/reset simulation and verify goal age semantics. A wall timer for startup may be intentional. |
| Mission readiness is cached and waypoint advancement uses the latest stored XY. | Same file, `_belief_cb`, `_maybe_advance`. | Deliver valid belief, then stale/invalid covariance or a different frame. Check whether readiness clears and an old pose can advance the mission. |
| Config snapshots are keyed and named by basename. | `src/experiments/experiments/core/manifest.py::snapshot_configs`. | Snapshot two distinct `config.yaml` files and verify that both identities and bytes survive. Establish whether active launch paths can supply collisions. |
| Git provenance is cached for a repository root for the life of a process. | Same file, `_GIT_PROVENANCE_CACHE`, `write_manifest`. | Determine whether the process ever creates another run after source changes; compare the manifest with the code actually imported. |
| Some plan checks use permissive defaults for absent validation fields. | `src/planning/planning/nodes/efe_agent_node.py::_result_safe_to_execute`. | For every active planner/controller result type, remove or corrupt rollout/clearance evidence and test whether execution is admitted. Establish producer guarantees first. |
| Plan age is checked, but a plan also depends on a particular belief and goal. | Same file, `_after_plan_result`, `_result_safe_to_execute`. | Change belief revision or goal while solving; verify that obsolete controls cannot be installed or continued without revalidation. |
| Older architecture prose describes a different observation path. | `HOW_IT_WORKS.md` versus current runtime audit, launch and manager code. | Label which statements are historical and which selected path actually runs; check downstream figure captions for the same substitution. |

## Scope and order

| ID | Workstream | Primary ownership | Start |
|---|---|---|---|
| 01 | State estimation, stochastic process and correction mathematics | `planning/core`, estimator portions of planner node | Now; existing estimation chat |
| 02 | Clocks, callback transactions and belief publication | ROS scheduling/state boundaries | Now; coordinate with 01 |
| 03 | Commands, actuator, disturbance process and watchdog | Execution portions of EFE node; simulator noise adapter | Now |
| 04 | Camera acquisition, detector scheduling and batch identity | `perception` | Now |
| 05 | Observation geometry, reference point and camera calibration | `state`, observation parts of `reliability`, `unav_common` | After tracing 04 output |
| 06 | Admission, bootstrap, rejection and recovery | Manager gates and robot-filter gate policy | Audit now; changes after 01/07 |
| 07 | Camera aggregation and fusion covariance | `reliability/fusion.py`, manager fusion boundary | Now |
| 08 | Planner objective, future camera model and solver result | Planner core and network artifact | After agreeing 05/07 interfaces |
| 09 | Route tracking, waypoint handoff and mission completion | Tracking portions of EFE node; goal mission | Now; coordinate with 03 |
| 10 | Runtime logs and complete event accounting | Experiment logger and event producers | Now; coordinate with 01 |
| 11 | Offline alignment, replay, statistics and reporting | Analysis loaders and study scripts | Now; depends on 10 evidence |
| 12 | Capture, commissioning, learning and model export | Dataset and training/export tooling | Independent audit |
| 13 | Launch, configuration, artifacts and campaign lifecycle | Launch plumbing, manifests, campaign runner | Now |
| 14 | Simulator, world geometry, resets and physical outcomes | Simulation assets, bridge and geometry contracts | Independent audit |
| 15 | Complete-chain acceptance and documentation reconciliation | Integration report and focused harness | Last, after component reports |

The seven ROS packages are covered: `perception` (04), `state` (01/05), `reliability`
(05/06/07/12), `planning` (01/02/03/08/09), `experiments` (09/10/13), `sim` (03/14),
and `unav_common` (05/09/13/14). Offline scripts, training, campaigns and evidence consumers
are covered by 11–13 and 15. Optional/superseded branches should be identified, not mistaken
for the active runtime.

Reports can proceed concurrently. Editing shared files needs one owner at a time:
`unicycle_planner_node.py` is shared by 01/02/06/10, `efe_agent_node.py` by 03/08/09,
and `camera_manager_node.py` by 04/05/06/07. Give each chat a distinct report filename.
Keep simulation execution under one owner unless runs have verified process, ROS-domain,
Gazebo-transport and output isolation. Do not let one campaign's cleanup kill another.

## Common audit instructions

Each prompt below incorporates these instructions by reference.

1. Work in the repository above. Read root and repository `AGENTS.md`, `PLAN.md`,
   `docs/localization_metrics.md`, `docs/localization_metrics_registry.json`,
   `docs/open_questions.md`, `docs/runtime_integrity_audit.md`, and the current
   `docs/ICRA_STATUS.md`. Follow explicit current updates over preserved historical prose.
2. Inspect `git status`, active launch configuration, actual import/artifact paths, and
   existing investigations. Establish whether the suspected path is active, optional,
   diagnostic or obsolete. Preserve other chats' work and frozen experiment sources.
3. Start with the recent failure examples above. At each boundary record who owns the
   value, its event identity, units/frame, the instant it describes, its delivery time,
   its freshness/validity rules, and what the next component assumes.
4. Search for silent success: stamp advancement without matching integration, reuse of
   cached input as a new event, missing/duplicated outcomes, stale state overwrites,
   permissive missing fields, hidden defaults, swallowed exceptions, and a failure that
   prevents all later recovery. Follow sibling/alternative code paths for the same pattern.
5. Prove suspected bugs with the smallest meaningful reproduction, preferably without
   ROS and with an independent oracle. Include delayed, duplicated, missing, reordered,
   malformed and simultaneous inputs where relevant. Existing tests passing is not proof
   that the live wiring is correct. Do not write tests that only restate implementation.
6. First deliver an investigation report and concrete repair proposals. Small scratch
   reproductions are appropriate. Runtime edits must follow the ownership agreed in that
   chat; do not opportunistically change another workstream or a running experiment.
   Separate correctness repairs from policy changes, model assumptions and tuning.
7. For each finding provide severity, active-path reachability, precise file/line references,
   trigger, expected/observed behaviour, consequence, reproduction/test result, and the
   smallest repair. Label it confirmed-current, repaired-and-verified, documented-repair-
   not-reverified, hypothesis, or intentional-policy-limit. State what remains untested.
8. Save the report to `docs/module_audits/NN_<slug>.md`, with cross-workstream handoffs.
   Do not concurrently edit this index or other reports. If a fix is implemented in that
   chat, retain the failing case, run focused checks, and record exact source/config identity.
9. Ground truth is an offline reference only; it must not enter online state, admission,
   planning, goal or stuck decisions. Use `experiments/fusion_on_fixed_routes/aligned.py`
   and exact registered manifests before reporting run metrics. Keep camera readings,
   fused corrections and beliefs separate, score each at its own stamp, count events once,
   aggregate within runs first, and retain failures and reasoned refusals. Do not turn a
   repaired runtime or an idealized replay into a claim of learned-model navigation gain.

## Copyable prompts

### 01 — State estimation, stochastic process and correction mathematics

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 01.

Audit the robot's complete prediction/update recursion: dynamics, Q, measured odometry,
motion history, heading, cross-covariances, measurement linearization, gain scaling,
Joseph covariance updates and all commit/rejection paths. Start with the lost-outage-turn,
frozen-belief-clock and scaled-gain failures. Look for stale odometry extrapolated as
available motion, prediction that depends on callback subdivision, accumulated drift
state reset between steps, double application of motion/noise, shared mutable outcomes,
same-time camera handling and a posterior whose timestamp exceeds its integrated motion.
Trace optional pixel/direct-camera paths as well as the configured fused path. Produce
small independent numeric reproductions and a ranked report with concrete fixes. Own the
filter mathematics and update payload; hand scheduler and CSV issues to 02 and 10.
```

Start in `src/planning/planning/core/{dynamics,belief_correction,motion_history}.py`,
`src/planning/planning/planners/base_planner.py`, and estimator methods in
`src/planning/planning/nodes/unicycle_planner_node.py`. Inspect
`src/sim/sim/encoder_noise_node.py` as the input contract. Existing tests include
`tests/planning/test_coherent_drift.py`, `test_outage_motion_replay.py`,
`test_commissioning_joseph.py`, and `test_planner_node_state_correction.py`.

### 02 — Clocks, callbacks and belief publication

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 02.

Audit time and ownership from incoming odometry/corrections to published planner belief.
Map simulation time, wall time, monotonic time, capture time, receipt time and prediction
targets. Starting from the repaired three-clock publication and concurrent correction
bugs, force callback interleavings, equal timestamps with different state revisions,
slow prediction/solve, clock pause/reset and out-of-order delivery. Prove that publication
is a read-only prediction, stale work cannot replace a newer state, each message stamp
describes its actual state, and locks have a consistent order without starving input.
Include mission headers as a cross-module lead. Coordinate estimator edits with 01;
produce deterministic interleaving reproductions and explicit boundary contracts.
```

Start in `src/planning/planning/nodes/unicycle_planner_node.py` callback groups,
`_serialized_correction`, `_belief_publish_tick`, `_snapshot_plan_inputs`, and
`_resolve_belief_for_planning`; inspect EFE executor wiring and
`tests/planning/test_runtime_transactions.py`. Hand actuator clock decisions to 03 and
mission changes to 09.

### 03 — Commands, actuator, disturbances and watchdog

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 03.

Trace one command from the belief/goal used to solve, through plan acceptance, sequence
installation, latency compensation, periodic/immediate publication, ROS receipt, clipping,
disturbance sampling and actuator output. Start with stale commands after stop, obsolete
sequence expiry clearing a replacement, angular-limit disagreement and the new watchdog.
Test belief/goal changes during solving, duplicate or delayed commands, planner/adapter
death, clock reset, non-finite controls and stop precedence. Distinguish requested,
published, received and applied commands; identify where evidence is missing. Reproduce
how command frequency changes the seeded disturbance process. Propose precise correctness
repairs separately from a stamped-command or fixed-time-noise policy change.
```

Start in `src/planning/planning/nodes/efe_agent_node.py` command methods,
`src/sim/sim/actuation_noise_node.py`, `src/sim/launch/robot_description.launch.py`, and
`tests/sim/test_command_watchdog.py`. Noise generation belongs here; filter Q semantics
belong to 01 and route/controller behaviour to 09. A received adapter output is not by
itself proof of physical actuator application.

### 04 — Camera acquisition, detector scheduling and batch identity

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 04.

Trace physical image capture through detector scheduling to camera-result publication.
Use the mixed-round and repeated-invocation failures to search for stale latest-frame
caches, camera/result order swaps, repeated inference on one image, timestamp conversion
collisions, missing cameras that block every later batch, and silent detector failure.
Test staggered cameras, one camera absent, duplicates, out-of-order frames, slow inference,
node restart and simulation reset. Verify identity and stamps survive the manager handoff,
and distinguish a detector miss from unscheduled work, unavailable image or incomplete
batch. Check all configured camera IDs instead of assuming the historical four-camera
name describes the current network. Keep detector weights and selection rules fixed.
```

Start in `src/perception/perception/core/{four_camera_batch,four_camera_runtime_contract,
ros_image,yolo_selection}.py`, all three detector nodes under
`src/perception/perception/nodes/`, and the manager's receiving callbacks. Existing tests:
`tests/perception/test_batched_four_camera_yolo.py`, `test_scheduled_camera_registry.py`,
and `test_bbox_characterization_capture.py`.

### 05 — Geometry, observation reference and camera calibration

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 05.

Trace the same detector box through projection, robot reference-point interpretation,
the frozen NN, residual offset and full camera covariance. Verify camera identity,
calibration bytes, pixel conventions, image dimensions, floor height, frames, units and
Jacobian transformations. Search for offsets applied twice, a different reference point
between training and runtime, wrong-camera artifacts, covariance transformed at another
state/time, half-open pixel errors, near-horizon singularities and permissive fallbacks.
Use synthetic geometry and identical recorded boxes to compare runtime and offline output.
Treat hull/pixel/metric alternatives as distinct contracts. Do not train a replacement
model to conceal an implementation or reference mismatch.
```

Start in `src/unav_common/unav_common/{camera_model,robot_hull,geometry}.py`,
`src/state/state/core/pixel_to_bev.py`, `src/reliability/reliability/{projection,
silhouette_observation,learned_box_correction,reference_calibration,covariance_mapping}.py`,
and the observation construction in `camera_manager_node.py`. Relevant tests include
`tests/reliability/test_reference_calibration.py`, `test_commissioned_world_covariance.py`,
and `tests/state/test_pixel_to_bev_at_z.py`.

### 06 — Admission, bootstrap, rejection and recovery

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 06.

Build an explicit decision table for transport validity, geometry/image support,
bootstrap quorum, innovation rejection, outage handling, inflation and reanchoring.
Start with the historical bootstrap disagreement, undeclared tighter timeout and
refusal that froze the clock. Check wrong-prior lockout, a good camera returning after
a supported blind interval, incomplete motion history, repeated outliers, disagreeing
cameras and heading uncertainty. Every event needs one stated outcome and reason.
Determine whether repeated rejection inflation eventually accepts an unchanged outlier
and whether admission rejects recovery evidence because of an uncertain pose. Separate
bugs from intentional recovery policies; specify isolated policy experiments with
fixed mean/R/Q. Do not silently relax all gates or snap to camera observations.
```

Start in `src/reliability/reliability/{observation_gates,observation_opportunity,
camera_manager}.py`, bootstrap/admission methods in `camera_manager_node.py`, and
`CorrectionGates` plus rejection/reanchor methods in the planner. Coordinate changes to
shared nodes with 01/07. Hand label-definition consequences to 12.

### 07 — Fusion and covariance semantics

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 07.

Audit aggregation from unique camera observations to one fused event and filter handoff.
Start with mixed capture rounds, repeated batches, same-time camera loss and malformed
covariance. Verify common-time compensation uses supported causal motion, preserves
capture-time evidence, and cannot relabel uncompensated values. Exercise camera-order
permutations, duplicate cameras, zero/one/all cameras, identical readings, outliers and
singular inputs. Derive what each configured fusion rule actually does to mean and
covariance, including Huber weights, effective camera count and shared-error floors.
Check temporal dependence and prior reuse. Compare the direct-camera and robust paths
on identical events; explain the exact interface mismatch with planner precision addition.
```

Start in `src/reliability/reliability/{fusion,contracts,camera_manager}.py`,
`src/reliability/reliability/nodes/camera_manager_node.py`, and
`tests/reliability/{test_fusion_v2,test_fusion_arms,test_dynamic_occlusion_fusion,
test_camera_manager_node}.py`. The robot recursion remains owned by 01.

### 08 — Planner objective, future sensor model and solver

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 08.

Audit the future camera model and its use in the actual planner objective and rollout.
Keep detector-score reliability, usable-observation probability, conditional camera R
and robot P distinct. Test q=0/q=1, one/multiple cameras, unsupported map cells, grid edges,
camera order, fixed-chart transforms and interpolation. Compare expected information
with the separate hit/miss forecast and identify differences from live fusion, gate,
latency and update cadence. Verify derivatives, process propagation, constraints and
candidate selection, including iteration-limit results, missing validity fields, stale
beliefs, incomplete routes and safe stops. Preserve the objective and arm settings while
finding implementation faults. A numerical optimization probe is not navigation evidence.
```

Start in `src/planning/planning/core/{camera_network,casadi_efe,efe_utils,rollout,
visibility_gp_map,nogo_cost}.py`, `planners/base_planner.py`,
`experiments/icra_commissioning/{export_network_planner,verify_network_planner,future,
network_route_probe}.py`, and `tests/planning/test_camera_network.py`,
`test_efe_hit_miss_mixture.py`. Coordinate command acceptance with 03.

### 09 — Tracker, routes, goal and termination

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 09.

Trace global route to local waypoint selection, controller, final approach, success,
stuck and stop decisions. Reproduce the corner-cutting/arrival-radius failure with ideal
motion first. Audit short/duplicate waypoints, turn-in-place, route changes, final-waypoint
exhaustion, success hold times, tolerances at every consumer and stale belief readiness.
Test mission wall-clock stamps and invalid/frame-mismatched beliefs. Investigate a robot
stopping short of the goal without assuming the cause is localization; distinguish
controller stop, invalid rollout, exhausted route, stale state and mission completion.
Prove that completion and stuck decisions use operational belief, while physical contacts
and reference errors are scored separately. Coordinate shared EFE edits with 03/08.
```

Start in tracking methods of `src/planning/planning/nodes/efe_agent_node.py`,
`src/experiments/experiments/nodes/goal_mission_node.py`, termination portions of
`experiment_logger.py`, `src/unav_common/unav_common/{preselected_route,
lane_graph_routes}.py`, and `experiments/icra_commissioning/tracker_preflight.py`.
The short-of-goal observation is an ongoing thread lead; re-read its exact selected run
and outcome before treating it as a confirmed defect.

### 10 — Logging and event accounting

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 10.

Trace one physical capture through camera opportunity, accepted/rejected observation,
fusion, correction assimilation, belief publication, command and run summary. Audit
schema/version compatibility, identities, stamps, CSV field order, held messages,
asynchronous diagnostic pairing, queue loss, buffering, final flush and shutdown order.
Start with duplicate detections, wrong-time truth, missing assimilation and completed
runs without summaries. Every published fused correction must have exactly one terminal
assimilation record, including reasoned refusals; misses must remain observable. Compare
producer payloads with logged fields and independently reconstruct a tiny event sequence.
List what cannot be reconstructed, including command receipt/application and prediction
covariance if absent. Coordinate update-payload changes with 01 and scoring rules with 11.
```

Start in `src/experiments/experiments/nodes/experiment_logger.py`,
`src/experiments/experiments/core/camera_opportunity_log.py`, correction/command publishers,
and `tests/experiments/{test_logger_schema,test_logger_time_alignment,
test_camera_opportunity_log}.py`.

### 11 — Alignment, replay, statistics and reporting

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 11.

Audit the evidence consumers against the executable localization contract. Start with
held-frame duplication, wrong reference/time, inferred assimilation, reasoned refusals
incorrectly invalidating runs and silent CLI route substitution. Use synthetic logs with
duplicate/missing/extra events, timestamp ties, interpolation gaps, clock reset, multiple
updates between logger ticks and failure at shutdown. Check exact event-to-belief pairing,
initial/final outage windows, covariance/heading alignment and gt versus odometry fields.
Distinguish capture-time idealization from arrival-time live replay. Enforce exact frozen
manifests, within-run aggregation and paired run/seed comparisons with failures retained.
Trace generated tables/plots back to their inputs; do not infer performance from newest
folders or compare camera error with belief error.
```

Start in `experiments/fusion_on_fixed_routes/{aligned,replay,score,compare}.py`,
`experiments/icra_commissioning/{replay,network_replay,network_navigation_analysis,
thesis_evidence}.py`, `scripts/shared/metrics.py`,
`scripts/visibility_comparison/monitor_campaign.py`, and
`tests/experiments/test_fusion_study_alignment.py`, `tests/test_network_replay.py`.

### 12 — Capture, commissioning, learning and export

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 12.

Audit dataset identity from commanded pose and settled image through detector outputs,
splits, bias/residual fitting, covariance/availability models and runtime export. Look
for fresh timestamps on stale images, mixed rounds, repeated images treated as new
noise samples, resume holes/duplicates, lossy reconstruction changing detector inputs,
wrong labels, misses dropped from denominators and train/test leakage across positions,
headings or routes. Verify covariance fitting uses appropriate held-out mean residuals,
availability labels mean the same thing online and offline, and operational features
exclude reference truth. Test export/import equality, camera ordering, model hashes and
unsupported inputs. Keep the frozen NN and baseline intact; identify data/procedure bugs
separately from whether a richer sensor model is justified.
```

Start in `experiments/camera_observation_characterization/` capture and audit scripts,
`scripts/perception/dataset_split_utils.py`, `experiments/measurement_commissioning/`,
`experiments/icra_commissioning/{model,commissioned_field,study,field_study}.py`,
`src/reliability/reliability/{observation_exporter,observation_gp,operational_residual}.py`,
and tests for capture resume, dataset splits and model export. Treat research alternatives
such as `perception_filter_cascade` and learned covariance as separate optional branches.

### 13 — Configuration, artifacts, provenance and campaign lifecycle

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 13.

Trace campaign YAML through override precedence, launch arguments, node parameters,
resolved imports/artifacts, manifests and resume validation. Start with undeclared
timeout limits, missing run summaries, lost ledgers and silent route/default substitution.
Test typo/unknown keys, false/zero/None values, model replacement, dirty/untracked changes,
source versus installed code, same-basename config snapshots, cached provenance and
two runs starting together. Interrupt ledger/summary writes, resume partial runs and
verify exact task/arm/seed identity with failures retained. Audit startup readiness,
timeout clocks, process ownership, cleanup, ROS/Gazebo isolation and crash recovery.
Prove that the manifest describes what actually executed. Do not launch or kill another
chat's experiments; produce isolated unit or dry-run reproductions first.
```

Start in `scripts/visibility_comparison/{run_visibility_campaign,common}.py`,
`src/experiments/experiments/core/{visibility_launch_common,manifest,world_profiles,tasks}.py`,
`src/unav_common/unav_common/manifest.py`, launch files and package setup/entry points.
Existing tests include `tests/visibility_comparison/{test_campaign_ledger,
test_network_planner_config}.py` and `tests/test_no_machine_local_paths.py`.

### 14 — Simulator, world geometry and physical outcomes

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 14.

Audit agreement between the world actually loaded, robot body/collision geometry,
camera/TF calibration, driveable map, planned footprint/clearance, spawn pose and bridge
topics. Test a known clear pose, known contact and a tight turn using independent geometry.
Check reset readiness, simulation-clock restart, old bridge messages, topic namespaces,
sensor/physics rates and whether contact-channel silence is distinguishable from no
collision. Trace noisy and clean odometry sources and maintain the offline-truth firewall.
Determine whether headless/GUI and source/installed assets resolve to the same frozen
world. Coordinate actuation/noise changes with 03 and collision/termination interpretation
with 09/10. Use isolated probes under the agreed simulator owner.
```

Start in `src/sim/launch/`, `src/sim/sim/{reset_world,wait_for_clock,wait_for_odom,
clock_throttle_node}.py`, `src/sim/robot_description/urdf/warehouse_amr.urdf.xacro`,
`src/sim/gazebo_worlds/worlds/`, `src/unav_common/unav_common/` geometry, and
`experiments/warehouse_v2_sketches/` world contracts. Relevant tests include
`tests/experiments/test_warehouse_v2_world_contract.py`,
`tests/sim/test_gazebo_version_contract.py`, and `tests/test_icra_geometry.py`.

### 15 — Complete-chain acceptance and documentation

```text
Read /home/joostleliveld/Thesis/UnembodiedNavigation/docs/module_issue_investigations.md
and follow its common audit instructions for investigation 15 after component reports exist.

Reconcile the component findings and trace a complete capture-to-action-to-evidence chain
through the exact corrected runtime. Build a bounded integration scenario containing
normal motion, a blind turn with complete odometry, missing odometry, delayed/duplicate
camera delivery, a correction during solving, detector/command silence and a final goal
approach. Verify unique event outcomes, causal state stamps, coherent heading/covariance,
stop ownership, physical outcome liveness, durable summaries and faithful offline replay.
Use component regressions plus a small isolated live check; do not launch a broad study
until these contracts pass. Reconcile README/HOW_IT_WORKS, current status, registry and
figure captions with actual active/optional paths. Produce remaining blockers, a precise
change manifest and the evidence needed before any new navigation/model claim.
```

Read the completed `docs/module_audits/` reports first. This workstream owns the integration
report and acceptance scenarios, not unilateral edits to every component. In particular,
neither an all-green unit suite nor a single successful drive closes every timing and
evidence contract.
