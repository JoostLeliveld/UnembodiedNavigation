# Audit completion and runtime repair queue

Consolidated 2026-09-07, 00:36 Europe/Amsterdam (2026-09-06 22:36 UTC).
Coordinator: **Investigate issue-prone modules**, thread
`01a07849-8bf2-72c1-8c4f-66e3de7b1533`.

**All 15 audits are complete. The runtime is not accepted for a corrected matched
navigation pilot.** The [integration audit](15_end_to_end_acceptance.md) reconciles
the component findings and records the acceptance matrix. Audit completion,
implementation, scoped regression success, deployment and full acceptance are separate.

The latest inspected turns for previously pending owners completed successfully; all
15 main reports exist. The account usage interruption was resolved. The audit-completion
monitor reached its stop condition and is now paused. The earlier changing queue,
source hashes and handoff chronology are preserved in
[coordination history](COORDINATION_HISTORY_20260907.md).

## User priority and scope

**Implementation authorized:** the subsequent user request, “Fix the rest?”, starts
the repair phase and supersedes the report-only restrictions below. Existing audit
owners have bounded implementation assignments. 01 solely edits the unicycle/motion/
belief core; 02 independently tests/reviews timing; 03 owns EFE installation; 08 owns
pure planner/result validation; 07 owns manager integration; 04 detector acquisition;
05 geometry helpers; 09 mission; 10 logger; 11 offline consumers; 12 capture/export;
13 campaign/provenance; 14 simulator/actuator/reset. Coordinator owns the shared
`unav_common/correction_ledger.py` validator and repair integration. Production-file
ownership must remain explicit; independent helpers/tests can proceed in parallel.
The original audit-completion monitor remains paused; implementation is active in
this request. Preserve frozen data, declared model/gate/noise settings and A/B repairs.

Correctness of the entire unicycle runtime and its update lifecycle is the first priority:
every callback, recursive-state write, motion prediction, correction, publication,
planner snapshot, command installation and terminal record must have coherent ownership.

This coordination request prepares repair packages, monitors completion and coordinates
already-authorized work. It does not assign new shared-file mutations or new experiments.
Existing implementation permissions in the other chats remain in force. No broad campaign,
automatic commit of the shared working tree or deployment is implied.

Cameras remain enabled during turns and planning. Missing observations in probes are
synthetic failure cases. Keep Q/R, measurement model, detector settings and gate/recovery
parameters unchanged in correctness repairs. Changes to those choices require a separate
declared method/configuration treatment. A routine new camera observation must not
automatically cancel every slow solve and prevent progress.

## Completed audit inventory

| ID | Exact thread title | Thread ID | Audit / repair status | Report |
|---|---|---|---|---|
| 01 | Audit Belief Correction Pipeline | `01a0785f-350d-7bc0-a398-e6767f5cf232` | Complete; package A implemented, current verification delivered | [01](01_state_estimation.md), [implementation](01_repair_package_a_implemented.md) |
| 02 | Audit timing and belief ownership | `01a07860-74a5-7882-a07a-c7e039575911` | Complete; current callback/ownership handoff delivered | [02](02_timing_and_callbacks.md), [current handoff](02_current_source_handoff.md) |
| 03 | Trace command lifecycle regressions | `01a07862-d60a-7711-bed0-786172f4f299` | Complete; package B implemented, GLOBAL repair plan delivered | [03](03_command_execution.md), [implementation](03_command_execution_repair_implemented.md) |
| 04 | Audit camera batch identity flow | `01a07864-74ac-71d3-98bb-b3203ccc6636` | Complete; acquisition patch tested, later integration gaps remain | [04](04_camera_acquisition_and_batching.md) |
| 05 | Audit camera calibration pipeline | `01a07865-4cd1-7381-9304-3f0668af55be` | Complete; repair candidates remain | [05](05_observation_geometry_and_calibration.md) |
| 06 | Audit observation admission policy | `01a0786c-1500-7c80-a666-015ad4a8957b` | Complete; repair candidates remain | [06](06_admission_bootstrap_recovery.md) |
| 07 | Investigate multi-camera fusion | `01a0786d-25a9-7662-992d-82745b1e3895` | Complete; repair candidates remain | [07](07_multicamera_fusion.md) |
| 08 | Audit planner future model | `01a07871-a04d-7442-92cb-db37b1b1427f` | Complete; current result/cache verification delivered | [08](08_planner_and_future_camera_model.md) |
| 09 | Audit route tracking regressions | `01a07872-4398-7733-a7ff-87d1712068c2` | Complete; current tracking report delivered | [09](09_tracking_routes_termination.md) |
| 10 | Audit runtime logging pipeline | `01a07872-f2d5-73b1-b076-23fbbeaea2db` | Complete; logging repair outline delivered | [10](10_logging_and_event_accounting.md) |
| 11 | Audit offline alignment pipeline | `01a07873-9e1b-7b52-8335-a7642d26a613` | Complete; repair candidates remain | [11](11_alignment_replay_and_reporting.md) |
| 12 | Audit capture integrity pipeline | `01a07874-6b80-7eb1-ba2e-48b72726c579` | Complete; repair candidates remain | [12](12_capture_commissioning_and_export.md) |
| 13 | Investigate visibility campaign flow | `01a07875-10cf-73c0-bbea-437f88f1319a` | Complete; provenance repair candidates delivered | [13](13_configuration_provenance_campaigns.md) |
| 14 | Audit simulator integrity | `01a07875-e61a-7b41-a651-553b41b0c935` | Complete; simulator and profile-provenance findings delivered | [14](14_simulator_world_and_physical_outcomes.md) |
| 15 | Run final integration audit | `01a07876-c011-7ab0-a5a9-805cfd242bbe` | Complete; integration verdict NOT ACCEPTED | [15](15_end_to_end_acceptance.md) |

The current estimator and timing conclusions are in
[01 current verification](01_unicycle_current_verification.md) and
[02 current handoff](02_current_source_handoff.md). Their old defect-asserting probes
remain historical evidence, not passing desired-invariant tests.

## Implemented and independently scoped evidence

| Repair | State | Evidence and remaining limits |
|---|---|---|
| Immutable checked motion replay, identified envelope-only initialization, chronological/atomic odometry and encoder input | Local commit `5e844a89` | [Package A implementation](01_repair_package_a_implemented.md); 01/02 current rechecks pass the named properties. Missing-motion validity, capture-time heading and epoch handling remain open. |
| Ordinary-stop cancellation and real age gate on LOCAL/direct command installation | Local commit `f9e1a312` | [Package B implementation](03_command_execution_repair_implemented.md); current 03/08/15 checks verify the named properties. GLOBAL route installation and coherent request revisions remain open. |
| Camera batching and receipt handling | Patch and component handoff delivered | 04 reports 101 focused passes. Logical-cycle identity does not by itself satisfy per-physical-invocation identity; durable abort/outcome accounting remains open. |
| Planner cache identity and reload repair | Current 08 verification delivered | Q/H/R identity repaired; minimal CasADi reload failure explained and replacement checked for value/gradient/reload agreement. This does not validate route admission or navigation. |
| Shared input validation, fusion delegation and typed controller safety | Refactor changes present | Their exact validation/source scope must be reconciled with final handoffs before additional shared edits. Do not reimplement overlapping 05/06/07/09 baseline findings blindly. |

The preserved pushed checkpoint is `3c4ddeae4a7427bf374514dad6a1d65dc728a91c` on
`cleanup/remove-dead-filter-interventions`. A/B are verified local commits; this
monitor has not verified their remote push. Uncommitted work remains outside those
identities. A passing build or source symlink does not prove which bytes an existing
ROS process loaded.

01 reports 130 focused passes/3 skips; 02 reports 90 focused passes; 08 reports 139;
15 independently reports 104 scoped current A/B-related passes. These overlapping
counts must not be added. They are evidence from their named owners/source snapshots,
not a new suite run by this coordinator. The broad refactor transcript reported 1066
passed/4 skipped/2 failed at its own source boundary.

## Runtime-first implementation sequence

These are scoped proposals, not claims that the remaining repairs have landed.

| Order | Concrete package | Lead / reviewers and dependencies | Evidence / readiness |
|---|---|---|---|
| R1 | Commit command receipt and goal/progress metadata atomically; validate finite inputs and invalid/future ages; validate held predictions before any rejected-outcome commit | 01 state owner + 02 timing owner, coordinate active unicycle/refactor hunks | 01 U01 and 02 current handoff; independent reproductions and small repair boundaries ready |
| R2 | Central committed belief record: full mean/P, authoritative frame, integer anchor, epoch/revision and correction identity/outcome; preserve checked motion snapshot with prediction inputs | 01 leading; 02 ownership; 07 producer and 10/11 event schema | 01/02 current writer inventory; state commit must not depend on diagnostic publication succeeding |
| R3 | Validate motion support at prefix/interior/tail and capture-time heading; propagate unsupported state honestly to consumers | 01 leading; 02/07 motion API, 09/10 validity consumers | A fixes history races/order, not missing-motion semantics. Specify continuation/recovery behavior without silently changing Q/R or existing gates |
| R4 | Ordered belief publication and coherent consumer snapshots; preserve equal-time newer revisions; return prediction provenance instead of shared mutable scratch metadata | 02 leading after R2; 01/07/09/10 consumers | Current probes reproduce older-after-newer stamps and mixed mean/anchor/frame. Final epoch/revision/target check or ordered publication owner required |
| R5 | Atomic GLOBAL route admission across solver, fallback, geometric and preselected branches; validate finite result, completion and every added connection | 03 command/route transaction; 08 result contract; 09 route semantics; 02 epoch | [GLOBAL repair plan](03_global_installation_repair_plan.md). Preserve B; stop/negative age must invalidate. Do not reuse LOCAL duration as a new GLOBAL age threshold |
| R6 | Correction scheduling and durable terminal accounting; lossless event intake, diagnostic failure isolation, raw-ledger validation and exact committed-posterior evidence | 01/02 commit/scheduling; 07 producer; 10 logger; 11 validator; 13 reuse | One correction must retain identity/outcome even if diagnostics fail. Ship consumer validator with event schema; do not infer posterior from a periodic prediction |
| R7 | Physical command authority, coordinated epoch restart and terminal stop/drain | 14 actuator/bridge; 03 command; 09 mission; 10 logger; 13 isolation, 01–04 reset participants | 15 records physical failure evidence. Specify final actuator timeout, startup/shutdown zero, stop acknowledgment, old-epoch clearing and fresh readiness; no blanket wall-clock timeout without pause semantics |
| R8 | Complete loaded world/robot geometry and strict deployment/attempt identity before isolated integration | 14 world/contact; 13 provenance; 05/12 calibration; 09 route; 10/11 evidence | 15 identifies omitted collision-bearing props and attempt/asset identity gaps. Repair interpretation/closure, not the environment merely to obtain a better outcome |

R5 may proceed alongside estimator work only with a distinct agreed file owner. R7/R8
are mandatory acceptance prerequisites, not lower-severity optional cleanups. Before a
new simulator acceptance check, all required component contracts and source/asset identity
must be settled; do not combine arbitrary intermediate versions from separate chats.

The GLOBAL path currently mutates route/phase rather than immediately issuing a command.
A later LOCAL request can execute that obsolete route; passing LOCAL stop checks does not
close the upstream route defect. The expiry-side-effect concern remains API/inspection
evidence under the live single planning callback, not a confirmed overlapping-solve bug.

## Other deduplicated packages

| Package | Scope / dependencies |
|---|---|
| Sensor identity and geometry | 04/05/07: physical invocation/member identity, explicit abort/miss outcomes, bbox/pixel/frame/calibration binding, immutable fusion event and supported frame-correct common-time motion. Preserve existing robust fusion model. |
| Logging and shutdown | 10 with 01/07/11: distinct events at equal receipt time, producer/queue cutoff and drain, attempt every file closure, durable summary only after completion. Preserve raw refusals and failed attempts. |
| Offline validity and scoring | 11: one raw ledger validator before reference filtering; committed posterior unavailable unless explicitly recorded; strict selections, reference support, covariance validation and correct per-run aggregation. |
| Capture/export | 12 with 05/13: preflight on actual resume/writer path, source-image identity, failure denominators, exact loaded artifact lineage and crash-safe output. |
| Campaign execution | 13: bind attempt to its own summary; reuse full evidence validator on resume; one effective configuration, strict keys/types/precedence; scoped cleanup, collision-free IDs and durable concurrent ledger handling. |
| Tracking/mission | 09 with 02/03/08/14: valid operational belief for progress/success, persistent non-worsening clearance recovery, explicit stop cause, mission clock semantics and complete route validation. Current refactor/default changes require source-specific rechecks. |

No code edit in this monitor changed a scientific model, runtime policy, frozen run or
paper/thesis prose. Model choices such as first-return admission, repeated rejection
inflation, robust versus independent fusion, q/R forecast design and disturbance cadence
remain separate from invariant repairs.

## Provenance mismatch and performance configuration

14 traced the failed geometry-hash check to a new low-CPU profile in the live registry.
The frozen manifest hash `3559f0a78b` matches the original git HEAD bytes; the compared
original camera/world mappings were unchanged. This is not demonstrated historical
geometry corruption. Historical loading still needs verified exact configuration bytes
rather than rewriting the frozen hash or bypassing validation; 13 coordinates with 12/14.

The low-CPU world changes physics/contact rates, so it is a separate performance treatment
even when camera geometry matches. Preserve its own identity and do not silently pool it
with the original baseline. The other broad-suite failure is the existing machine-local
path scan. A current passing replacement test or full-suite handoff must name its source;
neither failure should be concealed by changing frozen evidence.

## Shared-file ownership and next handoffs

- `unicycle_planner_node.py`: 01 leads recursive repair, 02 timing/publication review,
  06 admission and 10 transaction schema review. Refactor owned config parsing and
  per-camera frame/error handling; agree exact handoff before further edits.
- `belief_correction.py`: refactor validation changes must be preserved; 01 owns the
  missing held-prediction validation and numerical invariants.
- `efe_agent_node.py`: 03 leads transaction boundary, 08 result contract and 09 route/
  controller review. Preserve B and reconcile typed-safety/default changes.
- `camera_manager_node.py`: 04 receipt repair was released, then refactor fusion
  delegation overlapped; sequence 05/07 changes after that handoff.
- Logger/offline schema: 10/11 jointly define it with 01/02/07 before cross-file edits.
- Performance runner/launch/cache and refactor work: latest inspected turns completed,
  but idle status alone is not an explicit shared-file release. Obtain a concrete
  source/test handoff when assigning the next patch.

Other existing owner threads:
**Speed Up Slow Runs** `01a0787c-ebd2-7572-930b-b75d27a57abd`;
**Refactor loose pivot code** `01a07886-2432-7422-89ca-4a698d07c8c2`;
**Advance ICRA Localization Paper** `01a06ff2-1a5a-7fd2-b387-0e59f08cce46`
(released its experiment/reporting ownership).

## Monitor completion

All 15 reports have been delivered, final integration reconciliation exists, and this
runtime-first queue is consolidated. Native heartbeat **Track module audits**
(`track-module-audits`, this thread) is confirmed **PAUSED**. Do not mark runtime repairs or physical acceptance
complete. Further implementation and its monitoring belong to the next work phase.

This pass read reports, completed thread statuses, implementation notes and the final
acceptance matrix. It did not run a new navigation trial or production repair. The
historical coordination record is retained for source/ownership traceability.
