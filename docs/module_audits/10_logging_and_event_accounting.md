# 10 — Runtime logging and end-to-end event accounting

2026-09-06. **The current files can account for ordinary delivered fused corrections, but do not establish complete capture-to-actuation accounting or an exact post-correction belief.** Deterministic probes reproduce lost decision rows, missing terminal records, incomplete validity checks, a summary that precedes its final rows, and a failed summary write that cannot be retried. These are software findings; they are not evidence that a camera model improved navigation or that every fault occurred in a drive.

The exact registered corrected-runtime P0/P1/P2 files pass the narrower accounting check: **813/779/598 assimilation rows respectively**, each matching its summary, with no missing, extra, duplicate or reasonless terminal IDs relative to the recorded fusion decisions. All selected file hashes match their frozen selection. This does **not** certify events absent from both files, exact posterior reconstruction, detector acquisition completeness, or physical stop/contact coverage.

A separate registered earlier tracking P1 run confirms the finalization tail in recorded evidence: summary contact count **1**, final CSV count **17** after the summary stop stamp. That is an accounting counterexample, not a cross-pilot accuracy comparison (L10-02).

At the investigation checkpoint, no production code, Q/R, gate, scoring semantics,
configuration, selected run, or other audit was edited. Only this report and its
dedicated reproduction directory were written. The authorized repair follow-up below
describes later working-tree changes; it does not retroactively change the frozen
reproductions or registered-run evidence.

## Authorized repair follow-up — 2026-09-07

The logger-owned part of the repair is implemented in the shared working tree. It is
not yet a newly commissioned campaign result. The change deliberately leaves filter
Q/R, gate thresholds and scoring definitions unchanged.

- [camera_opportunity_log.py](../../src/experiments/experiments/core/camera_opportunity_log.py)
  now writes the exact raw delivery before interpretation, rejects recursive
  NaN/Infinity and unsupported observation schemas, keys canonical camera evidence by
  `(camera_id, source_batch_id)`, distinguishes identical retransmission from a
  conflicting reuse of that identity, and advances row/dedup state only after the row
  is written and flushed. `JsonlDeliveryLog` supplies the same raw-first behavior for
  correction, decision, producer, mission and simulator outcome topics.
- [experiment_logger.py](../../src/experiments/experiments/nodes/experiment_logger.py)
  declares logging schema 8, writes `runtime_event_deliveries.jsonl`, subscribes to the
  actual fused correction envelope, and writes one canonical
  `correction_publications.v2` row with producer/event identity, exact and floating
  timestamps, frame, member IDs and payload hash. Distinct decisions no longer dedupe
  on logger receipt time. Schema-2 envelopes pass the fusion owner's public semantic
  validator, including hash, member, operand, covariance and motion-support checks.
  Duplicate or malformed terminal deliveries remain in the
  raw journal but cannot become a second canonical terminal row.
- Terminal schema 2 is validated for exact correction/apply times, belief epoch and
  revision, frame, boolean initialization/validity/motion support, and a full finite
  posterior mean/covariance/state time whenever initialized. A reasoned uninitialized
  refusal must explicitly carry null posterior fields. The CSV retains those fields
  and the logger maps the source producer
  epoch separately from the recursive belief epoch for publication/outcome agreement.
  Completion calls the shared `validate_correction_ledger`; only `accepted`,
  `accepted_bootstrap` and `reanchored` enter its update set. Both refusal classes need
  a reason. Mandatory fused-envelope runs without complete schema-2 posterior evidence
  are marked incomplete/invalid.
- Stop is now a request followed by a bounded quiet drain. All data streams are
  flushed, file-synced where supported and independently closed before a strict JSON
  summary is committed by same-directory atomic replacement. `_completed` is set only
  after that replacement succeeds. The simulator guard's flat
  `forwarded_linear`/`forwarded_angular` fields record a terminal zero reaching its
  native output boundary; the summary keeps `terminal_zero_forwarded` separate from
  `terminal_stop_verified` because the guard explicitly does not observe wheel-level
  physical application.
- Raw `/world_contacts` deliveries now receive a logger delivery identity and retain
  source stamp plus exact collision entity/link names in the runtime JSONL. The
  separate `/sim/contact_channel_status` heartbeat is also journaled and validated;
  its configured-silent state remains an explicit need for positive control, never a
  no-collision assertion.
- Mission-goal identity and coordinates are consumed atomically from
  `mission_goal.v1`. Intermediate tour goals cannot end a run, and changing goal ID
  resets arrival holds. The legacy `/goal_bev` remains a compatibility view.
- The run manifest now records the exact world SDF path/hash and the camera-network
  artifact hash, embedded source hashes and camera roster supplied by the launch path.
  A configured unreadable world, malformed source-hash map, duplicate roster or
  expected/artifact hash mismatch fails startup instead of recording intended-only
  provenance.

The new synthetic regression reconstructs publications and terminal outcomes from the
written CSV rows, independently invokes the shared ledger validator, and injects an
accepted update, reasoned rejection, reasoned drop, missing terminal, duplicate
delivery and malformed delivery. It finds exactly one accepted update, one of each
refusal, the explicit missing ID, nine raw deliveries, one malformed JSON object and
only three canonical outcomes. Dedicated logger/opportunity/ledger verification is
The final combined verification is **241 passed** across the four requested logger /
alignment files, the new logger accounting regressions, the shared correction ledger,
detector outcome journal, manager fusion-event contract, mission-goal contract,
simulator physical-outcome contract and campaign provenance/evidence checks. During
integration, two old offline fixtures omitted the explicit interpolation bound or used
four different fused values as one repeated batch; investigation 11 updated those
fixtures to the strict event contract rather than weakening the ledger. Three manager
fixtures likewise moved to the finalized nested motion-support contract.
[Test output](10_repair_test_results.txt) and the final
[source hashes](10_repair_source_sha256.txt) are retained beside this report.

Remaining limits are explicit. Normal process shutdown can close the detector's
durable journal, but DDS receipt alone cannot prove producer quiescence, and SIGKILL can
still interrupt between a detector start record and its terminal record. Command
request, adapter receipt and bridge receipt times remain null because `Twist` has no
command identity. No sensor confirms actuator state independently of the simulator
guard. Contact silence still means “no recorded contact”, not a positive no-collision
observation. The logger does not reconstruct callback execution order beyond the raw
delivery index and per-topic/source timestamps. Prediction covariance between
published belief snapshots remains unavailable unless supplied by the state owner.

## Evidence and active-path boundary

Read first: repository/root `AGENTS.md`, `PLAN.md`, `docs/localization_metrics.md`, `docs/localization_metrics_registry.json`, `docs/open_questions.md`, and `docs/runtime_integrity_audit.md`; subsequently `docs/ICRA_STATUS.md`, the investigation map, the earlier estimation review and completed audits [01](01_state_estimation.md), [02](02_timing_and_callbacks.md), [03](03_command_execution.md), [04](04_camera_acquisition_and_batching.md), [05](05_observation_geometry_and_calibration.md), [06](06_admission_bootstrap_recovery.md), and [07](07_multicamera_fusion.md). Old prose saying schema 4 or “only sensor characterization is active” is historical; the current source and selected runs use **logging schema 7**.

Source/configuration reference: [corrected runtime YAML](../../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml), its [protocol](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json), and exact [selection](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json). The separately registered [recovery YAML](../../experiments/icra_commissioning/network_navigation_recovery_pilot.yaml) uses the later command guards and `turn_then_go_recovery`; its [selection](../../logs/studies/icra_commissioning_20260905/network_navigation_recovery_evidence/selection.json) became available during final QA. A separate [raw accounting check](../../experiments/runtime_integrity/logging_accounting_20260906/recovery_run_accounting.json) verifies all frozen file hashes and 636 unique terminal rows matching that summary, with no missing/extra/duplicate/unclassifiable/reasonless IDs. This is the whole-run window; the recovery report's 453 outcomes use the post-first-command apply-time window. No accuracy or controller-effect comparison is made here. No run was selected by glob, modification time or “latest”.

The selected path is five-camera strict native YOLO, CPU, image size 960, masks off, 0.05 s batch spread; manager at 5 Hz, mandatory batch ID, `learned_nn` plus frozen residual offset/full constant `commissioned_reference_r`, prior-free robust `joint_network`; fused metric EKF, mandatory envelope, coupled heading, `/odom_noisy` replay, NIS 9.21, 1.5 s camera-gap cap, 0.05 m² rejection inflation and no hard reanchor. Belief/command/logger publication or sampling defaults are 10 Hz; LOCAL planning is 4 Hz. Publication-to-now compensation, pixel correction, direct per-camera updates, and live reset are disabled. Common-capture alignment remains enabled. The active executable is `efe_agent`, inheriting `UnicyclePlannerNode`, not the base planner executable in isolation.

Reachability is established by [launch construction:1709](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/core/visibility_launch_common.py:1709), [agent routing:1913](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/core/visibility_launch_common.py:1913), [logger construction:1332](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/core/visibility_launch_common.py:1332), selected manifests, and audit 01's evaluated launch wiring. Logger exit triggers launch shutdown; the detector/manager exit-policy gap was repaired by 04 during final QA, as qualified below. Source imports in this probe resolve to `src/` using the repository test bootstrap; audit 01 separately verified installed build symlinks. Neither this probe nor a current source hash proves which unrecorded module bytes a past process imported.

**Concurrent source boundary:** all original probe source hashes match checkpoint `3c4ddeae4a7427bf374514dad6a1d65dc728a91c`. Source-line links for the twelve retained files point to [verified frozen copies](../../experiments/runtime_integrity/logging_accounting_20260906/snapshot_manifest.json), preserving exact references despite concurrent edits. The [snapshot reproduction](../../experiments/runtime_integrity/logging_accounting_20260906/snapshot_reproduction_results.json) reruns the complete probe and all 90 tests in a temporary checkout; all three JSON outputs match the saved evidence exactly. [Final source comparison](../../experiments/runtime_integrity/logging_accounting_20260906/final_source_boundary.json) records subsequent drift.

Audit [04's repair follow-up](04_camera_acquisition_and_batching.md#repair-follow-up--2026-09-06-after-recovery-campaign-cleanup) adds bounded manager/detector buffers, identified expiry/conflict/supersession outcomes, detector process-session/cycle and per-chunk IDs, clock-rewind refusal, a manager preclaimed decision ID and detector/manager exit handlers. Those mechanisms are now repaired in the working tree within 04's stated test scope. Its new detector/manager `camera_batch_outcome.v1` topics also write prefixed process-log JSON. **The schema-7 logger is unchanged and subscribes to neither new outcome topic.** `decision_completed` is not proof of fusion publication or robot assimilation. The robot commit/terminal gap, logger deduplication, validation and finalization defects below remain. Later planner path/status publication optimizations and campaign first-command scan/cache changes do not repair those findings. The recorded drives precede the acquisition repair.

Retained evidence:

- [probe.py](../../experiments/runtime_integrity/logging_accounting_20260906/probe.py): actual `ExperimentLogger` and filter methods, real ROS message/time types, fake clocks/publishers and isolated files; no ROS graph, inference, simulator or physical actuator.
- [results.json](../../experiments/runtime_integrity/logging_accounting_20260906/results.json): independent reconstruction and fault results. Assertions establish observed behavior, including defects; successful execution does not mean repaired.
- [synthetic written files](../../experiments/runtime_integrity/logging_accounting_20260906/runs/mini_run/producer_oracle.jsonl): scripted producer oracle beside real logger CSV/JSONL files. This oracle exists only in the audit; production has no equivalent complete ledger.
- [selected_run_accounting.json](../../experiments/runtime_integrity/logging_accounting_20260906/selected_run_accounting.json): frozen-file hash verification, typed events loaded through `aligned.py`, raw-record counts and one trace. No run accuracy statistics were recomputed.
- [source_manifest.json](../../experiments/runtime_integrity/logging_accounting_20260906/source_manifest.json): exact audited sources; logger SHA-256 starts `6c03281600ad9060`, opportunity writer `ac4ce0693534f3d0`, planner `56100b37b8e9351d`, manager `5e11bd202de065a4`, EFE `a575ba53e74a835e`. Planner/manager match the corrected protocol; EFE matches the later recovery protocol. The protocol does not explicitly freeze the logger/opportunity writer.
- [existing_tests.txt](../../experiments/runtime_integrity/logging_accounting_20260906/existing_tests.txt): **90 passed**. The four requested logging/alignment files alone passed **33 tests** before the wider relevant verification.

## Ranked findings

P1 means a causal/evidence-validity defect to resolve before relying on that property. P2 denotes narrower faults or missing schema guarantees. Reachability describes code/configuration, not measured fault frequency.

| Rank | ID | Classification | Finding |
|---|---|---|---|
| 1 | L10-01 | P1 confirmed current | Summary validity does not reconcile published/terminal IDs; some invalid delivery evidence is erased |
| 2 | L10-02 | P1 confirmed current | Summary precedes callback drain and physical stop; write failure leaves completion latched |
| 3 | L10-03 | P1 confirmed current robot gap, shared 01/07; manager retry partly repaired | State commit, seen identity and outcome publication are not one recorded transaction |
| 4 | L10-04 | P1 confirmed current, previously identified | Distinct fused decisions at equal logger receipt time lose a row |
| 5 | L10-05 | P1 confirmed logger gap; acquisition outcomes added by 04 | No-fusion decisions and new producer batch outcomes lack a durable joined run ledger |
| 6 | L10-06 | P1 schema limitation plus confirmed consumer bug | Terminal row has no posterior/revision; a pre-assimilation prediction can be labeled post-correction |
| 7 | L10-07 | P2 confirmed on malformed input | Opportunity logging can itself raise after claiming an unwritten event |
| 8 | L10-08 | P2 confirmed current, shared 03 | Command diagnostics combine different events and never establish applied motion |
| 9 | L10-09 | P2 confirmed aggregation/label limitation | Held values weight summary means; correction-gap field counts refusals and omits the tail |
| 10 | L10-10 | P2 confirmed conditional identity/schema weaknesses; producer IDs strengthened by 04 | Opportunity writer and consumers still use incompatible identity/validation rules |
| 11 | L10-11 | P2 outcome/timing limitations; shared 09/14 | Contact silence and truth receipt timing do not establish physical outcome or timing accuracy |
| 12 | L10-12 | P2 provenance defects/limitations; shared 13 | Filename collisions and incomplete exact-source binding can misidentify evidence |

### L10-01 — A completed summary can call an unaccounted run valid

**Locations:** [logger:1853](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1853), [1883](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1883), [3256](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3256), [3477](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3477); [campaign validator:1147](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/scripts/visibility_comparison/run_visibility_campaign.py:1147), [outcome selection:1470](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/scripts/visibility_comparison/run_visibility_campaign.py:1470).

**Trigger/expected:** lose an assimilation, receive an extra one, or receive `rejected` without a reason. The contract requires invalid evidence, independent of goal/stuck/collision outcome. Refusals **with** reasons remain valid.

**Observed:** `logging_faults.missing`, `.extra` and `.unreasoned_rejected` all produce `summary.valid_run=true`. `_finish_run` never compares fusion and assimilation ID sets. The callback checks a missing reason only for `dropped`, not `rejected`. The campaign's separate row validator correctly catches these three cases, so the defect is not an assertion that every campaign automatically admits them.

Duplicate/malformed assimilation delivery has the opposite split: logger records only an in-memory invalid reason, returns before retaining the bad payload/duplicate row, and writes one canonical row. Independent file reconstruction and the campaign row validator see a clean chain; the summary says invalid. Audit 13 independently reproduced that campaign outcome selection can still classify `goal_reached` while `valid_run=false`; it also ignores several incomplete-summary/process-exit conditions. This makes retaining the raw fault and consistently consuming summary validity material.

**Consequence:** validity depends on which consumer is used; duplicate/fault counts and exact malformed payloads cannot be reconstructed after the fact. Accepted-boolean/status consistency and finite/causal timestamps are also not validated here.

**Smallest repair:** retain every received envelope in an append-only delivery log, including invalid/duplicate deliveries; separately maintain exactly one canonical terminal outcome per physical correction. At finalization use one shared ID/status/reason/field validator against an authoritative publication ledger. Have 11/13 use that result consistently. Do not classify a reasoned refusal as infrastructure failure or count duplicate transport delivery as a second filter update.

### L10-02 — Finalization is neither a stable snapshot nor a recoverable write

**Locations:** [logger:3256–3266](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3256), [3495–3533](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3495), [callbacks:1684](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1684), [tick:2466](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:2466).

**Trigger/expected:** queued data arrives between an operational stop decision and process termination, or disk/serialization fails. A final summary must describe a declared closed event interval and successfully written evidence. A stop request is not a physical-stop acknowledgement.

**Observed:** `_finish_run` sets `_stop_requested` and `_completed` before I/O, flushes only experiment/plan/perception handles, writes JSON directly to its final path, then starts a 0.15 s wall-time shutdown timer. Subscriptions and `_log_once` remain able to append and mutate counters. `post_summary_tail` writes one assimilation, finishes, then delivers another decision/assimilation/tick: **summary count 1, final CSV count 2**, with a row after stop time. No completed-summary rewrite occurs in `destroy_node`.

`summary_write_failure` injects `OSError` in the real JSON write: the final file exists with **zero bytes**, `_completed=true`, `_stop_requested=true`, parsing returns `None`, and a subsequent `_finish_run` immediately returns. The shutdown timer was never reached. If destruction runs, the completed latch suppresses the interrupted fallback. Separately, `headers_at_finish` proves fusion and assimilation headers can still be buffered at summary time: both files are zero bytes until close. Normal nonempty event callbacks do flush their rows immediately; this is not a claim that all event data routinely stays buffered.

**Registered-run confirmation, separate earlier protocol:** audit 14 identified tracking P1 `experiment_20260906_210032`; this audit independently follows the exact [tracking selection](../../logs/studies/icra_commissioning_20260905/network_navigation_tracking_evidence/selection.json), verifies every selected file hash, and reads rows through the required loader. The summary says `contact_messages_seen=1`, stop/first-crash time **165.184 s**. `experiment.csv` line **1640**, at **165.200 s**, says **17** contacts and still records nonzero command publication `(0.186216 m/s, −0.310306 rad/s)`. One raw tick is after summary stop. [Retained check](../../experiments/runtime_integrity/logging_accounting_20260906/tracking_contact_tail.json). This establishes an actual summary/contact tail in that registered earlier run; it does not establish physical applied velocity at contact. The three corrected-runtime selections above have no such observed count/tail mismatch.

**Consequence:** final counts, validity, contacts and metric accumulators can disagree with files; a completed run can again appear to have “no summary”. `json.dump` also permits NaN/Infinity here, so its output is Python-compatible JSON rather than strict interoperable JSON. `flush` does not provide crash/power-loss durability (`fsync` is absent).

**Smallest repair:** define `stop_requested → producers_quiesced → drain_complete → files_closed → summary_committed`. Record the cutoff and any unresolved producer IDs. Request/observe stop through the command owner; record whether physical stop was established. Snapshot counters only after drain, flush **all** handles, atomically replace a temporary summary, and set finalized only after successful commit. Preserve a structured incomplete/error record if finalization fails. A longer arbitrary sleep alone cannot guarantee drain.

### L10-03 — The state can consume evidence without a truthful outcome row

**Locations:** [manager:1827–1899](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1827); [planner:827–876](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:827), [2508–2529](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2508), [2655](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2655). Active fused-envelope path.

**Expected:** exactly one immutable terminal outcome describes whether a published physical correction changed the belief; diagnostic/display failure cannot change that answer.

**Confirmed prior/current findings:** 01/02 reproduce planning bootstrap from anonymous `/state/bev` before its envelope; the later identified event is recorded `dropped/not_newer_than_belief` despite being consumed. Audit [07 F02](07_multicamera_fusion.md) injects a diagnostic-publish failure **after state commit but before assimilation publication/seen-ID insertion**: state advances, no terminal exists, ID remains unseen. A controlled retry yields `dropped/not_newer_than_belief`; this is not an automatic live retry claim. It also reproduces manager envelope publication followed by compatibility-publish failure: one envelope, no decision. Logger subscribes to the later decision, not to the envelope itself. The later 04 preclaim/error-outcome wrapper prevents same-process manager retry after that failure; it does not make the robot outcome transaction atomic or cause the logger to retain the published envelope.

This audit's malformed-envelope fixture confirms a fatal receiver rejection produces **no assimilation row**; a duplicate envelope does not change state or create a second assimilation. Those integrity failures need separate durable evidence. Their production behavior is process termination; the probe isolates fault cases and intercepts that external effect, rather than pretending a live process keeps running after fatal.

**Consequence:** a CSV equality check can miss an event absent on both observer channels. Reliable topics, per-process sets and locks do not establish durable exactly-once commit across failure/restart.

**Smallest repair:** with 01, make validated correction intake the only bootstrap writer and commit state, seen ID and immutable outcome together; publication retries the stored outcome rather than rerunning filtering. With 07, record actual fusion publication identity independently of subsequent display/decision publication. Record structured integrity failures even when no usable ID can be parsed; relate them to raw delivery bytes/hash and sequence. Preserve the five existing terminal statuses; define delivery/integrity metadata separately rather than adding casual scoring statuses.

### L10-04 — Logger clock is used as an event-deduplication key

**Location:** [logger:1699–1703](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1699). Active decision subscriber; confirmed previously in 01/07 and independently here.

**Trigger:** distinct decisions are serviced under one `/clock` value, including queued callbacks after a pause/backlog. **Expected:** each batch is recorded once regardless of identical receipt times. **Observed:** `receipt_collision` delivers batches `a` and `b` at logger time 10 s, with captures 1.0 and 1.2 s and two assimilations. Only `a` reaches `fusion_observations.csv`; `b` appears to be an extra assimilation. A clock rewind can suppress decisions until the old receipt watermark is exceeded. No live reset was performed; active campaigns set `reset_world=false`.

**Consequence:** valid filter behavior can fail evidence checks or disappear from camera/fused statistics. **Smallest repair:** record each delivery with a monotonically increasing logger sequence and separate receipt stamp; canonicalize by explicit event ID, preserving duplicate/conflict records. Do not suppress a distinct event because its receipt time is equal or older.

### L10-05 — Misses are retained, but their downstream fate is often not

**Locations:** [opportunity subscription:1335](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1335), [decision callback:1687–1698](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1687); [manager pending batches:1191](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1191), [refusals:1635](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1635), [1684](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1684), [1708](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1708), [1729](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/reliability/reliability/nodes/camera_manager_node.py:1729).

**Trigger/expected:** all cameras miss; bootstrap quorum, geometry/common-time support or disagreement rejects every observation; one member is missing; a newer complete batch supersedes an older one. These are distinct outcomes, each needing an ID/reason. No-fusion batches do **not** need fictitious filter assimilations.

**Observed:** schema 7 correctly preserves received hit/miss camera contracts. However, all manager decisions without `observations` return before any CSV write, and the callback never persists the full decision payload. Even successful-decision CSVs omit most per-camera rejection reasons, frame/model metadata and gate diagnostics. The mini-run retains five misses and four of five members of a partial batch, but its explicit `no_eligible_synchronous_observations` decision has no durable row. Raw opportunities cannot identify which downstream gate refused a hit.

Audit 04's baseline proves manager incomplete batches lack expiry/cap or per-batch closure, and complete-batch replacement silently deletes pending work. **Its subsequent repair adds those bounds/outcomes**, emitted through the two new topics and prefixed process logs. Their success/error meaning must still be joined to the actual fused publication and filter terminal outcome. The existing logger never subscribes to them. Acquisition queue losses and unscheduled frames still precede the opportunity log; its manifest correctly scopes it to **received detector outputs**, not all physical captures. Normal detector misses must not be confused with absent frames, inference exceptions or missing ROS delivery.

**Smallest repair:** retain raw manager decisions for **every** source batch, with expected/present/missing/admitted/used camera sets and explicit terminal no-fusion reasons. Collect the existing new 04 outcome streams and coordinate their joins with 07 rather than inventing a competing batch schema. Add producer counters/sequences if all acquisitions must be accounted; do not silently alter the current strict all-camera scheduling policy.

### L10-06 — Exact post-correction beliefs and asynchronous diagnostics cannot be reconstructed

**Locations:** [assimilation payload:2341](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2341), [belief publication:2777](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/planning/planning/nodes/unicycle_planner_node.py:2777); [logger held fields:1763](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1763), [1850](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1850), [3107](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3107); [consumer:406–459](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/experiments/fusion_on_fixed_routes/aligned.py:406).

**Expected:** posterior plots/replay join one terminal update to its actual resulting state/covariance; a planning snapshot has its own revision/target. **Observed schema:** terminal rows contain only batch ID, correction/apply times, status/reason, accepted flag, NIS and belief timestamp. They lack pre/post mean/full P, R, frame, state revision and logger receipt time. `apply_stamp` is a publisher clock read after commit/diagnostic work, not a measured callback-entry or exact commit instant. The public belief has state time and full ROS covariance but no batch/revision/commit provenance; it is sampled by the logger, not logged for every publication.

`belief_at_fusion_events` filters by accepted status but then picks the first public **state timestamp after capture**, ignoring assimilation apply time. 01 identified this; audit [11](11_alignment_replay_and_reporting.md) independently reproduces capture 1.0/apply 1.3 selecting belief 1.1, which predates assimilation. Two corrections can also receive the same public sample. Selecting a sample after apply would still not recover the exact posterior after intervening prediction/corrections. The shared schema, not merely the timestamp predicate, needs repair.

The main CSV similarly holds correction diagnostics, manager camera sets, planner text/numeric arrays, state and belief independently. They are contemporaneous snapshots of latest arrivals, not one transaction. A bootstrap may leave older correction diagnostics held. `pixel_corr_pred_*`/`next_*` are partial diagnostics, not a complete per-batch posterior ledger; prediction covariance and actual replay support cannot be recovered from them. Audits 01/02 also establish out-of-order/equal-time revision problems that logger last-arrival storage cannot repair.

**Smallest repair:** 01 owns one immutable correction output with revision, pre/predicted/post state and full P, R, innovation/NIS, support and reason. 02 owns state target, callback/commit/publish/receipt clocks and ordering. 07 owns source/member identity; 11 must fail closed for an “exact posterior” query when that record is absent. Keep diagnostic tick snapshots explicitly labeled as held asynchronous fields.

### L10-07 — Malformed raw evidence can crash its audit writer

**Location:** [CameraOpportunityLog:16–40](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/core/camera_opportunity_log.py:16), called by an unguarded logger subscription lambda at [1341](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1341).

**Trigger:** JSON parses but has NaN/Infinity in an unchecked nested field, or output write/flush fails. Normal `CameraObservation.to_json` converts optional NaN to null and rejects nonfinite output, so the nonfinite example requires a malformed/replacement producer or corrupted payload; it is not a normal native-detector output.

**Expected:** preserve raw bytes/reason as invalid evidence, or explicitly fail the logger with an auditable write failure. **Observed:** `opportunity_nonfinite` inserts NaN in `detector_score_raw`. The writer increments `rows` and updates `seen`, then strict `json.dumps` raises outside the validation handler. **Zero rows are written; retry of the corrected first record is marked duplicate.** An I/O exception can likewise leave counters advanced before durable output. The logger's launch-exit policy can end the whole run after such a callback exception.

`valid_contract=true` additionally validates only camera/batch/stamp/boolean basics. A `schema_version='future'` record is marked valid by this writer but rejected by `CameraObservation.from_dict` in the real manager.

**Smallest repair:** distinguish raw-delivery retention from contract validation; serialize an invalid record containing the original string and structured reason for any contract/serialization failure. Validate through the shared camera contract or rename the narrower flag. Update canonical seen/count state after successful write; retain failed-write metadata independently. Bound in-memory identity storage by run/session policy without losing duplicate detection.

### L10-08 — Command rows mix events and cannot mean executed commands

**Locations:** [logger callbacks:1772](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1772), [assembly:2736–2759](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:2736), [execution fields:2937](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:2937); [adapter:169–188](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/sim/sim/actuation_noise_node.py:169), [watchdog:190](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/sim/sim/actuation_noise_node.py:190). Active with command noise.

**Trigger/expected:** raw request/output/noise diagnostic arrive asynchronously. Use the adapter's internally paired input/output and associate every time with that same message; keep requested, published, received and physically applied distinct.

**Observed:** the logger takes output from latest `/cmd_vel`, raw receipt time from latest `/cmd_vel_raw`, but overwrites raw **values** from the held noise diagnostic. It discards that diagnostic's output values and producer timestamp, then subtracts mixed values. The real-method fixture has a new raw 0.10 m/s at 2.1 s, output 0.09, and old diagnostic input/output 0.20/0.18 at 1.0 s. Written raw is **0.20 stamped 2.1**, and noise error is **−0.11 m/s**, versus **−0.02** for the internally paired old diagnostic. These are scripted values, not observed actuator errors.

`exec_cmd_v/w` are planner tape-publication diagnostics; `/cmd_vel` is the adapter's requested output, not observed wheel/body motion. Watchdog zero and invalid-command stop lack normal noise diagnostic records. Latest-value sampling can miss intermediate commands completely. Audit 03's held execution diagnostic after stop is repaired in the later EFE source; logger event mixing and the actuation boundary remain open.

**Smallest repair:** append each adapter diagnostic as an event preserving its own input/output/time; keep topic receipts separate until explicit command IDs exist. With 03, add command/tape ID, belief revision, issue/validity times, adapter receipt/output times and stop reasons. Applied time/state requires a bridge/actuator acknowledgement or measured physical state; do not relabel publication as execution.

### L10-09 — Summary weighting and gap semantics are not event-update statistics

**Locations:** [logger means:3088–3100](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3088), [gap accumulator:1870–1879](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1870), [gap summary:3490](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3490); [monitor:72](../../scripts/visibility_comparison/monitor_campaign.py:72).

**Observed:** alignment repairs correctly compare each held estimate to truth at its own stamp, but `_log_once` adds the same value again on every tick. In `held_and_diagnostics`, synthetic belief errors 100 cm and 0 cm, with the first held for four rows, produce summary mean **80 cm**, versus **50 cm** over the two distinct states. Both statistics are mathematically definable; the summary does not declare the tick weighting or sample counts. A held `/state/bev` reading has the same risk. Do not reinterpret existing summaries as independent detections or exact event-weighted results.

`longest_correction_gap_s` advances on every finite terminal correction stamp, including rejected/dropped events, in arrival order. It neither filters accepted updates nor closes initial/trailing no-update intervals. Fixture: accepted at 1 and 5 s, rejected at 2/3/4, stop at 10: **reported 1 s**, accepted-update interval 4 s and trailing absence 5 s. Calling this field “how blind its worst stretch was” or monitor `blind_s` overstates its meaning. Delayed out-of-order terminal stamps can also move the remembered stamp backwards.

**Smallest repair:** explicitly name and retain tick-weighted diagnostics, unique-publication statistics, fused-arrival gaps and accepted-update gaps separately, with counts/window endpoints. Agree the desired gap endpoint convention with 11 and version any new fields. Do not change old scoring or discard reasoned refusals as a shortcut.

### L10-10 — Event identity and CSV compatibility are only partly enforced

**Locations:** [detector identity:947](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/perception/perception/nodes/batched_four_camera_yolo_node.py:947), [opportunity identity:32](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/core/camera_opportunity_log.py:32), [fusion identity:1722–1737](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1722), [offline reading identity:329](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/experiments/fusion_on_fixed_routes/aligned.py:329); [schema defaults:77](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/experiments/fusion_on_fixed_routes/aligned.py:77), [tests](../../tests/experiments/test_logger_schema.py).

The selected-run/baseline detector source ID preserves camera IDs and exact capture nanoseconds but has no session/epoch or invocation sequence. It names a full inference cycle; default chunking makes that cycle several backend calls (04). **The concurrent 04 repair now supplies process-session/cycle identity and separate chunk invocation IDs.** Raw opportunity dedup still uses `(camera, batch, float stamp)`. Fusion/readings still use `(shortened camera, round(stamp,6))`; the planner uses source ID alone. The mini precision fixture's 1.0000000/1.0000004 s **distinct batch IDs become repeat 0/1** in fusion CSV. Conversely, same camera/batch with a changed capture stamp is fresh in the raw opportunity log. These conflicting logger/consumer meanings are confirmed; the 0.4 µs trigger is outside normal selected 5 Hz cadence and reset is disabled, so no such selected-run collision is claimed.

CSV widths currently match for the three writers covered by the existing AST tests. That protects header-versus-row length, not swapped meanings, duplicate header names, data types, every other writer, message-array versions or partial rows. Assimilation receiver ignores `schema_version`; camera opportunity validator ignores the full camera schema. Positional diagnostic arrays use length checks. `aligned.schema_version` can silently fall back to legacy schema 1 on a missing/malformed manifest; audit 11 owns stricter consumer cases. These are confirmed validation limitations, not an invented current column-offset failure.

**Smallest repair:** agree epoch-scoped capture ID, inference-cycle ID, per-camera observation ID and delivery ID with 02/04/07. Keep timestamps as fields, not substitutes for identity. Preserve canonical camera IDs; detect conflicting payloads for the same ID. Version each record family, validate named schemas/types/widths and reject unsupported versions for current evidence. Keep historical legacy parsing an explicit diagnostic mode.

### L10-11 — Physical outcome and truth-time certainty exceed the recorded evidence

**Locations:** [contacts:1995](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1995), [geometry stamp:3012](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3012), [first-command liveness:3233](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3233), [truth fallback:1661](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:1661), [summary claims:3405](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:3405).

Publisher/message counts repair the earlier omission, but `collision_contact=false` is still emitted when no contact messages were ever received. A publisher is not evidence of a functioning full path. The three exact runtime summaries each report **0 received contact messages and 46 publishers**. The warranted statement is “no recorded contact”. Audit 14's separate [native simulator fixture](../../experiments/runtime_integrity/simulator_world_20260906/physics_results.json) confirms the clear case emits zero contact bytes, while a known overlap yields [2,500 contact events](../../experiments/runtime_integrity/simulator_world_20260906/contact.contacts.jsonl). This is an event-only channel; silence alone cannot distinguish clear from broken delivery. Positive controls or an independent liveness signal are needed for that stronger interpretation. No new validity policy is imposed here.

Contacts have no append-only event file, callback receipt, source link/complete contact payload or physical snapshot. A collision callback can finish before the next experiment tick. Logger geometry uses latest GT coordinates but assigns `odom_map_stamp` to the shared crash timestamp; final goal distance similarly uses latest GT without recording its final reference stamp in the summary. Audit 09 additionally reproduces summary creation while command publication remains nonzero. Operational completion is not verified rest.

For zero-stamped truth transforms, logger labels samples with ROS callback receipt. The comments claim inter-sample intervals bound transport delay. **That bound is false:** `truth_receipt_latency` injects a constant **5 s** delay with **1 ms** sample spacing; the real callback reports `receipt_sim_clock` and a 1 ms maximum interval. This is a counterexample, not a measurement of live delay. Equal/backward truth stamps can also update the held latest pose while being excluded from the interpolation buffer. The full-rate truth buffer is not persisted, and `gt_samples` is its capped length, not lifetime sample count.

**Smallest repair:** distinguish contact observation state from no-contact evidence; record contact and terminal snapshots with source and receipt times, and declare channel semantics. Use the GT sample's own timestamp for geometric/reference records. Propagate the simulator's actual Pose_V time through the bridge for truth; until available, state transport latency as unmeasured rather than bounded by sample spacing. Agree these fields with 02/09/14. Keep truth out of goal/stuck/control decisions.

### L10-12 — Exact path/artifact provenance is stronger than HEAD, but incomplete

**Locations:** [manifest wrapper:10–33](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/core/manifest.py:10), [common manifest:10–16](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/unav_common/unav_common/manifest.py:10), [snapshot/write:91–106](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/unav_common/unav_common/manifest.py:91); [logger hashes:701](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:701), [806–830](../../experiments/runtime_integrity/logging_accounting_20260906/source_snapshot/src/experiments/experiments/nodes/experiment_logger.py:806).

**Present and verified:** schema declaration; source-label/settings manifest; content-sensitive git diff/untracked hashes; YOLO/compiled-model, camera-network, calibration, world-covariance, geometry and campaign-config hashes; config snapshots. The active reference-calibration artifact indirectly binds the learned NN hash. It would be incorrect to report YOLO hashing absent.

**Confirmed shared findings:** 13 independently reproduces two different `config.yaml` inputs overwriting the same basename snapshot/key; same-second run IDs share a directory because `exist_ok=true`, while CSV writers open with `w`; git provenance is cached per repository root for process lifetime. Active normal config basenames are distinct and the campaign is sequential, so those collision triggers are conditional, not demonstrated destruction of these selected runs. Cache staleness matters if one process generates another manifest after source changes; caching can intentionally preserve its startup snapshot if that is what the field promises.

**Limitations:** config snapshot return mapping is discarded; no resolved module-file hashes or loaded-process source fingerprint proves the exact imported code. Logger/opportunity writer, launch, simulator description and other transitive source inputs are not all enumerated in the corrected protocol. A global dirty hash can detect a checkout difference but cannot reconstruct its bytes or identify which were loaded. Current source/artifact files could change after one process loads them and before another hashes the path. That race was not exercised.

**Smallest repair:** with 13, create directories exclusively with a collision-resistant run/session ID; retain path-keyed config snapshots and hashes; bind each executable's resolved source/config/artifact identity at startup and freeze bytes for the selected run. Make startup provenance versus current filesystem provenance explicit. Use atomic manifest replacement. Do not refit or substitute artifacts to repair metadata.

## One physical camera capture through the available evidence

Exact example: registered runtime P0, `.../P0/seed210/experiment_20260906_213015`, first fused batch:

```text
strict:camera_A@3400000000,camera_B@3400000000,camera_C@3400000000,camera_D@3400000000,camera_E@3400000000
```

Camera **B** is a hit; camera **A** in the same physical cycle is a miss. `camera_opportunities.jsonl` lines 2/1 respectively retain them. `fusion_observations.csv` line 2 records B; `correction_assimilations.csv` line 2 records the batch. Full values and hashes are in the retained trace; numeric coordinates below are record contents, not accuracy claims.

| Stage | Identity and timestamp ownership | Frame/units/covariance meaning | Terminal record / reconstruction limit |
|---|---|---|---|
| Capture | Camera B image header exact ns=3,400,000,000; source ID embeds all five members. Capture float ≈3.4 s | Image pixels in B's original camera coordinates; no measurement covariance on the image | No runtime image hash/capture sequence or all-acquisition ledger. Cannot prove every physical frame reached detector |
| Detector receipt/inference | B image callback receipt 3.427; inference start/finish both 3.476 ROS s; inference wall duration 754.799879 ms; publication preparation 3.476 | ROS simulation timestamps and monotonic elapsed wall ms are different clocks. Equal ROS start/finish is not zero compute time | `camera_opportunities` observation fields; no per-chunk invocation ID/timing; publication stamp is not acknowledgement |
| Detector result/miss | B pixel=(65.0143,459.3118), bbox and confidence 0.949896; A `detection_valid=false`, null pixel/bbox. Same batch ID | B bottom-centre pixels; raw `conditional_cov_uv≈0.614904 I` px² is the detector contract's statement. Manager replaces it with configured observation-model covariance; A's miss covariance is not a filter update | Valid hit/miss CameraObservation, schema `phase0.v1`; detector output to logger receipt 3.693, delivery index 1 for B. Manager receipt time absent |
| Camera metric observation | B capture `obs_stamp≈3.4`; camera string becomes `B` in CSV. Raw projection≈(-7.688169,-8.998426), operational mean≈(-7.886615,-8.633010) | `map_bev`, robot floor/metric reference, metres; full B R `[[.0333131,.0126175],[.0126175,.2790908]]` m². Active NN then frozen residual offset/R, not filter P | B is used; A absent from fusion rows but its miss survives JSONL. Admission prior/revision, exact manager receipt and full per-camera terminal reasons not retained |
| Fusion | Same source ID; common capture/fused stamp≈3.4; manager decision logger receipt 3.802; used cameras B and C | Prior-free robust metric fused mean≈(-7.892793,-8.614565), full R `[[.0537255,.0165344],[.0165344,.0862621]]` m². This is robust aggregate R, not robot P or independent-precision forecast | Envelope schema 1 carries `map_bev`; logger does not retain envelope directly. CSV decision has no explicit fusion start/finish/publication timestamp or frame |
| Assimilation | Same source ID; correction stamp 3.4; `apply_stamp=3.802`; `belief_stamp_after=3.4` | Metric EKF bootstrap uses fused R for XY P and odometry-derived heading uncertainty. No pixel yaw measurement | `accepted_bootstrap / bootstrap`, accepted=1, NIS unavailable. No callback-entry, exact commit instant, prior/posterior, revision, frame or logger receipt in row |
| Belief publication | First retained sample with state time after apply is stamped 4.001; logger row also 4.001 | Published predicted mean/full planar/yaw covariance, state valid at header target; CSV retains the symmetric full planar/yaw block across columns | No source-batch/revision join or publication sequence. This row cannot certify the exact bootstrap posterior or every intervening publication |
| Command | First nonzero topic sample is experiment CSV line 482: logger tick 49.0, raw/output receipt 48.927, belief state time 48.902 | Raw/output linear values≈0.202277/0.187800 m/s and angular≈0.167984/0.136214 rad/s; body-forward/z-turn convention, no frame field in `Twist`; these are observed publications | Many corrections intervene. **No causal command-to-capture ID exists.** Adapter/bridge/actuator receipt and applied time/state are absent; cannot claim this capture caused that command |
| Physical outcome / summary | Operational completion `stuck`, stop stamp 240.301; summary has 813 assimilations; last apply 240.207; last logger row 240.301 | Belief decides stopping; GT fields are offline reference/geometry. Contact counts distinguish lack of messages from counted contacts | No recorded contact (0 messages, 46 publishers), no actuator-rest acknowledgement. Raw event counts agree for this selected run; no complete physical outcome chain can be inferred |

The strict inference batch is timestamp-based rather than an independently assigned simulator capture-round ID. Timing/geometry repairs do not remove that identity limitation. Active robust fusion is not independent Gaussian precision addition; no covariance-model change is proposed by this audit.

## Synthetic mini-run and independent accounting

The probe invokes the real camera contract serializer, opportunity writer, logger callbacks, fused-envelope receiver, correction computation/status publisher, public belief timer and summary writer. Controlled messages supply boundary inputs; camera image inference and manager geometry are not rerun. The filter fixture uses a transparent stationary prediction test model and scripted positive R, not commissioned sensor statistics. Producer/delivery faults are injected explicitly. Duplicate and malformed fatal cases use separate receiver fixtures so the audit does not claim normal execution continues after a fatal process stop.

Independent reconstruction reads the written CSV/JSONL and scripted producer oracle rather than logger counters. Physical batch IDs in the table are abbreviated by capture seconds; the files preserve complete strings.

| Scripted event | Written outcome | Counting result |
|---|---|---|
| Capture 1.0, receipt 1.1 | `accepted_bootstrap/bootstrap` | One update |
| Capture 1.2, receipt 1.3 | `accepted/accepted` | One update |
| Capture 1.4, gross outlier | `rejected/nis_too_large` | Zero updates, explicit reason |
| Capture 1.6, delayed receipt 2.0 | `accepted/accepted` | One update, original capture retained |
| Capture 1.8, stale receipt 3.0 | `dropped/stale_age` | Zero updates, explicit reason |
| Capture 2.2, terminal delivery omitted | Filter emits outcome; logger receives none | One missing terminal; raw fused row reveals it |
| Malformed published envelope | Receiver fatal; no terminal assimilation | One missing terminal plus unretained receiver fault in production |
| Repeated 1.2 detector delivery | Extra JSONL delivery marked duplicate | No new physical observation |
| Repeated 1.2 fused envelope | Fatal duplicate guard; state unchanged | No second assimilation/update |
| Five-camera miss batch | Five `detection_valid=false` JSONL records | Misses observable; no invented filter update; manager no-fusion reason discarded |
| Missing camera E | Four camera records | Missing member observable only with expected-member oracle; no manager terminal |
| Malformed camera JSON | Raw string and reason retained | Invalid delivery, not a hit or miss |

Result: **7 unique scripted fused publications, 6 generated assimilation messages, 5 written assimilation rows, 3 accepted update events, 2 missing rows**. The 41 opportunity deliveries contain five misses, one duplicate and one malformed record. Written status counts are accepted=2, bootstrap=1, rejected=1, dropped=1. Summary count 5 and dropped fraction 0.2 agree with the **received** rows, yet summary `valid_run=true` wrongly omits the two missing terminals. The campaign row validator rejects the mismatch. No event is treated as a second update by the independent oracle.

A separate five-status control calls the real terminal publisher for `accepted`, `accepted_bootstrap`, `reanchored`, `rejected`, `dropped`: exactly five rows, exactly **three update classes**, matching final summary and successful existing campaign validation. It verifies the status contract; it does not enable reanchoring in the active configuration. Additional isolated variants cover extra/duplicate/malformed/unknown/reasonless terminal deliveries, equal logger receipt times, post-summary tails, output failure, held fields and identity precision.

| Required property | Result |
|---|---|
| Every published fused correction has exactly one terminal row with source ID | Positive control passes; deliberately malformed/missing cases fail, and current summary fails to notice |
| Only the three accepted classes count as updates | Pass in independent reconstruction and producer-status control; timestamps advancing on refusals do not count |
| Rejections/drops have explicit reasons | Ordinary producer cases pass; malformed reasonless rejection exposes logger/validator discrepancy |
| Misses remain observable | Pass for delivered detector outputs in schema 7; all physical acquisitions and subsequent no-fusion reasons remain incomplete |
| No event counted twice | Normal physical duplicates do not add updates; fault delivery evidence is lost in canonical assimilation CSV; conditional identity mismatches remain |
| Final summary agrees with raw files | Clean control and three selected runs pass counts; deterministic shutdown tail and invalidity checks fail |

Reproduction from repository root, pinned to the audited implementation:

```bash
python3 experiments/runtime_integrity/logging_accounting_20260906/reproduce_snapshot.py
python3 experiments/runtime_integrity/logging_accounting_20260906/registered_boundaries.py
```

The snapshot runner uses a temporary checkout, reads the frozen selection and compares its outputs to retained evidence. The direct probe now refuses changed source hashes before writing anything, preventing later repairs from overwriting baseline evidence. It intentionally preserves reproduced broken summaries, including one empty file, as evidence.

## Documented repairs, remaining hypotheses and ownership

| Recent regression seed | Current assessment |
|---|---|
| Held camera values counted as repeated detections | Manager's ordinary once-per-source decision and capture-based offline reading dedup are reverified; raw deliveries retain duplicate flags. Held tick weighting/conditional identity exceptions remain L10-09/10 |
| Camera/fused/belief scored against wrong reference instant | Requested alignment tests pass: own stamps, GT buffer, no endpoint extrapolation. Reference receipt-delay uncertainty and post-assimilation belief association remain separate faults |
| One detector batch assimilated repeatedly | Mandatory-envelope duplicate guard and transaction tests pass; no second ordinary update. Bootstrap bypass, partial publication/commit and restart guarantees remain L10-03 |
| Offline inference of assimilations from timestamps | Explicit schema-4+ outcomes exist; accepted selection is required. Exact posterior is still guessed from public timestamps (L10-06, 11) |
| Completed run summary parser returned no summary | Existing `_read_run_summary` regression passes. New write/finalization failure L10-02 and 13's attempt/path attribution bugs remain |
| Reasoned refusal invalidated drive | Requested regression passes; the five valid statuses remain unchanged. Missing reason and inconsistent validators are separate failures |
| Wheel odometry called truth in campaign monitor | Current monitor reads `mean_belief_error_gt_after_first_cmd_m`; repaired by source inspection. Legacy perception/odom diagnostic names and held mean/gap interpretation still require care |
| Published command called executed/applied | No applied-command instrumentation exists. Later idle execution-diagnostic repair is verified by the 90-test suite; publication/receipt/applied distinction remains open |
| Missing/expired detector batches, restart ID collision, partial manager retry | 04's later 101-test repair package adds explicit bounded-buffer outcomes, process-session/cycle/chunk IDs, preclaimed manager decision and exit handlers. The existing logger does not collect the new topics; the robot commit/outcome gap is unchanged |

No actual DDS queue-pressure trial, SIGKILL/power-loss test, live reset, stuck inference, slow filesystem integration run, or actuator/contact bridge failure test was performed. Depth 10 decision/assimilation queues, depth 100 opportunity queues, volatile subscriptions, synchronous per-event flushes, header buffering, absent acknowledgements and shutdown without drain establish **missing guarantees**; they do not measure actual transport-loss rate. A crash can leave only a prefix or interrupted summary; equal counts cannot detect symmetric observer loss. Startup observations can precede logger subscription and are not replayed from these volatile topics. No specific amount of startup/queue loss is claimed.

Ground-truth firewall inspection: the logger subscribes to `/ground_truth_tf` and writes GT into evaluation files; active manager/planner subscriptions do not read that topic. Automatic goal/stuck calls receive operational belief distance, not GT. Geometry and final-goal statistics use GT offline. Planner receives the shared run directory to **write** global artifacts; no inspected active path reads the logger's GT columns into its controller/gate. Shared filesystem access is not a security isolation boundary, so files remain technically readable; no runtime leakage through them was demonstrated. Reference-calibration provenance/model commissioning is separate from feeding live GT into a gate. Keep future event envelopes truth-free and evaluation augmentation separate.

Still not reconstructable from the schema-7 structured run files and selected evidence:

- A complete ledger of every physical acquisition/eviction, missing detector member, partial publication or no-fusion outcome; original image bytes/hash and physical capture sequence. The newer 04 process-session/chunk outcomes now supply parts of this chain in separate topics/process logs, but are not joined into the structured run files.
- Manager callback receipt, fusion start/finish/output time, exact assimilation callback-entry/commit time, logger assimilation receipt, global cross-topic callback ordering and in-flight state at crash.
- Exact pre-fit prediction covariance, pre/post full state per correction, motion-history snapshot/support/fallback details, same-time belief revision, exact LOCAL planning belief/covariance/goal snapshot and which update each public belief contains.
- Exact command issue and adapter receipt for each request, bridge/Gazebo receipt, physical applied time, motor/wheel targets versus measured actuator state, command acceptance/rejection/expiry for every tape, and acknowledged terminal rest.
- Full contact event payload/source/receipt and channel liveness over the run; exact full-rate GT sequence or fixed transport delay for zero-header samples.
- Exact process-loaded transitive source/config/artifact bytes where only path/aggregate provenance is retained.

Coordinate the smallest repairs as a shared evidence contract: **01** owns filter transaction payload/commit and update meaning; **02** owns clocks, epoch/revision and ordering; **07** owns fused/camera identity and decision/publication outcomes; **10** owns append-only delivery/canonical logs, validation and finalization; **11** owns explicit consumer schemas/joins/scoring; **03/09/14** own command stop and physical evidence; **13** owns exact run/provenance/lifecycle identity. These findings were shared with those active investigations. Payload, timing and scoring changes are proposals here, not silently implemented fixes.

The first repair package can stay within logging ownership:

1. Replace receipt-clock suppression with source-ID canonicalization, preserving a separate ordered delivery record for duplicates/conflicts/invalid payloads. Keep existing scientific CSV fields unchanged; add explicitly versioned event files.
2. Retain actual fused envelopes, all manager decisions and the two existing 04 outcome streams. Check canonical publication/terminal ID sets and the five-status/reason contract through the common validator proposed by 11. Do not invent an accepted outcome for a no-fusion batch.
3. Coordinate producer stop/cutoff and drain with command/campaign owners. Flush/close every stream, write a temporary strict-JSON summary and atomically replace the final file; latch completion only after success. Retain an explicit incomplete finalization record on failure. A durable logger receipt proves receipt, not actuator execution or a producer commit.
4. Convert the equal-clock, missing/extra/reasonless, malformed/duplicate, post-summary-tail and write-failure probes into desired-invariant regressions. Keep the immutable committed-posterior/revision schema as the separate 01/02/07/11 package; no payload or scoring interpretation is invented here.
