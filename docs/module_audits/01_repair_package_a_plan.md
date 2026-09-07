# First state-estimation repair package: immutable motion and causal input ownership

Planning/revalidation only, requested by the audit coordinator after [audit 01](01_state_estimation.md). **Ready to implement as a small, sequenced package once assigned; no runtime edits made here.** This closes R02 and the active mandatory-envelope branch of R03, and fixes the chronological/atomic parts of R01/R06. It does not close missing-motion validity, wrong-time heading, epoch reset or recovery-policy findings.

[Revalidation evidence](01_repair_package_a_revalidation.json) contains six real-method cases and current hashes. The four requested defects still reproduce; the timing audit's callback barrier also reproduces mismatched latest yaw/velocity. The accepted-update covariance trace still agrees with the independent oracle to 6.94×10⁻¹⁸. These are successful reproductions of existing defects, not passing desired-invariant regressions.

The current node hash is `1191b8879b8515d7d2bd176015538cacd0bb9ed196af49da80d6cd8705bd8995`. Compared with the audited frozen source, only `_publish_plan_and_metrics`, `_publish_planner_diagnostics` and `_publish_planner_status_text` changed. The proposed functions below are unchanged. Preserve those concurrent publication edits; apply focused hunks to the current working tree. `motion_history.py` and `encoder_noise_node.py` retain their audited hashes.

**1. R02: pass the checked immutable history into prediction. Ready; no policy/schema dependency.**

Smallest production surface: `motion_history.py` plus `_advance_belief_over_outage` and `_predict_belief_to_now` in `unicycle_planner_node.py`.

- Add a small frozen `MotionHistorySnapshot`, holding tuples of primitive odometry and command entries and the selected source. Capture both histories because the existing prediction can fall back to commands; freezing only the preferred list would leave that fallback reading mutable state.
- Capture it inside the same existing `_data_lock` block as the outage anchor m/P/stamp/last command. Release the lock before coverage or numerical work.
- Add a keyword-only `motion_snapshot=None` parameter to `_predict_belief_to_now`. Normal callers take one snapshot when none is supplied. When supplied, it must not read either live history or a live source selector.
- `_advance_belief_over_outage` checks the selected entries from this snapshot, then passes that exact snapshot to prediction. Keep the current replay cap, source precedence, suffix fallback, reach inflation, catch behavior and correction refusal unchanged.
- Keep numerical dynamics and return shape `(m, P)` unchanged. Do not combine the two replay loops or alter the empty-history assumption in this first hunk; their different fallback policies are a separate open part of R01.

Convert `outage_support_snapshot_race` in `experiments/estimator_consistency/module_audit_01_20260906/probe.py:177` into a regression in `tests/planning/test_outage_motion_replay.py`: trim the live buffer after snapshot/check, require the original 0.2 rad turn and full expected P at 59.9 s, and retain the existing `dropped/replay_gap_too_large` terminal status. Use a barrier/hook with `*args, **kwargs` so the new snapshot parameter is forwarded. Also cover preferred-odom-absent command fallback: mutate both live histories after capture and require both source selection and replay to remain fixed. Retain the original supported blind-turn and unsupported-history regressions. The corresponding timing probe is `coverage_check_and_replay_use_different_buffers`.

**2. R03: mandatory-envelope mode cannot bootstrap from the compatibility pose. Ready; no policy/schema dependency.**

Smallest production surface: the `not has_belief` block of `_resolve_state_belief_ekf` at approximately line 2655.

When `require_state_correction_envelope` is true, do not call `_apply_state_correction(state_ref)` there. Return the existing unavailable result until the envelope callback has bootstrapped the belief. Preserve the explicitly configured legacy non-envelope fused fallback. No subscription or envelope-format change is required. Leave per-camera startup cleanup to its separately scoped repair; this package must not silently change that optional mode.

Convert `untracked_bootstrap` in `experiments/estimator_consistency/module_review_20260906/probe.py:64` into `tests/planning/test_planner_node_state_correction.py` coverage:

1. Deliver the compatibility pose, run the resolver, and require no committed belief or anonymous assimilation.
2. Deliver its valid envelope and require exactly one `accepted_bootstrap` row carrying the physical batch ID and capture time.
3. Exercise the reverse delivery order; it must produce the same initialized state and one identified event.
4. An invalid envelope plus a usable compatibility pose must not initialize the estimator; retain existing integrity-stop semantics. Keep a legacy non-envelope bootstrap test passing.

This changes scheduling to obey the already-required envelope contract. It does not change the freshness/NIS gate or bootstrap heading policy (R05).

**3. R01 chronology and atomicity: validate one odometry event, then commit all fields together. Ready with explicit late-input disposition.**

Smallest production surface: odometry watermark initialization and `_odom_cb`, approximately lines 1032–1048. Use primitive integer nanoseconds for a private last-accepted watermark; retain the existing replay-entry representation for this package.

Parse timestamp, velocity and yaw into local values before mutation. Reject malformed/non-finite inputs without inventing a receipt-time measurement stamp. Under `_data_lock`, check the timestamp against the accepted watermark and commit latest yaw, latest velocity, origin (when unset), history append, watermark and trimming together. An older or equal-stamp odometry message is a no-op for all these estimator fields. Exact duplicate and conflicting equal-stamp messages add no motion/noise; retain the first accepted event at that timestamp and expose a reason/count for diagnosis. Check the watermark inside the lock so two callbacks cannot both pass an earlier check.

Choose refusal of late odometry rather than sorted insertion for this first package: retrospective insertion could alter a history already used by an accepted correction without rewind/replay of that correction. The test and documentation must state this arrival policy. Rejecting old input does not establish that the remaining history has complete temporal support; the rest of R01 remains open.

Convert the `out_of_order_motion` probe into a node regression: callbacks 9.4, 9.7, 9.5 s leave accepted history 9.4, 9.7; the 0.6 s prediction gives x=0.12 and yaw process growth 0.00024, with actual replay duration 0.6. Convert audit 02's `odometry_callback_finishes_out_of_order` barrier: when 11 s commits before delayed 10 s, the 10 s callback must not overwrite yaw, velocity, origin or history. Assert all latest fields belong to the 11 s event; do not compare with a sorted-history oracle containing a deliberately refused 10 s input. Test duplicate delivery and invalid first/later messages without watermark poisoning.

The command callback has an adjacent atomicity issue. A tightly scoped companion hunk may move receipt-clock capture inside its `_data_lock` block so `_cmd_log` append and `last_cmd` belong to the same lock-ordered receipt. Equal-time command callbacks remain permitted: unstamped Twist does not supply physical duplicate identity, and two commands can arrive in one clock tick. Do not deduplicate commands or claim source-time chronology. Clock rollback/epoch invalidation and stamped command identity belong to audit 02/03 and are sequenced afterward.

**4. R06 chronology: old encoder input must not move its interval anchor or RNG. Ready; large-gap handling remains open.**

Smallest production surface: `_odom_cb` at approximately lines 251–274 in `encoder_noise_node.py`; new tests in `tests/sim/test_encoder_input_chronology.py`.

Validate/compute dt before mutating `_last_stamp`. For malformed or `dt<=0` input, return before changing any pose, covariance, slip state, scale Jacobian, timestamp or random-generator state. First-message initialization must similarly validate before committing fields. Retain a copied/owned stamp (or primitive time) rather than a mutable incoming message reference. The encoder currently uses a single-threaded executor/default callback group, so this repair does not require a new cross-callback lock.

**Do not simply move timestamp assignment after every dt gate.** The existing `dt>max_dt_s` branch resets the interval baseline so the next short interval can proceed. If a large gap leaves the old baseline forever, all later messages also exceed the cap and the encoder freezes. Preserve that positive-large-gap baseline reset explicitly in this chronology-only package; name/comment it as an interval baseline, not proof of continuous pose support. Normal valid steps advance it once. A full computed-state commit after successful arithmetic is appropriate, but publication failure must not make the same motion eligible for reintegration.

Split `encoder_order_and_gap` at probe line 243 into two tests. The old-input regression uses 0, .2, .1, .3 s at 1 rad/s with zero random innovations and requires 0.3 rad, not 0.4. Compare the entire pose/covariance/slip/RNG state of a stream with and without an inserted duplicate/old message. The large-gap compatibility test uses 0, .1, 1.1, 1.2 s: preserve the current skip/rebase behavior and resumption, while explicitly labelling the omitted interval as an unresolved validity defect. Do not turn that compatibility assertion into a claim that its pose timestamp is correct.

**Fields and policies deliberately outside this package.**

| Item | Needed for the four narrow fixes? | Follow-up dependency |
|---|---|---|
| Private immutable motion snapshot and accepted-input watermark | Yes; internal only | Audit 02 should extend these rather than create another history owner |
| Local input disposition/reason and dropped-input counters | Recommended, without changing existing CSV/envelope semantics | Audit 10 can persist them; no new validity meaning should be implied |
| Persistent motion support/unknown-interval record and pose validity | No for ordering/R02/R03; yes to close remaining R01/R06 | Must travel from filter/encoder to controller, publication and logger |
| Epoch and input/output belief revision | No for these mechanical fixes | Audit 02 reset/publication work and audit 10 transaction ledger |
| Capture-time odometry heading and uncertainty | No | R05, with timestamped heading support |
| Changed camera-gap acceptance, command fallback, missing-motion covariance, stop/relocalization policy | No; expressly excluded | Separately frozen policy package/ablation |

The encoder's skipped-gap pose still needs an explicit validity/unknown-motion state or a declared recovery before it can honestly be treated as current. A receive-time field and a last-supported-state field alone are insufficient unless consumers respond to validity. Likewise, ordinary replay still needs start/interior/tail support checks and unified reporting after the chronology patch. Neither remaining defect should be marked closed by this first package. Q/R, Joseph/coupled update algebra, the 1.5 s first-return refusal, rejection inflation and the 60 s buffer stay fixed.

**Verification and sequencing.** Implement as four small commits/hunks with the above converted regressions; keep original diagnostic probes/results as immutable baseline evidence. Run the new node/encoder tests plus `test_outage_motion_replay.py`, `test_planner_node_state_correction.py`, `test_runtime_transactions.py`, `test_planner_node_per_camera_correction.py`, `test_belief_correction.py` and `test_commissioning_joseph.py`. Retain the independent complete trace and read-only planning/publication invariant. Evaluate the actual launch descriptions using the existing `wiring.py`; these tests must import the installed paths that resolve back to edited sources.

No simulator restart or live-parameter change is needed for implementation/testing. A deployment must be assigned to a separate runtime version, with the experiment owner deciding when to start it; it must not change the identity of a running/frozen experiment. Coordinator should give this package exclusive ownership of `_odom_cb`, the outage snapshot seam, the EKF bootstrap block and encoder input chronology, then sequence audit 02's epoch/publication changes and audit 10's ledger schema on top. Recheck source hashes at implementation time and preserve unrelated publication/command work.
