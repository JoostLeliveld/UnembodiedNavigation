# First state-estimation repair package — implemented

Implements [the plan](01_repair_package_a_plan.md) against the working tree on 2026-09-06.
All four hunks are in place with converted regressions. This closes R02 and the active
mandatory-envelope branch of R03, and fixes the chronological/atomic parts of R01/R06.

**It does not close** missing-motion validity, wrong-time heading, epoch reset or the
recovery-policy findings. The encoder's skipped-gap pose still has no explicit
unknown-motion state, and ordinary replay still lacks start/interior/tail support
reporting. Neither is marked closed here.

## Source hashes after the repair

| file | before | after |
|---|---|---|
| `src/planning/planning/nodes/unicycle_planner_node.py` | `c56d51b4…` | `8aae15c4…` |
| `src/planning/planning/core/motion_history.py` | `4e2f71eb…` | `ac5ea439…` |
| `src/sim/sim/encoder_noise_node.py` | `e6a04fff…` | `f48cbb9d…` |

The planner node had already moved from the plan's `1191b887…` baseline; only
`_publish_plan_and_metrics`, `_publish_planner_diagnostics` and `_publish_planner_status_text`
differed, exactly as the plan predicted, and those edits were preserved.

## What changed

**1. R02 — one locked read feeds both the check and the replay.** `MotionHistorySnapshot`
(frozen dataclass, both buffers plus the source flag) is captured inside the same
`_data_lock` block as the outage anchor, and passed to `_predict_belief_to_now` through a
new keyword-only `motion_snapshot`. Callers that supply nothing still take their own
snapshot, so no call site changes behaviour. Both lists are frozen because prediction can
fall back from odometry to commands.

**2. R03 — the compatibility pose cannot bootstrap under the envelope contract.**
`_resolve_state_belief_ekf` no longer calls `_apply_state_correction(state_ref)` when
`require_state_correction_envelope` is set. The explicitly configured legacy non-envelope
fallback is unchanged.

**3. R01 — odometry is validated, ordered, then committed atomically.** `_odom_cb` parses
into locals, refuses malformed or non-finite input without inventing a receipt-time stamp,
and commits yaw, velocity, origin, history and a private nanosecond watermark together
under one lock. **Late input is refused, not inserted in order:** a retrospective insert
would alter a history an accepted correction was already computed from, without rewinding
and replaying that correction. Refusing old input does not establish that the remaining
history has complete temporal support. Duplicate and conflicting equal-stamp messages add
no motion and no process noise; the first event at a timestamp is retained, and
`_odom_refused_{old,duplicate,invalid}` expose the disposition for diagnosis only.

**4. R06 — old encoder input moves neither the anchor nor the RNG.** `_odom_cb` validates
and computes `dt` before mutating anything. A `dt <= 0` message returns having touched no
pose, covariance, slip state, scale Jacobian, timestamp or random-generator state. The
`dt > max_dt_s` branch **still rebases the interval baseline deliberately** — holding the
old baseline forever would make every later interval exceed the cap and freeze the encoder
permanently. That rebase is an interval baseline reset, not evidence the pose is supported
across the gap; the omitted motion remains an open validity defect.

## Regressions added

`tests/planning/test_outage_motion_replay.py` (+5): the coverage/replay snapshot race with
a callback that really does trim the verified turn; fallback entries and source choice both
frozen; late odometry refused (9.4, 9.7, 9.5 → history 9.4, 9.7 and x=0.12 over the real
0.6 s, not 0.8 s); equal-stamp conflict retains the first event; a malformed message does
not poison the watermark.

`tests/planning/test_planner_node_state_correction.py` (+5): the anonymous pose does not
initialize under the envelope contract; the envelope itself bootstraps with exactly one
`accepted_bootstrap` event carrying batch id and capture time; reverse delivery order gives
the same state; an invalid envelope plus a usable pose still does not initialize; legacy
non-envelope bootstrap still works.

`tests/sim/test_encoder_input_chronology.py` (new, 5): out-of-order gives 0.3 rad not 0.4;
an inserted old message and a duplicate each change *no* node state at all (pose,
covariance, both slip states, Jacobian, anchor, publication count compared against a clean
stream); a malformed message leaves everything untouched and the stream resumes; the
large-gap rebase and resumption are preserved, explicitly labelled as a compatibility
assertion rather than a correctness claim.

## Verification

Full suite: **1015 passed, 4 skipped**, up from 999 before the package.
`tests/test_no_machine_local_paths.py` fails, and **fails identically on the clean tree** —
it is pre-existing (audit artifacts and `/tmp` matplotlib config paths), unrelated to this
package.

The frozen probe `experiments/estimator_consistency/module_audit_01_20260906/probe.py`
intentionally asserts the OLD behaviour, so it now fails where the defects were fixed. Its
snapshot and `results.json` are preserved as immutable baseline evidence rather than
rewritten. Re-running it through its own `node()` fixture confirms the repair: the
interfering callback still trims the buffer to 0.3 s, but heading is now **0.2 rad instead
of the recorded 0.0**, and the encoder gives **0.3 rad instead of 0.4**.

Two probe fixtures build nodes with `object.__new__`, so they lack attributes `__init__`
sets. The new callbacks read `_odom_accepted_stamp_ns`, `_last_stamp_ns`,
`_odom_origin_stamp_s` and the refusal counters through `getattr` defaults so a partially
constructed fixture cannot make the callback throw. Production behaviour is unaffected.

No simulator restart, live-parameter change or campaign run was involved. Deployment must
be assigned to a separate runtime version and must not change the identity of a running or
frozen experiment.
