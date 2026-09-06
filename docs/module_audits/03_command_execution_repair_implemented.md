# Command correctness repair package — implemented

Implements [the repair package](03_command_execution_repair_package.md) (audit 03
findings C03-01 and C03-05) against the working tree on 2026-09-06. Both parts A
(ordinary-stop cancellation) and B (the computation-age gate) are in place with
regressions. The already-repaired fatal publication latch, negative tape-age stop
and idle diagnostics were left alone, as the package instructed.

Source hash after the repair: `src/planning/planning/nodes/efe_agent_node.py`
→ see the commit; the file also carries unrelated pre-existing uncommitted work
(the `ControlSafetyResult` refactor), which this package did not touch.

## A. Ordinary-stop cancellation

`_command_stop_generation` is owned by `_data_lock` and incremented by every
explicit safe stop, atomically with clearing the tape and publishing zero. The
fatal latch is independent and stays latched.

`_plan_once` is now a thin wrapper that captures an immutable request context —
stop generation plus ROS computation-start time — **before** the planning callback
snapshots goal/belief or does expensive prediction, and clears it in `finally`.
The hierarchical body is `_plan_once_impl`; the non-hierarchical route still calls
`super()._plan_once()` from inside the wrapper, so both execution paths share one
mechanism. The existing mutually exclusive planning group makes this scoped
context single-writer.

A cancelled result is discarded **entirely**, including its rejection and stop side
effects: it must neither publish nonzero nor clear a newer replacement tape. A
request captured *after* an ordinary stop is genuinely new and resumes normally.

## B. The computation-age gate, actually wired

`_plan_request_expired` applies the existing duration rule with the strict `>`
preserved (`age > steps * max(dt, 1e-3)`), evaluated **under the install lock** so
safety-validation and scheduling time count. It measures the tape that actually
installs — the LOCAL **safe prefix**, not the solver's untruncated output. It runs
independently of whether latency compensation is enabled, and a missing execution
context returns `missing_request_context` rather than silently disabling
validation. Non-finite and negative ages are refused; a backward clock jump during
computation rejects the request even when no old tape existed for the timer to
invalidate. That is deliberate and is **not** complete filter-epoch reset handling.

Newly populated age metadata is **not** reused to activate the previously inert
latency-skipping path. The active flag stays false and the skip counters stay zero;
elapsed time does not establish executed motion.

## One shared installation boundary

The package described the same transaction in two places. Keeping two copies is how
the age gate ended up assigned on the LOCAL route (`_pending_plan_started_at`) and
read on neither, so both routes now call `_install_control_tape`: generation
comparison → expiry → tape replacement → immediate publication, in one lock, with
the generation checked first so a stale result can never issue the fail-closed stop
over another owner's tape. Logging happens outside the lock and cannot break a stop.

## Regressions

`tests/planning/test_command_installation.py` (new, 21), a separate file to avoid
concurrent edits to `test_runtime_transactions.py`. Event-controlled interleavings
and fake publishers; no sleeps, no live ROS. They assert absence of nonzero
publication and the tape's identity, not only a boolean return.

Cancellation: stop during computation; stop injected between `_result_safe_to_execute`
returning and the install lock; a cancelled result cannot erase a replacement; a
request captured after an ordinary stop resumes; fatal still latches; no-stop
request installs; the wrapper clears its context when planning raises.

Age gate, over both routes: fresh short tape accepted; delayed one-step tape
rejected and failing closed; the equality and just-over boundaries (0.5 vs 0.5001 s
on a two-step 0.25 s tape); delay introduced during safety validation; negative age
from a backward clock jump; missing context; the shortened LOCAL safe prefix
determining the duration where the untruncated tape would have hidden the delay;
and a stale over-age result that must cancel rather than stop.

**Mutation-checked.** Neutralising the generation comparison fails 6 of the 21;
neutralising the expiry comparison fails a different 6. Both mechanisms are
genuinely exercised rather than incidentally satisfied.

## Verification

Full suite **1036 passed, 4 skipped** (1015 before this package). The required set —
new installation tests plus `test_runtime_transactions.py`, `test_command_watchdog.py`
and `test_tracker_guard.py` — is green (27 passed). No campaign processes were run.

Diff inspection confirms cadence, controller and adapter parameters are unchanged:
the only `dt` references added are reads inside the duration rule; no parameter
declaration, timer or publisher configuration was modified.

`tests/test_no_machine_local_paths.py` fails, and fails identically on the clean
tree — pre-existing and unrelated.

## Explicitly out of scope, still open

Belief/goal revision policy (C03-04), complete reset epochs, stamped
validity/sequence transport, a downstream independent watchdog, and fixed-time
noise. Age rejection and cancellation **do not prove physical stop delivery** or
disturbance matching (C03-08, T09-04).
