# Command correctness repair package — remaining work

Prepared for the audit coordinator after rechecking current source. Proposal only; no shared runtime edits made.

Current hashes: EFE `a575ba53e74a835ee955625b14e5718c09939fc36d206ef8b35b4a0c405818ae`; base planner node `1191b8879b8515d7d2bd176015538cacd0bb9ed196af49da80d6cd8705bd8995`. The base node has changed since audit 03's original snapshot. EFE still contains the already-repaired fatal publication latch, negative tape-age stop and idle diagnostics. Do not implement those again.

## A. Ordinary-stop cancellation

Two installers remain: hierarchical LOCAL at `efe_agent_node.py:958` and direct solver `_after_plan_result` at `:1329` (installation `:1361`). Both can install work originating before an ordinary safe stop. `_publish_safe_stop_command:1371` clears current ownership but has no generation counter.

1. Add a command-stop generation, owned by `_data_lock`. Every explicit safe stop increments it atomically with tape clearing/zero publication; preserve the existing fatal latch.
2. Capture an immutable request context before the planning callback snapshots goal/belief or does expensive prediction: stop generation and ROS computation-start time. Both execution paths must use the same mechanism. One EFE-only implementation is a small `_plan_once` wrapper around the existing body renamed `_plan_once_impl`, with context cleared in `finally`; the existing mutually exclusive planning group makes that scoped context single-writer. Direct `super()._plan_once()` remains inside this wrapper. Explicitly passing the context to installation helpers is preferable where practical.
3. At the final locked install boundary, compare the request generation. A cancelled result is discarded, including its rejection/stop side effects. It must neither publish nonzero nor clear a newer replacement. A genuinely new request captured after an ordinary stop may resume; fatal remains latched.
4. Keep the comparison, tape replacement and immediate publication in one transaction. Do not treat `_result_safe_to_execute` alone as the ownership barrier; a stop can occur after it returns.

Minimal regressions, parameterized over LOCAL/direct solver: stop during computation; stop after acceptance but before lock acquisition; old cancelled return after a fresh replacement (replacement retained); new request after ordinary stop resumes; no-stop request still installs. Retain existing fatal, old-timer, old-expiry and idle-diagnostic tests. Use event/barrier-controlled interleavings and fake publishers, not sleeps or live ROS.

## B. Wire the actual computation-age gate

LOCAL assigns `_pending_plan_started_at:918` but bypasses `_result_safe_to_execute`. Direct base `_plan_once` calls the solver without assigning that field. The gate at EFE `:1261` is therefore absent on the real LOCAL installer and inert on the normal direct solver path.

Use the same request context's ROS start time and a final installation-time read for both paths. Check finite/nonnegative age and the existing duration rule (`age > accepted_control_count * max(dt, .001)`), using the LOCAL safe prefix's duration. Preserve the existing strict `>` threshold unless a different expiry policy is explicitly agreed. Gate independently of whether latency compensation is enabled. Missing execution context must not silently disable validation; update test fixtures and any real direct callers to supply it.

Evaluate the final age under the install lock after computational validation, so validation/scheduling time is included. Check generation first: a stale result must not issue an extra stop that erases another owner's tape. For a current-generation over-age result, preserve fail-closed safe-stop behaviour. A backward time jump during computation rejects the request even if no old tape existed for the timer to invalidate; this is not complete filter-epoch reset handling.

Do not automatically reuse newly populated age metadata to activate previously inert latency skipping. Keep age validation and compensation bookkeeping separate. The active flag stays false; changes to the optional compensation path require their own explicit tests and semantics because elapsed time does not establish executed motion.

Minimal regressions over both paths: fresh short tape accepted; delayed one-step tape rejected; equality and just-over-duration boundaries; delay introduced during safety validation; negative age during computation; shortened LOCAL safe prefix determines duration; rejection cannot clear a replacement after generation changes. Assert absence of nonzero publication and valid idle diagnostics, not only a boolean return.

## Ownership and verification

Reserve `src/planning/planning/nodes/efe_agent_node.py` with the runtime/performance owners before edits. With the scoped EFE wrapper, no `unicycle_planner_node.py` change is needed; if common base hooks are chosen instead, reserve that file with the timing/filter owner first. Use a new `tests/planning/test_command_installation.py` to avoid concurrent edits to `test_runtime_transactions.py`; import/reuse fixtures carefully without changing their contracts for other tests.

Run the new installation tests plus existing runtime transactions, watchdog and tracker-guard tests; inspect the source diff to verify cadence, controller and adapter parameters are unchanged. Do not run campaign processes. The frozen audit probe intentionally asserts old behaviour; preserve its snapshot/results rather than rewriting history. Add desired-invariant regressions for the repaired working tree.

Later work: belief/goal revision policy, complete reset epochs, stamped validity/sequence transport, downstream independent watchdog, and fixed-time noise. None belongs in these two patches. Age rejection and cancellation do not prove physical stop delivery or disturbance matching.
