# Stage-09 rerun: termination and scoring parameters

Written 2026-09-22. Applies to any rerun of the stage-09 navigation campaign.
Values below are the intended configuration, not a record of a completed run.

## Why this changes

In `stage09_final_five_seed_campaign`, the stop criterion and the success
criterion were both 0.35 m. The run stops on the BELIEF distance and is scored
on GROUND TRUTH, so with equal thresholds any terminal belief error in the
wrong direction converts a normal stop into a failure. One spatial-model
dropout run stopped at 0.353 m ground truth with a 0.040 m median belief error
and was scored a failure.

The manuscript now scores success at 0.40 m against a 0.35 m stop, which gives
a 5 cm tolerance for terminal belief error. The rerun replaces both numbers.

## Executed values in the campaign being replaced

Read from `run_manifest.json` of a completed run:

| Parameter | Value |
| --- | --- |
| `goal_success_radius` | 0.35 |
| `goal_success_hold_s` | 2.0 |
| `goal_stable_radius` | 0.2 |
| `goal_stable_hold_s` | 2.0 |
| `goal_stable_max_displacement_m` | 0.04 |
| `run_timeout_after_first_cmd_s` | 300.0 |
| `stuck_window_s` | 8.0 |
| `stuck_max_displacement_m` | 0.08 |
| `stuck_max_goal_improvement_m` | 0.05 |

Outcomes were 55 `goal_reached`, 4 `stuck`, 1 `collision`. No run timed out.
Longest run 43.1 s, median 34.6 s. The 300 s timeout was 7.0x the longest run.
All 55 successes fired the `goal_success_radius` branch; the
`goal_stable_radius` fallback never terminated a run.

## Changes for the rerun

### 1. Split the stop and scoring thresholds

    goal_success_radius: 0.10        # was 0.35, belief frame
    # scoring threshold, applied offline on ground truth
    strict_success_goal_distance_m: 0.30   # was 0.35

Scoring is applied by the analysis step, not by the logger. The logger's
`goal_success_radius` is the stop criterion only.

Keep `goal_success_hold_s: 2.0`.

### 2. Lower the timeout

    run_timeout_after_first_cmd_s: 90.0    # was 300.0

90 s is about 2x the longest observed run and leaves headroom for the slower
routes a 0.10 m stop radius will produce. It bounds the worst case at a cost
no observed run approaches.

Note the campaign runner also carries `first_cmd_timeout_s: 400.0`, which is
the wait for the FIRST command, a different quantity. Leave it.

### 3. Close the dead zone between the two terminators

`_maybe_finish_for_stuck` in
`src/experiments/experiments/nodes/experiment_logger.py` returns early when
the belief is inside `goal_stable_radius`:

    if math.isfinite(goal_dist) and goal_dist <= self.goal_stable_radius:
        return False

A robot whose belief is inside that radius but which never satisfies the hold
is then unreachable by both terminators and runs to the timeout. At a 0.35 m
stop radius nothing landed there. At 0.10 m the robot must hold its belief
inside 10 cm for 2 s, and hovering just inside without stabilizing becomes
plausible.

Add a separate terminator with its own reason code rather than widening the
stuck rule, so the outcome stays distinguishable in the analysis:

    goal_loiter_timeout_s: 15.0

If the belief has been within `goal_success_radius` for longer than
`goal_loiter_timeout_s` without satisfying the hold, finish the run with
reason `goal_loiter_timeout`. Score it as a failure, and count it separately
from `stuck` and from `timeout_after_first_cmd`.

Keep `stuck_window_s: 8.0` and the displacement thresholds. The stuck detector
caught 4/4 cases at 31.4-32.3 s and needs no change.

### 4. Analysis

`strict_success` becomes: no collision, complete evidence ledger,
`campaign_outcome` not in {`stuck`, `goal_loiter_timeout`,
`timeout_after_first_cmd`}, and `final_goal_distance_m` < 0.30.

Report `goal_loiter_timeout` as its own failure mode in the navigation table
if any run hits it.

## What to check after the rerun

- No run should reach `run_timeout_after_first_cmd_s`. If any does, the
  loiter terminator is not firing.
- Compare the count of `goal_loiter_timeout` against zero. A nonzero count on
  the global and per-camera dropout arms is a real result, not a defect.
- 49 of 55 successful runs already ended under 0.30 m ground truth, so the
  0.30 m scoring threshold should not by itself change the intact arms.
