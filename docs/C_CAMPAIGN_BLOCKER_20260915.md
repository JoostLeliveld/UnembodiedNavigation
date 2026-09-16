# The four-arm navigation campaign cannot run as configured

Written 2026-09-15 06:00 after the campaign failed on its first run and a
follow-up timing probe reproduced the cause three times.

## Symptom

`final_navigation_campaign_v6_locked.yaml` run 1 (C00, T1, seed 91500) ended
`infra_invalid / no_first_cmd_timeout`: no nonzero velocity command in 480 s.

Raising `--first-cmd-timeout` to 1500 s does **not** fix it. It produces a
different failure with the same root cause, and a longer deadline makes the
cause worse rather than better.

## Cause

The hierarchical EFE global solve blocks the belief pipeline for its whole
duration. While it runs, the `efe_agent` process saturates the machine, the
camera manager assimilates no corrections, and the belief goes stale. When the
solve finishes the local tracker has a valid plan and a pose estimate several
hundred seconds old, and it refuses to drive on it. That refusal is correct.

Measured on the timing probe
`logs/studies/reference_controlled_commissioning_v1/solve_timing_probe_20260915_045514`,
one run per arm with the deadline raised to 1500 s so nothing is guillotined:

| arm | last fresh belief | run end | stale gap | nonzero commands | outcome |
|---|---|---|---|---|---|
| C00 | 162.8 s | 167.6 s | **4.8 s** | 443 | goal_reached |
| C10 | 162.6 s | 236.8 s | 74.2 s | 0 | (killed at deadline) |
| C01 | 164.0 s | 615.9 s | 451.9 s | 0 | infra_invalid |

Belief stops updating at sim t = 162.6-164.0 s in **every** run. That is when the
global solve starts. The freeze is universal; only its consequence varies.

C00 did not survive the freeze. It finished 4.8 s after it, before staleness
mattered. Its success is timing, not health.

In the C01 run at sim t = 500 s: `state_available = 1.0`, `state_fresh = 0.0`,
`state_age_s = 336.4`, belief frozen at `state_stamp = 163.6`, and
`operational_belief_timeout_s` is 1.5 s.

## Solve time is not the variable

Three clean global solves: **409.6 s, 444.6 s, 448.9 s** (all
`nit=0, nfev=21`, all selecting the same route as the offline probe). A fourth
measurement of 574.0 s is discarded: it ran against orphaned processes left when
this session killed an earlier campaign with SIGTERM on the parent instead of
SIGINT on the runner.

The spread across clean runs is under 10 %. Both the run that reached the goal
and the run that stalled solved in about 445 s. Solve duration does not separate
success from failure; the length of the post-solve staleness gap does.

Offline, the identical solve takes 86.7 s. The 5x inflation is live-system
contention, not a different computation.

## Why the U campaign was unaffected

The 60-run frozen-route campaign
(`three_uncertainty_frozen_route_campaign_v2.yaml`) reached the goal 60 times out
of 60. It uses `global_planner_mode: preselected_route`, so there is no global
solve, so belief never freezes. That is direct corroboration: same simulator,
same perception stack, same machine, no solve, no failures.

## What this does not affect

The offline route-selection results stand. Selection is deterministic given the
planner field and seed, the live run chose the same corridor as the offline
probe (`above_connector`), and the +2.40 m result on
`thesis09_lane08E_to_lane12W` reproduced identically across all five seeds.

## Directions, none chosen

Each of these changes what the campaign measures, so none was applied:

- reuse the planner cache across seeds, so only the first run of a cell pays the
  solve
- time-box the global solve and accept the incumbent plan
- let the belief predict through the solve instead of going stale, which weakens
  the localization claim the campaign is making
- solve before the simulator clock starts, so the solve costs no sim time
- run the solve on a machine that is not also running the simulator

The measured facts above are what a choice should be made against.
