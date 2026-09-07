# 03 — Remaining command and GLOBAL admission repairs

2026-09-07. Implementation follow-up to [the ranked audit](03_command_execution.md). This records software-method verification on a concurrently edited working tree, not physical actuator acceptance or a navigation result. The original audit and frozen defective-behaviour reproductions remain historical evidence.

## Scope and ownership

This tranche changes `src/planning/planning/nodes/efe_agent_node.py` and adds `tests/planning/test_global_route_installation.py`. Existing stop-generation/common-installation repairs and other owners' tracker/footprint edits were preserved. Investigation 01 owns the authoritative belief API, 08 owns solver-result validation, 09 owns mission envelopes and route/controller policy, and 14 owns downstream actuation protection. No process was launched or terminated for an experiment; no configuration, Q/R, camera artifact, noise cadence, or controller tuning was changed. No commit was made.

## Implemented boundaries

- **Request identity and stop precedence:** capture stop generation, exact goal/frame/mission identity, configuration identity, clock start and originating supported belief. A stop, fatal latch, goal/configuration change or belief epoch change cancels admission. Solver exceptions stop only the still-current request. A delayed rejection cannot clear a replacement. Diagnostic failures cannot prevent the stop command.
- **Atomic mission consumption:** consume investigation 09's `/mission/goal_state` envelope. Stable resends preserve identity; older tour positions, retired epochs and delayed active resends after completion cannot resurrect motion. Once the envelope owns goals, legacy pose messages cannot compete. Small goal changes revoke the tape immediately.
- **GLOBAL admission:** all EFE, geometric and preselected paths use one installation boundary. Solver output must pass investigation 08's complete-result validator with its original state/covariance and GLOBAL timestep. Empty, malformed, partial or inconsistent results fail closed. Removed unchecked straight-route exception/empty-seed fallback and implicit goal appending. Registered preselected coordinates are retained exactly when accepted.
- **Correction during solving:** a same-epoch newer correction is allowed only after fresh supported-belief revalidation. Validate the entire installed polyline, including the current-pose entry connector, turns and intervening segments, with the configured swept rectangular geometry. Commit under correction/data ownership and check the fresh revision again. No LOCAL tape-duration threshold is applied to GLOBAL planning: a slow result must pass fresh geometry and support checks.
- **Tape admission and continuation:** finite bounded controls and a fresh safe prefix are required before installation. Active tapes revalidate supported corrected belief and remaining controls at publication. Truncation preserves the original start time, so a correction cannot renew expiry. Expiry and stop ownership are decided under the same lock. Non-finite outgoing commands become zero and revoke the tape.

Source entry points: `efe_agent_node.py` methods `_capture_plan_request`, `_mission_goal_cb`, `_global_solver_result_safe`, `_global_route_candidate_safe`, `_fresh_request_belief`, `_install_global_route`, `_install_control_tape`, `_plan_once`, `_publish_command`, and `_publish_active_plan_command`. The main ranked audit retains lifecycle file/line references; method names here remain stable under concurrent edits.

## Deterministic reproduction and verification

The new tests use fake clocks/publishers, explicit solver barriers and callback interleavings, and one real supported BeliefRecord correction integration. Initial new cases produced **8 failures and 1 pass before repair**. Cases cover stop/goal/epoch/reset during solving; safe versus unsafe corrections; fatal stop; stale expiry versus replacement; malformed/partial solver results; unsupported belief; valid and invalid geometric/preselected routes; unsafe segments between clear endpoints; mission replay; exception paths; and diagnostic failures.

**Focused scope: 79 passed in 0.89s.** Python compilation and scoped whitespace checks passed.

Latest expanded check: **90 passed, 3 failed** across the dedicated installation, command installation, preselected route, tracker guard, nonrenewable penetration, watchdog and runtime transaction files. The unresolved tests are:

1. `test_belief_header_is_the_prediction_target_even_if_computation_takes_time`: fixture prediction callback lacks the new `motion_snapshot` keyword.
2. `test_prediction_from_superseded_anchor_is_not_published`: same fixture/API mismatch.
3. `test_metric_corrections_cannot_compute_from_the_same_prior_concurrently`: correction test does not enter its instrumented replay barrier with the concurrent belief implementation.

These shared investigation-01 integration failures were reported to the coordinator; their production files and shared tests were not changed here. Earlier broad planning checkpoint was **309 passed, 13 failed, 3 skipped** while other owners were integrating. This is not a full-suite green claim.

Reproduce the focused scope:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests/planning/test_global_route_installation.py tests/planning/test_command_installation.py tests/planning/test_preselected_route.py tests/planning/test_tracker_guard.py tests/planning/test_no_renewable_penetration.py tests/sim/test_command_watchdog.py
```

Add `tests/planning/test_runtime_transactions.py` for the expanded check. These tests do not exercise DDS scheduling or physical response.

## Remaining limits and separate changes

Request/belief/mission identity and fresh-correction revalidation are explicit admission-policy changes beyond the earlier mechanical stop-generation repair. They require a new frozen runtime before comparing experimental runs. GLOBAL geometry validation establishes a feasible candidate polyline; it does not prove the controller will follow it exactly. Controller performance remains investigation 09.

Commands are still unstamped Twist. A live receipt watchdog bounds silence since receipt, conditionally on its clock/scheduling/output path; it cannot establish original command age. Delayed queued nonzero messages, competing publishers and adapter/bridge death require downstream integration evidence and/or a stamped epoch/expiry protocol. Physically applied wheel commands are not established by planner publication or adapter output logs. Logging/event attribution remains investigation 10.

Immediate plus periodic publications still advance correlated actuation noise per message. The original deterministic duplicate/rate comparisons remain applicable: equal seeds do not imply equal time-indexed disturbances. A fixed simulated-time disturbance grid, command acknowledgement/age protocol, and acceleration/deadband modelling remain separate changes. This tranche does not claim to repair them or retune filter Q.

## Source checkpoint

SHA-256 captured at report creation; shared sources may subsequently change:

- `src/planning/planning/nodes/efe_agent_node.py`: `6aa2779b14b6135702b336c2c6f76fd8fd3eafa94a99f5cf70931dc7eecd74f3`
- `tests/planning/test_global_route_installation.py`: `345d514c735cd7613b97520e29263fd91bd77368d6de7da69f0d7f78f3cb1045`
- `src/planning/planning/nodes/unicycle_planner_node.py`: `1f37d544143ec62f6ac5209c08b19f390ed335f844f43b0a6f46deae7fc460cf`
- `src/planning/planning/core/plan_validation.py`: `00cf28b8ea2e167073f160c875717c5faf4721dfff4d19f6bb63ff24f800b332`

### Line references at checkpoint

- [_capture_plan_request](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:624)
- [_mission_goal_cb](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:657)
- [_global_solver_result_safe](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:721)
- [_global_route_candidate_safe](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:736)
- [_fresh_request_belief](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:784)
- [_install_global_route](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:817)
- [_install_control_tape](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:858)
- [_plan_once](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:964)
- [_publish_command](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:1509)
- [_publish_active_plan_command](/home/joostleliveld/Thesis/UnembodiedNavigation/src/planning/planning/nodes/efe_agent_node.py:1612)
