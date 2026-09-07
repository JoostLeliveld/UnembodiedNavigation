# GLOBAL installation — remaining correctness boundary

2026-09-07. Report-only coordination package; no shared runtime edits.

Rechecked EFE SHA-256 `f3750f4a4dcb0969de7bd114b696f1dd536d1a10d47f14b80c53a755fae4bd45`, matching investigation 08's [current checks](08_current_result_checks.json). Its 19 groups distinguish observed behaviour from desired invariants. The ordinary-stop/computation-age command repair is already present; do not duplicate or replace it. [Its implementation account](03_command_execution_repair_implemented.md) belongs to the implementing chat, not this audit.

## Problem and scope

`_plan_once` captures request generation/start, but GLOBAL success at `efe_agent_node.py:972` directly mutates waypoints, phase, goal and warm start without consulting that context. Exception fallback at `:961`, geometric mode at `:853–897` and preselected acceptance at `:784` have additional route-installation paths. Only LOCAL/direct command tapes use `_install_control_tape`.

Investigation 08 confirms GLOBAL partial, invalid and NaN routes install, and GLOBAL results originating before stop or clock changes still install. These do not immediately publish a nonzero command; they become inputs to a later LOCAL computation with a fresh command request. Therefore passing direct-command cancellation does not establish cancellation of the route that drives the next request.

## Minimal implementation sequence

1. **Make candidate construction side-effect free.** Construct waypoints, phase transition, route goal and optional warm-start seed in local variables. Do not assign `_waypoints`, `_wp_idx`, `_hier_phase`, `_global_goal_xy` or `_global_solve_done` until admission. Do not publish/save a candidate as the accepted route before admission. Diagnostic rejected candidates may be retained only with explicit rejection labels.
2. **Validate the producer contract.** Require finite correctly shaped states and the validity fields promised by the EFE producer. Do not use permissive missing-field defaults for solver results. Reject invalid rollout before extracting waypoints. Geometric/preselected routes require their own typed validator, not fake solver fields.
3. **Validate the route actually handed to LOCAL.** An appended terminal connector is not covered by validation of the solver's shorter path. Either require the solver endpoint to satisfy the existing configured goal tolerance and reject incomplete candidates, or validate the entire appended connector using the same route geometry contract. Investigation 08 owns producer validity; 09 owns endpoint/route policy. Do not silently fall back to an unchecked straight route on rejection or exception.
4. **One atomic route-install transaction.** Under `_data_lock`, require a present request, unchanged stop generation, nonfatal state and valid clock epoch/nonnegative computation age. Commit the complete route record only if all checks pass. Cancelled returns have no clearing/stop side effects on a newer route or tape. Keep existing command-tape transaction intact.
5. **Specify positive GLOBAL age separately.** The LOCAL rule `control_count * self.dt` must not be applied to a global route whose horizon/time step differ and whose controls are not executed. Negative-age and stop-generation refusal are immediate correctness repairs. Positive maximum route-computation age requires an explicit global validity contract (or revalidation against fresh supported state); capture/log actual age now and have 08/09 choose that bound before claiming arbitrarily slow GLOBAL results are safe. A new threshold is a declared policy, not implicit reuse of LOCAL duration.
6. **Cover every route-producing branch.** Pass EFE success, global-replan fallback, geometric route and preselected route through request admission. Preserve preselected provenance/geometry rules. A stopped/cancelled replan must not use exception fallback to bypass admission. No runtime/API behaviour should allow missing context to mean current request.

Belief, goal and configuration revisions remain an additional design: record and revalidate an immutable route request against those revisions, with a declared correction policy. The stop/clock repair must not claim that it settles these revisions. Check the old-epoch filter-state defect from investigation 02: starting a new request after reset does not prove its belief is new-epoch.

## Minimal acceptance tests

Use actual `_plan_once`, fake solver, controlled ROS clocks and barriers as in 08's existing probe, converting confirmed failures into desired invariants:

- Valid complete fresh result installs once; later LOCAL can execute normally.
- NaN/wrong-shape/missing-required-validity/explicit invalid results never enter LOCAL, update warm start or publish an accepted route.
- Partial endpoint either rejects or passes an explicit full-connector validator; unsafe appended connector cannot install.
- Stop during solve and stop between validation/commit discard the old route; a newer replacement route/tape remains unchanged.
- Backward clock during solve rejects, even with no prior tape for the command timer to invalidate.
- Missing request context fails closed; genuinely fresh post-stop request can install.
- Exception fallback, geometric mode and preselected acceptance obey the same ownership checks.
- Positive global age boundary tests follow the explicitly chosen GLOBAL policy, not LOCAL dt.
- Retain all current direct/LOCAL command tests, fatal/negative-tape-age/idle diagnostics, and controller checks.

Inspection also confirms the expiry-side-effect window flagged by 02: `_install_control_tape` determines expiry under lock, releases it, logs, then calls safe stop. A replacement may install in between. Keep this with 02/current transaction owner; require either the stop to complete in the same ownership transaction or a recheck before its side effects. GLOBAL admission must not copy that pattern.

## Ownership and coordination status

Reserve EFE with the active refactor/runtime owner before any edit. Let 08 own solver-result/route validation criteria and fixtures; 09 owns completion/connector policy; 02 owns expiry transaction and epoch handling. Prefer a pure route-candidate validator plus a small EFE atomic installation helper, avoiding solver/controller rewrites. Use a dedicated `tests/planning/test_global_route_installation.py` rather than concurrently modifying shared command tests.

No ownership grant is inferred from this package. Outgoing coordinator disclosure was previously blocked pending explicit destination approval; this plan is saved locally for review. Stamped transport, downstream watchdog redesign and fixed-time disturbances stay outside this patch. No campaign or simulator interventions are needed to verify the proposed method-level invariants.
