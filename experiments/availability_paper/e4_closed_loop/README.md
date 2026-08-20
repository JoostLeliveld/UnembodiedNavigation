# E4 — matched closed-loop campaign

**STATUS: STOPPED AFTER 12/45 PLANNED RUNS.** `GATE-ONLINE-EFE` passed on
2026-08-18. The matched campaign then produced 12 plan-bearing runs (C1: 5, C2: 4,
C3: 3); a thirteenth C3 attempt was interrupted before a plan was persisted. Every
saved `mc_blind_L/global_plan.csv` contained the same 76 coordinates, with maximum
pointwise deviation 0.0 m. Continuing the campaign could therefore compare
stochastic driving on one common route, but could not test whether
availability-aware planning changes route choice.

The runtime-objective audit reproduces this outcome on all three tasks: every field
selects the same seed class. Availability reaches the objective, but its ambiguity
term is dominated by risk and obstacle costs. The campaign was stopped rather than
spending the remaining 33 runs on a contrast the frozen planner did not express.
Because the preregistered 45-run matched set is incomplete, these runs support no
comparative navigation-performance claim.

Evidence: `logs/studies/availability_paper/e5_offline_efe_solve/`,
`logs/studies/availability_paper/e6_offline_map_plan_audit/`, and figures 06, 08,
11, and 12.

## Before launching any redesigned successor

```bash
pgrep -a "ros2 launch|ign gazebo|run_visibility_campaign"
```

**Non-empty means do not launch.** Campaigns run for hours, and a second Gazebo
collides with the live one on the same ROS topics and gz partition, corrupting both.
Campaigns are often started from a different session, so "I didn't start one" is not
evidence that none is running. To inspect a live run, read its CSVs.

## Original order of operations (preserved)

1. **Gate first, one seed.** Run condition C1 only, on `mc_blind_L`, one seed. Then:

   ```bash
   python3 experiments/availability_paper/e0_online_efe_readiness/check_gate.py \
       --run 'logs/visibility_comparison/<campaign>/mc_blind_L/C1/seed0/experiment_*' \
       --goal <goal_x> <goal_y>
   ```

   The gate checks six things, and the sixth is the one the pilot silently failed
   before it got stuck: that the executed path actually enters ground where the arms
   differ (minimum fused availability ≤ 0.2). A run that arrives while staying in
   well-observed ground proves the stack works and still cannot discriminate the
   observation models.

2. **On pass**, promote `EXP-AVAIL-CL` to READY in `research/registry.yaml`, then run
   the 45-run campaign in `campaign.yaml`.

3. **On fail**, fix the planner or controller and re-run the single seed. Do not widen
   the clearance, do not relax `optimizer_terminal_goal_tolerance_m`, and do not
   substitute pre-solved waypoints — the waypoint-replay protocol was considered and
   explicitly rejected in favour of full online global EFE.

## The design, and why each part is fixed

Three arms differing **only** in the planning observation model. `campaign.yaml` is
copied key-for-key from the 2026-08-15 pilot except the task list, the route seeds and
`ros_domain_id_base`, so any outcome difference is attributable to the model.

| Arm | Planning observation model |
|---|---|
| C1 | availability-blind, constant `R` |
| C2 | availability-aware, folded `R/p` |
| C3 | availability-aware, explicit Bernoulli hit/miss mixture |

Route seeds come from `logs/studies/availability_paper/e3_route_discrimination/e3_selected_routes.json`.
The planner still solves online; the seeds only replace a cold-start route search,
which is what failed in the pilot.

**Task roles were fixed from E3's measurements, not from intuition:**

- `mc_blind_L` — **primary**. The blind route has minimum fused availability 0.011 and
  a 5.6 s unobserved stretch; an availability-aware route reaches 0.444 for +0.32 m.
- `mc_m2_w2e_traverse` — **negative control**. Minimum availability 0.00012 for every
  candidate; there is nothing to avoid. If C2 or C3 beats C1 here, the campaign has
  found a confound, not an effect.
- `route_tall_shadow_west_safe_start` — pilot continuity.

**Primary endpoint: longest continuous interval with no accepted camera correction.**
Not terminal σ. E3 measured terminal σ moving 0.6 mm across all arms while the longest
unobserved stretch halved — the belief re-converges after a blackspot, so terminal σ
would report a null by construction.

## What a null will and will not mean

The C2-vs-C3 contrast is expected to be small, and that expectation is registered in
advance rather than discovered afterwards: the runtime precision blend is linear in
precision, so it already carries the availability-weighted mean information. Registry
C3 is falsified if the two agree within 10%. A null there is a bounded null about mean
information — the explicit mixture still needs no `R_miss` at all, which dissolves the
unreconciled 40 px offline vs 120 px runtime miss endpoint that
`reliability.covariance_mapping.MissEndpointPolicy` refuses to bless.

The C1-vs-C2 contrast is the headline. A null there bounds the effect in this world,
with these four cameras and this detector. It is not proof of no effect, and per
registry C4 a no-difference result is recorded as a bounded null.

## Pilot post-mortem, kept so it is not repeated

All three conditions selected `solver:warm_or_cold`, produced a near-stationary global
plan, executed about one metre, ended roughly 12.9 m from the goal and tripped the 20 s
stuck detector. The local controller reported
`driveable_clearance_violation_step_0` and safe-stopped. In the corrected diagnostic
run all three optimizer candidates were properly marked `goal_feasible=False` and the
planner still selected the least-bad infeasible one.

No belief, dropout, NEES or success number from that pilot is evidence about the
observation models.
