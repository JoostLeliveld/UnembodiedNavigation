# Global planner objective: duration-dependent accounting, diagnosis and correction

Goal of this work: make the global planner drive from A to B along the **shortest
safe and sufficiently observable route**, and make the reason for every route
choice visible in a cost decomposition.

Status of this document: it records the trace, the diagnosis, the correction, the
regression tests and the offline validation. Numbers quoted here come from the
versioned artifacts under `paper_artifacts/planning/global_objective_v2_*/`.
Nothing in the existing runs, logs, routes or manifests was modified.

---

## 1. Trace of the complete global objective

### 1.1 Where the global route decision is actually made

* `src/planning/planning/nodes/efe_agent_node.py::_plan_once` -- the hierarchical
  planner solves the global route **once** (`self._hier_phase == 'GLOBAL'`) and
  freezes it; the result becomes the waypoint list the local tracker follows.
* Candidate route basins come from `unav_common/lane_graph_routes.generate_route_seeds`
  (map-derived, condition-neutral: below the rack block, above it).
* `base_planner.UnicyclePlannerBase.plan` runs L-BFGS-B from each seed, then picks
  the best candidate: unsafe rollouts lose to safe ones, and among equals the
  lower `total_cost` wins (`base_planner.py`, the `keep = ...` block).

### 1.2 The NumPy evaluator

`base_planner.UnicyclePlannerBase._evaluate_controls` is the reference accounting,
and it is what candidate selection compares. Per step `t` of a fixed horizon `H`:

1. propagate the belief: `m, S = predict(m, S, u_t)` (exact integrated `Q_d`);
2. observability: `p_vis = q(m, S)` from the fitted GP, blended into a
   planner-facing measurement covariance by precision blending
   `R_plan = (q/R_visible + (1-q)/R_miss)^-1`;
3. observation transform `ET1`: `mu = g(m)`, `Sigma = J S J' + R_plan`,
   `Gamma = S J'`;
4. **risk** `= risk_weight_obs * observation_risk_scale * KL(N(mu, Sigma) || N(y*, S*_t))`
   with a progress-annealed goal prior `S*_t`;
5. **ambiguity** `= ambiguity_weight * ambiguity_term_scale * 0.5*(d*log(2*pi*e) + logdet(Sigma - Gamma' S^-1 Gamma))`;
6. **no-go** penalty on the (belief-tube) driveable clearance;
7. **control** `= control_weight * |u_t|^2` (`control_weight = 0.0` in the campaign);
8. everything discounted by `gamma^t` and summed.

### 1.3 The CasADi objective

`planning/core/casadi_efe.py::visibility_aware_unicycle_efe_ca` mirrors 1-7
symbolically and is what the optimizer actually minimises. Before this work it
differed from the NumPy evaluator in one respect: it divided the per-step sums by
the effective discounted horizon `H_eff` and the NumPy evaluator did not. At a
fixed horizon that is a constant factor and does not change a ranking, but it
means the two "same" objectives returned different numbers. They are now
identical (test `g`).

### 1.4 What is *not* in the objective

There is **no route-length or travel-time term at all**. `control_weight` is
actuator effort (sum of squares of `[v, w]`) and is set to `0.0`. The only thing
that penalises a slower route is the discounted goal/risk integral.

---

## 2. The defect

### 2.1 Under ET1 the ambiguity is a pure function of q and R

With the first-order extended transform the conditional covariance collapses
exactly:

```
Sigma - Gamma' S^-1 Gamma = (J S J' + R_plan) - J S S^-1 S J' = R_plan
```

so, exactly (verified numerically to 1e-15):

```
A_t = 0.5*d*log(2*pi*e) + 0.5*logdet(R_plan(q_t, R_t))                      (1)
```

The ambiguity term carries **no information about the trajectory** beyond `q` and
`R` at the visited state. That is a useful property -- it makes the observability
term clean -- but it also exposes the problem.

### 2.2 The route-independent constant becomes a duration term

Split (1):

```
A_t = A_ref + 0.5*log( det R_plan(q_t,R_t) / det R_ref )
      \___/   \_________________________________________/
    constant            the part that depends on q and R
```

`A_ref` is the same for every step of every route. In a **fixed-horizon**
comparison (as in IWAI) every candidate accumulates exactly `H` copies of it, so
it cancels and is invisible. In a **full-route** comparison -- candidates with
different arrival times, objective gated after arrival -- it contributes
`A_ref * T`, i.e. a pure duration term. Its sign is not a modelling choice:

```
A_ref < 0  <=>  det(R_ref) < (2*pi*e)^-d
```

which is decided by the units the measurement is written in. Concretely, for the
2-D pixel measurement of this system:

| measurement units | r | `A_ref` | effect of taking longer |
|---|---|---|---|
| pixels | 2.5 px | **+4.67 nats/step** | penalised |
| image widths (same camera, same q, same R) | 2.5/1280 | **-9.64 nats/step** | **rewarded** |

`scripts/planning_validation/minimal_duration_bias_repro.py` runs exactly this:
two safe routes between the same endpoints, identical constant q and R, one 4 m
longer. The legacy accounting picks the short route in pixels and the **long**
route in image widths. Same robot, same physics; the winner flips because a
coordinate was renormalised.

So the answer to "how do the arrival gate and negative ambiguity interact with
route duration" is: **the arrival gate converts a route-independent additive
constant into a linear duration term, and a negative constant makes that term a
reward for taking longer.** In the shipped pixel-scale configuration the constant
happens to be positive, so the defect is latent rather than active there -- but it
still means the planner's de-facto travel cost is an accident of the measurement
units rather than an explicit quantity, which is not defensible.

### 2.3 Three further couplings found in the trace

* **Discounting.** `gamma = 0.995` with an arrival-truncated sum makes later steps
  cheaper, so the extra steps of a longer route are systematically discounted.
  Discounting is appropriate for a receding-horizon controller; it is not
  appropriate for comparing whole routes. The full-route selector uses
  `gamma = 1`.
* **q and R leaking into the goal term.** The risk term used `Sigma = J S J' + R_plan`,
  so poor observability changed the *goal* cost as well as the observability cost
  (`0.5*tr(S*^-1 Sigma)` grows with `R_plan`, `0.5*(logdet S* - logdet Sigma)`
  shrinks). Requirement: q and R must affect the localization part only.
* **No travel baseline.** See 1.4.

### 2.4 What is *not* the fix

Setting `control_weight > 0` would penalise actuator effort, not route length. A
rotate-in-place turn has `v = 0` and would be free; a fast straight leg would be
expensive. It also does not remove the `A_ref * T` term. The diagnosis in 2.2 is
the accounting bug; a control cost does not touch it.

---

## 3. The correction

Implemented in `src/planning/planning/core/localization_cost.py` and used by both
back-ends.

**(a) Anchor the observability term to a fixed reference measurement quality.**

```
c_loc(t) = max( A_t - A_ref , 0 )
         = max( 0.5 * log( det R_plan(q_t, R_t) / det R_ref ), 0 )      [nats]
```

with `R_ref` the best attainable measurement covariance (`r_visible_uv` by
default, overridable via `r_reference_uv`). Properties:

* exactly `0` when `q = 1` and `R = R_ref`, so constant perfect observability
  contributes no route-dependent preference at all;
* `>= 0` always, so no route can ever be rewarded for lasting longer;
* strictly increasing in `det R_plan`, so lower q or worse R strictly increases it;
* invariant to any constant offset in the ambiguity, because the offset appears in
  both `A_t` and `A_ref`.

Where each form is used matters, and is deliberate:

* the **fixed-horizon MPC** uses `c_loc` per step, exactly like the risk term,
  because that objective is a per-step discounted average. At a fixed horizon
  `c_loc` differs from the raw ambiguity by a route-independent constant, so the
  optimum and the gradients are provably unchanged -- the anchoring cannot
  destabilise the running controller (test
  `test_anchoring_shifts_the_fixed_horizon_objective_by_a_constant`, and
  confirmed on the real solves in section 6.6);
* the **full-route selector** integrates `c_loc` over time (nat-seconds), because
  there the candidates have different durations and the objective is in absolute
  seconds. That is exactly where the correction bites.

**(b) Add an explicit travel baseline.** Route length / travel time, not actuator
effort: `route_length_weight` (per metre) and `travel_time_weight` (per second) in
the per-step objective; the full-route selector uses travel time directly.

**(c) Evaluate the goal/risk term at `R_ref`** (`risk_uses_reference_R = true`),
so q and R enter the objective through `c_loc` alone.

**(d) Make the two back-ends numerically identical** -- the NumPy evaluator now
applies the same `1/H_eff` normalisation as the CasADi objective.

The full-route objective is then

```
minimize   travel_time
         + localization_weight * INTEGRAL c_loc(q(t), R(t)) dt
subject to collision-free, inside the driveable region,
           the complete body fits, kinematically feasible, reaches the goal
```

`localization_weight` is the single declared exchange rate: seconds of extra
travel worth avoiding one nat of excess measurement entropy sustained for one
second. It is declared (1.0), never fitted to make a route flip; the validation
reports the break-even weight for every decision so the sensitivity is visible.

### 3.1 Safety is a hard gate, evaluated on the complete body

`src/planning/planning/core/route_safety.py` models the robot as an oriented
rectangle (default 0.80 x 0.55 m) and checks **every pose of the executed
trajectory, including the poses swept during in-place rotations**, against

* the obstacle geometry (no body/obstacle contact),
* the driveable-region union (no departure from the driveable region),

plus kinematic feasibility and goal attainment. Unsafe candidates are rejected
before any objective value is compared, so no weighting can buy an unsafe route.
The safety re-sampling resolution is decoupled from the control tape, so
refining the geometric check can never change a route's travel time or cost.

### 3.2 Backwards compatibility

`localization_cost_mode = 'raw_ambiguity'` plus `risk_uses_reference_R = false`
reproduces the pre-correction objective exactly, so old runs remain reproducible
and old-vs-new comparisons run on identical candidates, geometry, q and R.

---

## 4. Regression tests

`tests/planning/test_global_route_objective.py` (synthetic world, every number
hand-checkable):

| test | guarantee |
|---|---|
| `test_a_*` | constant q + constant R selects the shortest safe route; localization cost is exactly 0 |
| `test_a2_*` | uniformly *degraded* q still selects the shortest safe route |
| `test_b_*`, `test_b2_*` | an unsafe short route is rejected, even when it scores best |
| `test_c_*` | lowering q on part of a route increases that route's localization cost |
| `test_d_*`, `test_d2_*` | worsening R increases localization cost where measurements are available; the rate is anchored and monotone |
| `test_e_*` | a longer observable route wins only above the break-even weight |
| `test_e2_*` | with equal q, no localization weight (up to 1000x) can flip the route |
| `test_f_*`, `test_f2_*` | a constant offset in the ambiguity cannot change the ranking (and the legacy accounting demonstrably does flip) |
| `test_g_*`, `test_g2_*` | NumPy and CasADi objectives and candidate rankings agree |
| `test_footprint_turn_feasibility_*` | an in-place turn is checked on the swept body, not the centre |
| `test_reaching_the_goal_is_a_hard_gate` | a route that stops short is rejected although it costs less |

---

## 5. Offline validation

`scripts/planning_validation/run_global_route_validation.py` writes a new
versioned artifact directory and never overwrites an existing one. It runs:

* **tasks**: the blind-corridor diagnostic `occlusion_transit_a4`, the four
  principal thesis tasks (`route_apron_to_a3_mid`, `route_apron_to_a2_mid`,
  `route_west_to_a1_upper`, `control_west_to_a1_low`) and the sanity crossing;
* **conditions**: q in {1, commissioned GP} x R in {constant/global, commissioned};
* **objectives**: legacy accounting and corrected objective, on identical
  candidates, geometry, q and R;
* **footprints**: the required 0.80 x 0.55 m body and the deployed TurtleBot3
  burger body (0.140 x 0.178 m) that the Gazebo campaign actually drives;
* **gate sets**: `full` (every gate hard, including body-inside-driveable) and
  `deployment` (body-inside-driveable reported, not vetoing).

Results, the full candidate table (safety status, length, minimum body clearance,
q/R localization cost, total cost, selected/not selected), the old-vs-corrected
comparison and the break-even weights are in the artifact directory's
`summary.md`, `candidates.csv`, `selection_comparison.csv` and `breakeven.csv`.

See section 6 for the findings, including the ones that did not go the way we
would have liked.

---

## 6. Findings

Numbers below are from `paper_artifacts/planning/global_objective_v2_20260915b/`
(`candidates.csv`, `selection_comparison.csv`, `breakeven.csv`). The earlier
`..._20260915/` directory is the same grid with a smaller summary and is kept as
evidence.

### 6.1 The validation gates pass

| gate | result |
|---|---|
| q = 1 selects the shortest safe route (every task, both R modes) | **PASS** |
| q = 1 with constant/global R gives exactly zero localization cost | **PASS** |
| localization cost is never negative -- no route is rewarded for lasting longer | **PASS** |
| no unsafe route is ever selected | **PASS** |
| NumPy and CasADi objectives agree (relative error <= 3e-7 over random tapes) | **PASS** |

### 6.2 q and R are cleanly separated

With the deployed footprint, the corrected objective's selections are:

| condition | a3 | a2 | west_upper | blind corridor | control (no-detour) |
|---|---|---|---|---|---|
| q = 1, R constant/global | short | short | short | short | short |
| q = 1, R commissioned | short | short | short | short | short |
| q commissioned, R constant/global | short | short | short | short | short |
| q commissioned, R commissioned | **long** | **long** | **long** | **long** | short |

The third row is the important one: the commissioned q field is active, but
because R does not respond to availability (`r_miss = r_visible`), the
localization cost is identically zero and the shortest safe route wins on every
task. Availability only costs something when it changes the achievable
measurement quality -- which is exactly the intended semantics of q and R.

The control task `control_west_to_a1_low` keeps the short route in **all four**
conditions: the longer route is less observable, not more, so no localization
weight can justify it (break-even = infinity). No spurious detour.

### 6.3 Every detour is paid for, and the price is visible

Decomposition for the commissioned-q / commissioned-R condition, deployed
footprint (travel cost in seconds, localization cost in seconds-equivalent at the
declared weight 1.0):

| task | route | length [m] | T [s] | q mean | q min | loc cost | total | selected |
|---|---|---|---|---|---|---|---|---|
| a3 | below_main_aisle | 7.93 | 18.1 | 0.958 | 0.670 | 0.86 | 18.96 | **yes** |
| a3 | above_connector | 5.05 | 13.4 | 0.420 | 0.000 | 37.02 | 50.42 | no |
| a2 | below_main_aisle | 9.93 | 21.4 | 0.964 | 0.665 | 0.88 | 22.28 | **yes** |
| a2 | above_connector | 7.05 | 16.7 | 0.423 | 0.000 | 40.19 | 56.89 | no |
| west_upper | below_main_aisle | 9.88 | 21.4 | 0.774 | 0.028 | 10.26 | 31.66 | **yes** |
| west_upper | above_cross_aisle | 8.65 | 19.3 | 0.272 | 0.000 | 55.93 | 75.23 | no |
| blind corridor | below_main_aisle | 7.28 | 18.7 | 0.574 | 0.000 | 32.80 | 51.50 | **yes** |
| blind corridor | above_connector | 7.20 | 15.4 | 0.236 | 0.000 | 58.75 | 74.15 | no |
| control | below_main_aisle | 5.63 | 14.4 | 0.975 | 0.833 | 0.39 | 14.79 | **yes** |
| control | above_cross_aisle | 12.90 | 26.4 | 0.293 | 0.000 | 67.07 | 93.47 | no |

Break-even localization weights (the weight at which each detour stops paying
for itself): a3 0.130, a2 0.120, west_upper 0.046, blind corridor 0.127. The
declared weight is 1.0, so every detour decision has roughly an 8x-20x margin;
none of them is a knife-edge produced by weight tuning, and none of the weights
was chosen to produce a flip.

### 6.4 The corrected objective did not change any route -- but it changed the reason

`selection_comparison.csv`: for the deployed footprint under the full gate set,
**legacy and corrected select the same route in all 24 task x condition cases.**
So no previously recorded Gazebo run is invalidated by a changed route choice.

That agreement is not, however, evidence that the legacy objective was sound. In
the commissioned-q / commissioned-R condition the legacy total is 98.5-100% soft
no-go penalty: the legacy accounting reaches the observable route because the
belief tube grows while the robot is blind and then violates the driveable
standoff band, not because it measured a localization benefit. Strip that safety
shaping term and the legacy objective prefers the **short, blind** route on all
four route-choice tasks (see "What actually drove the legacy choice" in
`summary.md`). The corrected objective reaches the same routes through an
explicit, auditable localization cost.

### 6.5 The 0.80 x 0.55 m body does not fit any of these routes

This is the finding that did not go the way we wanted, and it is not something
the objective correction can fix.

With the required 0.80 x 0.55 m footprint, **no candidate route on any task
passes the hard safety gates**:

| task | route | body-vs-obstacle clr [m] | body-in-driveable clr [m] | failing gate(s) |
|---|---|---|---|---|
| blind corridor | below_main_aisle | **-0.080** | -0.100 | obstacle contact + outside driveable |
| blind corridor | above_connector | **-0.080** | -0.100 | obstacle contact + outside driveable |
| a3 / a2 | both | +0.123 / +0.124 | **-0.110** | outside driveable (during the in-place corner turn) |
| west_upper / control | below_main_aisle | +0.185 | **-0.085** | outside driveable |
| west_upper / control | above_cross_aisle | **-0.140** | -0.160 | obstacle contact + outside driveable |
| sanity crossing | below_main_aisle | +0.500 | **-0.185** | outside driveable |

Two distinct problems:

* **Real collisions, both against the north wall** (not the racks):
  * blind corridor, both candidates: -0.080 m **at the goal pose itself**
    (3.10, 4.60) facing north. The task goal is 0.08 m too close to
    `warehouse_walls/wall_north` for a 0.80 m-long body to stand there. No
    driveable-map or route change can fix this one -- the task goal would have to
    move.
  * `route_west_to_a1_upper` / `control_west_to_a1_low`, `above_cross_aisle`:
    -0.140 m at (-5.25, 4.58) while turning into the upper cross-aisle.

  These are physical infeasibilities, not conservative-lane artefacts.
* **In-place turns.** Where there is no collision, the binding constraint is the
  turn: the body sweeps a 0.485 m radius disc when it rotates on the spot, so it
  needs a **0.971 m** turning diameter. Of the 13 declared driveable prisms only
  two are that wide -- `rack_aisle_A4` (1.10 m) and `west_service_lane` (1.05 m).
  The rest are 0.60-0.95 m: the rack aisles A1-A3 are 0.90, the connectors 0.95,
  the lower main aisle 0.82, the upper cross-aisle 0.65, the apron 0.60. On top of
  that, the lane-graph candidates place their corners at the start/goal column
  rather than at the lane centre, which costs a further 0.10-0.18 m. The straight
  legs fit; the corners do not (the "turn" column of `candidates.csv` equals the
  overall minimum for a3/a2/west/control).

With the footprint the Gazebo campaign actually drives (TurtleBot3 burger,
0.140 x 0.178 m -- consistent with the shipped `robot_collision_radius_m = 0.125`)
every candidate passes every gate with 0.19-0.29 m of body clearance to the
driveable boundary and 0.23-0.69 m to obstacles. The corrected-objective
validation above is therefore reported against the deployed footprint, with the
0.80 x 0.55 m result reported alongside rather than quietly dropped.

If the 0.80 x 0.55 m body is the real requirement, three things have to change,
and none of them is an objective change: the candidate generator must place
corners at lane centres rather than at the start/goal column
(`unav_common/lane_graph_routes.py` uses `x_start` / `x_goal` directly); the
declared lanes must be widened to at least the body's 0.971 m turning diameter;
and the blind-corridor and upper-cross-aisle routes must be re-cut away from the
rack geometry. Those are deliberate changes to tasks/candidates/geometry, which
ground rule 6 forbids while an objective correction is under test, so they are
reported here rather than made.

### 6.6 The deployed solve path agrees with the selector, and is unchanged by the anchoring

`scripts/planning_validation/check_global_solve.py` runs the code the robot
actually executes -- `UnicyclePlannerBase.plan`, CasADi + L-BFGS-B, multistarted
from the lane-graph seeds -- with the corrected objective. Evidence:
`deployed_solve_check_corrected.txt` and `deployed_solve_check_raw_ambiguity.txt`
in the artifact directory.

* **The winning route seed is identical in all 10 task x condition solves**,
  whether the observability term is the anchored excess or the raw ambiguity.
  That is the fixed-horizon constant-shift property (section 3) confirmed on the
  real solver, not just in a unit test.
* **The winning seed matches the full-route selector's choice in every case**:
  `above_connector` / `above_cross_aisle` under q = 1, `below_main_aisle` under
  commissioned q and R, `below_main_aisle` on the control task under both. The
  offline selector is a faithful predictor of the deployed solve.
* The `loc` column shows the correction directly: raw ambiguity sits at ~4.67
  per step (the constant) plus a small excess; the anchored term is 0.000-0.424
  and is *exactly* 0.000 in every q = 1 condition.

Two honest caveats, both pre-existing and present under **both** objectives:

* `occlusion_transit_a4` under commissioned q/R does not converge within
  `optimizer_maxiter = 60`: the predicted terminal distance is 3.67 m (corrected)
  vs 2.69 m (raw ambiguity), and the cold start beats both route seeds. The
  blind-corridor global solve is not reliable at this horizon/iteration budget.
  This is a solver-budget problem, not an objective problem, and it is not a
  regression.
* Four other solves stop 0.30-0.38 m short of the 0.25 m success radius
  (raw ambiguity: 0.25-0.37 m). The runtime appends the mission goal as a final
  waypoint, so the tracker closes that gap, but it is worth noting that the
  global plan does not itself terminate inside the success radius on those tasks.

---

## 7. Which Gazebo runs must be repeated

* **On route-choice grounds: none.** Legacy and corrected select the same route in
  all 24 task x condition cases at the deployed footprint (section 6.4).
* **On cost-reporting grounds: all of them, for any figure that quotes the EFE
  decomposition.** `risk_cost` and `ambiguity_cost` mean different things now
  (risk is evaluated at `R_ref`; the observability term is the anchored excess
  entropy; both back-ends are normalised by `H_eff`). Any plot or table of the
  cost split must be regenerated; trajectory, timing and localization-error
  metrics are unaffected by the objective change.
* **Sanity re-run recommended, not required**, for the four principal tasks plus
  the blind corridor at one seed, to confirm the runtime path behaves as the
  offline solve predicts.
* **The blind-corridor task (`occlusion_transit_a4`) should not be re-run under
  commissioned q/R until its global solve converges** (section 6.6): the frozen
  route is sound, but the deployed solve reaches it only 3.67 m short of the
  goal, so a run would exercise the tracker's straight-line fallback rather than
  the planned route. Raising `optimizer_maxiter` (currently 60, hit on 9 of 10
  solves) is the first thing to try; that is a solver-budget change, not an
  objective change, and was deliberately not made while the objective correction
  was under test.

## 8. Provisional campaign status

`scripts/planning_validation/freeze_corrected_routes.py` writes
`frozen_routes.json` into the validation artifact directory: per campaign arm and
task, the selected route, its waypoints, its safety verdict, its cost
decomposition and the planner overrides to reproduce it.

**The one-seed provisional campaign has not been run.** This environment has no
ROS 2, no Gazebo and no GPU (`rclpy`, `/opt/ros`, `gz` and `nvidia-smi` are all
absent), so the simulator cannot be launched from here. The frozen routes and the
exact campaign command are in `frozen_routes.json`; the campaign must be started
on a machine with the simulator stack.

## 9. Limitations, stated plainly

* **No spatially varying R model exists in this stack.** `R` is the single scalar
  `r_visible_uv` (quality when an observation is available) and `r_miss_uv`
  (quality when it is not); all spatial variation enters through `q`. The
  "constant/global R vs commissioned R" axis is therefore implemented as
  `r_miss = r_visible` vs the commissioned `(2.5, 40)` px pair. The corrected
  localization cost already accepts a spatially varying `R_plan`, so a future
  commissioned R model plugs in without touching the objective.
* **q is clipped to `[1e-4, 1-1e-4]`**, so when `r_miss > r_visible` the
  localization cost of a "perfectly observable" route is O(1e-3) s rather than
  exactly 0. It is positive on both routes and larger on the longer one, so it
  cannot reward duration; the exact-zero guarantee holds when R does not depend
  on availability.
* **The full-route selector uses a turn-then-go reference execution model**, not
  the closed-loop tracker. It is identical for every candidate and condition, so
  it is a fair basis for comparison, but the absolute travel times are idealised
  (no tracking error, no acceleration limits).
* **The driveable-body clearance is evaluated on a 0.05 m footprint sample grid**,
  so reported clearances are accurate to about that resolution.
* **`goal_risk_weight` and `obstacle_weight` are 0 in the corrected full-route
  total**, because goal attainment and safety are hard gates there. Both terms are
  still computed and reported per candidate.

