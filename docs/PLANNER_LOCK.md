# Planner lock — 2026-09-14

Every value here was established by measurement on 2026-09-14 and is enforced in
code: `UnicyclePlannerBase.__init__` emits a `RuntimeWarning` naming this file if
any of them is overridden. **A warning means a stale config is in play. Fix the
config, do not silence the warning.**

## The objective

    J(U) = sum_j gamma^(j-1) [ w_risk J_risk + w_amb J_amb + J_nogo ]

gamma = 0.98, horizon 75, dt = 1.0 s, with a smooth arrival gate that zeroes
every term once the predicted mean enters the goal region.

| term | what it is |
|---|---|
| risk | Kouw (IWAI 2024) Eq. 27, Gaussian KL to the goal prior, **clipped on one side** |
| ambiguity | Kouw Lemma 1 under ET1 (Thm 1) on the availability-weighted commissioned covariance |
| no-go | single continuous clearance penalty on the **rectangular body**, heading-aware |

### Risk is clipped on one side
The KL is V-shaped in the predicted covariance, with its minimum where
prediction and goal prior agree. Taken unmodified it PAYS the planner to let the
belief drift whenever the prediction is sharper than the prior — which is the
entire operating range here (belief 2-13 cm against a 0.35 m tolerance; measured
d(risk)/d(sigma) about -1.0 across 2-20 cm). Clipping the covariance argument up
to the prior leaves certainty neither rewarded nor punished; widening past the
tolerance carries the full penalty.

### The goal prior is annealed
Held fixed, the mean term charges (distance/sigma*)^2 per step from step zero:
measured **1633** against an ambiguity of **4** at 20 m from the goal, i.e. the
belief terms decided 0.2% of the step cost. Annealing 5.0 m -> 0.35 m cuts that
first step to **8.0**. Meera, Lanillos & Kouw (arXiv 2608.14466) anneal the
preference variance the same way as their sole exploration control, tau^2
20 -> 0.6.

### Ambiguity is Kouw's ET1 term on a commissioned sensor
Under ET1 the state covariance cancels EXACTLY (verified numerically, residual
~1e-15), so ambiguity reduces to 0.5(Dy log 2*pi*e + log|R_eff|). Kouw's Theorem 1
makes this constant over states for a FIXED sensor — the Koudahl/Kouw/de Vries
(Entropy 2021) collapse. It varies here only because commissioning makes R and q
spatial fields:

    R_eff(p, psi) = ( sum_i q_i(p) R_i(p,psi)^-1  +  1/(Q dt) I )^-1

Measured: blind cell -2.42, covered -3.38, main aisle -5.12 nats. The floor
1/(Q dt) is the motion model, not a chosen constant: an unobserved step is not
infinitely uncertain, the belief grows by exactly one step of process noise.

### The clearance cost uses the robot's body
It previously inflated the belief MEAN by a fixed disc and never saw the extent
or heading, so it read **exactly zero on every candidate of every task**. It now
uses the heading-aware support distance of the rectangle against the nearest
lane normal:

    h(theta) = a|cos theta| + b|sin theta| + body_margin

0.450 m driving aligned, **0.535 m through the diagonal** — the heading a robot
holds while turning. Lane rectangles are axis-aligned, so the nearest normal is a
coordinate axis and the expression stays differentiable.

The penalty is ONE continuous function of the clearance deficit in warning-band
units, replacing a hinged-log warning plus a separate quadratic violation that
overlapped below zero clearance and mixed two unrelated scales:

    d = (warning_band - clearance) / warning_band
    penalty = near_weight * ( d^2 + contact_gain * max(d - 1, 0)^2 )

Zero beyond the band, 50 at contact, 272 one centimetre past it. Both pieces are
quadratic, so it is continuous and C1 at contact.

## The values

| parameter | was | now | why |
|---|---|---|---|
| `nogo_safe_distance` | 0.55-0.585 | **0.325** | half-width 0.275 + 0.05 lane-keeping. The old value left the 1.10 m lanes a NEGATIVE budget, infeasible before any uncertainty. |
| `nogo_logbarrier_eps` | 1e-3 | **0.05** | = `warning_band`. At 1e-3 a 1 mm violation cost 2,036, more than the entire risk term. |
| `use_belief_nogo_cost` | false | **true** | the covariance -> clearance channel. |
| `network_goal_std_m` | 0.15 | **0.35** | the declared `goal_success_radius`. At 0.15 risk is ~5x stronger and the objective collapses to shortest path. |
| `network_goal_std_start_m` | none | **5.0** | anneal start; see above. |
| `kouw_et1_ambiguity` | false | **true** | the thesis method. |
| `process_noise_xy` / `_theta` | 0.01 / 0.02 | **0.02 / 0.08** | conservatively bounds the simulated drift; see PROCESS_NOISE_LOCK.md. |

`camera_network_objective: metric_expected_belief` must be set PER CAMPAIGN — it
is the only objective that loads the per-arm planner fields. With
`global_planner_mode: preselected_route` the planner does no solve and never
queries a camera field, which is why the paused campaign's four arms were
identical.

## Route seeds land on their waypoints

At the global layer one step is v_max*dt = 1 m. The seeder used to switch target
within one step of a corner (turning a metre early, so the swept body left the
lane) and to pivot in place above a heading gate (which a 0.80 x 0.55 m body
cannot do in a 1.10-1.30 m aisle). Every lane-graph seed was rejected, the solver
fell back to a nominal control vector, and it converged 22.6 m from the goal.

The seed now rotates OR advances, never both, never overshooting, landing on each
waypoint exactly. Seed feasibility on the blind-corridor task: **0/6 -> 6/6**,
each reaching the goal to machine precision. The free CasADi solve then CONVERGES
(`optimizer_success: true`, goal distance 1.8e-15) in 131-159 s per condition,
where every earlier attempt hit the iteration cap.

## Two accounting bugs — do not reintroduce

1. **Fixed-horizon tail.** The objective summed a fixed 75 steps regardless of
   route duration, so a short route banked its remaining steps parked at the goal
   and accumulated that cell's ambiguity. Symptom: a constant-parameter arm gave
   an IDENTICAL total on every route. Fixed by the arrival gate.
2. **Summed per-step terms encode route LENGTH.** Always inspect per-step values
   along a route, never only totals. A constant-R arm must give a constant
   per-step ambiguity once discount-normalised.

## Verification that must pass before trusting any result

Hand calculation, the NumPy evaluator and the CasADi objective must agree on the
ambiguity term to ~1e-6 at a fixed pose, and the full-rollout total must
reproduce by hand to ~1e-3.

## Status

- Candidate scoring: the commissioned model changes route choice on **2 of 4**
  contrast tasks, paying 2.3 m and 9.2 m to avoid unobserved driving. The flips
  are carried by the clearance term, NOT by ambiguity — state it that way.
- The rejection is independently verified: at the rejected routes' worst poses
  the real rectangular body leaves the DRIVEABLE LANE at 1-2 sigma, checked by
  the footprint validator. Declared collision prisms are NOT intersected, so the
  mechanism is lane departure, not a proven rack strike. Do not overstate it.
- Free solve on the blind-corridor task: converges, both conditions select the
  same route. Free solves of the two flipping tasks are pending.
- No closed-loop Gazebo evidence for any repaired configuration.

## optimizer_control_block_steps = 1 (LOCKED 2026-09-15)

Controls are optimised one step at a time. Anything above 1 averages each block
of consecutive controls and repeats the average
(`compress_controls`/`expand_controls` in `base_planner.py`).

**Why it must be 1.** The lane-graph seeder emits alternating pure turns
`[0, omega]` and pure drives `[v, 0]`. Averaging a turn step with a drive step
destroys both, so a seed that lands on the goal exactly no longer does. Measured
on the two corner-to-corner tasks, terminal miss of the best seed:

    block_steps   west_to_east_north   east_to_west_south
    1                   0.000 m              0.000 m
    2                   0.398 m              1.035 m
    3                   1.669 m              2.127 m

The terminal gate is 0.35 m, so at 2 every seed fails it. With NO candidate
passing the gate the lexicographic rule falls through to raw cost, where a
do-nothing plan wins: standing still cost 8,093 against 86,258 for the route
that reaches the goal, because the clearance term is charged per step and a
parked robot drives past nothing. The planner then commands zero velocity for
the whole run. This is what happened in the first Gazebo pilot.

**This was never a free parameter.** `block_steps = 2` came from a solve-time
experiment and was never derived or recorded here. It is not a speed/quality
trade-off: at 1 the same four tasks solve in 46-122 s, FASTER than the broken
setting, because the solver starts from a seed that already satisfies the gate.

**Status after the fix.** All 4 of 4 tasks reach the goal (0.000, 0.000, 0.000,
1.9e-15 m). Three report `optimizer_success: false`: the L-BFGS-B polish does not
converge, but the SELECTED plan is a seed route that reaches the goal and passes
the hard terminal gate. Do not read that flag as a failed plan; read the selected
source and the terminal distance.

## Goal arrival is a HARD CONSTRAINT, not a cost term

`optimizer_terminal_goal_tolerance_m = 0.35` with the lexicographic rule in
`_prefer_candidate`: safety first, then terminal-goal feasibility, then EFE.
A plan that does not arrive can never beat one that does, whatever its EFE.
EFE therefore ranks only plans that already complete the task, which is the
claim the thesis makes. The mechanism was already correct and needed no change
-- the 2026-09-14 failure was the seed being destroyed before reaching it.

## Campaign timeouts, sized from measurement (2026-09-15)

`run_visibility_campaign.py` defaults: `--first-cmd-timeout 480`,
`--run-timeout 900`. The previous 270/420 pair was sized against a "contended
tail to ~220 s" that measurement falsifies.

Global solve wall time with `block_steps = 1`:

    8 uncontended solves    median  96 s, worst 238 s
    1 solve sharing the machine with a live campaign   304 s

304 s exceeds the old 270 s first-command deadline, so a slow solve was killed
mid-optimization with no command and scored `infra_invalid`. That is the failure
mode the deadline exists to avoid, and it was doing the opposite.

The deadline must exceed the worst CONTENDED solve, not the uncontended one: a
live run shares the machine with Gazebo, five camera streams and YOLO. Re-measure
before lowering either cap. At the new caps 80 runs take about 4.5 h at the
typical rate.
