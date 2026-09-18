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

### The ambiguity is ANCHORED (2026-09-16)

Those measured values are NEGATIVE, and that is the whole problem. Written out,

    J_amb = 0.5 ( D_y log 2*pi*e + log|R_eff| )

is an ABSOLUTE differential entropy. It carries a large route-independent
additive constant, and the SIGN of that constant depends only on the units
R_eff is written in -- here state units, because of the 1/(Q dt) I floor, which
is what makes it negative.

On a fixed horizon the constant cancels: every candidate accumulates it the same
number of times. But the smooth arrival gate makes duration a FREE VARIABLE, so
it instead contributes `constant * T`. With the constant negative, a longer route
is paid for being longer, on identical q and R. That is a duration term wearing
the sign of an arbitrary unit choice, and it swamps the q/R differences that are
supposed to decide the route.

Note the diagnostic in "Two accounting bugs" below PASSES under this defect: a
constant-R arm does give a constant per-step ambiguity. A constant per-step value
is exactly what the bug produces. The problem is what that constant multiplies.

The term is now measured against a fixed FLOOR:

    J_amb = 0.5 * max( log( |R_eff(q,R)| / |R_floor| ), 0 )

    R_floor = (1.5 mm)^2 I,  ONE covariance shared by every arm

**R_floor is a threshold, not an estimate.** It enters only as a constant
subtraction, so ANY value below the tightest reachable R_eff produces identical
route rankings -- verified: a floor 100x tighter reproduces the same spread and
the same (zero) clip count. The only thing that matters is that it stays BELOW
the operating range. A floor inside that range clips real poses to zero and
deletes the very signal the term carries.

It is chosen inside a two-sided window:

- **upper** the tightest R_eff reachable on any of the seven arms is det
  2.39e-11 (~2.2 mm sd). Above that, the floor clips real poses.
- **lower** `casadi_efe._logdet_small_pd` clamps det at 1e-12. At or below that
  clamp the CasADi log-determinant stops matching the NumPy one (measured
  8.3e-6 per evaluation, compounding across opportunities) and the two
  back-ends silently disagree.

1.5 mm sits 5x above the clamp and 4.7x below the tightest reachable pose. The
window is narrow only because the commissioned cameras are very precise. If a
future arm is tighter still, widen the clamp rather than lowering the floor, and
re-check both bounds.

**One floor for ALL arms.** Anchoring each arm to its own commissioned best-R
was the first attempt and it is WRONG: the per-arm floors span 3.5 nats across
the current seven arms, so each arm would be shifted by a different constant and
the arms would no longer be comparable -- the one thing a q/R comparison must
not do.

Properties, in the order they matter:

- **constant, NOT zero, for a constant-parameter arm.** q = 1 means every camera
  reports, not that the pose is perfectly localized; the resulting finite R_eff
  is real residual uncertainty and the objective should still charge for it.
  What makes such an arm pick the shortest safe route is that the SAME amount is
  added to every candidate at every step. Measured on W0 (q = 1 and constant R):
  per-step ambiguity 1.6679-1.6700 nats, spread 2.2e-3.
- **never negative**, so no route can be rewarded for lasting longer;
- **dimensionless** -- it is a ratio, which kills the m^2 vs px^2 scale problem
  flagged in the `expected_posterior_uncertainty_ca` docstring;
- differs from Kouw Lemma 1 by an ADDITIVE CONSTANT only, so on a fixed horizon
  it is provably rank-identical to the published objective. The running MPC's
  optimum is unchanged; only the free-duration global route decision moves.

Both back-ends carry it and must stay numerically equal:
`casadi_efe._anchored_ambiguity_ca` (the live frozen path under
`camera_network_objective: metric_expected_belief` with `kouw_et1_ambiguity
true`) and the NumPy twin in `base_planner._evaluate_controls`.

The MIXTURE path (`expected_posterior_uncertainty_ca`, E[H(P+)]) needed the same
treatment and is anchored to the IDEAL-AVAILABILITY POSTERIOR reached from the
same prior -- not to the prior. Anchoring to the prior gives -q I(x;y), which is
<= 0 and reintroduces the duration reward with the opposite sign. Its NumPy twin
`CameraNetwork.expected_belief` previously returned the raw absolute entropy
while its own docstring claimed prior-differencing; both now do the same
anchored thing.

**Why a constant-parameter arm is not exactly flat.** R_i is stored per heading
and position, and in the q = 1 arms the per-camera R has constant eigenvalues but
ROTATING orientation, so fusing five differently oriented precisions gives an
R_eff whose logdet still varies slightly with pose. Measured on W0: 2.2e-3 nats
across the whole field. That is a property of the commissioned geometry, not of
the accounting. Re-measure with `verify_anchored_ambiguity.py`, which reports the
spread, the clip count (must be 0) and the floor margin.

**q is still necessary.** q is availability (probability a camera returns an
admitted measurement); R is reliability GIVEN a report. They enter as
`sum_i q_i R_i^-1`, so q decides whether precision arrives at all. R does inflate
in low-availability regions -- measured ~1.3x between q < 0.05 and q > 0.5 cells
-- but q itself swings by 100x, so R alone cannot stand in for it.

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
3. **An ABSOLUTE entropy is a duration term once duration is free.** (2026-09-16)
   The arrival gate fixed bug 1 and created this one: with the horizon no longer
   fixed, the route-independent constant inside `0.5(D_y log 2*pi*e + log|R|)`
   became `constant * T`, signed by the units R is written in. Diagnostic 2 does
   NOT catch it -- a constant per-step value is precisely the symptom. Any term
   summed over a free-duration rollout must be a RATIO or a difference against a
   fixed reference, never an absolute entropy. See "The ambiguity is ANCHORED".

## Verification that must pass before trusting any result

Hand calculation, the NumPy evaluator and the CasADi objective must agree on the
ambiguity term to ~1e-6 at a fixed pose, and the full-rollout total must
reproduce by hand to ~1e-3.

## The legacy golden was regenerated for the one-sided risk clip (2026-09-18)

`tests/planning/test_flag_off_is_bit_identical_to_golden` pins the LEGACY
precision-blend path (`casadi_efe.make_efe_valgrad_fn`) bit-for-bit. It had been
failing since `85d94a1b`, the commit that made `risk_ca` one-sided. That commit
changed the frozen path deliberately but did not regenerate the golden, so the
test was red for a benign reason -- which is the worst state for a regression
test, because it masks a real break on that path.

**The cause was established before rewriting anything, not assumed.** Forcing
`one_sided=False` reproduces the OLD golden on all four cases with ZERO
mismatches, so the old numbers are exactly the pre-clip method and nothing else
had drifted into them. Every recorded change is a DECREASE (the clip removes the
reward for letting the belief drift), and evaluation point 1 is unchanged in all
four cases because there the belief is already wider than the goal prior and the
clip is inactive -- the signature the clip should have, and a check worth
repeating if this is ever regenerated again.

    case                  point   old         new         delta
    et1_vis               0       14.897593   13.315536   -1.582056
    et1_vis               1      309.257577  309.257577    0.000000
    et1_vis               2      287.140708  286.293118   -0.847590
    et1_novis             0       13.291797   11.605726   -1.686071

Regenerated with `experiments/efe_hit_miss_mixture/regenerate_golden.py --write`,
which had itself been deleted in `66d34f4d` and was restored from
`66d34f4d^` for this. It imports the harness from the test file, so generator
and test cannot drift apart. The rewrite touched 144 hex literals and no logic or
comment, and a second run produced no further change -- the golden is a fixed
point of the current method.

**This does not affect the live planner.** The campaign runs
`make_metric_network_efe_valgrad_fn` (`base_planner.py:1410`) under
`camera_network_objective: metric_expected_belief`; the regenerated golden guards
`make_efe_valgrad_fn` (`base_planner.py:1431`), the legacy path. The standing
rule in the generator's docstring still holds: if this test fails, the default
assumption is that you broke the frozen path -- revert. Regenerate only when the
change was deliberate and the cause has been demonstrated, as above.

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

## The objective weights are NOT derived (open, 2026-09-16)

`nogo_safe_distance`, `nogo_logbarrier_eps`, `network_goal_std_m` and
`optimizer_control_block_steps` are locked and enforced. The four weights that
set the trade-off between the EFE terms are not:

    ambiguity_weight   3.0    no derivation; entered under a commit "Launch updates"
    risk_weight_obs    1.0    no derivation
    nogo_weight       40.0    no derivation
    control_weight     0.0    (term disabled; this one is a decision, not a number)

**This decides every route.** Gates 4-5 on the seven-arm route-selection config,
2 tasks x 4 arms, anchored against legacy ambiguity:

    w_amb    1.0    2.0    3.0    5.0
    flips    2/8    5/8    7/8    8/8

Every one of the seven flips at the deployed weight breaks even between
w_amb 0.91 and 2.85, and four of them within a factor 2 of 3.0. A weight chosen
anywhere in that band selects different routes.

**Do not fix this by scanning w_amb against the route split.** That is fitting a
constant to the outcome it is judged by. The defensible options are to derive the
weight from the units the terms are in (risk and ambiguity are both in nats, so
the relative weight should be 1 unless there is a stated reason), or to report
the route choice as a function of w_amb and show over what band the conclusion
holds.

Re-measure with `score_anchored_route_contrast.py` then
`analyse_anchored_route_contrast.py --w-amb <W>`.
