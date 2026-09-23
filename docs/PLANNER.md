# Planner lock — amended 2026-09-20

The objective form and locked constants are enforced in code:
`UnicyclePlannerBase.__init__` emits a `RuntimeWarning` naming this file if
any of them is overridden. **A warning means a stale config is in play. Fix the
config, do not silence the warning.**

Numerical route contrasts from the superseded availability/covariance factorial
are historical diagnostics, not current thesis evidence. The final Stage-09
design is exactly six matched conditions: M0/R0, M1/R1 and M2/R2, each under an
intact five-camera roster and a task-relevant declared camera removal. The
blind-corridor task removes camera C; the lane-08-to-lane-12 task removes camera
B. Planning and runtime fusion must use the matched
member of each pair; no separate availability model or stale/updated-removal arm
exists in the canonical design.

## The objective

    W(U) = sum_j gamma^(j-1) g_j

    J(U) = sum_j gamma^(j-1) g_j
           [ J_risk + J_amb + J_control + J_nogo ] / W(U)

gamma = 0.995, horizon 75, dt = 1.0 s, with a smooth arrival gate `g_j` that
zeroes every term once the predicted mean enters the goal region. The same
active discounted weight normalizes every running component. This is the locked
IWAI-like fixed-horizon comparison: a spatially constant per-step term is equal
between routes instead of becoming an accidental duration penalty.

| term | what it is |
|---|---|
| risk | Kouw (IWAI 2024) Eq. 27, Gaussian KL to the goal prior, **clipped on one side** |
| ambiguity | anchored ET1 term on direct matched-covariance precision |
| no-go | single continuous clearance penalty on the **rectangular body**, heading-aware |

### Risk is clipped on one side
The KL is V-shaped in the predicted covariance, with its minimum where
prediction and goal prior agree. Taken unmodified it PAYS the planner to let the
belief drift whenever the prediction is sharper than the prior — which is the
entire operating range here (belief 2-13 cm against a 0.35 m tolerance; measured
d(risk)/d(sigma) about -1.0 across 2-20 cm). Clipping the covariance argument up
to the prior leaves certainty neither rewarded nor punished; widening past the
tolerance carries the full penalty.

Risk is a running term and is normalized by `W(U)` with the other stage costs.
`terminal_risk_only` is forbidden in the locked method. The effective risk
multiplier is exactly `risk_weight_obs * observation_risk_scale = 1 * 1`.

### The goal prior is annealed
Held fixed, the mean term charges (distance/sigma*)^2 per step from step zero:
measured **1633** against an ambiguity of **4** at 20 m from the goal, i.e. the
belief terms decided 0.2% of the step cost. Annealing the mean-preference width
from 5.0 m to 0.10 m prevents the distant-goal term from dominating early steps
while ending at the locked localization preference. Meera, Lanillos & Kouw
(arXiv 2608.14466) anneal the
preference variance the same way as their sole exploration control, tau^2
20 -> 0.6.

The locked tightening power is **0.9**.

### Ambiguity is Kouw's ET1 term on the camera network
Under ET1 the observation term can be represented through the total predicted
camera information. The current method exports one matched-covariance precision
grid per camera. The diagnostic effective covariance is

    R_eff(p) = ( sum_i Lambda_i^plan(p) + epsilon_amb I )^-1

    epsilon_amb = 1.0 m^-2

where the sum contains only active planning cameras. Each field is the direct inverse of
the matched runtime covariance queried for that camera and position, then rotated into the
world frame. No second model is fitted and opportunity outcomes do not enter the export.
The actual belief rollout uses the predicted prior covariance and adds the summed
field in information form. `epsilon_amb` is a fixed, shared information
regularizer that keeps the ambiguity-only inverse finite when camera information
is zero. It gives `R_eff = 1.0 m^2 I` in that limit. It is not learned, does not
depend on process noise, and never enters belief propagation.

### The ambiguity is ANCHORED (2026-09-16)

Those measured values are NEGATIVE, and that is the whole problem. Written out,

    J_amb = 0.5 ( D_y log 2*pi*e + log|R_eff| )

is an ABSOLUTE differential entropy. It carries a large route-independent
additive constant, and the SIGN of that constant depends only on the units in
which R_eff is written.

On a fixed horizon the constant cancels: every candidate accumulates it the same
number of times. But the smooth arrival gate makes duration a FREE VARIABLE, so
it instead contributes `constant * T`. With the constant negative, a longer route
is paid for being longer, on identical information fields. That is a duration term wearing
the sign of an arbitrary unit choice, and it swamps the field differences that are
supposed to decide the route.

The fixed information regularizer above solves only the singular inverse at zero
camera information. It does not remove this unit-dependent entropy constant;
anchoring is therefore still required.

The term is now measured against a fixed FLOOR:

    J_amb = 0.5 * max( log( |R_eff(Lambda)| / |R_floor| ), 0 )

    R_floor = (1.5 mm)^2 I,  ONE covariance shared by every condition

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

**One floor for ALL conditions.** Anchoring each condition to its own best
effective covariance shifts the conditions by different constants and makes them
incomparable.

Properties, in the order they matter:

- **constant, not zero, for a spatially constant finite-information field.** A
  camera report is not perfect localization; finite expected information leaves
  finite posterior uncertainty. A constant field contributes the same amount at
  equal priors.
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

The direct-information NumPy and CasADi paths must remain numerically equal. The
historical Bernoulli-mixture path is compatibility code and is not part of the
active method.

**No separate q field is active.** Detector opportunities, misses, admission refusals and
NIS outcomes do not alter planner precision. Weak residual support lowers M2 precision
through the broad covariance prior. Runtime admission still decides whether a realized
measurement is available to the estimator.

### The clearance cost uses the robot's body
It previously inflated the belief MEAN by a fixed disc and never saw the extent
or heading, so it read **exactly zero on every candidate of every task**. It now
uses the heading-aware support distance of the rectangle against the nearest
lane normal:

    h(theta) = a|cos theta| + b|sin theta|

For the 0.80 x 0.55 m body this is 0.275 m against a lateral wall when driving
aligned, 0.400 m when facing the wall, and at most 0.485 m through a diagonal
turn. Lane rectangles are axis-aligned, so the nearest face normal is a
coordinate axis and the expression stays differentiable.

The penalty is ONE continuous function of the **oriented rectangular body's**
clearance deficit in warning-band units, replacing a hinged-log warning plus a
separate quadratic violation that overlapped below zero clearance and mixed two
unrelated scales. There is no additional body margin: the cost is exactly zero
at body-to-region clearance >= 0.05 m and starts inside that distance:

    d = (warning_band - clearance) / warning_band
    penalty = near_weight * ( d^2 + contact_gain * max(d - 1, 0)^2 )

Zero beyond the band, 50 at contact, 272 one centimetre past it. Both pieces are
quadratic, so it is continuous and C1 at contact.

## The values

| parameter | was | now | why |
|---|---|---|---|
| `nogo_safe_distance` | 0.55-0.585 | **0.0 for the thesis pipeline** | Shape-aware paths use the oriented 0.80 x 0.55 m body and the map's explicit 0.10 m geometric inset. Adding a centre-distance radius would count clearance twice. |
| `nogo_warning_band` | 0.05 | **0.05 m from the body** | zero at >=5 cm body clearance; rises only inside the band. |
| `nogo_logbarrier_eps` | 1e-3 | **0.05** | = `warning_band`. At 1e-3 a 1 mm violation cost 2,036, more than the entire risk term. |
| `use_belief_nogo_cost` | false | **true** | the covariance -> clearance channel. |
| `network_goal_std_m` | 0.35 | **0.10** | preferred position standard deviation used by EFE risk. This is deliberately separate from the 0.35 m terminal arrival tolerance. |
| `network_goal_std_start_m` | none | **5.0** | anneal start; see above. |
| `kouw_et1_ambiguity` | false | **true** | the thesis method. |
| `process_noise_xy` / `_theta` | 0.01 / 0.02 | **0.02 / 0.08** | conservatively bounds the simulated drift; see PROCESS_NOISE.md. |

`camera_network_objective: metric_expected_belief` must be set PER CAMPAIGN — it
is the objective that loads the per-camera planner fields. Canonical Stage-09 uses
`global_planner_mode: efe`; `preselected_route` is retained only for replaying
superseded sealed campaigns and is not a current navigation arm.

## Route seeds land on their waypoints

At the global layer one step is v_max*dt = 1 m. The seeder used to switch target
within one step of a corner (turning a metre early, so the swept body left the
lane) and to pivot in place above a heading gate (which a 0.80 x 0.55 m body
cannot do in a 1.10-1.30 m aisle). Every lane-graph seed was rejected, the solver
fell back to a nominal control vector, and it converged 22.6 m from the goal.

The seed now rotates OR advances, never both, never overshooting, landing on each
waypoint exactly. Before optimization, seeds are repaired against the same
oriented 0.80 x 0.55 m footprint used by the no-go test. Candidate routes that
reverse over a collinear segment are rejected. When a feasible cross-aisle lies
between the start and goal ordinates, cross-aisles beyond both endpoints are not
offered as away-from-goal detours. The final two-task offline matrix reaches the
goal to numerical precision in all six conditions per task and every retained
rollout passes the geometric validity checks. `optimizer_success` is diagnostic:
a valid seed remains admissible when L-BFGS-B reaches its iteration limit or
reports a line-search failure without improving it.

## Two accounting bugs — do not reintroduce

1. **Fixed-horizon tail.** The objective summed a fixed 75 steps regardless of
   route duration, so a short route banked its remaining steps parked at the goal
   and accumulated that cell's ambiguity. Symptom: a constant field gave
   an IDENTICAL total on every route. Fixed by the arrival gate.
2. **Summed per-step terms encode route LENGTH.** Always inspect per-step values
   along a route, never only totals. The locked objective divides every running
   term by the same active discounted weight. A constant information field
   therefore gives the same ambiguity contribution on every route.
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

- Canonical seven-task development screening evaluated every individual camera
  removal with the spatial planner. The final campaign retains two distinct
  single-camera interventions that change the spatial route: camera C on the
  blind-corridor task and camera B on the lane-08-to-lane-12 task.
- On the blind-corridor task, M2 changes from `above_connector` when intact to
  `above_cross_aisle` after removing C. On the lane task, M2 changes from
  `below_main_aisle` to `below_south_cross_aisle` after removing B. M0 and M1
  retain `below_main_aisle` across the B-removal lane comparison.
- These route choices are development selection evidence, not closed-loop
  navigation performance. Routes must be regenerated under the final campaign
  hash before Gazebo execution.
- No canonical six-condition closed-loop Gazebo evidence exists yet.

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
trade-off: at 1 the evaluated tasks solve in 46-122 s, FASTER than the broken
setting, because the solver starts from a seed that already satisfies the gate.

**Status after the fix.** Every condition in the final two-task offline matrix
reaches the goal to numerical precision. Some report `optimizer_success: false`:
the L-BFGS-B polish can reach its iteration limit or stop its line search, but the
selected plan remains a valid seed or solver rollout that reaches the goal and
passes the hard terminal gate. Do not read that flag as a failed plan; read the
selected source, `rollout_valid`, and terminal distance together.

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

## The ambiguity weight is now DERIVED (2026-09-18)

`ambiguity_weight` is **1.0**, set in `PAPER_LAUNCH_DEFAULTS`. The derivation is
the units: risk (Kouw Eq. 27, a Gaussian KL) and ambiguity (Kouw Lemma 1 under
ET1, now an anchored log-ratio) are BOTH in nats, so their relative weight is 1
unless there is a stated reason otherwise. There is none.

The previous 3.0 had no derivation, entered under a commit called "Launch
updates", and route choice is a direct function of it -- an undeclared tuning
knob deciding the result. It is also NOT set by any campaign config, so the code
default governs every run; changing it here changes every arm.

**What moves at 1.0 rather than 3.0**, measured on the four thesis tasks x seven
arms by re-decomposing the scored candidates (`total - 3.0*amb + w*amb`, exact):
13 of 28 cells select a different route, and the number of TASKS on which the
seven arms disagree goes 1/4 -> 2/4. Two of the new splits are near-ties (0.27%
and 0.28% margins) and should not be reported as effects.

**Do not re-tune this against the route split.** That is fitting the constant to
the outcome it is judged by. If 1.0 is ever changed, the reason must be a stated
property of the objective, not a better-looking result. Re-measure with
`score_anchored_route_contrast.py` then `analyse_anchored_route_contrast.py
--w-amb <W>`.

`nogo_weight` is deliberately NOT reset: the clearance term is not in nats, so
the units argument does not reach it, and its penalty shape (0 beyond the band,
50 at contact, 272 one cm past) was derived against the deployed value. Note the
fallback in `visibility_launch_common.py` is 2000.0 with the comment "At 40 the
clearance term cannot compete", while every campaign config sets 40.0 -- the
campaigns govern, and that disagreement is unresolved.

`control_weight` stays 0.0; the term is disabled, which is a decision rather
than a number.

## The objective weights were NOT derived (historical, 2026-09-16)

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
