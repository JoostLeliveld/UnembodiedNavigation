# Planner lock — 2026-09-14

Every value here was established by measurement on 2026-09-14. The defaults in
code now match. **A run launched with the old defaults gets a broken planner**,
so if you see any of the "was" values in a config, that config is stale.

## The objective

    J(U) = sum_j gamma^(j-1) [ w_risk J_risk + w_amb J_amb + J_nogo ]

with gamma = 0.98, horizon 75, dt = 1.0 s, and a smooth arrival gate that stops
all four terms once the predicted mean enters the goal region.

| term | what it is |
|---|---|
| risk | Kouw (IWAI 2024) Eq. 27, Gaussian KL to the goal prior. Unmodified. |
| ambiguity | Kouw Lemma 1 under ET1 (Thm 1): `0.5(Dy log 2*pi*e + log\|R_eff\|)`, with `R_eff = (sum_i q_i R_i^-1 + 1/(Q dt) I)^-1`. Constant over states for a fixed sensor; state-dependent here ONLY because commissioning makes R and q fields. |
| no-go | hinged-log warning band plus quadratic violation on the belief tube. The covariance -> clearance channel. |

## The values, and why

| parameter | was | now | why |
|---|---|---|---|
| `nogo_safe_distance` | 0.55–0.585 | **0.325** | half-width 0.275 + 0.05 lane-keeping. The old value left the 1.10 m lanes a NEGATIVE lateral budget — infeasible before any uncertainty. |
| `nogo_logbarrier_eps` | 1e-3 | **0.05** | equals `nogo_warning_band`. At 1e-3 a 1 mm notional violation cost 2,036 — more than the entire risk term — and the no-go term reached 10^6 on routes the footprint validator called clear. |
| `use_belief_nogo_cost` | false | **true** | with it off the obstacle term sees only the mean path, so predicted belief growth costs nothing. |
| `nogo_weight` | 40 | **2000** | camera-ready IWAI value. |
| `network_goal_std_m` | 0.15 | **0.35** | the declared `goal_success_radius`. At 0.15 the risk term is ~5x stronger and the objective collapses toward shortest path. |
| `kouw_et1_ambiguity` | false | **true** | the thesis method. |
| `camera_network_objective` | legacy_pixel_chart | **metric_expected_belief** | the ONLY objective that loads the per-arm planner fields. Must be set per campaign. |

## Two accounting bugs that were fixed — do not reintroduce

1. **Fixed-horizon tail.** The objective summed a fixed 75 steps regardless of
   route duration, so a short route banked its remaining steps parked at the
   goal and accumulated that cell's ambiguity. A smooth arrival gate now zeroes
   all terms after arrival. Symptom: a constant-parameter arm produced an
   identical total on every route.
2. **Summed per-step terms encode route LENGTH.** Always inspect per-step values
   along the route, never only totals. A constant-R arm must give a constant
   per-step ambiguity once discount-normalised; if it does not, the accounting
   is wrong.

## What is NOT settled

- Route choice is currently a **NULL**: on 11 of 11 contrast tasks both arms
  picked the identical route. Earlier positive results from the same day are
  retracted — they predate the arrival-gate fix.
- Whether Kouw's preference covariance `Sigma*` is a goal tolerance or a
  belief-scale parameter. Ask W. Kouw.
- The free CasADi solver cannot use lane-graph seeds (0/8 feasible), so results
  so far are candidate scoring, not solver output.
- No closed-loop Gazebo evidence for any repaired configuration.

## Verification that must pass before trusting any result

Hand calculation, NumPy evaluator and CasADi objective must agree on the
ambiguity term to ~1e-6 at a fixed pose, and the full-rollout total must
reproduce by hand to ~1e-3. Script pattern: `verify_one_truth.py`.
