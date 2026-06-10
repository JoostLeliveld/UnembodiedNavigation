# Planner Hyperparameter Reference

The EFE planner has ~30 tunable knobs. This document explains what each one does, current
values in the three active campaign configs, and recipes for eliciting specific behaviours.

**The clean-EFE invariant**: do NOT add new cost terms to
`src/planning/planning/core/casadi_efe.py`. The objective stays
`risk + ambiguity + control + nogo`. Everything below is
*tuning* of existing knobs and *meta-strategy* (initialisation, planning resolution,
optimiser), never new terms.

References:
- Code: `src/planning/planning/planners/base_planner.py` (constructor signature has the
  full list), `src/planning/planning/core/casadi_efe.py` (math).
- Active configs: `scripts/visibility_comparison/paper_campaign_config.yaml`,
  `parcel_sortation_smoke_config.yaml`, `putaway_route_choice_smoke_config.yaml`,
  and `aws_smoke_config.yaml`.

---

## EFE objective scaling

The CasADi objective is
`J = (1/H_eff) · Σ_t γ^t [risk_scale·risk_t + amb_scale·amb_t + ctrl_t + nogo_t]`
with `risk_scale = risk_weight_obs · observation_risk_scale` and
`amb_scale = ambiguity_weight · ambiguity_term_scale`.

| name | smoke | probe | dark-probe | what it does | tune up when | tune down when |
|---|---|---|---|---|---|---|
| `ambiguity_weight` | 6 | 10 | 10 | base weight on the entropy of the conditional observation covariance per step | you want C2 to detour around shadow more aggressively | C2 refuses to enter shadow even when shadow is unavoidable |
| `ambiguity_term_scale` | 1 | 1 | 1 | second multiplier on ambiguity (kept as 1 so we tune only the weight); change only if porting math from another formulation | — | — |
| `risk_weight_obs` | 1 | 1 | 1 | base weight on KL-to-goal-observation per step | only if goal-attraction is too weak overall | rarely — usually you want to *widen the goal prior* instead |
| `observation_risk_scale` | 1.25 | 1.25 | 1.25 | second multiplier on risk; kept at 1.25 across configs | only if you want to amplify goal-pull without touching the goal-prior width | — |

**Practical rule**: keep `goal_prior_u_std_final` and `goal_prior_v_std_final` no smaller
than the shadow covariance scale unless you intentionally want a very aggressive
goal-attraction term. A tight final goal prior can make the trace term dominate when the
planner enters shadow, so changes here must be treated as method-level tuning and rerun
through the same evidence pipeline.

---

## Visibility model (R_plan blending)

| name | smoke | probe | dark-probe | what it does |
|---|---|---|---|---|
| `r_visible_uv` | 2.5 | 2.5 | 2.5 | observation-noise std (px) when `p_vis_eff ≈ 1`. Lower = sharper sensor where visible. |
| `r_miss_uv` | 15 | 15 | 15 | observation-noise std (px) when `p_vis_eff ≈ 0`. Higher = fuzzier sensor in shadow. |
| `visibility_sigma_kappa` | 1.0 | 1.0 | 1.0 | UT spread for the sigma-point expectation `E[p_vis(m, S)]`. Higher = more conservative widening of p_vis under belief uncertainty. |

The precision-blending formula is `1/R_plan = p_vis·(1/R_vis) + (1-p_vis)·(1/R_miss)`, so
R_plan transitions sharply with p_vis (precisions add linearly; r_plan recovers like
1/sqrt(prec)). To soften the shadow boundary in the planner without retraining the GP,
reduce `r_miss_uv` (e.g., 15 → 10) — that compresses the dynamic range of R_plan and the
ambiguity term becomes less aggressive at shadow entries.

---

## Goal prior (scheduling of the goal observation distribution)

| name | smoke | probe | dark-probe | what it does |
|---|---|---|---|---|
| `goal_prior_u_std_start` | 80 | 80 | 80 | goal-pixel std at horizon step 0 (wide) |
| `goal_prior_u_std_final` | 20 | 20 | **80** | goal-pixel std at horizon step `goal_progress_n_steps` (tight) |
| `goal_prior_v_std_start` | 80 | 80 | 80 | same as `_u_` but in v-axis |
| `goal_prior_v_std_final` | 20 | 20 | **80** | same as `_u_` but in v-axis |
| `goal_tightening_power` | 0.45 | 0.45 | 0.45 | exponent on `progress` before smoothstep — lower = tightens earlier in the horizon |
| `goal_progress_n_steps` | 90 | 90 | 90 | number of steps over which the schedule interpolates start→final |

**The wider the goal prior, the weaker the goal-pull**, because `(μ-μ_goal)^T Σ_goal^{-1}
(μ-μ_goal)` shrinks quadratically with σ_goal. This is the lever to make C2 willing to
detour. Re-run `scripts/visibility_comparison/efe_offline_lab.py` when a fresh sweep is
needed; generated plots are scratch outputs and are not maintained as paper artifacts.

The dark-north probe uses σ_final=80 specifically because offline analysis showed A4 wins
by ~1500 EFE units at σ_final=20 and only ~30 at σ=80.

---

## Look-ahead and time discretisation

| name | smoke | probe | dark-probe | what it does |
|---|---|---|---|---|
| `horizon` | 40 | 60 | 80 | number of optimisation steps |
| `dt` | 0.25 | 0.25 | 0.25 | seconds per step |
| `v_max` | 0.15 | 0.15 | 0.22 | max forward speed (m/s) |
| `w_max` | (default) | (default) | (default) | max angular speed; kept at the launch default |
| `discount_gamma` | 0.98 | 0.98 | 0.98 | per-step discount factor; `H_eff = Σ γ^t ≈ 27` at h=40 |

**Effective look-ahead distance = `horizon · dt · v_max`**. This is the single number that
matters for "can the planner see the decision-relevant feature ahead?". Visible-goal probe
at (h=60, dt=0.25, v=0.15) reaches 2.25 m; dark-probe at (h=80, dt=0.25, v=0.22) reaches
4.4 m.

**Coarse horizon / sparse-planning** (allowed, not yet tried): increase `dt` (e.g., 0.25 →
0.5) so the same horizon covers twice the distance. Caveats: process noise grows
proportionally with dt (already correct in `process_noise(dt)`), and any resulting
controller change should be treated as a separate planner variant.

---

## No-go / collision

Current paper-facing values are from `aws_f31b1_final_config.yaml` (2026-06-10). The
no-go is now a hinged-log **`warning_band`** keep-in penalty (replaces the old softplus /
`log_barrier` interior-biased barrier, which collapsed the C2 route split at weight>200).

| name | aws_final | what it does |
|---|---|---|
| `nogo_mode` | `keep_in` | penalize leaving the driveable prism union (vs `keep_out` obstacle distance) |
| `nogo_penalty_type` | `warning_band` | hinged-log: `near_weight*log(1+(max(b-c,0)/b)^2) + weight*(max(-c,0)/eps)^2`; exactly 0 for valid interior (clearance c ≥ b) |
| `nogo_weight` | 2000.0 | scalar on the violation term; high weight crushes violations without biasing valid-route choice |
| `nogo_warning_band` (b) | 0.05 | width of the soft warning band inside the boundary |
| `nogo_near_weight` | 50.0 | weight on the in-band log term |
| `nogo_safe_distance` | 0.15 | clearance offset baked into the signed distance |
| `robot_collision_radius_m` | 0.125 | radius used for trajectory diagnostics |

`use_belief_nogo_cost: false` for the global solve (the belief-nogo route mechanism is
retired). To make the planner "more willing" to push through tight spots, reduce
`nogo_safe_distance` — but only if the footprint allows it.

---

## Optimiser

| name | smoke | probe | dark-probe | what it does |
|---|---|---|---|---|
| `optimizer_maxiter` | 150 | 180 | 180 | L-BFGS-B iteration cap |
| `optimizer_maxfun` | 900 | 1200 | 1200 | L-BFGS-B function-evaluation cap |
| `optimizer_ftol` | 1e-6 | 1e-6 | 1e-6 | relative tolerance |
| `optimizer_gtol` | 1e-4 | 1e-4 | 1e-4 | gradient infinity-norm tolerance |
| `optimizer_warm_start` | true | true | true | reuse previous solution shifted by one step |

When solver wall time is close to `maxfun`, the solver is not converging; raise `maxfun`
first, then `maxiter`. When solver time is fast but the trajectory looks suboptimal, treat
that as a diagnostic local-minimum issue rather than adding route-specific initial seeds to
paper-facing runs.

---

## Optimizer Initialisation

Paper-facing runs may use neutral optimiser initialisation:

- shifted warm-start from the previous optimized control sequence when available;
- all-zero controls on the first plan or after a goal reset;
- small local escape maneuvers from the current heading;
- modest lane-graph route seeds generated from the known 2D driveable floor;
- condition-neutral multistart candidates, if every compared condition receives
  the same generic candidate set and the paper reports this as optimizer basin
  handling.

Robotics reason: this is a nonlinear receding-horizon planner. Local optima can
dominate the result, especially in aisle/cross-aisle layouts. A real AMR stack
would use the known map/lane topology to propose plausible route basins before
letting the local optimizer refine them. That is scientifically faithful here
when the seeds use only the known driveable floor.

Do not add visibility-derived route seeds or mission waypoints to the live
controller. Multistart is acceptable only when it is neutral, shared by C1/C2,
and reported as optimizer basin handling.

---

## Process and command noise

| name | smoke | probe | dark-probe | what it does |
|---|---|---|---|---|
| `process_noise_xy` | 0.01 | 0.01 | 0.01 | XY process-noise std rate (m/√s); scales correctly with dt |
| `process_noise_theta` | 0.02 | 0.02 | 0.02 | heading process-noise std rate (rad/√s) |
| `command_noise_*` (family) | ON for AWS | ON for AWS | ON for AWS | execution disturbance / slip on issued commands |
| `encoder_noise_*` (family) | ON for AWS | ON for AWS | ON for AWS | dead-reckoning / odometry integration error |

For AWS diagnostics, command and encoder noise are part of the realism claim.
Command noise changes the true executed motion; encoder noise changes the
dead-reckoning estimate used when camera updates are weak or absent.

---

## Recipes (behaviour → which knobs to touch)

These are the headline tunings. Use them as starting points; always re-check on offline
rollout via `scripts/visibility_comparison/efe_offline_lab.py` before running Gazebo.

### "C2 hugs the visible side of an aisle (lateral preference)"

Goal: when traversing a uniform aisle whose left side is more visible to the camera, C2
should bias to that side; C1 should drift to centre.

- `ambiguity_weight`: increase modestly (6 → 10–12).
- `goal_prior_u_std_final`: widen (20 → 50) so per-step lateral position matters more than
  per-step distance to goal.
- Effective look-ahead: keep modest (h≈60, v≈0.15) so the planner doesn't try to commit to
  a multi-aisle replan within one horizon.
- Verify offline: sample p_vis along left edge vs centerline vs right edge; the
  per-step ambiguity gradient should point clearly toward the brighter side.

### "C2 commits to the dark only at the end (late commit)"

Goal: stay in visible region throughout the approach; enter the dark cap-region only for
the last 1–2 m to a dark goal.

- `goal_prior_u_std_final` / `_v_std_final`: widen (20 → 50–80). Tight goal prior pulls the
  robot into the dark zone too early.
- `goal_tightening_power`: increase (0.45 → 0.7) so the tight prior kicks in late in the
  schedule, not from the start.
- `goal_progress_n_steps`: match to expected total run length so the schedule's "final"
  state lines up with the actual goal approach.
- Verify offline: the optimiser's chosen trajectory should have terminal-progress > 0
  throughout and only enter the low-`p_vis_eff` zone in the last quarter of the horizon.

### "Reduce stall risk at a (mostly-visible) goal"

Goal: when the goal is visible but the approach passes near a brief dark region, C2 should
push through rather than stall.

- `ambiguity_weight`: lower modestly (10 → 6–8) — at the same time, to keep the safety
  property on truly dark goals.

### "Optimiser is too slow / hits maxfun every step"

- First raise `optimizer_maxfun` (1200 → 1800).
- If still pinned, raise `optimizer_maxiter` (180 → 300).
- If still pinned, the EFE landscape is too non-convex — softening `r_miss_uv` (15 → 10)
  helps gradient stability.

### "Look-ahead is too short to see the decision-relevant feature"

- Increase `horizon` first (cheap if the decision can fit in 80–100 steps).
- Then consider sparse planning: raise `dt` (0.25 → 0.5) and possibly reduce `horizon` to
  keep optimisation variable count manageable.
- Do not add route-specific cold starts or mission waypoints to compensate for short
  look-ahead in paper-facing C1/C2 runs.

---

## What this doc deliberately does NOT cover

- New cost terms (forbidden by the clean-EFE invariant).
- World, GP, or YOLO retraining — see `docs/CONSISTENCY_CHECKLIST.md` for that.

---

## Math-level interventions (advanced)

In addition to the per-knob tuning above, there is a second layer of "math-level"
levers: changing the parametric form of an EXISTING term while preserving its role.
These are NOT forbidden by the clean-EFE invariant; they are not pure hyperparameter
tuning either. They require small code edits in `casadi_efe.py`. Each one's effect on
the GP→p_vis→R_plan inflation chain (the visibility-aware mechanism) is the gating
question — if it breaks that chain, it is forbidden.

| Intervention | What changes | Effect | Touches | Preserves R-inflation? |
|---|---|---|---|---|
| **Laplace goal prior** | Goal-prior switches from Gaussian to Laplace on the goal observation. Per-step distance penalty grows linearly in `|μ_y − μ_goal|` instead of quadratically. | Softer goal-pull far from goal. Addresses "C2 stalls because risk dominates ambiguity mid-route". | `risk_ca` in `casadi_efe.py` | yes — the ambiguity term is independent of the risk distribution family. |
| **L1 distance** | Replace Mahalanobis (with Σ_goal) with L1 in pixel space (still scaled by σ_goal). | Linear distance growth; simpler implementation than Laplace; similar qualitative effect. | `risk_ca` mean-term | yes |
| **Room-distance normalisation** | Divide the goal-distance term by the room diagonal (or start-to-goal straight-line distance). | Same hyperparameters work for short and long missions; goal-pull becomes scale-invariant. | `risk_ca` | yes |
| **Visibility-conditioned goal-prior schedule** | Replace the time-only smoothstep on `goal_progress_n_steps` with a schedule that ALSO depends on the look-ahead's mean p_vis: σ_goal stays wide while the planner sees a shadow band, tightens once past it. | Direct fix for "C2 stalls at shadow boundary because tight σ_goal arrives before C2 has cleared the shadow". | `goal_obs_cov_ca_for_progress` | yes |
| **Hierarchical decomposition** | A coarse path planner (A* on GP-weighted graph) produces sub-goals; the EFE planner controls between sub-goals. EFE inflation untouched. | Lets EFE focus on local maneuvering while a global planner handles long-horizon route choice. Heavier engineering. | New file outside `casadi_efe.py`; EFE planner takes sub-goals as input. | yes (EFE untouched). |
| **Coarse / sparse-resolution trajectory** | Increase planning `dt` (e.g., 0.25 → 0.5). Same horizon, longer look-ahead, fewer optimisation variables. | Extends look-ahead cheaply; gives EFE access to the shadow earlier. Process noise scaling already correct. | YAML config only — already supported. | yes |

**Forbidden** (would break R-inflation):
- Anything that decouples `R_plan` from `p_vis_eff`.
- Removing the ambiguity term from the objective.
- Adding a per-step term that overrides the ambiguity gradient (e.g., a hard "stay
  on centerline" penalty that ignores visibility).

Each of these math-level interventions is a **proposal** that flows through the
root agent loop: `planner-diagnostician` recommends it after a diagnosed failure
mode, the main session implements the code edit, `rollout-runner` re-runs, and
`figure-analyst` verifies.
