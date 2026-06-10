# F86a Method & Runtime Contract

Status: maintained reference for the F86a `camera_xy_only` route-choice benchmark.
This document is the detailed companion to `aws_f86a_camera_xy_config.yaml` and the
root `CLAUDE.md` evidence standard. It records exactly how the runtime works,
how every plotted quantity is computed, and what must stay aligned. It is
**diagnostic/method documentation**, not a results claim.

Benchmark task: `a0_west_to_a1_upper_blocked_mid` on `warehouse_aws.world.sdf`
(F86a v4 geometry: continuous left shelf R1, R4 occluding crate stack).
Conditions: **C1** `constant_R_efe` (constant observation covariance), **C2**
`visibility_aware_efe` (GP-modulated observation covariance).

---

## 1. Architecture — hierarchical belief loop (`hierarchical_beliefloop_v1`)

Two-stage planner in `src/planning/planning/nodes/efe_agent_node.py`:

- **GLOBAL** (`global_horizon: 120`, `dt: 0.25`): long-horizon EFE plan solved
  once at the start of each global phase. Its state sequence is converted to a
  2D waypoint polyline by `extract_waypoints(..., spacing_m=waypoint_spacing_m)`.
  The global plan provides the **route shape** (which corridor). `global_use_ambiguity:
  true`, `global_optimizer_multistart: true` with two condition-neutral named
  route seeds (`nw_blind_to_a1`, `south_visible_to_a1`).
- **LOCAL** (`local_horizon: 6`, `local_plan_rate: 5 Hz`): a simple tracker that
  follows the waypoint polyline and emits velocity commands at `cmd_publish_rate:
  10 Hz`. `local_use_ambiguity: false`, `local_use_visibility_model: false` — the
  local layer is condition-neutral execution plumbing; it never queries the GP and
  is identical for C1/C2. It carries a `log_barrier` belief-nogo safety device
  (`local_use_belief_nogo_cost: true`) purely to keep the executed tube inside the
  driveable region; it does not choose routes.

Rationale (memory `project_beliefloop_architecture`): a local nonlinear EFE MPC
was abandoned because Gazebo contention made per-step nonlinear solves ~30×
too slow and offline could not predict runtime timing. Belief-closed reasoning
lives at the GLOBAL stage only.

The route choice is therefore a property of the **global EFE objective**, not of
the executor. No mission waypoints, route scripts, or condition-specific seeds are
used; multistart is neutral basin handling (root `CLAUDE.md`).

---

## 2. State, prediction, belief

State `m = (x, y, θ)`, covariance `S` (3×3 EKF).

- **Prediction**: unicycle model driven by odometry (`use_odom_for_predict: true`,
  `use_odom_heading_correction: false`). Odometry is used for dead-reckoning
  prediction only — there is no separate yaw measurement update.
- **Process noise**: `process_noise_xy: 0.01`, `process_noise_theta: 0.046`.
  (Diagnostic note: the real encoder noise injected in sim — `encoder_noise_angular_slip_std:
  0.16`, `encoder_noise_angular_additive_std: 0.1` — is larger than `process_noise_theta`,
  so the filter can be heading-overconfident. This is a known NEES>1 effect, recorded
  here, not silently corrected.)
- **Init belief**: `init_belief_sigma_xy: 0.05`, `init_belief_sigma_theta: 0.05`.

---

## 3. Camera (x, y) update — the only global correction

Pipeline: external camera image → robot detection pixel `(u, v)` →
homography / back-projection to the ground plane → measured `(x, y)`. Heading is
**not** measured by the camera in the current paper-facing runs.

- `heading_update_mode: camera_xy_only`. The pixel `(x, y)` update corrects the
  full state through the Kalman gain, so θ is corrected **only indirectly** via the
  prediction-Jacobian cross-covariance terms `S_xθ, S_yθ`. There is no explicit
  yaw anchor (`use_odom_heading_correction: false`,
  `use_state_bev_heading_correction: false`). See memory `project_f86a_correct_odom`.
- **Gate**: a NIS gate exists but is disabled by default
  (`pixel_correction_nis_threshold = 0.0` ⇒ the condition `threshold>0 AND nis>threshold`
  never fires ⇒ 100 % of corrections accepted). **Therefore `pixel_corr_accepted`
  is NOT a visibility/availability signal** — it is ~always 1. Use the YOLO
  detection flag for availability (Section 6), never `pixel_corr_accepted`.
- Other guards: `pixel_max_correction_jump_m: 0.5`, `pixel_timeout_s: 1.25`,
  `skip_stale_pixel_correction: true`.

### GP role (C2 only)
The GP (`P_conservative_plan_map`, `gp_artifact`) modulates the **camera `(x, y)`
observation covariance** used inside the EFE objective: lower observation
reliability ⇒ larger planner-facing `(x, y)` covariance. The GP affects `(x, y)`
covariance only — **it does not improve heading directly**. C1 uses a constant
observation covariance (`r_visible_uv`/`r_miss_uv` without GP modulation).

---

## 4. EFE objective (global)

Per-step expected free energy terms (see `base_planner.py`):

- **observation risk** (`risk_weight_obs: 1.0`, `observation_risk_scale: 1.25`,
  `r_visible_uv: 2.5`, `r_miss_uv: 40.0`): penalises predicted observation
  uncertainty; for C2 this is GP-modulated so camera-poor regions look risky.
- **ambiguity** (`ambiguity_weight: 18.0`, `ambiguity_term_scale: 1.0`): predicted
  posterior uncertainty / information term.
- **goal prior** (`goal_prior_*_std_*`, `goal_tightening_power`): pulls the plan
  toward the goal in observation space.
- **control** (`control_weight: 0.0` for the global EFE route shape).
- **no-go warning-band** (Section 5): keeps the plan inside the known driveable
  region.

Route choice must emerge from **risk + ambiguity**, not from no-go interior bias
or from inflating `ambiguity_weight`. The offline route-split gate
(`timing_presentation/scripts/make_f86_heading_compare.py`) verifies C1→NW-blind,
C2→south-visible from the EFE cost alone, and that the split is robust to the
no-go weight.

### Nonlinear solver / local optima
L-BFGS-B (`optimizer_maxiter: 120`, `optimizer_multistart: true`). The objective is
non-convex; multistart with the two named route seeds is condition-neutral basin
coverage. Long multistart solves are expensive — a known scalability limit, not a
result.

---

## 5. No-go warning-band penalty (F86a v4)

`nogo_mode: keep_in`, `nogo_penalty_type: warning_band`. With clearance

```
c_i = d_trav(m_xy,i) − d_safe − κ·sqrt(λ_max(S_xy,i))
```

(`d_safe = nogo_safe_distance = 0.15`, `κ = nogo_belief_kappa`), the per-step
penalty is the **hinged-log warning + quadratic violation** form:

```
phi(c) = near_weight · log(1 + (max(b − c, 0)/b)^2)  +  weight · (max(−c, 0)/eps)^2
```

with `b = nogo_warning_band = 0.05`, `near_weight = 50`, `weight = nogo_weight = 2000`,
`eps = nogo_logbarrier_eps = 0.01`. Properties (verified numerically + numpy↔CasADi
agreement to 1e-6):

- `c ≥ b`: penalty exactly **0** — valid interior states are unbiased, so two valid
  routes (e.g. narrow vs wide aisle) compete on risk/ambiguity only.
- `0 ≤ c < b`: soft log-shaped warning (keeps the familiar log "look").
- `c < 0`: strong quadratic violation penalty.

**Why this replaced `log_barrier`** (the root cause of the "weight>200 collapses
the C2 split" bug): `log_barrier` applies `w·log(1+scale/c)` to *every* valid
interior state, so it always prefers the wider aisle (south, clearance ≈0.26 m)
over the narrower one (NW upper-cross-aisle, ≈0.11 m) by ~2.3× per step. Raising
the weight amplified this width bias until both C2 multistart seeds escaped the NW
basin into the south global minimum → identical J → gate tie. The warning-band
penalty is zero for valid interior states, so weight can be raised arbitrarily to
crush violations without biasing valid-route choice. Implementation:
`src/planning/planning/core/nogo_cost.py` (numpy `_penalty_from_clearance_np`,
CasADi `make_penalty_state_casadi`, and both `make_penalty_belief_casadi` branches).

---

## 6. Localization error & filter-consistency definitions

- **Position error** `e_p,i = || (truth_x, truth_y)_i − (planner_belief_x, planner_belief_y)_i ||`.
  Truth from `experiment.csv` (`truth_x`, `truth_y`); belief from
  `planner_belief_x`, `planner_belief_y` in the same file.
- **Planner 2σ radius**: from `(planner_cov_x, planner_cov_y, planner_cov_xy)`,
  `λ_max = (Sxx+Syy)/2 + sqrt(((Sxx−Syy)/2)^2 + Sxy^2)`, `r_2σ = 2·sqrt(λ_max)`.
- **NEES** (position Mahalanobis, table only): `(Syy·ex² − 2Sxy·ex·ey + Sxx·ey²)/det`,
  `det = Sxx·Syy − Sxy²`; mean over samples **after first command**. NEES≫2 ⇒
  filter overconfident (see Section 2 note).
- **NIS**: normalized innovation squared of the pixel correction (gate input);
  disabled by default (Section 3).

### perception.csv parse hazard (must follow)
`perception.csv` has 81 data values per row but 79 header names (two anonymous
leading timestamp columns). Read with `pd.read_csv(path, index_col=[0,1])`; take
stamps from `perc.index.get_level_values(0)`. Column-name→value alignment is shifted,
so only use it for YOLO flags/timestamps (`yolo_detected_after_threshold`,
`yolo_score_selected`, which land correctly) — **always take truth positions from
`experiment.csv`, never from `perception.csv`**.

---

## 7. Relevant plots & exact computation (F86 paper figure)

Generated by `timing_presentation/scripts/make_f86_paper_figure.py`:

- **Map panels (a) C1, (b) C2**: GP `P_conservative_plan_map` background; driveable
  rectangles + shelf overlays from the same geometry the planner used; camera
  marker; start/goal; initial planned path (first `plan_samples.csv` stamp); truth
  path; belief-mean path; 2σ covariance ellipses; **YOLO-detection markers** at
  truth position for frames with `yolo_detected_after_threshold>0` after first cmd
  (NOT `pixel_corr_accepted`).
- **(c) YOLO score / availability**: `yolo_score_selected` vs time; grey bands where
  not detected; threshold line at `yolo_conf_threshold`.
- **(d) position error + 2σ**: `e_p` (solid) and `r_2σ` (dashed) per condition.
- **(e) belief-tube clearance**: `c = d_trav − d_safe − κ·σ_major` per condition;
  zero line is the constraint boundary.

Offline route-choice + θ-variance audit: `make_f86_heading_compare.py`
(F83 odom-anchor vs F86a camera_xy_only θ-std).

---

## 8. Timing hygiene (root contract)

All runtime plots and aggregate means use samples **at or after the first
non-trivial velocity command** (`first_cmd_stamp` in `run_summary.json`). Launch
wait, global-solve time, and estimator warm-up are reported separately and never
mixed into localization / belief-error / yaw-error / tracking / detection-rate /
clearance / command-timing means. Goal success uses `goal_region_success`, not
`completed`.

---

## 9. Artifact chain (must all align before any claim)

world SDF + camera pose → detector (`aws_yolo_simseg_v2`) → visibility samples in
that same world → GP fit (`aws_gp_v6/yolo_score_raw_gp.npz`, with
`P_conservative_plan_map` and embedded `geometry_json`/`geometry_sha256`) →
campaign config records exact detector + GP → seeded logs (no hidden fallbacks) →
figures/metrics from those logs → paper wording matches the data. If any link is
missing the result is diagnostic/exploratory, not paper-ready.
See `docs/final_figure_alignment_checklist.md`.
