# Multicam extension — state of play & how to continue (handoff, 2026-07-29)

Written to hand a clean overview to a fresh session. Read this first, then
`docs/multicam_vs_paper1_correction_parity.md`.

## 1. The architectural lesson (read this first)

The 4-camera extension was built as a **parallel implementation** rather than by extending the
paper-1 (single-camera) method:

- paper-1 correction path: `_apply_pixel_correction` in `unicycle_planner_node.py`
  (consumes `/state/bev` from `pixel_to_bev_state_node`).
- multicam correction path: `_apply_state_correction` **(separate, newly written)**
  (consumes the fused `/state/bev` from `camera_manager_node`; bypasses `pixel_to_bev_state_node`).

**Every confirmed parity gap traces to that fork**, and two real bugs already came from it this
month: (a) the heading-frame bug (multicam bypassed the node that applies `odom_yaw + offset`), and
(b) the missing correction guards below. **Recommendation: consolidate onto ONE gate chain
(paper-1's), treating the fused observation as just another measurement source.** Do this before
adding features.

## 2. What is SOLID (does not depend on the open questions)

- **Multicam stack runs closed-loop**, ~5 Hz fused corrections, 4 cameras simultaneously.
- **Fused 4-cam localization beats single-camera where coverage runs out** (fig23,
  `RESULTS_localization.md`): W→E fused 0.177 m / 3-3 goal vs single-A 0.606 m / 0-3 (drifts once
  past A's field). E→W: fused 0.128 m ≈ single-C 0.137 m (camera C covers that whole route) —
  honest asymmetry, shown not hidden. 3 seeds, both directions.
- **Coverage/handover is the strongest observed multi-camera effect** (fig23). The apparent
  selection≈fusion / GP≈non-GP result in fig24/fig26 is **provisional**, not a settled negative
  result: `conditional_cov_uv` is validated then discarded, the fusion input is a constant
  isotropic `diag(0.08², 0.08²)` (or a scalar GP rescaling), and projection propagates no
  covariance. The combine-rule ablation must be re-run with a physically meaningful map-frame
  noise model before it supports a paper claim.
- **New knobs (plumbed, tested):** `manager_camera_ids`, `manager_fusion_mode`,
  `manager_require_gp_artifacts`, `goal_replan_move_m`, `local_controller_type: ff_fb`.
- **Multi-goal missions work**: goal-advance triggers a global replan (`efe_agent_node`), with a
  non-fatal fallback if the replan solve fails.
- **`ff_fb` local controller** cuts tight-aisle lateral drift 0.66 m → 0.04 m (fig29).

## 3. What is PROVISIONAL (do not trust until §4 is done)

All **navigation-failure attributions**: fig25 (grand tour), fig30 (robustness 4/9), fig31/fig32
(failure diagnosis), and the narrative "tight-aisle execution is the bottleneck". Reasons:
- The correction path is missing guards (§4.1), and a **1.87 m one-sample belief jump** was measured
  — physically impossible (≤0.06 m at v_max) and it moved *away* from both GT and the good
  correction available at that instant. Some failures are likely this bug, not navigation limits.
- `nogo_weight` is 1200 here vs 2000 in paper 1 (weaker obstacle penalty).
- **fig24/fig26's combine-rule conclusion is also provisional:** the covariance model discards the
  published SPD `conditional_cov_uv` and does not propagate pixel covariance through the ground
  projection. Equal, isotropic map covariances can make selection and fusion look artificially
  similar and can misweight far-range oblique observations.

NOTE: an earlier claim that the 4-cam world was "missing driveable lanes / should use keep_in" was
**retracted** — the world is open-floor by design with all 28 rack segments + 4 walls registered as
collision prisms, so `keep_out` is correct. See the correction section of the parity audit.

## 3.1 Noise model to freeze before the re-run

To remain consistent with paper 1, keep its **precision-blended planning interface**, but do not
mistake that interface for the covariance of a measurement that actually arrived. The factorised
model should be

```text
a_c(s) ~ Bernoulli(p_use,c(s))
z_uv,c | a_c=1 ~ N(h_uv,c(x), R_cond,uv,c(s))
```

where availability and conditional localisation error are separate random objects.
`conditional_cov_uv` must therefore describe residuals **conditioned on a valid, associated
detection**. It must not itself be produced by blending `p_available` between visible and miss
endpoints; the current single-camera adapter and BEV reliability model do exactly that despite the
field name, and should be corrected.

For a strict paper-1 planning baseline, retain the old image-space precision blend, per camera:

```text
R_plan,uv,c(s)^-1 =
    p_use,c(s) R_good,uv,c(s)^-1
  + (1-p_use,c(s)) R_bad,uv,c(s)^-1 .
```

Historical reproduction uses `R_good=(2.5 px)^2 I` and `R_bad=(40 px)^2 I`. The stray 120 px
runtime default and the newer bounded **variance** interpolation are separate ablations, not silent
replacements for the locked paper configuration. With a learned anisotropic `R_cond`, use it as
`R_good` and construct `R_bad >= R_good` while preserving its ellipse shape and the frozen miss
endpoint. `R_miss` is only a planning surrogate for low expected observability: a miss produces no
filter update.

For an observation that actually arrives, project the full conditional covariance:

```text
J_c = d g_corr,c(u,v) / d(u,v)
R_geom,xy,c = J_c R_cond,uv,c J_c^T

R_update,xy,c =
    R_geom,xy,c
  + Sigma_projection,c(range, bearing)
  + J_z sigma_z^2 J_z^T
  + J_ext Sigma_ext,c J_ext^T
  + Sigma_time,c(delta_t).
```

`g_corr` includes the per-camera along-bearing correction. The result is generally a rotated,
range-dependent ellipse with a non-zero cross term. `Sigma_projection` is a held-out commissioning
residual covariance after removing the fitted mean bias. `Sigma_time` should come from motion
uncertainty when observations are moved to a common time; otherwise fuse at the measurement stamp
or reject excessive skew. A small eigenvalue floor is acceptable for numerical SPD, but neither
`diag(0.08^2)` nor the newer `fusion_report_std_m=0.3` diagonal floor is a measurement model.

Inflation should represent a measured uncertainty source, not a policy event:

- switching camera ID alone adds no noise; disable the current arbitrary 1--9x handover multiplier
  for the primary baseline;
- where pairwise registration is imperfect, add a calibrated anisotropic
  `Sigma_handover,A->B` (or maintain a camera-bias state), rather than multiplying all of `R`;
- association collapse, excessive age, and gross disagreement are gates/rejections because their
  errors are non-Gaussian; do not hide them inside a large Gaussian covariance;
- health/epistemic inflation is allowed only as a separately calibrated, logged robustness layer,
  and must not reuse availability or confidence already represented in `p_use`/`R_cond`;
- NIS rejection or Student-t reweighting remains a filter-robustness ablation after the nominal
  covariance is fixed. It must not be used to calibrate the nominal model on the same evaluation
  run.

Naive sequential fusion also assumes conditional independence. Pixel noise may be independent, but
contact-point bias, detector bias, world registration, and camera calibration can be shared. The
primary multi-camera baseline should therefore either use one selected camera, use a block
measurement covariance with estimated cross-camera terms, or use covariance intersection. A useful
common-mode approximation is

```text
R_fused = (sum_c R_ind,c^-1)^-1 + R_common,
```

with `R_common` added once, not once per camera. Only after this model is frozen is
selection-versus-fusion an interpretable ablation.

### Implemented first arm: strict historical reproduction

The selectable runtime profile is `manager_covariance_profile: paper1_historical`. The four
`_mc_2x2_{sel,fus}_{gp,nogp}.yaml` configs now freeze it together with
`r_visible_uv: 2.5` and `r_miss_uv: 40.0`. In this arm:

- the detector's historical 2.5/40 px precision-blended matrix is consumed as published and
  propagated through the complete corrected projection as `J R_uv J^T`;
- the resulting full map covariance, including its cross term, reaches selection/fusion;
- the multicamera-only 1--9x handover inflation and `fusion_report_std_m` floor are not applied;
- fusion forms a measurement by precision weighting the accepted camera measurements, with no
  invented identity state prior; the planner remains the one belief filter.

This arm intentionally retains paper 1's scalar availability/quality conflation: the published
field is named `conditional_cov_uv`, but is still produced by the historical trust-to-R precision
blend. It therefore answers the narrow reproducibility question, not the later factorised
`p_use + R_cond` question. The latter must be a separately named follow-up arm so its improvement
cannot be attributed to selection/fusion.

## 4. Recommended next steps, in order

1. **Run the strict historical arm above**, replacing the constant map covariance and
   reporting-floor substitute while holding the old 2.5/40 precision interface fixed.
2. **Implement and calibrate the factorised §3.1 covariance path**. Log both genuine
   `R_cond,uv` and projected `R_update,xy`.
3. **Consolidate the correction paths** (§1) and while doing it restore the paper-1 guards:
   - jump limiter: `pixel_max_correction_jump_m` (0.5 m) is SET in config but **never read** on the
     multicam path — restore it, but keep advancing the belief stamp + mild inflation so a rejection
     cannot *freeze* the belief (the runaway the multicam path was written to avoid);
   - NIS gate: stop the +1.0 m² inflate-on-reject from neutering the gate for the next correction;
   - add a kinematic plausibility cap on the prediction (`_replay_cmd_log_interval` integrates the
     LAST COMMAND for up to 1.5 s when odom samples are absent → up to ~0.9 m invented motion);
   - add reason-coded rejection diagnostics (paper 1 has 7 codes; multicam has none — this is why
     the bug was invisible without CSV forensics).
4. **Set `nogo_weight: 2000`** to match paper 1.
5. **Re-run the robustness campaign** (`_rob_rob_{easy,hardA,hardB}.yaml`, 3 tasks × 3 seeds) and
   re-derive the failure profile. Only then update fig30/31/32 conclusions.
6. Optional, if uncertainty plots are wanted: log a **growing** belief covariance (currently a fixed
   reported 2σ ≈ 0.36 m from `manager_fusion_report_std_m`), which also unlocks NEES/consistency
   claims. Note the Option-A smoothing (NEES 16.8→2.8) is offline-only.

## 5. Hard constraints to respect (from memory)

- Real Gazebo only, no synthetic data; report failures immediately.
- **Never tune the visibility term / R_plan / EFE observation-risk weight** to change navigation
  outcomes — it is the frozen method. Legitimate levers: task/mission design, execution/tracker,
  camera placement, EKF robustness guards.
- Two-world rule: method dev in `warehouse_aws`; `warehouse_full_4cam` evaluates.
- GT is evaluation-only (the user relaxed this for convenience during this phase: "you can use GT
  for now"). Calibration constants are commissioning-time, never refit at runtime.

## 6. Key file map

- Planner/EKF: `src/planning/planning/nodes/unicycle_planner_node.py`,
  `efe_agent_node.py`
- Fusion manager: `src/reliability/reliability/nodes/camera_manager_node.py`
  (+ `fusion.py`, `replay.py`, `providers.py`)
- Detector: `src/perception/perception/nodes/batched_four_camera_yolo_node.py`
- Launch/config plumbing: `src/experiments/experiments/core/visibility_launch_common.py`,
  `src/experiments/launch/warehouse_primary_comparison.launch.py`,
  `src/experiments/config/tasks.yaml`
- Campaign configs: `scripts/visibility_comparison/_mc_*.yaml`, `_rob_*.yaml`
- Figures + results notes: `logs/studies/multicam_nav_demo/figures/` (fig22–fig32 + RESULTS_*.md)
- Per-camera GPs: `logs/visibility_comparison/spawn_grid_20260727/gp/camera_{A,B,C,D}/`
- Old paper reference figures: `../RobotControlExternalCamera/results/OLD_PAPER_FIGURES/`

---

# START HERE if you are the Gazebo session (appended 2026-07-29, end of day)

## Gazebo update — solve showcase completed

The R sweep and a waypoint-free three-seed `rob_easy` showcase have now run.
The sweep did not rescue the difficult route: all three R arms solved globally
but later collided. The subsequent `rob_easy` run exposed two implementation
bugs in the per-camera correction arm:

1. `allow_reanchor=False` was incorrectly passed as
   `metric_measurement=False`, selecting paper-1's 0.5 m pixel jump limiter for
   metric observations; and
2. `camera_xy_only` spliced prior XY-heading correlations into the posterior XY
   block, making the committed covariance indefinite and NIS negative.

Both are fixed and regression-tested. The real fixed showcase reached the goal
in **3/3 seeds**, collision-free, with **526/526 accepted corrections**, no
negative NIS, and pooled belief-error p95 **0.174 m**. See
`docs/multicam_solve_showcase_2026-07-29.md` and
`logs/studies/multicam_nav_demo/figures/fig33_multicam_solve_showcase.png`.

The straight `rob_easy` result is now treated only as the estimator regression
check. The harder waypoint-free offline gate has also passed. For the
`rob_hardA` and `rob_hardB` start/final-goal pairs, the old H75 solve missed by
5.62 m / 3.21 m; collision-aware condition-neutral map seeds plus the corrected
seed tracker and H120/iter120 converged to valid curved routes ending 0.17 m /
0.12 m from goal. See
`docs/multicam_offline_hard_route_showcase_2026-07-29.md`. No new hard-route
Gazebo claim has been made yet.

Everything below describes the state before that Gazebo result and is retained
as the investigation record.

## Immediate next action: the R sweep (3 runs, ~5 min each)

```bash
for t in r08 r14 r21; do
  python3 scripts/visibility_comparison/run_visibility_campaign.py \
    --config scripts/visibility_comparison/_rsweep_$t.yaml \
    --log-root logs/visibility_comparison/rsweep_$t
done
python3 scripts/visibility_comparison/read_correction_sweep.py \
  logs/visibility_comparison/rsweep_{r08,r14,r21}
```

`rob_hardA`, 1 seed, per-camera mode, effective R = 0.080 / 0.144 / 0.215 m
(`state_measurement_inflation_std_m` = 0 / 0.12 / 0.20, added in quadrature).

**Read:** NIS p50 (well-specified ≈ 1.4-2; paper 1 measured **0.29**), rejection % (want low single
digits), belief error p50/p95. **Baseline from the 2026-07-28 rob_hardA runs, measured: belief error
p50 0.106 / 0.110 / 0.148 m, p95 0.252 / 0.273 / 0.569 m.** Those runs predate the reason codes, so
only belief error is comparable; rejection % has no historical counterpart.

If the three are indistinguishable, R does not matter here -- stop tuning it and run the full
campaign at r08.

## What changed since the 2026-07-28 rob campaign

Nothing was committed (`HEAD` == the sha in those manifests), so git cannot separate pre/post-run
edits; this list is from file mtimes plus the session record.

- **One shared correction chain.** `planning/core/belief_correction.py`; both the pixel and metric
  paths run it. Golden trace: 6365 locked corrections replay with **zero verdict changes**.
- **Multicam guards**: NIS reject-inflation 1.0 -> **0.05 m²**; reason-coded diagnostics (were
  absent); prediction clamp `max_predict_speed_mps = v_max`; quorum re-anchor.
- **Jump limiter OFF on the metric path** (`state_max_correction_jump_m`, default 0). It deadlocks:
  rejecting inflates S -> raises gain -> raises update -> fires again. NIS is the correct gate for a
  linear-H measurement. Pixel path keeps its 0.5 m limit (locked method).
- **`nogo_weight` 1200 -> 2000** in all 30 4-cam configs.
- **Per-camera sequential updating** is now the DEV DEFAULT (`state_correction_mode: per_camera`) for
  `_rob_*` and `warehouse_full_4cam_{tours,campaign,missions,missions_smoke}`. Only `_mc_2x2_*` stay
  `fused` (historical reproduction).
- **`state_measurement_inflation_std_m`** (new): `R' = R + σ²I`. Needed because per-camera mode
  bypasses the `fusion_report_std_m` floor.
- **Codex**: `paper1_historical` covariance profile (J R Jᵀ, anisotropy, cross terms) + a **0.25 s
  fusion stamp-spread window and manager eligibility gates now applied to the fused path for ALL
  profiles** -- that last one changes rob behaviour and is unmeasured.
- **CSV**: new `pixel_corr_measurement_space` (0 = px, 1 = m), `_predict_clipped_m`, `_camera_index`;
  reject code 7 = `diverged`. 220 -> 221 columns.

## Facts already measured -- do not re-derive

- Paper 1 (locked, 43 runs, 6370 corrections): pixel innovation median **1.53 px** vs R 2.5 px,
  NIS median **0.29**, **1 rejection in 6370**. Its R was slightly LOOSE, not overconfident.
- Belief error vs GT p95 **0.127 m** while pixel residuals were 1.5 px ⇒ single-camera pixel updating
  absorbs calibration bias into the POSE; NIS never sees it.
- 4-cam fused-vs-GT ≈ **0.19 m**. This is **inter-camera disagreement** -- a term that does not exist
  in a single-camera system. That is why the frozen 2.5/40 blend transfers in shape but not scale.
- `paper1_historical` propagates to **0.022-0.114 m** (anisotropy only 1.07x near -> 1.70x at 7.7 m).
  **Do NOT enable it for performance runs**: 29-39 % predicted rejection in every model tried. It is a
  reproduction profile. Keep `legacy_fixed_metric` for development.
- The legacy `fusion_report_std_m: 0.18` floor is WHY the old setup worked -- it was an empirically
  calibrated total error budget. Codex's profile removed it, which is the whole regression.

## Open, in order

1. Freeze the passed offline hard-route settings, create waypoint-free hard
   task entries, and run the real hard-route Gazebo showcase. Do not reuse the
   existing mission-waypoint outcomes as evidence.
2. Measure `time_skew_rejected_camera_ids` from the manager decision payload -- the new 0.25 s window
   may be silently cutting effective camera count.
3. Commission `Σ_unmodelled` from the between-camera spread in overlap regions (`overlap.py`,
   `fusion.camera_disagreement_m`) and add it to `J R_uv Jᵀ`. Correct shape + correct scale, no tuning.
4. `_mc_2x2_*` historical reproduction (12 runs) -- only after 3.
5. Config base+overlay: a parameter currently needs declaring in FOUR places.
