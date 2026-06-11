# Decision Log

Short, dated decisions that prevent the project from re-litigating the same
scientific choices.

## 2026-06-11 — ★ CLOSED-LOOP F31_b1 ROUTE SPLIT IN GAZEBO (locked config, belief-tube, ambiguity=1)

Campaign `logs/visibility_comparison/f31b1_split_gazebo/` (run_visibility_campaign on
aws_f31b1_final_config.yaml, cleaned aws_gp_v7b, marker-off urdf, rebuilt stack). **2/2 runs
goal_reached, 0 collisions, 0 infra-invalid.** The route split emerges in CLOSED LOOP:

| cond | route | outcome | L(m) | f_shadow | belief err ē(m) | det_rate | NLL | NEES |
|---|---|---|---|---|---|---|---|---|
| C1 constant-R | blind mid-connector (global J=848, min_clear 0.063) | goal d=0.188 | 5.12 | 0.36 | 0.17 | 0.73 | 73 | 157 |
| C2 visibility-aware | observable lower-sweep (global mid J=6374 vs lower J=1189) | goal d=0.109 | 6.11 | 0.00 | 0.11 | 1.00 | 13 | 38 |

- Both REACH the goal; the difference is OBSERVABILITY/localization quality: C2 trades a longer
  path (6.11 vs 5.12 m) for staying fully observable (f_shadow 0 vs 0.36, detection 1.0 vs 0.73),
  achieving lower belief error (0.11 vs 0.17) and far better-calibrated belief (NLL 13 vs 73,
  NEES 38 vs 157). Exactly the "stay observable" thesis claim, in closed loop.
- GP-driven via the belief-tube at ambiguity_weight=1 (no inflation); both route seeds offered to
  both conditions, C1 chose blind, C2 chose observable. No route-forcing.
- Figures regenerated from this campaign: `logs/paper_figures/f31b1_markeroff_v2/`
  (paired_mechanism_taskA.pdf, compare_taskA.pdf, paths_per_seed.pdf, paper_metrics.csv, paper_summary.txt).
- PIPELINE NOTE: `make_thesis_figures.py` READS `--metrics-csv` (precomputed by
  `compute_paper_metrics.py`); it does NOT recompute. Must run compute_paper_metrics on the new
  campaign FIRST, else figures silently use a stale metrics CSV (hit this once: the first render
  showed C2 as a collision from the OLD markeroff campaign before recompute).

## 2026-06-11 — aws_gp_v7b corrected: 2 stale R0-connector training points removed + refit

Deleted 2 training samples at the REMOVED R0 mid-gap connector (row133 (-4.172,1.676) score 0.002,
row134 (-3.733,1.676) score 0.747) from the fit CSV
(`aws_gp_targets_v7b_col461/gp_targets_xy_combined.csv`, restored to active from archive), and
refit with the locked params (ls 0.90, noise 0.05, beta 0.5, grid 220x200). v7b X_train: 240 -> 238.
These points sampled the old R0 mid-gap which no longer exists (R0 is continuous); they were
previously only hidden in the plot. Pre-clean npz backed up to /tmp/aws_gp_v7b_preclean.npz.
- A1 PRESERVED (driveable): A1 (-3.0,-1.0) 0.801->0.799, (-3.0,0.0) 0.753->0.755; A2-mid 0.863->0.857,
  A3-mid 0.705->0.704, west 0.905->0.904 — all essentially unchanged. Only the NON-driveable R0 body
  (0.08->0.51) and R0->A1 gap (0.56->0.90) brightened (lost their anchors; GP interpolates from bright
  surroundings) — planning-irrelevant (keep-in forbids the rack footprint). gp_pipeline_aws_v7.pdf regenerated.
- OWED: the locked-config splits + f31b1_split figure were computed on the pre-clean v7b; re-verify
  (quick) on the cleaned GP. A1/A2/A3 rho unchanged so the splits are expected to hold.

## 2026-06-11 — RESULT: LOCKED config (belief-tube mechanism) splits all 3 discriminators at ambiguity_weight=1

Locked baseline (user-specified): `use_belief_nogo_cost: true`, `nogo_belief_kappa: 1.0`,
`ambiguity_weight: 1.0`, `goal_tightening_power: 0.9`, `goal_prior_*_std_final: 12`,
`process_noise_xy/theta: 0.012/0.05`, GP `aws_gp_v7b`, moderate command/encoder noise,
`yolo_use_masks: false` (bbox bottom-centre runtime point), `visibility_weight: 0.0` explicit.
The route mechanism is now the **belief-tube keep-in** (clearance − κ·σ_max(S_xy)), driven by
C2's learned covariance — NOT ambiguity-weight inflation.

Offline EFE route choice across the 6-task suite (`diag_route_suite.py --gfs 12`):
| task | kind | C1 | C2 | |
|---|---|---|---|---|
| F31_b1 | discriminator | blind/short (d0.10) | observable/long (d0.05) | **SPLIT** |
| b5 | discriminator | blind/short (d0.08) | observable/long (d0.26) | **SPLIT** |
| b2 | discriminator | blind/short (d0.05) | observable/long (d0.11) | **SPLIT** |
| b3 | control-like (low-band start) | observable (d0.30) | observable (d0.27) | same ✓ |
| b6 | west control | observable (d0.10) | observable (d0.04) | ctrl-ok ✓ |
| b7 | west control | observable (d0.16) | observable (d0.11) | ctrl-ok ✓ |

- **All 3 discriminators split; both controls + b3 keep both conditions on the visible route; all reach; NO stalls.**
  (The κ=2.0 belief-nogo stall recorded earlier does NOT recur at κ=1.0.)
- **Stronger than the ambiguity=18 version**, which only split F31_b1+b5 and missed b2. The belief-tube
  catches b2's mildly-camera-poor shortcut (C2 belief covariance grows there → tube clearance reacts)
  even though the ambiguity contrast was too small to deter it. Defensible: split is from learned
  covariance + belief-tube feasibility at ambiguity_weight=1, no inflation, no route-forcing.
- CODE CHECKS (user-requested) all PASS: R_visible/R_miss = diag(std²) with precision-blend on
  covariances (no rename needed); warning_band interiors = 0, no weight double-multiplication;
  runtime pixel is always bbox bottom-centre (`use_masks` only diagnostic); NIS gate is
  `threshold>0 AND nis>threshold` so 0.0 = disabled.
- Decomposition figure: `logs/paper_figures/suite_decomposition.png` (paths + per-seed cost
  breakdown). Offline open-loop only; Gazebo closed-loop still owed (approval-gated).

## 2026-06-11 — RESULT: cross-task (6-task) offline route-choice sweep

Tool: `scripts/paper_figures/diag_route_suite.py` (per-task short/long feasible seeds offered
to BOTH conditions via multistart; goal-prior `*_std_final` sweep; classifier = match realized
trajectory to nearer seed). Suite (visible start + visible goal): discriminators F31_b1, b5, b2;
controls b6 (a0→A1-low), b7 (a0→A2); b3 reclassified as control (starts in the visible lower band).
New tasks b6/b7 added to `src/experiments/config/tasks.yaml`.

| gfs | F31_b1 | b5 | b2 | b3 | b6 | b7 |
|---|---|---|---|---|---|---|
| 12 | same(blind) | same | same | both-obs | ok | ok |
| 18 | same(blind) | same | same | both-obs | ok | ok |
| 24 | **SPLIT** | **SPLIT** | same(blind) | both-obs | ok | ok |
| 30 | **SPLIT** | **SPLIT** | same(blind) | both-obs | ok | ok |

- **Split generalizes to F31_b1 + b5** at the same condition-neutral setting (goal_final_std ≥ 24);
  C2 takes the observable route (reaches d≈0.09/0.36), C1 always the blind shortcut. GP-driven:
  C1 never takes the observable route on any discriminator at any gfs. Levers = goal-prior schedule
  (condition-neutral) + feasible multistart seed (offered to both; C1 rejects observable). No
  ambiguity-weight inflation, no route-forcing.
- **b2 does NOT split** (A1→A2 shortcut only mildly camera-poor, Δp_vis≈0.09) — honest scope limit:
  the detour mechanism needs sufficient occlusion contrast.
- **b3 + controls b6/b7 always take the visible route, both conditions** — no spurious detour.
- **gfs≈24 is the sweet spot**: gfs=30 starts hurting goal-reach (b3 d=1.52, b7 d=0.99). Current
  committed `goal_prior_*_std_final` = 12 (no split).
- CAVEAT: sweep loaded the live config; `process_noise_xy/theta` were edited to 0.012/0.05 during
  the session, so a confirmation re-run of gfs=24 under the final process-noise values is owed
  before locking. Offline open-loop only; Gazebo closed-loop still owed (approval-gated).

## 2026-06-11 — RESULT: offline F31_b1 route split emerges (GP-driven, condition-neutral levers)

Tool: `scripts/paper_figures/diag_route_split_probe.py` (multistart over {mid_cross_lane,
lower_sweep_feasible} offered to BOTH conditions; goal-prior `*_std_final` sweep; F31_b1 / v7b).
Two condition-neutral levers vs the blocker found earlier: (1) a lane-FEASIBLE lower-sweep
seed (held nogo 0.0 vs the crude seed's ~134k), (2) a looser goal-prior final std (the
allowed schedule lever) so the goal-directed risk stops over-penalizing the longer route.

Result (route chosen by the EFE solve):
| goal_final_std | C1 constant-R | C2 visibility-aware |
|---|---|---|
| 18 | mid | mid |
| 21 | mid | **lower-sweep** |
| 24 | mid | lower-sweep |
| 27 | mid | lower-sweep |
| 30 | mid | lower-sweep |
| 36 | mid | lower-sweep |

- **C1 never takes the observable route** across 18–36 (even with the lower-sweep seed
  offered) → the split is NOT created by the goal-prior or the seed; both are condition-neutral.
- **C2 flips to lower-sweep at ≥21 and stays** (stable regime, not a tie-break); C2's
  lower-sweep ambiguity ~8360 < mid ~9300 → the GP/visibility drives the choice.
- At gfs=21 BOTH reach the goal via different routes (C1 d=0.11 blind-short, C2 d=0.15
  observable-long) — the contract's intended "fair visible-start→visible-goal, route differs
  only by condition" demonstration. This is OFFLINE/open-loop; the closed-loop drift/collision
  for C1 on the blind route is the Gazebo validation (still owed, approval-gated).
- Honesty: no route-forcing waypoints (seeds offered to both; C1 rejects lower-sweep), no
  ambiguity-weight inflation. Levers = multistart seed (optimizer basin handling) + goal-prior
  schedule, both condition-neutral. Current config has `goal_prior_*_std_final: 12` (no split);
  the split regime is ≥21. Proposed committed value not yet applied (methodology change to confirm).

## 2026-06-11 — offline route-split diagnosis under the new Q_d (blocker pinpointed)

Tool: `scripts/paper_figures/diag_route_split.py` (per-route, per-term EFE costs on F31_b1 / v7b;
single-seed solves + held-trajectory eval). Findings:

- **Visibility mechanism works directionally.** Held on each route, C2 sees the observable
  lower-sweep as MORE visible and LESS ambiguous than the blind mid-connector
  (mean_p_vis 0.60 vs 0.48; ambiguity 8412 vs 9719). The GP→ambiguity coupling has the right sign.
- **Goal-directed risk acts as an implicit length penalty that cancels the gain.** The longer
  lower-sweep accumulates more goal-prior-mismatch risk over its extra steps (risk 2120 vs 641),
  so risk penalty (≈+1480) slightly exceeds the ambiguity advantage (≈−1300). Net: the objective
  marginally prefers mid even on the clean terms. (Held-trajectory `nogo` is a confound — the
  crude `_controls_for_waypoints` seed is lane-infeasible: 134k lower-sweep vs 17k mid — so only
  risk/ambiguity/p_vis are clean signals.)
- **Single optimizer basin.** Both route seeds converge to connector/mid (identical cost, gap 0.0)
  for C1 and C2; there is no feasible lower-sweep local minimum the optimizer settles in.
- **Conservative behavior IS present on the shared route.** Converged: C2 stops at d=0.57 with higher
  risk/ambiguity (866/9312); C1 drives in to d=0.06 (401/7600). This is the "delay commitment / stop
  short of the blind approach" outcome — a valid thesis result even without a route split.

CONCLUSION: the route SPLIT does not yet emerge; the margin is small (~1300 vs ~1480) and the
contrast is modest (Δp_vis≈0.12). Non-cheating levers to try next (condition-neutral): (a) reduce
the goal-prior risk's implicit length penalty via the goal-prior tightening schedule / discount γ;
(b) give the optimizer a feasible (lane-following) lower-sweep seed so a clean lower-sweep basin can
be held; (c) widen the mid-vs-lower observability contrast in the GP/world (locked → recapture).
NOT allowed: route-forcing waypoints, ambiguity-weight inflation.

## 2026-06-11 — analytical Q_d process noise + union-boundary keep-in + archive sweep

- **Analytical process-noise covariance Q_d is now active.** Belief propagation uses the
  exact integrated `Q_d(θ,v,Δt)` (nilpotent unicycle Jacobian → 3-term closed form, matches
  appendix `app:heading`) in both the NumPy EKF predict and per-step in the CasADi EFE loop.
  `process_noise_xy/theta` are now read as the actuation PSDs σ_v/σ_ω (std per √s), integrated
  over Δt — the retired diagonal treated them as per-step std (no Δt, no cross-terms).
  **All single-version EFE results predating this change are superseded.** Offline F31_b1
  (v7b) delta: C1 d=0.06/total≈8001, C2 d=0.34/total≈9886 vs prior diagonal baseline
  C1 0.09/11179, C2 0.09/11823 (totals drop from the ×Δt PSD scaling; C2 stays the more
  conservative). Removed the now-dead `Q` field from `CasadiEfeParams` (+ its kwarg).
- **Test fix**: `test_unicycle_process_noise_analytical` passed `ca.MX` (symbolic) to
  `np.asarray` → crash; switched to `ca.DM` and added a `ca.Function` symbolic round-trip so
  the MX path the EFE actually uses is verified. 3/3 green; `colcon build planning unav_common` clean.
- **Keep-in obstacle cost** now measures clearance to the TRUE driveable-union boundary
  (`occlusion_geometry._get_union_boundary_segments` + `signed_distance_to_union_xy(keep_in=True)`),
  not the nearest single prism — removes phantom seam penalties. Verified on real F31_b1
  geometry (connector seam reads deep-inside, no zero). Belief-inflated clearance
  (κ·σ_max) remains OFF (`use_belief_nogo_cost: false`) → mean-only clearance today.
- **Archive sweep (newest-version-only):** moved to `_archive_nonpaper/` (reversible) —
  `aws_gp_v7` (superseded by v7b), `aws_gp_targets_v7b_col461` (intermediate),
  `aws_f31b1_final_v2`, and the tainted `aws_f31b1_final_v3`. v7b + `aws_capture_v7b_col461`
  remain. Docs reconciled to v7b: `paper_runtime_contract.yaml` (now v0.6),
  `paper_alignment.md`, `runtime_dataflow.md`, `active_research_state.md`,
  `experiment_registry.md`, `CONSISTENCY_CHECKLIST.md`. `PLANNER_HYPERPARAMETERS.md`
  process-noise rows corrected (θ 0.02→0.046; PSD-unit + family-A/B clarification).
- **New canonical doc** `docs/uncertainty_propagation.md`: three noise families
  (model vs sim-corruption vs measurement), Q_d, heading=odometry dead-reckoning (camera_xy_only
  fuses no explicit yaw; correction only via cross-covariance), GP scope (camera x,y cov only),
  and the obstacle-cost / global-planning coupling. Also lists the mis-used terms.
- Unrelated WIP (`src/sim/launch/*`, urdf, `pose_keypoints.py`, `tests/perception/*`) left untouched.

## 2026-06-11 — aws_gp_v7b promoted + Figure 2 split (current world)

- **v7b promotion complete.** All active configs (`aws_f31b1_final_config.yaml`,
  `aws_f86a_camera_xy_config.yaml`) and figure-script defaults now point to
  `aws_gp_v7b` (the v7 GP + the added A0 west-corridor column at x=-4.61):
  `make_aws_gp_pipeline_figure.py` (v5→v7b), `make_f88_stepwise.py` (v7→v7b + suptitle),
  `make_localization_pathway_figure.py` (already v7b). Regenerated
  `figures/campaign/gp_pipeline_aws_v7.pdf` and `figures/localization_pathway.pdf`
  (projected (-4.70,-1.72) vs true (-4.61,-1.67), ≈0.10 m). Filenames kept to avoid TeX breaks.
- **localization_pathway DEFAULT_IMAGE repointed** to a current-world v7b capture frame
  (`aws_capture_v7b_col461/.../000012_xy0003_h00.jpg`, true (-4.61,-1.67)); the old
  aws_simseg_v2 training image was removed in cleanup.
- **Figure 2 split (combined → two figures), both from the current world.**
  `make_aws_problem_setup_figure.py --split` now emits:
  `problem_setup_camera.pdf` (panel a, external-camera view) for the **Introduction**, and
  `problem_setup_snapshots.pdf` (panels **b,c**, top-down constant-$R_0$ rollout) for the
  **Problem Statement**. Panel (a) image = current-world v7 frame near the F31_b1 start
  (camera z=4.8; copied to `logs/paper_figures/inputs/problem_setup_panel_a_aws.jpg` so the
  figure does not depend on archived capture data — the prior panel (a) used a 2026-05-13
  pre-camera-move detector image, which is why its map did not match). Spawned
  driveable-region markers in panel (a) are acceptable (user confirmed). Lettering kept as
  (b),(c). Suggested (un-applied) TeX edits in
  `thesis-report/figure2_split_tex_suggestions.md` (no-TeX-by-default).
- **Snapshot run reverted to the ORIGINAL** (user request): panels (b,c) use the original
  constant-R C1 rollout `_archive_nonpaper/.../paper_final_v1/.../C1/seed1/experiment_20260603_091302`
  (fuller start→goal route) rather than the v3 C1 run. NOTE: this run lives under
  `_archive_nonpaper`; `make_aws_problem_setup_figure.py` DEFAULT_COV_RUN now points there.
- **Figure micro-edits (plot-only, no data/GP change):**
  - `gp_pipeline_aws_v7.pdf` panel (a): the two aggregated training dots that rendered
    inside/on the continuous R0 (leftmost) shelf are masked out of the scatter (GP fit
    untouched; downstream panels identical).
  - `localization_pathway.pdf`: removed the "separate heading branch / odometry fallback"
    box from panel (b) (heading is odometry-supplied by architecture, not a fallback;
    "$\hat\theta$ from odometry" label in panel (c) kept); and the profile `mid_cross_aisle`
    band is no longer drawn across the continuous R0 footprint (R0 has no mid gap, so no
    driveable "cross" is shown there). Caption "odometry fallback"→"odometry".

## 2026-06-11 — pred_world homography diagnostic fixed (two defects)

- **Bug A (wrong camera):** `experiment_logger` built its `ObliqueCameraModel` from
  `cam_pos`/`look_at` node parameters that the launch never passes, so it silently used
  the declared defaults `[-3,-3,6]`/`[1.5,1.5,0]`. The real camera is `[0,-5.5,4.8]` /
  look_at `[0,-1.845,0]`. **Fix:** the logger now builds the camera from the world profile
  it already loads (`load_profile` + `compute_look_at_from_pose`), the same source the
  state/planner nodes use. Offline-validated: round-trip world→pixel→world err = 0.0000;
  real detector pixel (961.46,299.23) → (3.267,-1.069), err **0.077 m** (matches the
  ~0.06 m static detector accuracy). Old default camera projected the same pixel to
  (5.906,-1.002) — the garbage that filled v3 `pred_world`.
- **Bug B (stale-install column shift):** the `aws_f31b1_final_v3` campaign ran an
  installed logger whose perception-CSV **header had 79 cols** while the data row wrote
  **81** (`state_age_s`/`state_fresh` were in the row but missing from the header),
  shifting every field after `state_yaw` by +2. That is why the real pixel `961.46`
  appeared under the `obs_yaw` column. Current src AND build are aligned 81/81 (AST-checked).
- **Consequence:** `pred_world_x/y` / `localization_error_m` in the **v3 perception CSVs are
  tainted** (wrong camera + shifted columns) and must NOT be used. No current figure depends
  on them — `make_thesis_figures` plots YOLO det/miss at the TRUE position and panel (d) uses
  `truth_belief_error_m` / `truth_state_error_m`, not `pred_world`. Runs themselves were
  unaffected (the `/state` localizer projects independently and worked, error ~0.08 m).
- Rebuilt `experiments` (symlink-install; egg-link → `build/experiments`, refreshed). Next
  Gazebo run logs correct, aligned, profile-projected `pred_world`.

## 2026-06-10 — current state snapshot (read first)

- **Locked setup:** world `warehouse_aws.world.sdf`, camera z=4.8/y=-5.5, GP `aws_gp_v7`,
  detector `aws_yolo_simseg_v2`, MAIN task `F31_b1_apron_a3_mid` (a0 = saved secondary),
  runtime = global EFE H120 + simple local tracker, `camera_xy_only` heading, `warning_band`
  keep-in no-go (w=2000). Single runtime contract = `docs/paper_runtime_contract.yaml` v0.5.
- **OPEN (F88):** F31_b1 route-split does not yet emerge — objective has no path-length
  term (`control_weight=0`, `goal_progress=0`), so both C1/C2 prefer the lower-sweep.
  Connector seam artifact fixed. No closed-loop F31_b1 split evidence exists yet.
- **Repo cleanup (pass 1+2):** non-paper material moved to `_archive_nonpaper/` and stale
  logs/docs deleted; docs consolidated (3 runtime contracts → 1). See `DEPRECATED_REGISTRY.md`.

## 2026-06-10

- Lock the external camera at **z=4.8 (+0.3), y=−5.5 (+0.6 back)** to fix corner/FOV
  coverage and open A1; the south wall (+0.4 height) + side walls + dock bumpers
  moved south with the camera so the raised wall stays behind it (no self-occlusion).
  Full GP recapture on the locked geometry → **`aws_gp_v7`** (912 frames, 647/912
  detected = 71%, driveable-only sample filter, tuned **length_scale 0.90, noise_var
  0.05, beta 0.5**). camera_pos in the artifact = (0,−5.5,4.8); consistency OK.
- **A1 made observable by hyperparameters, not de-occlusion.** The A1-mid "blind band"
  was a GP over-smoothing artifact: the R2-occluded A1-**east** lane (raw YOLO ≈0.002)
  was bleeding into the genuinely-visible A1-**west** lane (raw YOLO 0.74–0.83) via the
  1.20 m length scale. Shortening to 0.90 m respects the occlusion boundary and lifts
  A1-west ρ_plan 0.24→0.55 while keeping the genuinely-occluded A4 rack-shadow low
  (≈0.005) — faithful to the data, not inflated. (ls=0.70 over-pulled C2 to hug the
  west boundary, seg_clear=−9 mm; 0.90 is the clean choice.)
- **F87 offline rollout (aws_gp_v7, ls=0.90): Gate PASS.** C1→NW-blind reaches
  (d=0.19); **C2→south-visible drives through A1** (d=0.35, seg_clear=+0.002 clear),
  preferring the visible route over NW-blind (J 5299 < 5310). C2 went from stuck at
  d≈4.25 (old camera) to reaching via A1 — the route-split now holds with C2 actually
  traversing A1. Figures regenerated from v7: `problem_setup_camera.pdf`,
  `gp_pipeline_aws_v7.pdf`, `driveable_region_alignment.{png,pdf}`,
  `F87_offline_rollout_v7.png`.

- Paper camera-view cleanup of `warehouse_aws.world.sdf`: replace the hand-authored
  `known_driveable_green_boundary` (62 green segments, not matched to the planner)
  with a generated `known_driveable_boundary` model — **blue** outer driveable
  boundary (prism-union bbox x[-5.65,5.15] y[-3.45,4.8]) + **green** no-go region.
  The green no-go is the *complement of the driveable corridors*, not the tight
  obstacle outline: each inter-aisle column spans corridor-to-corridor (so green
  touches the driveable aisles), running the full rack band top-to-bottom but
  **split at the physical rack mid-gap** so the open R2-R5 connectors stay
  driveable (R1, whose mid-gap is filled, is one solid column). Emitted from
  `driveable_geometry_json` (column x-extents) + the world `<collision>` rack
  geometry (mid-gap split) by
  `scripts/paper_figures/generate_driveable_overlay_sdf.py` (no hand-tuned coords).
  Removed visual-only clutter: `mission_floor_markers` (disks),
  `low_floor_stock_and_apron_context` (red apron spots, black label stripes, label
  panels, charging pad, cone, low crates), `aws_staging_mesh_visuals`, the 5
  `box_spot_R*` red disks, and the `aws_pallet_jack_receiving` include. Collision
  link set is **byte-identical** (23 links) → v6 GP `geometry_json` (collision-based)
  stays valid; only the render changes, so the pending route-split GP re-recapture
  uses this clean world as its base. Alignment verified by
  `scripts/paper_figures/make_driveable_region_alignment.py` →
  `logs/paper_figures/driveable_region_alignment.{png,pdf}`. Colour semantics
  locked: blue = outer driveable boundary, green = internal no-go.
- Make the figure match the PLANNER ground truth on the rack mid-gaps: added 4
  mid-gap connector prisms (`connector_A1_A2_R2gap`, `..._A2_A3_R3gap`,
  `..._A3_A4_R4gap`, `..._A4_east_R5gap`; x = inter-aisle column, y[1.25,2.2]) to
  `driveable_geometry_json` in `aws_f86a_camera_xy_config.yaml`. R1 has NO connector
  (its mid-gap is physically filled), so A0→A1 still requires going around R1. The
  planner now allows the R2-R5 mid-gaps (driveable), matching the physical world
  and the figure. Gate re-checked (`make_f86_heading_compare.py`): route choice
  UNCHANGED vs pre-connector v6 — C1→NW (d=0.13, reaches), C2 both seeds safe-stop
  at d=4.34 (identical to before; the pre-existing camera-poor-goal finding, not a
  regression — connectors are east of the A0→A1 corridor). The SDF green no-go
  (split columns, open mid-gaps) already equals the planner no-go (complement of
  aisles+connectors), so no camera re-render was needed.

- Replace the no-go `log_barrier` penalty with a hinged-log `warning_band`
  penalty (`nogo_cost.py`). Root cause of the "nogo_weight>200 collapses the C2
  route split" failure: `log_barrier` penalizes every valid interior state, so it
  always prefers the wider aisle; raising the weight amplified this width bias
  until both C2 seeds fell into the south basin. The warning-band penalty is
  exactly zero for valid interior states (clearance ≥ b=0.05), a soft log warning
  inside the band, and a strong quadratic violation term — so `nogo_weight` (now
  2000) crushes real violations without biasing valid-route choice. Offline gate
  PASS at weight=2000 with the nogo term negligible; route choice driven by
  risk/ambiguity. The earlier `project_to_driveable` post-solve waypoint clip is
  reverted/removed. See `docs/F86_method_and_runtime_contract.md` §5.
- F86a v4 world geometry: replace the ugly 2.6 m `rack_R4_highstack_occluder`
  monolith with a sensible tapered occluding crate stack (~1.9 m) on the
  box_spot_R4 pad (still occludes the A4 lane by height); fill the R1 left-shelf
  mid gap (`rack_R1_mid` + rails + shelf boxes + DAE mesh) so the left shelf is
  one continuous barrier — a physical reason the robot cannot cut west-service →
  A1 mid-shelf. Both edits stay in non-driveable bands (no driveable prism
  touched; `driveable_geometry_json` unchanged and re-verified). World geometry
  edit invalidates aws_gp_v5 → GP recaptured as aws_gp_v6.

## 2026-05-20

- `warehouse_aws.world.sdf` is now the paper benchmark. `warehouse_occ_light` was the original candidate but superseded before seeded Gazebo validation. It is the simplest
  validated setting for showing state-dependent observation uncertainty.
- Keep `warehouse_aws.world.sdf` exploratory. It requires final geometry,
  detector retraining/validation, visibility capture, GP fitting, smoke tests,
  seeded logs, and figures before it can support a claim.
- Remove mission waypoint support. Route choice must emerge from the planner
  objective, not from a mission script that changes the goal sequence.
- Reject the AWS visible-goal route-choice probe as paper evidence. The baseline
  already took the detour-like route, while the learned condition stalled when
  ambiguity was weighted aggressively.
- Reject the AWS dark-final-goal route-choice probe as paper evidence. It mixed
  route visibility with a camera-poor final goal, making the result hard to
  interpret.
- Treat sparse planning as future work. A scientifically fair version may score
  coarse route candidates with the same objective terms, but it should not
  inject route-forcing waypoints into the local controller.

## 2026-05-27

- Move AI/research authority to `/home/joostleliveld/Thesis/CLAUDE.md`. The
  `UnembodiedNavigation` and `thesis-report` guidance files are supplements only.
- Keep broad Claude permissions for speed, but encode stronger behavioral rules:
  no destructive cleanup without explicit delete lists, no YOLO/GP recapture
  before accepted geometry, no Gazebo campaigns before offline sanity checks, and
  no paper claim without the full artifact chain.
- Retire repo-local agent prompts in favor of root agents:
  `experiment-designer`, `rollout-runner`, `planner-diagnostician`,
  `figure-analyst`, and `paper-rigor-writer`.
- Preserve multistart. It is allowed as condition-neutral optimizer basin
  handling and must be reported. It is not a mission waypoint mechanism.
- Treat long-horizon/multistart timing results as a useful diagnostic:
  they can show that the visibility-aware solution exists in the objective, but
  current solve times are a scalability limitation.
- Prefer general planner mechanisms such as goal-prior scheduling/annealing and
  normalized costs over simply increasing ambiguity weight.

## 2026-05-28

- Lock the runtime method contract in `docs/runtime_method_contract.md`.
- Define C1 as constant-observability EFE with both risk and ambiguity active.
  C1 differs from C2 by not querying the GP and by using spatially constant
  observation covariance, not by removing ambiguity.
- Define C2 as learned-observability EFE with both risk and ambiguity active.
  The GP affects planner-facing camera `(x, y)` covariance only.
- Use condition-neutral multistart as optimizer basin handling. Candidate
  generation may use the known 2D driveable floor and local maneuvers, but not
  learned visibility or the condition label.
- Use a shared 2-sigma belief-tube driveable-region log barrier for AWS
  diagnostics. Non-driveable floor is a forbidden-zone/traversability layer, not
  an observation-reliability tradeoff.
- Lock AWS Gazebo diagnostics to a robotics-faithful hierarchical runtime:
  longer global EFE route solve, short local tracker, command/encoder noise on,
  and crash/contact as terminal tracked failures.
- Permit modest lane-graph optimizer seeds generated from the known 2D
  traversability layer. This addresses supervisor feedback about local optima
  without scripting the desired visibility-aware route. The seeds must be shared
  by C1/C2 and must not use GP visibility.
- Treat any older diagnostic in which `constant_R_efe` has `ambiguity_cost=0`
  as stale for C1/C2 interpretation.

## Stable Wording Decisions

- Use `known driveable / forbidden-zone layer` for 2D planner constraints.
- Use `learned observation reliability` for the GP-derived reliability map.
- Use `3D occlusion affecting camera observations` for shelves, boxes, distance,
  perspective, and calibration effects.
- State that the GP affects camera `(x, y)` observation covariance only; heading
  is odometry-backed in the paper-facing runs.
