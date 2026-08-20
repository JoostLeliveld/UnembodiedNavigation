# Metric definitions & ground-truth audit (2026-07-01)

> This is the ground-truth/column audit. For current versus historical projection contexts,
> camera-measurement versus belief metrics, run IDs, and allowed statistical comparisons, use
> [`localization_metrics.md`](localization_metrics.md).

**Headline:** the experiment's reference pose `truth_*` is **`/odom` = DiffDrive wheel
odometry**, not the true robot pose. Wheel-odom drifts (worst in turns). So **every
metric measured against `truth_*` is contaminated by odom drift** — most importantly
the **collision** metric, which flags collisions from the *odom* position penetrating a
rack while the true robot is clear. Ground-truth logging (`gt_*`, `*_gt_m`, from
`/ground_truth_tf`) was added 2026-07-01 to give honest references.

Legend: **[ODOM]** = uses `truth_*` = `/odom` (contaminated) · **[GT]** = true Gazebo
pose (correct) · **[EST]** = an estimate (not a reference) · **[PHYS]** = real physics.

---

## References / poses
| column | definition | note |
|---|---|---|
| `truth_x/y/yaw` | `/odom` (DiffDrive wheel odometry) → `map_bev` frame | **[ODOM]** the experiment's "truth"; **drifts, esp. turns** |
| `gt_x/y`, `gt_available` | `/ground_truth_tf` (true Gazebo pose, `dynamic_pose/info`) | **[GT]** the real pose (verified: matches spawn 0.000 m) |
| `odom_x/y`, `odom_yaw` | raw `/odom` (odom frame) | wheel odometry |
| `odom_noisy_x/y` | `/odom` + injected encoder noise | EKF motion-prediction input |
| `planner_belief_x/y/yaw` | planner **EKF belief** (predict `/odom_noisy` + correct pixel) | **[EST]** — the controller navigates by this. Accurate ~0.05–0.10 m vs GT |
| `state_x/y` (`/state/bev`) | **camera-only** per-frame world pos (pixel→homography) | **[EST]** legacy/diagnostic, **NOT used by planner** (camera_xy_only) |
| `pred_world_x/y_calibrated` (historical perception.csv) | YOLO bbox-bottom → ground → retired fitted correction | **[EST]** historical camera measurement; never use its old ~0.03–0.06 m diagnostic as current IPM accuracy |

## Localization / belief error metrics
| column | definition | flag |
|---|---|---|
| `truth_belief_error_m` | `‖truth − planner_belief‖` (logger:2045) | **[ODOM]** inflated by odom drift in turns |
| `truth_state_error_m`, `state_pos_error_m` | `‖truth − /state/bev‖` | **[ODOM]** |
| `localization_error_m` / `_calibrated_m` | `‖camera_pred − truth‖` at log time | **[ODOM]** latency+drift inflated |
| `localization_error_captime_m` | `‖camera_pred − truth‖` at capture stamp | **[ODOM]** removes latency, **not** odom drift |
| `belief_error_gt_m` | `‖planner_belief − gt‖` | **[GT] ✅ honest belief error** |
| `state_error_gt_m` | `‖/state/bev − gt‖` | **[GT] ✅** |
| `odom_truth_drift_gt_m` | `‖truth(/odom) − gt‖` | **[GT] ✅ measures the drift itself** (0 at spawn → grows in turns) |

## Uncertainty / calibration
| column | definition | flag |
|---|---|---|
| `mean_loc_nll`, `mean_loc_nees`, `mean_overconf` (paper_metrics) | Gaussian NLL/NEES of the belief error **vs truth** (compute_paper_metrics:281) | **[ODOM]** "overconfidence" is really belief-vs-odom mismatch |
| `state_cov_*`, `planner_cov_*`, `*_sigma_*`, `*_entropy_*` | belief covariance (self-reported) | [EST] fine (no reference), but NEES that pairs them with `truth` is [ODOM] |

## Code audit — odom-as-truth removed everywhere (2026-07-01)
Root cause: the logger's `_latest_truth_pose()` reads `/odom`, so every column
named `truth_*` (truth_x/y, truth_belief_error_m, truth_state_error_m) is ODOM,
not ground truth — and `/odom` drifts up to ~0.4 m from the true pose. All
current readers now use the GT columns; there is NO odom fallback.

FIXED (now ground-truth only):
- `make_paired_mechanism.py` — error panel AND executed trajectory → gt_x/gt_y,
  belief_error_gt_m; **no fallback** (hard-fails if a run lacks GT).
- `compute_paper_metrics.py` — loc error, NLL, NEES, overconfidence, ρ-along-path,
  and the summary mean → GT (gt_x/gt_y, mean_belief_error_gt_*).
- `experiment_logger.py` — collision/clearance/goal/stuck/path already GT; ADDED
  GT error means (mean_belief_error_gt_m + after-first-cmd) and GT heading
  (gt_yaw + belief_yaw_error_gt_rad columns); corrected comments that wrongly
  called `truth_*` "ground truth".
- `run_visibility_campaign.py` — campaign_log carries mean_belief_error_gt_m.
- `make_robustness_spread.py`, `make_aws_problem_setup_figure.py`,
  `make_problem_setting_figure.py` — plotted paths/error tube → gt_x/gt_y.

STILL odom-based — flagged, NOT auto-fixed (historical / diagnostic):
- `scripts/visibility_comparison/diag/*` (diag_belief_drift, diag_route_animation,
  diag_side_by_side, analyze_run, prove_c2_limit, compare_gate, plot_turn_rootcause,
  plot_failure_story, diag_common) run on OLD keepin_clean data that has NO GT.
  **Consequence:** the "why C2 ≠ 100 %" conclusions built from truth_belief_error_m
  (belief err "0.05→1.8 m in turns") are odom-contaminated and must be re-derived
  on GT data before use.
- `showcase_failure_frames.py` + the perception node's `localization_error_m`:
  detector accuracy is computed vs `true_x`=/odom (capture-time, so odom≈GT to
  within instantaneous drift ~1–6 cm, but technically should be GT — perception
  node would need to subscribe to /ground_truth_tf).
## RENAME — the misnamed `truth_*` columns are gone (2026-07-01)
To make the mistake impossible to repeat, the odom-as-truth columns were renamed
at the source (logger) and across all 19 consumers (275 whole-word renames):

| old (misleading) | new (honest) | meaning |
|---|---|---|
| `truth_x/y/yaw/stamp/available` | `odom_map_x/y/yaw/stamp/available` | /odom expressed in the map frame (= raw `odom_*` + spawn offset) |
| `truth_belief_error_m` | `belief_error_odom_m` | ‖belief − odom_map‖ (parallels `belief_error_gt_m`) |
| `truth_state_error_m` | `state_error_odom_m` | ‖state − odom_map‖ |
| `mean_truth_*_error_*` | `mean_*_error_odom_*` | run_summary means |
| `yaw_error_truth_{odom,state,belief}_rad` | `yaw_error_odom_map_vs_{…}_rad` | heading diffs vs odom_map |
| `odom_truth_drift_gt_m` | `odom_map_gt_drift_m` | ‖odom_map − GT‖ (the odom-drift diagnostic) |

Naming convention going forward: **`gt_*` = ground truth (the only valid
reference); `odom_map_*` / `*_error_odom_m` = the /odom stream, an odom-drift
diagnostic ONLY — never a truth reference.** No active code reads an `odom_*`
column as truth. Verified: `grep truth` in active code returns only prose
labels/docstrings, no data-column references. Old CSVs still carry the old names
(historical), but the honest `gt_*` columns in them are what all readers use.

## Collision / safety  ← the decisive contamination
| column | definition | flag |
|---|---|---|
| `collision_contact` | real Gazebo physics contact on `/world_contacts` (logger:1502, bridge_contacts=true) | **[PHYS]** correct signal, but was **structurally silent** in the paper/old campaign (0/505 was uninformative, NOT "never touches" — see below). **FIXED 2026-07-01**: contact sensors added to all 22 rack/wall collisions + bridge repaired; now a real independent channel |
| `collision_geom` | `obstacle_penetration > 0` from **`truth_x/y`** vs rack geometry (`_geometry_safety_at_truth`, logger:1516) | **[ODOM] ← the false-collision source**; fires when odom drifts into a rack |
| `collision_any` | `collision_contact OR collision_geom` | **[ODOM-dominated]** — all 91/505 collisions are geom-only |
| `min_obstacle_distance_m`, `obstacle_penetration_m` | signed dist from **`truth_x/y`** to obstacle prisms − robot_radius | **[ODOM]** |
| `min_wall_distance_m`, `wall_penetration_m` | same from `truth_x/y` to wall prisms | **[ODOM]** |
| `inside_no_go` | `obstacle_penetration_m > 0` (from odom) | **[ODOM]** |
| `off_map` | `truth_x/y` outside world bounds | **[ODOM]** |

## Outcome
| column | definition | flag |
|---|---|---|
| `goal_dist`, `min_goal_distance`, `final_goal_distance` | distance to goal from `truth_x/y` | **[ODOM]** (goal-reaching is less drift-sensitive than collision, but still odom-based) |
| `goal_reached`, `is_clean_success` | goal region entered / held | derived from goal_dist **[ODOM]** |
| `is_collision` / campaign "collisions" | `collision_any` | **[ODOM]** → the C1-vs-C2 & paper collision numbers are odom-drift artifacts |

## Control / plan (not reference-based, fine)
`cmd_v/w`, `exec_wp_*`, `plan_*`, `optimizer_*`, `efe_*`, `p_vis_*`, `solve_time_ms`,
`planner_pixel_correction_age_s` — self-consistent (no external reference); trustworthy.
The global plan (`global_plan.csv`) is the planner's actual output (not an estimate) —
verified at the aisle **center** (x≈−3.03), clear of racks.

---

## What must change (the base fix)
Recompute all **[ODOM]** metrics against **`gt_*`** (now logged) or real
`/world_contacts`:
- **collision/clearance** → from `gt_x/y` (or `collision_contact`), not `truth_x`.
- **belief/loc error, NLL/NEES** → vs `gt`, i.e. use `belief_error_gt_m`.
- Then re-run the campaign: collision counts should largely collapse (robot stays
  ~0.5 m clear; the "collisions" were odom drifting ≥0.5 m).

## Sanity checks performed (all pass)
- **GT correct:** `gt_*` matches known spawn (−5.25,−0.75) to 0.000 m (3/3 seeds).
- **Frames consistent:** `‖belief−odom‖` == `‖(belief−gt)+(gt−odom)‖` exactly (triangle).
- **Belief accurate vs GT:** 0.05–0.10 m incl. at the "collision" (belief-vs-GT 0.095 m).
- **Odom drift grows:** 0→0.03 m (near-idle seeds) vs 0→0.53 m (loaded seed that "collides").
- **Real-contact channel was STRUCTURALLY SILENT** in the paper/old campaign, so its
  0/505 proved nothing: (a) the world SDF had NO `<sensor type="contact">` on any
  rack/wall, and (b) the bridge subscribed to `/world/<w>/physics/contacts`, which
  gz-sim never publishes (it emits one topic PER contact sensor). **FIXED 2026-07-01**:
  added a contact sensor to all 22 rack+wall collisions and rebuilt the bridge to parse
  the SDF and bridge every per-sensor topic → `/world_contacts`.
- **Forced-contact fire test PASSED (2026-07-01):** stationary robot → 0 contacts (no
  static-static/ground flood); robot teleported into rack R1 → `/world_contacts` fires
  `warehouse_rack_occluders::rack_R1_lower::collision`. So the honest re-run's
  `collision_contact` is now a genuine independent physics cross-check of the GT-geometric
  channel, not a structural zero.
