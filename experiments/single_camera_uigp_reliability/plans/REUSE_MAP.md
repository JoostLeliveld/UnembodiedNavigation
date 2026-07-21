# REUSE_MAP — reuse code, run NEW data

Everything the algorithm needs already exists. This file is the authoritative
"reuse, don't reimplement" map + the exact runbook. See `NO_SHORTCUTS.md`.

## The real single-camera pipeline (corrected)

There is **no `drive_study_route` / no separate recorder** in `warehouse_aws`.
The EFE planner drives the robot to each task goal; `experiment_logger` writes
two CSVs per run; the GP-training `events.csv` is built **offline** afterwards.

```
run_visibility_campaign.py  (planner drives, experiment_logger writes
   │                         perception.csv + experiment.csv per run)
   ▼
build_belief_gp_events.py   (join perception↔experiment by stamp → events.csv)
   ▼
fit_belief_aware_gp.py      (fit GP modes; route-disjoint held-out eval)
```

(`drive_study_route.py` / `record_multicamera_views.py` are **big-warehouse
4-cam only** — not this study.)

## Reuse table (reimplementing any of these is a bug)

| need | reuse | note |
|---|---|---|
| GP fit (all modes) | `scripts/visibility_comparison/fit_belief_aware_gp.py` | `MODES=(naive, uncertainty_weighted, belief_spread, expected_kernel)` — fit all in one invocation |
| campaign run | `scripts/visibility_comparison/run_visibility_campaign.py` | `--config`, `--log-root`; self-clears zombies via `_force_fresh()` |
| events build | `scripts/visibility_comparison/build_belief_gp_events.py` | `--campaign <log-root> --out <log-root>/belief_gp_events` |
| metrics | `scripts/shared/metrics.py` | Brier/NLL/AUROC/ECE — never hand-roll |
| log loading | `campaign_metrics.load_run/load_detections` | never read `state_x/truth_x` raw from CSV |
| trust→covariance | `reliability.covariance_mapping` | single source of truth (reconciled ~1e-9); L/P conditions use this |
| confidence calibration | `reliability.confidence_calibration` | isotonic/logistic — from last session |
| trust stacking | `reliability.trust_stacker` | L4/P4 stacked τ |
| camera model | `unav_common.camera_model.ObliqueCameraModel` | projection / support region |

## U-grid ↔ implemented modes (IMPORTANT accuracy note)

`fit_belief_aware_gp.py` implements 4 modes. Mapping onto ch.03's U-grid:

| U-id | ch.03 model | status in fitter |
|---|---|---|
| U0 global constant | — | trivial (compute directly; not a fitter mode) |
| U1 point-input GP | `naive` | ✅ implemented |
| U2 point-input, larger ℓ | `naive` w/ `--gp-length-scale` bump | ✅ config of U1 |
| U3 Gaussian smoothing | ≈ `belief_spread` (small) | partial — confirm/add tiny smoothing baseline |
| U4 covariance-weighted | `uncertainty_weighted` | ✅ implemented |
| **U5 uncertain-input expected-kernel** | `expected_kernel` | ✅ **implemented — the champion** |
| U6 GT-position GP | `naive` fed eval-only `gt_x/gt_y` | evaluation-only ceiling (firewall — separate offline scorer, never operational) |

So U0/U6 are computed outside the fitter; U2 is a length-scale config of U1;
U3 needs a small smoothing baseline if not already present. Do NOT add these by
reimplementing the GP — U0 is arithmetic, U6 reuses `naive` with GT inputs under
the firewall, U2 is a CLI flag.

## events.csv schema (produced by build_belief_gp_events.py)

```
event_id, run_dir, route, condition, seed, run_id, diag_stamp, log_stamp,
matched_experiment_stamp, stamp_delta_s, m_x, m_y, S_xx, S_xy, S_yy,
sigma_major_m, sigma_minor_m, trace_S_xy, det_hit, yolo_score_raw,
yolo_detected_after_threshold, pixel_pose_available, pixel_pose_fresh,
localization_error_captime_m, state_source, eval_gt_x, eval_gt_y,
eval_belief_error_gt_m
```
`m_x/m_y` = belief mean; `S_xx/S_xy/S_yy` = belief covariance (the uncertain
input `P_t`); `det_hit` = hit label; `yolo_score_raw` = confidence; `eval_*` =
GT (**evaluation-only**). `--holdout-run-id <run_dir>` (repeatable) = the
route-disjoint held-out mechanism (reserve every run_dir under a route).

## ⚠ Data source for the reliability GP = commissioning drive, not this campaign

`DATA_SOURCE_commissioning_drive.md` supersedes the runbook below **as the
GP-training data source**. The `run_visibility_campaign.py` path below is the
EFE *navigation* stack — kept here because it is still the current-result
reproduction and a candidate replay source for Experiment C (localization). It
is NOT how Experiment A/B collect reliability-map training data (that is the
dedicated coverage drive — planner-agnostic, unbiased spatial coverage).

## RUNBOOK — navigation campaign (reproduction / Exp-C replay only; NEVER overwrite honest_campaign_v1)

```bash
# (a) preconditions
colcon build --symlink-install && source install/setup.bash
pgrep -af 'drive_study_route|efe_agent|yolo_robot_detector_node|encoder_noise_node|gz sim'   # must be empty
ls logs/perception_models/warehouse_yolo_detector_v1/model.pt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2     # P2000: help the single-threaded solve win the 270s first-cmd race

# (b) NEW capture (fresh log-root). Config bakes headless + warehouse_aws + C1/C2.
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/warehouse_visibility_campaign.yaml \
  --log-root logs/visibility_comparison/single_cam_uigp_capture_v1 --dry-run   # preview first
#   then drop --dry-run for the real run; --resume to fill gaps

python3 scripts/visibility_comparison/build_belief_gp_events.py \
  --campaign logs/visibility_comparison/single_cam_uigp_capture_v1 \
  --out      logs/visibility_comparison/single_cam_uigp_capture_v1/belief_gp_events

# (c) fit all modes on the NEW events (score target = apples-to-apples)
python3 scripts/visibility_comparison/fit_belief_aware_gp.py \
  --events   logs/visibility_comparison/single_cam_uigp_capture_v1/belief_gp_events/events.csv \
  --out      logs/visibility_comparison/single_cam_uigp_capture_v1/belief_aware_gp_score_v1 \
  --grid-from paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz --target score
#   route-disjoint: add --holdout-run-id <run_dir> for every run under the held-out route
```

## Machine reality (P2000, from diag/FINDINGS + CAMPAIGN_REPRODUCE)
- detector ~1 Hz (100 ms floor); global EFE solve single-threaded ~120–220 s.
- do NOT shorten `run-timeout` (420 s) / `first-cmd-timeout` (270 s).
- full config = 40 runs (C1+C2 × 4 tasks × 5 seeds) ≈ multi-hour; pilot a
  trimmed `tasks:` copy first to validate plumbing, then scale — a pilot is a
  real run, not a shortcut, but the reportable result needs the grouped design.
- outputs must stay under `logs/visibility_comparison/`.
