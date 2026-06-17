# Superseded Handoff

This historical handoff is retained for traceability only. It may reference
archived detector artifacts and should not be used as the current workflow.

# Handoff — visibility-aware EFE navigation (v3), figure generation

You are continuing work on a thesis/IWAI-workshop project: a turtlebot3_burger in an
AWS warehouse Gazebo world, localized by an EXTERNAL overhead camera (YOLO detector →
homography → EKF belief). Two conditions are compared:
- **C1 = constant_R_efe** (baseline, assumes constant camera reliability)
- **C2 = visibility_aware_efe** (uses a learned GP camera-reliability model in the EFE
  predictive covariance, so it prefers observable routes)

Working dir: `/home/joostleliveld/Thesis/UnembodiedNavigation`. Read the persistent
memory in `~/.claude/projects/-home-joostleliveld-Thesis/memory/` first — especially
`project_localization_and_controller_fix.md`, `project_gp_detection_rate_methodology.md`,
`project_stale_install_gotcha.md`, `project_campaign_failure_diagnosis.md`.

## Immediate task
Produce per-task C1-vs-C2 "paired-mechanism" figures (like the paper's
`paired_mechanism_taskA.pdf`) for all 4 tasks (F31, b5, b2, b6) — 3 panels: C1 route map,
C2 route map (over the GP reliability field), and truth–belief error with 2σ radius.
Then a full 40-run campaign for statistics. **A decision is pending (see below) — do NOT
mass-produce figures until the user picks a direction.**

## What is DONE and VERIFIED (do not redo)
1. **Detector retrained** on the current camera (`local_artifacts/perception_models/
   aws_yolo_simseg_v3/model.pt`, mAP50 0.888). The v2 detector was out-of-distribution
   (camera moved z4.5→4.8 after training).
2. **GP** switched from raw-confidence to a **calibration-invariant accurate-detection-rate**
   field (`paper_artifacts/gp/aws_gp_v3/yolo_score_raw_gp.npz`, corr 0.84 with old v7b,
   near-camera reliability 0.99). v3 detector is reliable but LOW-confidence-calibrated, so
   raw score understated reliability.
3. **Localization** — found a LATENT BUG: `bev_y_calibration_offset_m` was plumbed to the
   logger+planner but NOT to `pixel_params` (the projection node), so the state node ran
   with 0.0 = NO correction. Fixed + added a position-dependent **affine** calibration
   (`bev_affine_calibration` param, 6 floats, fit on the teleport grid). Static error
   0.074→0.023m. Plumbing: `pixel_to_bev_state_node.py`, `visibility_launch_common.py`
   (PAPER_LAUNCH_DEFAULTS + cfg + pixel_params), `run_visibility_campaign.py` passthrough,
   config.
4. **Controller recovery fix** in `efe_agent_node.py::_simple_plan_safe_to_execute`: the
   old hard `clearance<0 → (0,0) stop` gate deadlocked when a transient EKF-prediction
   excursion during a hard turn put the BELIEF mm-outside the narrow keep-in band (truth
   stayed centered). New gate returns `n_safe` (leading safe steps), rejects a step only if
   it drives an already-negative clearance strictly WORSE than the plan start; executes the
   safe prefix. Only safe BECAUSE localization is now tight (it drifted into walls when loc
   was bad).
5. **NIS outlier gate** enabled: `pixel_correction_nis_threshold: 9.21` (chi-squared 2-DOF
   99% innovation gate — standard validation/Mahalanobis gating, cite Bar-Shalom). Rejects
   detector outlier measurements (normal NIS<8, outliers NIS 13+). Implemented already in
   `unicycle_planner_node.py` ~line 1703.

With all of the above (config `scripts/visibility_comparison/aws_f31b1_v3_config.yaml`),
seed-0 results: **F31, b5, b6 reach goal; b2 fails** (its goal is in total shadow → 0
useful detections → belief diverges → collision; user said that's acceptable, the
controller is intentionally dumb/one-shot, no replanning).

## THE KEY FINDING (the open problem)
With localization CORRECT, the figures show **C2 with HIGHER belief error than C1** on
most tasks — the opposite of the thesis:
| task | C1 mean err | C2 mean err | note |
|---|---|---|---|
| F31 | 0.05 | 0.11 | C2 worse |
| b5  | 0.10 | 0.16 | C2 worse |
| b2  | 0.15 | 0.36 | C2 fails (shadow) |
| b6  | 0.09 | 0.08 | C2 marginally better |

Reason (structural, not an artifact): **the GOAL sits in a low-observability region for
both conditions.** C2 correctly picks the longer, MORE-observed route (detection rate 96%
vs C1 87%) — the mechanism works and the route maps show it — but because C2's route is
LONGER it spends more time approaching the unobserved goal, where its error grows. C1's
shorter route reaches the same shadowed goal faster with less total exposure.

IMPORTANT: the user's ARCHIVED figure (`paper_artifacts/figures/_archive_nonpaper_20260612/
f31b1_markeroff_v2/paired_mechanism_taskA.pdf`) cleanly showed C2 WINNING — but that was
LARGELY the (now-fixed) localization bug, which inflated C1's error more. The genuine
"error grows when unobserved" mechanism (EKF dead-reckoning: no detection → odom-only →
error AND 2σ covariance grow together → snap back on update) IS real and present in v3 for
both conditions (e.g. v3 C1 goes unobserved in its occluded upper stretch).

**Pending decision (user is choosing):** (1) reframe the figure claim around route-choice /
observability (true & verified) rather than lower error; (2) build ONE discriminating task
(goal in an OBSERVED spot, C1's shorter route cutting through an avoidable shadow C2 avoids)
— the clean "C2 wins" figure (recommended, matches the supervisor's "longer-but-visible vs
shorter-but-shadowed" critique); or (3) report the honest negative result. The user was
mid-review of the 4 figures when this handoff was written.

## How to run things
- Figure data campaign (C1+C2, seed0, 8 runs): `aws_f31b1_v3_fig.yaml` (regenerate from
  the main config so it inherits all fixes). Run via the campaign runner with a LONG
  timeout: `python3 scripts/visibility_comparison/run_visibility_campaign.py --config <cfg>
  --log-root <dir> --run-timeout 700 --first-cmd-timeout 400`. LAUNCH IT VIA THE HARNESS
  `run_in_background: true` Bash param (reliable); do NOT use `nohup ros2 launch ... &`.
- Generate a figure: `PAIRED_CAMP=<logdir-basename> PAIRED_TASK=<task> PAIRED_SEED=0
  PAIRED_OUT=paired_mechanism_taskX_v3 PAIRED_COPY_TO_THESIS=0
  PAIRED_GP=paper_artifacts/gp/aws_gp_v3/yolo_score_raw_gp.npz
  python3 scripts/paper_figures/make_paired_mechanism.py`. View PDFs with `pdftoppm -png`.
- Offline (no-Gazebo) global-solve check: `scripts/visibility_comparison/offline_solve_check.py`.
- Tasks/conditions: 4 tasks × C1/C2 × 5 seeds = 40 runs in the main config.

## WHAT TO BE WARY OF (these cost hours)
1. **STALE BUILD (worst):** the runtime imports nodes from the egg-link target
   `build/<pkg>/<pkg>/...`, NOT src. An incremental `colcon build --symlink-install` does
   NOT reliably refresh that copy. After ANY src edit under `src/*/`, you MUST
   `rm -rf build/<pkg> install/<pkg> && colcon build --packages-select <pkg> --symlink-install`,
   THEN verify the actual file:
   `tgt=$(head -1 install/<pkg>/lib/python3.10/site-packages/<pkg>.egg-link); grep -c "<marker>" "$tgt/<pkg>/nodes/<file>.py"`.
   Packages touched: `state`, `experiments`, `planning`.
2. **CONTENTION → flaky `warm_or_cold` divergence:** the campaign's OWN perception+solve
   (Gazebo 1.5c, efe_agent solve 2.4c during one-shot global solve, YOLO/pixel_to_bev/logger
   ~2.5c) saturates the machine. Under load the global solve is slow, loses the
   first-command-timeout race, and falls back to `solver:warm_or_cold` (risk huge, wrong
   route → stuck). This is the original "instability." Mitigate: close firefox/heavy tabs
   (frees ~1 core), use long timeouts, run one thing at a time. F31/C2 and b2/C2 flip
   goal↔stuck run-to-run because of this — re-run them when the load is low. Check load with
   `uptime`; the campaign nodes show via `ps -eo pid,pcpu,comm --sort=-pcpu`.
3. **Bare `nohup ros2 launch ... &` dies empty / writes no log.** Use harness
   `run_in_background: true`. And `bringup_sim.launch.py` needs `headless:=true` or the
   camera never renders / the bridge hangs (do NOT bridge the segmentation topic).
4. **Detection-rate counting:** the CSV column `yolo_detected_after_threshold` is "1.0" not
   "1" — parse as `float(v)>=0.5`, not string equality, or you'll get a false 0% detection.
5. **`pkill ...` returning exit 1** (nothing matched) makes the WHOLE compound Bash command
   report "Exit code 1" even though the later commands ran. Run pkill separately or ignore
   the exit code. Also `pgrep -f "<name>"` self-matches your own command line — count real
   procs with `pgrep -x ruby` (Gazebo) / `ps -eo comm | grep -c gz-sim`.
6. **NIS gate trade-off:** it assumes a consistent filter. In small-covariance (well-
   localized) regions NIS inflates and it can over-reject; and once the belief has drifted,
   a CORRECT measurement looks like an outlier (high innovation) and gets rejected, keeping
   the belief stuck. 9.21 made 0 spurious rejections in the runs checked, but watch for it.
7. **`offline_solve_check.py` is NOT perfectly faithful** to the online setup (different
   route seeds / goal-prior handling → term_gd 2–4m even for tasks that reach goal online).
   Trust it for RELATIVE/convergence conclusions, not absolute terminal distance.
8. **GP detection-rate vs accuracy:** the GP encodes detection RATE, not localization
   ACCURACY. C2 prefers "where am I detected," not "where am I detected PRECISELY" — far/
   oblique positions get detected but localize imprecisely (the source of C2's error spikes).
   If the user wants C2 to genuinely win on error, the principled change is a GP over BEV
   measurement covariance/accuracy, not detection rate (the user has NOT approved this).
9. **Locked config / no param-searching** (`feedback_lock_in_no_search`): apply the user's
   decided values; fix genuine BUGS, but don't go on exploratory parameter sweeps.

## Paper context
The paper is already SUBMITTED; results can change only for camera-ready/arXiv/rebuttal.
Disclosure notes for the detector + GP improvements are in `docs/camera_ready_notes.md`.
Generated figures so far: `paper_artifacts/figures/paired_mechanism_task{A,B,C,D}_v3.pdf`
(A=F31, B=b5, C=b2, D=b6).
