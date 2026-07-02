# Paper vs. Current — configurations, settings, and artifacts

Snapshot date: **2026-07-01**

This folder freezes, side by side, **everything that differs** between the
IWAI-paper campaign and the honest re-run campaign we are about to launch.

```
docs/paper_vs_current/
├── README.md                                  <- this file (the full diff + artifact provenance)
├── paper/
│   └── aws_f31b1_final_config.yaml            <- the exact paper campaign config
└── current/
    └── warehouse_visibility_campaign.yaml     <- the exact current re-run config
```

The two config files are **~95 % identical** — same world
(`warehouse_aws.world.sdf`, camera z=4.8 y=-5.5), same command/encoder noise
model, same no-go mechanism (`warning_band` belief-tube keep-in, weight 2000,
κ=1.0), same heading mode (`camera_xy_only`), and the **same four routes / five
seeds** (the tasks were only *renamed*, waypoints are byte-identical). Everything
that actually changed is listed below.

---

## 1. Perception artifacts

### 1a. YOLO detector — RETRAINED (paper `aws_yolo_simseg_v2` → current `warehouse_yolo_detector_v1`)

| | Paper | Current |
|---|---|---|
| Model path | `local_artifacts/perception_models/aws_yolo_simseg_v2/model.pt` | `logs/perception_models/warehouse_yolo_detector_v1/model.pt` |
| md5 | *(file deleted — superseded; config path still references it)* | `61d425867c1a7cb7800e50356e4bb466` |
| Base model | — | `yolo11n-seg.pt` |
| Training dataset | (contaminated: ~70 % duplicate frames, ~252 floor/rack mislabels) | `logs/perception_datasets/warehouse_yolo_dataset_v1` (541 samples, 0 dup, 268 occluded frames rejected by occlusion gate) |
| Image size | **640** | **960** |
| Runtime conf threshold | **0.10** | **0.05** |
| Epochs | — | 30 |
| Trained | (paper era) | 2026-06-17 18:40 |
| Box mAP50 | — | 0.995 |

**Why retrained:** the paper detector was trained on a contaminated dataset
(duplicate frames + floor/rack labels leaking in). That is the true cause of the
"box-bottom periphery" localization error — *not* projection geometry (red-seg
uses the identical projection and was already flat at 0.038 m). The clean
occlusion-gated retrain at imgsz 960 drops west-periphery box-bottom error from
0.127 m → 0.027 m (now flat = the red-seg baseline). Provenance for the current
model lives in `logs/perception_models/warehouse_yolo_detector_v1/`
(`manifest.json`, `results.csv`, `confusion_matrix.png`, `model.torchscript`).

### 1b. Visibility GP — REFITTED (paper `aws_gp_v7b` → current `warehouse_visibility_gp_v1`)

| | Paper | Current |
|---|---|---|
| GP path | `paper_artifacts/gp/aws_gp_v7b/yolo_score_raw_gp.npz` (archived `..._v7b_superseded`) | `paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz` |
| md5 | `97e9a2733939c54d2cf05d503f06af2d` | `673a8b29043b96e570520af647873a7e` |
| Capture run | `aws_capture_v7` | `warehouse_visibility_capture_v1` |
| Target table | `aws_gp_targets_v7b_col461/gp_targets_xy_combined.csv` | `warehouse_visibility_targets_v1/gp_targets_xy_aggregated.csv` |
| Reliability signal | raw YOLO **confidence** | calibration-invariant **accurate-detection rate** |
| GP hyperparameters | length_scale 0.9, noise_var 0.05, β 0.5, grid 220×200 | **identical** (length_scale 0.9, noise_var 0.05, β 0.5, grid 220×200) |

**Why refitted — and what did NOT change:** the GP *fitting* is identical
(same kernel, same hyperparameters, same grid, same schema). Only the **input
data** changed. The paper GP used raw YOLO confidence as its reliability target;
the v3 detector is reliable but *low-confidence-calibrated*, so a confidence
target wrongly makes visible aisles look ~73 % "miss". The refit switches the
target to the accurate-detection **rate** (calibration-invariant) captured with
the clean detector on the current camera. Field correlation with v7b is 0.84,
near-camera 0.99.

> **Note:** the paper YOLO `.pt` file itself is deleted (only the config path
> remains as a record); the paper GP is preserved under
> `paper_artifacts/gp/archive/aws_gp_v7b_superseded/`. Both current artifacts are
> present with full provenance.

---

## 2. Config knob differences (paper YAML → current YAML)

| Key | Paper | Current | Reason |
|---|---|---|---|
| `yolo_imgsz` | 640 | **960** | clean retrain trained at 960 |
| `yolo_conf_threshold` | 0.10 | **0.05** | matched to the clean, low-confidence-calibrated detector |
| `yolo_model` | `aws_yolo_simseg_v2` | `warehouse_yolo_detector_v1` | §1a |
| `gp_artifact` | `aws_gp_v7b` | `warehouse_visibility_gp_v1` | §1b |
| `pixel_correction_nis_threshold` | **0.0 (disabled)** | **9.21** | standard χ²(2 DOF, 0.99) innovation gate — rejects detector outlier spikes (NIS median ~13, up to 363) the EKF would otherwise swallow. This is a textbook, non-tuned value. |
| `pixel_correction_nis_reject_cov_scale` | *(absent → irrelevant, gate off)* | **1.0 (self-heal OFF)** | the custom "self-heal" band-aid is dropped. Its original "validation" was measured against the odom-as-truth artifact; belief is accurate vs. ground truth without it. Only the standard χ² gate is kept. |
| `global_horizon` | **120** | **75** | see below |
| `global_dt` | *(absent → 0.25 default)* | **0.4** | 75 × 0.4 = **30 s** lookahead — SAME route coverage as paper (120 × 0.25 = 30 s), but 75 steps instead of 120 (dt 0.4 vs 0.25) → ~27 % faster global solve. The global plan is one-shot/frozen (`efe_agent_node`: "chosen once and never replanned"); if it stops short of goal the local tracker drives a straight line to it, so the horizon must cover the whole route. An offline sweep (`diag/offline_horizon_check.py`, 2026-07-01) showed a 12 s cut (30 × 0.4) left the a3/a2 tasks 2.7–2.8 m short; 75 × 0.4 reproduces the paper reach (a3 C2 0.08 m, a2 C2 0.28 m). |
| task names | `F31_b1_*`, `b5_*`, `b2_*`, `b6_*` | `route_apron_to_a3_mid`, `route_apron_to_a2_mid`, `route_west_to_a1_upper`, `control_west_to_a1_low` | **rename only** — waypoints and seeds are byte-identical. |

Everything else in the two YAMLs is the same.

---

## 3. Behavioural / instrumentation differences NOT in the YAML

These matter for honesty of the results but do not live in the config file:

1. **`nogo_mode: keep_in` now actually takes effect.** Both YAMLs *say*
   `keep_in`, but in the paper campaign the runner never forwarded `nogo_mode`,
   so the runtime silently defaulted to `keep_out`. The paper's reported
   "C2 > C1" was largely a keep_out corner-cut artifact. The current runner
   passes `keep_in` through, so the re-run is a true keep_in comparison.

2. **All success/collision/clearance metrics are now GROUND-TRUTH based.** The
   paper logger derived "truth" from `/odom` (DiffDrive wheel-integrated
   kinematic odometry), which diverges from the true pose in turns (odom-vs-GT
   up to 0.53 m). That inflated in-turn "localization error" and produced
   **false-positive collisions** (all 91/505 logged collisions were
   `collision_geom` computed from odom; **0** were real physics contacts). The
   current logger subscribes to a real ground-truth pose bridge
   (`/ground_truth_tf` from `/world/warehouse_aws/dynamic_pose/info`) and
   computes every outcome (goal distance, clearance, collision, stuck, path
   length) from ground truth, with **no odom fallback** — the metric is NaN if
   GT is unavailable, never silently substituted.
   - Launch change: `bringup_sim.launch.py` adds a **separate**
     `ros_gz_groundtruth_bridge` node (an earlier inline version took down the
     whole bridge).
   - Logger change: `experiment_logger.py` adds `_ground_truth_cb`, GT columns
     (`gt_x/gt_y`, `belief_error_gt_m`, `state_error_gt_m`,
     `odom_truth_drift_gt_m`), and GT-only collision/goal/stuck/path evaluation.
   - Full metric taxonomy (with `[ODOM]/[GT]/[EST]/[PHYS]` flags) is in
     `docs/metric_definitions_and_gt_audit.md`.

   Sanity-checked honest re-run (3 seeds): 3/3 goal_reached, **0 collisions**,
   true clearance 0.44–0.55 m — vs. the old odom metric's 5/5 "collisions".

3. **The physics-contact cross-check is now functional.** In the paper campaign
   `collision_contact` was **structurally silent**: the world SDF had no
   `<sensor type="contact">` on any rack/wall, and the bridge listened on
   `/world/<w>/physics/contacts` — a topic gz-sim never publishes (it emits one
   topic *per* contact sensor). So the paper's "0 physics contacts" proved
   nothing. The current world adds a contact sensor to all 22 rack+wall
   collisions, and `bringup_sim.launch.py` now parses the world SDF at launch
   and bridges every per-sensor topic onto `/world_contacts`. Verified
   2026-07-01: stationary robot → 0 contacts (no static/ground flooding); robot
   teleported into a rack → `/world_contacts` fires the real contact. This gives
   the re-run an **independent** physics collision channel alongside the
   GT-geometric one (they cross-check each other).

---

## 4. One-line summary

Same world, same routes, same noise, same navigation method. What changed since
the paper: **(a)** a clean-retrained YOLO detector (imgsz 960, uncontaminated
dataset), **(b)** a GP refitted on detection-*rate* instead of confidence,
**(c)** a standard χ² NIS gate replacing the disabled gate + custom self-heal,
**(d)** a lighter 30 s planning horizon (75×0.4 = same coverage as the paper's
120×0.25, ~27 % faster), and
**(e)** honest ground-truth metrics replacing odom-as-truth (which is what
manufactured the phantom collisions in the first place), and **(f)** a now-working
physics-contact cross-check (contact sensors on all racks/walls + a fixed bridge)
that was structurally silent in the paper. The runner now actually honours
`keep_in`.
