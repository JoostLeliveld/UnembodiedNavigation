# D4 demo — camera method (per-camera projection bias + calibration)

First instance of the demo grammar in `research_story/DEMO_LAYER_PLAN_2026-07-16.md`
(P1 problem happening / P2 mechanism overlaid / P3 with-vs-without paired /
P4 verdict anchored / one animation), applied to the projection-bias fix from
`GATE_PROVENANCE.md`. Chosen first because it needed zero new data collection
— everything below is rendered from the two GT-attached commissioning runs and
the frozen `projection_calibration_v2` artifact already on disk.

## Run it

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
python3 experiments/multicamera_commissioning_bigwarehouse/tools/render_d4_camera_method_demo.py
```

No Gazebo, no ROS. Outputs (gitignored, regenerate on demand) go to
`logs/studies/multicamera_commissioning_bigwarehouse/demos/d4_camera_method/`:
`p1_problem_happening.png`, `p2_p3_bias_before_after.png`,
`p4_verdict_anchored.png`, `manifest.json` (claim + data-source labels + gate).

## What each panel shows

- **P1** — one real instant (t=58.0s, run 2, central overlap sweep): cameras
  B, C, D each report the SAME robot pose differently, spread over ~32 cm,
  against the ground-truth star. Establishes the problem before any method
  appears.
- **P2 / P3** — bias arrows (reported → truth) across run 2, all four
  cameras, before and after `projection_calibration_v2`. Mean arrow length
  drops 16 cm → 4 cm. Arrow directions confirm the mechanism: each camera
  pulls its estimate toward its own wall (A/C south, B/D north).
- **P4** — the actual verdict: C↔D synchronized disagreement, raw vs
  calibrated, both runs, plotted against the frozen 0.30 m D2 gate and a
  TB3-footprint-width scale bar. Reproduces `GATE_PROVENANCE.md`'s
  0.247 → 0.078–0.107 m result as a picture instead of a table cell.
- **Animation** — not built yet. Needs a full-route frame sequence (raw vs
  corrected arrows evolving as the robot moves), which is cheap once a
  dedicated calibration/qualification route is recaptured (research plan
  Module 2) — the current two runs are short single passes, not enough
  frames for a convincing sweep.

## Implementation notes for whoever extends this pattern

- All projections are recomputed from raw `obs_u/obs_v` pixels via
  `reliability.projection`, never read from a CSV's `pred_world_x/y` — those
  columns may carry whatever calibration (or none) was active when the row
  was recorded. This is the same discipline `fit_projection_calibration.py`
  uses; the figure and the number it illustrates cannot silently diverge.
- Run 1 (`gt_validation_smoke_20260716`) predates the per-camera GPU-device
  fix and has **zero camera_A rows** — don't reuse it for an all-four-cameras
  panel. Run 2 (`gt_validation_smoke2_20260716`, central_overlap_sweep) has
  all four active in one shared aisle and is the right source for P2/P3.
- Ground truth is read only from `evaluation_inputs/*_perception.csv`
  (GT-eval-only, produced by `record_evaluation_truth.py` +
  `attach_evaluation_truth.py`) — never a model input, consistent with the
  leakage firewall.
- `experiments/demos/` already exists in this repo for an unrelated purpose
  (single-camera-campaign README media, `outcome_counts_by_condition.png`
  etc.) — do not repurpose it. This demo's code lives in this study's own
  `tools/`, outputs under this study's own `logs/studies/.../demos/`. See the
  correction note at the top of `research_story/DEMO_LAYER_PLAN_2026-07-16.md`.
