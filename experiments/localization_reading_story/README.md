# localization_reading_story

**Separated on purpose.** This folder owns ONE job: explain, in pictures, how the camera
reads the robot's position — every method tried, what each one bought, and what the
current reading means for the filter and the planner. It produces presentation and
thesis figures. It does not re-run captures and it does not touch any other study.

Outputs go to `logs/studies/localization_reading_story/`. Scripts stay local to this
folder until a figure is promoted.

> NOT YET REGISTERED in `research/registry.yaml`. Register a chapter + claim before any
> number here is quoted as evidence. Nothing in this folder is a new measurement —
> every figure recomposes an existing, dated result.

---

## Part 1 — the method ladder (build this first; it is the presentation)

Every row was measured. Scored **on the same images** where the table says so, so this is
a like-for-like ladder, not a comparison across captures.

| # | how the camera reads position | typical miss | systematic part | gives heading? | evidence |
|---|---|---:|---|:--:|---|
| 1 | bottom of the detected box → floor (**deployed**) | **7.78 cm** | **−6.34 cm, same direction every time**; swings −2.76→−8.85 cm with heading | no | `logs/studies/keypoint_measurement/RESULTS.md` |
| 2 | fitted projection corrections (v2/v3/v4) | — | RETIRED: every fitted correction was worse than none | no | `experiments/legacy_projection_corrections.py`, `logs/studies/pixel_ground_path/` |
| 3 | plain IPM, zero parameters (current pixel→ground) | — | the honest baseline; no fitted terms survive | no | `logs/studies/pixel_ground_path/e7_ipm_zero_parameter` |
| 4 | mesh / silhouette bias correction on the box bottom | — | lean 9.50 → **1.93 cm** | no | `logs/studies/filter_notebook/RESULT_learning_r_after_the_bias_fix_2026-08-14.md` |
| 5 | offset-state filter (estimate the offset as state) | — | recovers offsets **GT-free** to 0.7–2.4 cm | no | `logs/studies/offset_state_model/` |
| 6 | marked point, model trained for the *old* camera | 1.51 cm | −0.42 / −0.28 cm | yes | `logs/studies/keypoint_measurement/` |
| 7 | marked point, **retrained for this camera** | **0.74 cm** | **−0.02 / −0.41 cm** | **yes, 3.8° median** | `logs/studies/keypoint_measurement/RESULTS.md` |

**The headline for the slide:** 7.78 cm → 0.74 cm, a factor of ten, and the systematic part
goes from −6.34 cm to a few millimetres — *without a calibration step*. Mean pixel residual
is (+0.09, +0.16) px, i.e. already within a fifth of a pixel of zero.

**The one-sentence reason rows 2–5 exist:** the box bottom is not a fixed point on the robot.
Where the bottom edge of a silhouette sits depends on orientation and range, which is why its
lean swings by 3× with heading and why every attempt to correct it (2, 3, 4, 5) was fighting a
symptom. A marked point *is* fixed to the robot, so rows 6–7 remove the cause instead.

### Figures for Part 1 — **BUILT 2026-08-18**, in `logs/studies/localization_reading_story/figures/`
- **L1** `L1_method_ladder.png` — the ladder: the three like-for-like readings as bars split
  into systematic and random, next to the seven-row ladder in which rows 2–5 are marked
  "measured on other data" (they are corrections to row 1, not replacements).
- **L2** `L2_why_the_box_bottom_leans.png` — the mechanism slide. Lean by world heading, then
  the sharper version: resolved along the camera's line of sight and binned by how the robot
  faces *relative to that line*, the box bottom always reads too close to the camera, by
  **+2.2 cm facing it and +7.9 cm driving away**. Panel C is the diagram: the body's visual
  centre sits 3.2 cm behind `base_link` and turns with the robot, which predicts a **6 cm**
  swing between those two headings against **5.7 cm** measured.
- **L3** `L3_what_the_camera_sees.png` — one held-out pose (#119, 5.9 m, robot 24 × 25 px):
  the frame, the magnified crop with both readings' pixels, and the same pose on the floor
  (7.89 cm vs 0.18 cm).
- **L4** `L4_error_over_the_floor.png` — error maps on one colour scale, plus what is left of
  the marked point (clean readings only: at most half a centimetre, and no useful range trend).

---

## Part 2 — what the new reading means for R (redo the separation on keypoints)

The whole bias/variance/availability argument was developed on the box-bottom reading. It has
to be re-derived on row 7, because two of its premises **changed**:

| premise, box-bottom era | on the keypoint reading | consequence |
|---|---|---|
| R is anisotropic, aspect 1.1–2.0 (up to 3.1) | **still anisotropic — 1.45:1 in the CAMERA frame** (0.88 along the ray × 0.61 across, n=369 clean). The map-frame 0.80 × 0.73 is pooling over a ±35° ray fan, not roundness | the anisotropy argument SURVIVES, and matches aspect = slant/height (1.42 at 4.8 m → 1.77 at 11 m). Measured 2026-08-18 |
| the bias dominates R by 8× | bias is **−0.30 cm along the ray and near-constant**; the −1.10 cm "range trend" was single-marker readings (−1.79 cm, sd 3.21) contaminating the far bins | **the framing flips WITH the reading method** — see the box below. On keypoints R is the state-dependent part; on the box bottom the bias is. Both measured 2026-08-18 |
| σ_px = 0.5 px, total R 6.4× too small | **gap gone.** Pushing the measured 4×4 pixel covariance (front/rear correlation −0.40 in `u` included) through the reading's own Jacobian gives measured/predicted = **0.94** per reading, 1.42 cm² predicted vs 1.15 cm² measured, and it predicts the **shape** too (1.32 vs 1.45 along/across). Measured 2026-08-18, figure P4 | the old 6.4× compared a variance model against a bias-contaminated target. With the bias at 0.27 cm the comparison no longer depends on that choice: including the bias moves the measured total 1.15 → 1.24 cm² |

**The live question this feeds (the supervisor thread).** On this reading R is **small
(0.88 cm along the ray × 0.61 across), anisotropic along the ray, and much the same size
everywhere on the floor**; the bias is **0.27 cm and near-constant** once readings with a
hidden marker disk are kept out. So neither term has much spatial structure — a *map* of R
is close to uninformative for planning — while `p_use(x)` has plenty: **28% of floor cells
this one camera can never read at any heading, 56% readable at all twelve**. That is where
the planning-relevant structure now lives:

```
sigma*(x)^2 = floor(x)^2  +  q_rate / (f * p_use(x))
              does not shrink   shrinks with every observation
```

That is the `achievable_precision_map` model. Its mechanism stands; its numbers are
historical-v2 (a 77 mm floor from the box-bottom era) — **quote the model, not those numbers.**

### Figures for Part 2 — **BUILT 2026-08-18**, in `logs/studies/localization_reading_story/figures/`
- **P1** `P1_bias_versus_noise.png` — systematic against random per range band, with the
  random part also drawn after 4 and 16 looks and the number of looks after which averaging
  stops paying (22 at 5 m, **1–3 past 7 m**). Panel B is the correction that matters: the
  pooled far-range pull is mostly readings with a hidden disk (**−2.18 cm at 8–9 m, −3.46 cm
  at 9–12 m**, a fifth to a third of readings there), while clean two-disk readings stay
  inside 1 cm everywhere.
- **P2** `P2_r_over_the_floor.png` — block ellipses for both readings at one scale, and then
  the axis-free test: along-sight against across-sight spread per block, against the band a
  genuinely round R would produce at that sample size. Pooled **1.45:1, p = 7×10⁻¹³**;
  8 of 9 blocks elongated along the ray; the pixel geometry predicts 1.36. **Not round** —
  the world-axis 0.73 × 0.80 cm was a frame artefact of pooling over a fan of sight lines.
- **P3** `P3_achievable_precision.png` — `p_use(x)` measured from this capture (280 cells ×
  12 headings, a heading counting only if both disks rendered), `sigma*(x)` = **1.3–4.4 cm**,
  and the term that binds. With the box bottom's 6–7 cm floor any cell above **3.8%**
  availability was bias-bound; with this reading's 0.11–0.84 cm floor **none is** — even seen
  at every heading it sustains 1.5 cm, 1.8× its own floor. Model constants are imported from
  `experiments/achievable_precision_map/exp1_precision_vs_coverage.py`, not copied.
- **P4** `P4_does_geometry_explain_the_scatter.png` — the whitening check in the camera's
  frame: measured 0.89 × 0.60 cm against 0.79 × 0.59 cm predicted, whitened mean squared size
  **0.94** (1.00 exact), and per range band 1.11 / 0.97 / 0.62 / 0.58 / 0.96 — the worst band
  underpredicts by 11%.

---

### Which reading you are talking about decides who is right (measured 2026-08-18)

Both readings, scored on the same images, decomposed along/across the camera's viewing ray.
"Bias variation" is the noise-corrected spread of per-cell means over the workspace.

| | **box bottom** (n=413) | **keypoint, retrained** (n=369 clean) |
|---|---|---|
| bias along the ray | **−4.75 cm** | −0.30 cm |
| bias across the ray | +0.17 cm | −0.02 cm |
| noise sd along / across | 3.16 / **5.11** cm (wider ACROSS) | **0.88** / 0.61 cm (wider ALONG) |
| bias variation, heading | **1.36 cm** (span 4.50) | small |
| bias variation, position | **1.23 cm** (span 5.42) | 0.25 E–W / 0.38 N–S |
| bias variation, distance | **0.71 cm** (span 2.79) — real, but half of heading | near-constant, d²/H law rejected |
| noise falls below bias variation after | **~5 readings** (2.95 cm ÷ √n vs 1.36 cm) | many |

**So the supervisor argument holds on the box-bottom reading and not on the keypoint one.**
On the box bottom the bias is the state-dependent part (5.4 cm span, driven by HEADING, not
range) while R is nearly uniform — exactly the "a state-dependent R optimises the wrong term"
case. All bin comparisons use EIGHT EQUAL-COUNT bins per axis: coarse quantile bins at short
range hid the distance trend entirely on a first pass (reported 0.23 cm, actually 0.71 cm). On the keypoint reading it reverses: R carries the state dependence (1.45:1 along the ray,
growing with range) and the bias is small and flat. **State which reading before making either
argument.**

Figures built 2026-08-18 (`scratchpad/fig_boxbottom_bias.py`, promote into `figures/`):
`fig_bias_heading_not_range.png` (heading vs distance, shared y-scale) and
`fig_bias_vs_noise.png` (the √n crossover, and the perpendicular scatter in the camera frame).

Mechanism worth keeping: on the box bottom, bias runs ALONG the ray (a wrong contact point) and
noise is widest ACROSS it (the silhouette's horizontal centre wobbles with heading) — the two are
perpendicular, which is a second reason one covariance cannot carry both.

### Part 2b — and what the filter then does with each reading: **BUILT 2026-08-18**

`B1_filtered_box_bottom.png`, `B2_filtered_marked_point.png`,
`B3_r_is_right_the_mean_is_not.png`, `B4_what_R_learns_versus_the_error.png`,
`B5_error_and_claim_over_time.png` — the same R-learning loop (the notebook's prior and
ELBO terms, its trajectory removed), the same places, only the reading swapped. Each floor
cell holds 2–6 readings of one position at different headings, so each cell is a standing
robot read several times; there is no odometry and none is invented. Rows: what the loop
believes R is · what it predicts · **what it then says about its own position**.

| after ten passes | box bottom | + lean corrected per heading | marked point |
|---|---:|---:|---:|
| learned R | 2.85 × 2.68 cm | 1.31 × 1.38 cm | 1.05 × 1.12 cm |
| within-place scatter it is fitting | 2.78 × 2.60 cm | 1.07 × 1.16 cm | 0.72 × 0.82 cm |
| per-place error it cannot see | **7.04 cm** | **1.39 cm** | **0.58 cm** |
| what the belief claims (1σ) | 1.60 cm | 0.78 cm | 0.63 cm |
| truth inside its own 95% region | **8.3%** | **71.1%** | **90.2%** |

**B5 puts time on the x axis** (a standing robot, sightings at the deployed 3 Hz): learning R
does not move the belief at all — averaging a place's readings does not depend on R, so the
measured error line is the same at every pass and stops falling at the place's own average
error (7.04 / 1.39 / 0.58 cm) — it only lowers the claim, which falls as R/√n. The claim
crosses below the error **before the first sighting** (box bottom), after **0.9 s**
(heading-corrected) and after **4.3 s** (marked point). The drive-based version of this
picture, box bottom only, is `smoothing_bands` in
`experiments/filter_notebook/notebook_views.py`.

**B4 is the one-picture version:** what the loop learns R is (2.77 / 1.35 / 1.08 cm, no truth
used) · the scatter it is fitting (2.69 / 1.11 / 0.77) · **the part it never sees** (7.04 /
1.39 / 0.58) · what actually happens (7.91 / 1.67 / 0.74). So the stated number is 2.9× too
small on the box bottom, 1.2× too small heading-corrected, and 1.5× too big — conservative —
on the marked point.

**And the framing that survives the supervisor's objection:** learning R does not make
anything worse, it makes R *right*. R's own test — the squared innovation over the covariance
it forecast, which needs no ground truth and where 2 is exactly right — climbs 0.58 → 1.89,
0.10 → 1.36 and 0.05 → 1.01 over the passes. Push it further with a 1 cm prior and R lands
**exactly** on the within-place scatter with its test at 2.04 / 2.01 / 1.97, and position
coverage falls *further* (8.3 / 57.9 / 82.0%). So coverage is a verdict on the **mean**, not
on R, and the marked point's 90% under the 5 cm prior was partly a prior holding R 40% too
high. Coverage falls 56 → 21 → 10 → 8% (box bottom), 98 → 94 → 85 → 71%
(heading-corrected), 98 → 98 → 95 → 90% (marked point) while each one's distance from truth
never moves — which is arithmetic, not a finding. The heading correction
is the one worth arguing about: it kills the pooled lean (−6.46 → −0.09 cm) but what is left
is *place*-dependent (1.39 cm median per place, 1.79 × 1.75 cm spread), which is invisible to
the loop for the same reason the lean was. It buys four fifths of the way and hits the same
wall — and it needs a heading the box bottom cannot supply.

**The loop is not learning R wrongly** — it recovers the within-place scatter to a few
percent on both readings. What it cannot see is the part of the error that is the same at
every heading in a place, and on the box bottom that part is 6.5 cm. Coverage of its own 95%
region falls **56% → 21% → 8%** over passes 0, 1, 10: every pass makes the belief tighter
around the same wrong place. Meanwhile readings land 4.31 cm from its prediction while it
forecasts ±3.39 cm — the innovation picture stays healthy, which is Part 3's mechanism
appearing in miniature. Evidence:
`logs/studies/localization_reading_story/filter_on_both_readings/RESULTS.md`.

The marked point is *nearly* honest: 90.2% instead of 95%, because after averaging several
readings its stated σ (0.63 cm) is about the size of its own leftover bias. Same mechanism,
12× smaller, no longer decisive — which is why the safety statement is `|bias| + k·σ/√n`.

### Part 2c — the drive: **CAPTURED AND SCORED 2026-08-19**

`D1_drive_estimate_over_time.png`, from
`logs/studies/localization_reading_story/drives/markers_aisle_east_north_20260819_095440/`
(its own RESULTS.md). The static capture cannot answer "what does learning R do over time",
because a standing robot gives R nothing to trade against — the belief is the average of the
readings whatever R says. So one drive was captured **with the marker disks rendered**
(`capture_drive_with_markers.sh`; the existing notebook drives have them off, and keep no
usable frames for the keypoint model), 291 frames up the central aisle in 62 s, and every
frame was read twice: the frozen detector for the box bottom, `yolo_pose_aws_v4` for the
marked point.

**The long route with turns** (`comb_four_aisles`, 0.3 m/s, 152 s of valid driving,
`D1_drive_comb_four_aisles.png`) adds the part the straight route could not show: the box
bottom's error **changes by a factor of four inside one drive**, purely with heading —
median miss 10.33 cm facing away from the camera, 9.76 west, 6.86 towards it, **2.41 cm
facing east** — while the marked point stays between 0.97 and 1.43 cm at every heading. Its
belief ends 9.64 cm from truth against 1.01 cm, inside its own 95% region 18% against 83%,
and the gate starts throwing away the readings that would pull it back (13 → 17 → 57
rejections over the passes, against 0 → 5 → 15). Both still learn nearly the same R (0.81 and
0.75 cm).

Getting that drive cost three captures: the cross aisle has only 0.26 m of clearance, and the
waypoint follower steers to the next point rather than to the line, so lateral error from a
turn is never corrected and the robot ends up against a rack with its wheels turning. The two
failed captures are kept with notes, and `drive_filter.driving_window` now cuts any recording
where the truth covers less than 60% of the ground the odometry claims.

| on the same frames, same motion, same loop | box bottom | marked point |
|---|---:|---:|
| median miss of the readings | 9.23 cm | 1.08 cm |
| learned R after 12 passes | 0.90 × 1.06 cm | 0.91 × 0.97 cm |
| belief error from truth | **9.39 cm** | **1.10 cm** |
| truth inside its own 95% region | **0%** | **83%** |
| readings rejected by the χ² gate | **0 of 290** | 0 of 290 |

**The two readings learn almost the same R** — the frame-to-frame scatter really is about a
centimetre for both — so the same number is "about right" on one and ten times too small on
the other, and nothing in the data says which. **The gate never fires**, so no health check
in the stack notices a belief 9.4 cm out; that is Part 3's mechanism, measured on a drive.
The filter, gate, process noise and R loop are `notebook_model.py` unmodified — the only
difference between the two columns is which pixel the reading came from.

### Part 2d — where the camera can be trusted at all: **BUILT 2026-08-19**

`V1_surveyed_gp_monodepth_and_fusion.png` — the three availability fields of THIS world, on
one colour scale, rather than the four estimator sources of the availability paper's A0–A3
panel (those are the four-camera world):

| | mean over the driveable floor | below 0.5 | agreement with the survey |
|---|---:|---:|---|
| the GP that was surveyed (139 driven-to places, detection RATE) | 0.63 | 20% | — |
| **P from the camera's own picture** (one frame → monocular depth → floor-anchored → raycast) | 0.59 | 22% | **Spearman 0.773**, R² 0.581 |
| P + GP, precision-weighted in logit space | 0.61 | 20% | — |

`--geometry` draws the CAD prior instead (`V1_surveyed_gp_geometry_and_fusion.png`), and it
is the weaker of the two: **Spearman 0.730, R² 0.515**. So the field that needs no CAD model
also predicts the survey better — the depth prior sees the crates the CAD file does not know
about. Either way the map can exist on day zero and be corrected later; the fusion leans on
the prior exactly where the survey is thin (median prior weight 0.52). Fields are rebuilt into
`logs/studies/localization_reading_story/availability_fields/` by
`scripts/geometry_visibility/build_geometry_visibility_prior.py` (CAD) and
`build_mono_depth_prior.py` (monocular depth, Depth Anything V2 Metric-Indoor-Large on one
drive frame, floor-anchored `d_true = 0.86·d_pred + 1.74` at 81% inliers) from the locked GP
artifact `paper_artifacts/gp/warehouse_visibility_gp_v1`; that folder's VALIDATION.md carries
the CAD agreement numbers. This is the `p_use(x)` that P3's achievable-precision model needs.

## Part 3 — why rows 1–5 took so long: the filter cannot see its own bias

This is the explanation slide, and it is the reason the method ladder has five dead ends
in it. **Not a defect of the readings — a defect of the diagnostics.**

The single-camera path updates the belief in **pixel space**: `z = (u,v)`, `h(x)` nonlinear
via the camera model (`PixelMeasurementSource` in
`src/planning/planning/core/belief_correction.py`). The multicam path is metric. Both call
their residual "innovation" and both gate on "NIS".

In the pixel path the innovation is `measured_px - h(belief)`, and both sides pass through
the same calibration. A bias in `h()` pushes the belief the same way, so the belief settles
at whatever pose drives the pixel residual to zero. **The error becomes a bias in the
estimate, never a large innovation.** Measured on the locked campaign:

| | paper-1 pixel path |
|---|---|
| pixel innovation | median 1.53 px |
| `R` applied | 2.5 px |
| NIS (gate 9.21) | median **0.29** (a healthy 2-DOF filter sits at 1.4-2, so `R` was *loose*) |
| NIS rejections | 1 / 6370 = **0.016 %** |
| **belief error vs truth** | **p95 0.127 m** |

Every internal health check passed while the belief sat 13 cm off truth.

### What this explains, that was previously three unconnected results

1. **Why "learning R backfired."** The learner fits innovations; in pixel space the lean is
   not in the innovation, it has been absorbed by the belief. Hence `needed/visible = 8.4x`
   — the learner measured the scatter correctly and was asked an impossible question.
2. **Why `sigma_px` comes out identical on the raw and mesh-corrected paths** (0.47 vs
   0.51 px). Same mechanism. That was recorded as "a robust GT-free measurement"; it is
   equally a symptom.
3. **Why the 6.34 cm box-bottom lean survived undetected.** Nothing in a single-camera
   pipeline could see it. It took four cameras disagreeing to make it observable — so the
   4-camera 0.19 m was **not** the cameras being worse, it was the bias becoming visible for
   the first time. (`docs/notes/multicam_vs_paper1_correction_parity.md`)

### Why it still matters now that the reading is unbiased

The keypoint reading has a pixel residual of (+0.09, +0.16) px, so at present there is
**nothing being concealed**. But the concealing mechanism is untouched and latent: if
calibration drifts, NIS still will not see it and the filter will still look healthy.

⚠️ **Direct consequence for `calibration_drift_lifecycle`:** a drift detector that keys on
innovations or NIS is, in the pixel path, structurally blind to exactly the fault it exists
to find. Check this before trusting any drift-detection result on the single-camera stack.

### Figures owed for Part 3
- **F1** the two-panel health check: pixel innovation / NIS looking healthy on the left,
  metric residual vs truth on the right, same run, same time axis. The whole point in one
  picture.
- **F2** the mechanism as a diagram: bias in `h()` -> belief slides -> pixel residual returns
  to zero -> NIS sees nothing. No data needed; this is the explanation.

### The cheap fix — do not refactor
Keep the fork: paper-1 is locked to it and changing the measurement space invalidates the
locked campaign. Make it **visible** instead. Report the pixel residual and the metric
residual side by side, in the same table, always (`eval_res_x/y` already exists). The failure
was never that the numbers were missing — it was that NIS 0.29 and 0.127 m lived in two
different documents.

## What is in this folder

| file | what it does |
|---|---|
| `GLOSSARY.md` | one name per quantity; read before writing a caption |
| `reading_data.py` | the only loader: joins the three scored readings to the capture, checks the join on the true pose, and holds the figure house style |
| `keypoint_geometry.py` | the reading as a function of four pixels, its Jacobian, and the pixel→floor covariance push (no fitting) |
| `recompute_reading_stats.py` | recomputes every number the figures state and records it: `reading_stats.json`, `by_world_heading.csv`, `by_relative_heading.csv`, `by_range.csv` |
| `plot_bias_vs_noise.py` | the box-bottom side: bias vs heading vs distance, and the √n crossover → `logs/studies/localization_reading_story/bias_vs_noise/` |
| `filter_on_both_readings.py` | the R-learning filter on both readings (cells, VB loop, closed-form ELBO, honesty scoring) → `.../filter_on_both_readings/` |
| `capture_drive_with_markers.sh` | one drive in `warehouse_aws` with `show_pose_markers:=true`, frames + odometry + truth → `.../drives/<tag>/` |
| `score_drive_frames.py` | reads every drive frame twice (box detector, keypoint model) → `<drive>/readings.csv` |
| `drive_filter.py` | builds the notebook's Sequence from a drive and runs its `learn_R` unmodified, once per reading |
| `figures/fig_L1..L4, fig_P1..P4` | one script per figure; each prints the numbers it drew |
| `figures/fig_B_filter_both_readings.py` | B1–B5 in one pass, with identical scales so the two readings can be laid side by side |
| `figures/fig_D_drive_over_time.py` | D1: the drive, estimate against time, per pass, one column per reading |
| `build_mono_depth_prior.py` | P from one camera frame: monocular depth → floor anchor → raycast → calibrated onto the detection-rate scale, then fused with the GP |
| `figures/fig_V_availability_fields.py` | V1: the surveyed GP, P (monocular depth by default, `--geometry` for CAD), and their fusion |

Everything writes to `logs/studies/localization_reading_story/`, whose `RESULTS.md` is the
run record. Part 3's two figures (F1, F2) are still owed — B1's middle and bottom rows are
the "health check passes while the belief is wrong" picture that F1 asked for, but on static
poses rather than on the locked campaign, so F1 is not discharged by it.

## Rules for this folder

0. **Read `GLOSSARY.md` before writing any caption.** One name, one quantity;
   every number carries its reading method.
1. Every figure states its finding in the title and names its evidence file in the caption.
2. A figure that recomposes a retired-input study says so on its face.
3. No new numbers. If a figure needs a number that does not exist yet, write the script,
   run it, and record the result under `logs/studies/localization_reading_story/` first.
4. Nothing here is a navigation or safety claim.
