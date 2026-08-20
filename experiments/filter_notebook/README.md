# `filter_notebook` — Bayesian filtering and smoothing, on a real capture

**Four** teaching notebooks that follow the arc of the BMLIP course notebook *PP4 —
Bayesian filtering and smoothing*, on recorded runs of this system: a TurtleBot3 driving
while ceiling cameras try to see it. Two questions, asked first of **one camera** and
then of **all four**:

| | one camera | four cameras |
|---|---|---|
| *how much should I trust it?* | `pp4_1_learning_r` | `pp4_3_learning_r_four_cameras` |
| *where does it think I am?* | `pp4_2_the_offset` | `pp4_4_the_offset_four_cameras` |

### Two worlds, and which notebook runs where

Notebook 1 runs in the **single-camera AWS warehouse**, which is where the two-world rule
puts method development (`research/06_world_camera_design.md`). That is not a detail: the
one camera there sees the robot over a 1.4 m to 10 m range sweep in a single drive, so one
pixel of detector error is worth **0.64 cm of floor near it and 2.40 cm far away — a
factor of 3.7 within one drive**, and the notebook can show that a single number for `R`
is the wrong shape of answer without needing a second camera to argue it.

The loaders are world-aware and switch themselves. `notebook_data.World` describes each
world (cameras, world file, camera includes, detector weights, commissioning file), every
capture carries a `raw/capture_manifest.json` naming its world, and `nd.load_capture` sets
the active world from it — so a notebook says which *drive* it is looking at and the
cameras, the projection and the detector follow. Captures recorded before the manifest
existed are all four-camera ones, which is what an absent manifest means.

| | notebook 1 | notebooks 2–4 |
|---|---|---|
| world | `warehouse_aws.world.sdf` | `warehouse_full_4cam.world.sdf` |
| cameras | `camera_A` only | `camera_A`–`camera_D` |
| detector | `warehouse_yolo_detector_v1` (trained in that world) | `warehouse_yolo_detector_4cam_v3_960` |
| drives | 4, one per route (see below) | `notebook_5hz` + 3 earlier captures |
| baseline `R` | the drive's own true errors (`nm.oracle_noise`) | commissioned on other captures |

### The single-camera data collection protocol

Four drives, **one per route**, so that "held out" means a viewing geometry the fitted
covariance has never seen rather than a different stretch of the same line. Routes live in
`config/aws_study.yaml`, checked against the rack collision boxes; one invocation captures
one drive:

```bash
ROUTE=aisle_east_north  SEED=20260813 RUN_TAG=aws_aisle_east_north \
  bash experiments/filter_notebook/capture_aws_notebook_dataset.sh
```

| drive | route | role | detections |
|---|---|---|---|
| `aws_aisle_east_north` | north up the R3/R4 aisle, 1.4 → 10.0 m | **analysed**, held out of the fit | 276 in the window, 100% of frames |
| `aws_apron_west_to_east` | near-field lateral, 1.2 → 4.7 m | held-out test | 272, 100% |
| `aws_aisle_west_north` | the mirror aisle, opposite bearing | held-out test | 224, 100% |
| `aws_mid_cross_east` | far-field lateral over the lower racks | held-out test | 97 of 167 frames (59%) |

**Notebook 1 does not use a commissioned `R` at all** — that baseline turned out to be a
confusing thing to compare against, because it mixes two questions (is the fit right? does
it transfer between drives?). Everything is now judged against **what the camera's errors
actually are on the drive being filtered**, measured with ground truth (`nm.oracle_noise`):
the best a zero-mean model could possibly be told. The three other drives remain, purely as
held-out *test* data for the prediction question. Commissioning still exists as a script
and is still used by the four-camera notebooks:

```bash
python3 experiments/filter_notebook/commission_observation_noise.py \
  --world warehouse_aws --captures aws_apron_west_to_east aws_aisle_west_north aws_mid_cross_east
```

Three things the protocol had to get right, each of which cost a run somewhere:

1. **`pgrep -a` never matches a live Ignition server** (`ign` is a Ruby script, so
   comm=`ruby`); the guard uses `pgrep -f` plus a VRAM pre-flight, and escalates teardown
   past SIGTERM.
2. **The route driver does not exit after `route_complete`.** It logs completion, writes
   its artifact, and keeps spinning. The script backgrounds it, waits for
   `route_completion.json`, then SIGKILLs it — running it in the foreground hangs the
   capture until the outer timeout fires.
3. **The manifest is written before the drive, not after,** so a run that aborts half way
   is still identifiable and still loads.

Each opens with the same three-part box — **what it implements, what came out, what to do
about it** — so the answer arrives before the working. In order, the four decisions are:

1. **Do not deploy the same-drive zero-mean `R` fit shown here** — but not because the
   estimator is broken. The 12-pass variational routine is checked three ways (the gate
   is not what shrinks it; it recovers a covariance it generated itself; the update is
   pinned against exact references in the test suite) and it genuinely *does* predict the
   camera's next reading better. Scored on **held-out data** — the later half of the drive
   and three different routes — the two criteria come apart completely and stay apart:
   the fitted covariance wins the forecast on **all five** and loses the belief on **all
   five**. A camera lean cancels out of the innovation, so neither the fit nor the
   forecast can see it.
2. **Predict the pose-dependent geometric displacement; one camera cannot separately
   identify constant offset and absolute position.** Add a constant to one, subtract it
   from the other, and neither the observations nor odometry increments change.
3. **Per-camera learning is not the fix either.** The same reversal, harder: 293 nats
   better fit, median normalised error 43.7 against 9.21.
4. **Predict what geometry gives you and estimate a residual state with uncertainty.**
   Multiple cameras constrain relative offsets when the camera–time graph is connected;
   their common translation remains unobservable. On this diagnostic run, filter-belief
   median NEES moves 25.9 → 0.74 (conservative relative to the 1.386 reference).

Read them in order. 1 ends on the wall that 2 goes through; 2 ends on a limit that only
a second camera can lift.

### Notebook 1 measures fit and prediction separately, because they disagree

The question "is the fitted `R` better?" splits into three, and notebook 1 answers each
by measurement rather than by argument:

| question | how it is answered | result |
|---|---|---|
| is the implementation wrong? | learn again with the innovation gate off; then hand the loop readings drawn from its own model, where the right answer is known | **not the gate** — 276 of 276 readings kept either way, the answer moves by 0.00 cm. Recovery: 2 cm → 95%, 5 cm → 85%, 15 cm → 75%, so it under-reports more the noisier the camera really is; and a 9 cm lean added to otherwise perfect readings changes the answer by nothing |
| **why** does it fit better? | split the score into `−½vᵀS⁻¹v` (the reading missed) and `−½ log det 2πS` (the forecast was narrow) | credit for a narrower forecast **+2.62 nats**, extra penalty for missing **−0.13** — 95% of the gain is claimed confidence, not accuracy |
| can it **predict** better? | forecast `y_k ∼ N(m⁻, P⁻+R)` before each reading arrives — **no ground truth needed** — scored on the later half of the drive and on three other routes | predicts the **camera** better on **5 of 5** (+0.54 to +0.67 nats, and the margin does not fade when held out); knows where the **robot** is worse on **5 of 5** (median NEES 116 against 20 in-sample, 146 against 21 on another route, honest = 1.39) |

**And where the loop lands is the tell.** The true errors need σ = 6.46 cm; their *scatter
alone* is 1.24 cm; the loop learns **1.37 cm**. It has not failed to estimate something —
it has estimated the scatter, correctly, and its model has nowhere to put the rest. Drawn,
the learned R is a small circle centred on zero, and not one of the 276 errors landed
anywhere near there.

Even handed the right answer the model cannot be honest: filtering with the camera's true
error covariance still leaves the truth **3.3× further away than the belief allows**.
Inflating R to cover a lean is not the same as modelling the lean, and that is as good as
the trade gets.

The two criteria are cleanly and consistently **opposed**: choosing `R` by how well it
predicts the next reading picks the least honest belief available, on every drive tested —
and that is the only criterion a robot can compute for itself.

Why both can be true: over the 276 readings the camera is wrong by (−0.67, −8.62) cm and
the belief by (−0.66, −8.71) cm, while the filter's own surprise averages
(−0.03, +0.14) cm. The lean is in the sensor and in the estimate, and cancels out of the
only quantity either the fit or the forecast is scored against. Read the same fact the
other way: as a forecaster of the *reading* even the learned covariance is under-confident
(typical surprise 0.16 against an honest 1.39), while as a belief about the *robot* it is
a hundred times overconfident. The same matrix, at the same instant.

Last section: sweeping the assumed odometry noise moves the learned camera noise by 1.4×
and the belief's calibration by **55×**, so what one drive determines is the `R`/`Q`
*combination*, not `R`.

### The pictures come first

Notebook 1 opens on the measurement rather than the mathematics: the recorded frame with
the detector's box on it, the same frame magnified 24× so the few pixels between the box
bottom and the robot's real contact point are visible, and the same instant on the floor.
Then the same crop at five ranges, which is what turns "a few pixels" into "a lean": the
box bottom sits on the same side of the true contact point every time. The pixel gap
shrinks with range (17 px at 1.4 m, 2 px at 9.8 m) while the floor error does not
(10.2 cm, 7.4 cm) — because one pixel is worth 0.64 cm of floor near the camera and
2.40 cm far from it. The error is geometric, not photographic.

Everything is real. Recorded Gazebo sessions, one clock per drive, no synthetic data.

| file | role |
|---|---|
| `pp4_[1-4]_*.py` | the four notebooks, as `# %%` cell scripts — **edit these** |
| `pp4_[1-4]_*.ipynb` | generated from those, with outputs |
| `notebook_model.py` | the state-space model, every estimator, every scorer |
| `notebook_views.py` | every figure, animation and printed table |
| `build_notebook.py` | converts the scripts to `.ipynb` and executes them |
| `notebook_data.py` | loads the capture: frames, detections, odometry, truth |
| `capture_notebook_dataset.sh` | records a four-camera capture end to end |
| `capture_aws_notebook_dataset.sh` | records ONE single-camera drive end to end |
| `config/aws_study.yaml` | the four single-camera routes, checked against the racks |
| `record_demonstration_capture.py` | the recorder (observations + odometry + truth) |
| `commission_observation_noise.py` | fits the baseline `R` on held-out captures |
| `benchmark_detector_rate.py` | where the detector's milliseconds go |
| `measure_pipeline_rates.sh` | rate probe: clock, images, observations |

```bash
bash experiments/filter_notebook/capture_notebook_dataset.sh   # ~15 min
python3 experiments/filter_notebook/commission_observation_noise.py
python3 experiments/filter_notebook/build_notebook.py          # build + execute all four
python3 experiments/filter_notebook/build_notebook.py pp4_1_learning_r --no-exec
```

## The layering, and why it is worth keeping

None of the four notebook scripts defines an estimator or figure. Every code cell is a call:

```python
forward = nm.kalman_filter(seq, commissioned["R_total"])
nv.report_filter(forward, n_obs)
```

That is what makes four notebooks safe rather than a fork: each estimator has one shared
implementation in `notebook_model.py`, and all four notebooks call those implementations. The four notebook
scripts hold **42 to 55 lines of code** each between the prose; the estimators live in
`notebook_model.py` and every figure, animation and printed table in `notebook_views.py`. A test
asserts the notebooks define nothing of their own, so the split cannot quietly drift
into two copies.

## Checking the mathematics

The notebook states results in LaTeX and implements them in code a few cells later, and
nothing forces the two to agree. **`tests/experiments/test_filter_notebook_math.py`** pins
them together and runs in the ordinary suite (`python3 -m pytest`), so a change to the
mathematics fails a test rather than sitting unnoticed in a published figure.
It imports the **very objects the notebooks call** from `notebook_model.py` — not a copy —
and checks them against exact or Monte-Carlo references:

* the filter and smoother against the exact dense Gaussian posterior (agree to 1e-15);
* the accumulated log evidence against the joint marginal likelihood of the observations;
* the Joseph form against `(I − K)P`;
* the inverse-Wishart moments, the inverse-gamma marginal, and the KL against Monte Carlo;
* that the variational x-step uses the expected **precision**, which differs from both the
  mean and the mode — swapping it turns variational Bayes into MAP-EM;
* the homography, the round-trip projection, the radial/tangential basis, the 2σ ellipse
  mass, and the back-projection displacement formula;
* the exact two-dimensional common-offset gauge and observable relative-offset contrasts;
* that the numbers quoted in the prose are the ones the notebook actually produced.

It needs no capture — it builds its own small problems, so it stays green on a fresh clone
(the tests that read published artifacts skip when those are absent). 26 tests, ~10 s.
Two of its checks caught real bugs when first written; see below.

## Two math bugs these checks caught

1. **The inverse-Wishart KL came out negative** (−141 nats). Every sign in the closed form
   was flipped and a `d/2 (ν_q − ν_p) log 2` term was missing. A KL cannot be negative;
   the test now asserts zero-from-itself, non-negativity, and agreement with Monte Carlo.
2. **The log evidence was being compared across arms that gated different data.** The χ²
   gate admits a different subset of observations for every `R`, so the running evidence
   sums over different observations and is not a model comparison. Recomputed with the
   gate off over all 349 observations, the learned-`R` advantage falls from +389 to
   **+293 nats** — the conclusion survives, but the reported figure had been inflated by
   about a hundred nats.

## Evidence role

**Diagnostic.** This is a demonstration capture: no campaign ledger row, no
frozen-config hashes, no projection-calibration provenance (see the generated
`CAPTURE_ROLE.md` beside the data). It exists so the notebook can show the
perception stage on frames recorded beside the detections. Anything that has to
be citable goes through `record_operational_logs.py` instead.

The notebook back-projects pixels with the deployed parameter-free homography,
`camera.pixel_to_world(u, v)` — **not** through `rcond_common`, which still uses
the `projection_calibration_v2` corrections that were removed from the runtime in
August 2026 when plain inverse perspective mapping beat every fitted variant.

## What the notebooks do and do not contain

They are **variational inference**, not active inference: they approximate posteriors over hidden
things (the trajectory, the observation noise, the per-camera offsets) given data. Nothing in
it selects an action — the robot followed a fixed commissioned route at constant speed and
would have driven the same line whatever the cameras said. What it supplies to an
expected-free-energy planner is the ingredient that turns out to matter: an observation model
whose precision and bias depend on where the robot will stand and which way it will face.

## The offset: what it is, and three ways to get it

For a detection, the instantaneous residual is the back-projected *bottom-centre pixel of the
detector's box* minus `base_footprint`, the declared robot origin at the wheel-axis midpoint.
The notebooks reserve **constant offset** for the per-camera sample mean of those residuals
on a named dataset. The geometry-predicted displacement is a third, pose-dependent object.
In this dataset the usual suspects are exactly zero (the camera model
is built from the same world file the renderer uses, the camera SDFs carry no distortion term,
the floor is exactly z=0), so it is almost purely an object-model mismatch: the robot is
0.1909 m tall and its plan centroid sits **36.9 mm behind its own origin**, assembled from the
URDF meshes by `nd.robot_point_cloud()`.

**Predict it (`pp4_2_the_offset`).** Project the mesh at the robot's pose, take the bounding box's
bottom-centre, back-project. Zero fitted parameters. Median |observation − prediction| over
2404 detections in four runs: **4.69 cm → 1.98 cm** using heading from odometry. A *wrong*
heading gives 7.12 cm — worse than no correction — so it is the pose-dependence doing the work,
not a constant in disguise.

**Estimate it (`pp4_4_the_offset_four_cameras`).** Put a 2-D constant-offset approximation
per camera in the state (2 → 10 dimensions) and let one chronological Kalman pass estimate
it alongside position. This is recursive state estimation, not the 12-pass `learn_R` loop.

| arm | median NEES | RMSE | offsets recovered to |
|---|---|---|---|
| no offset handling | 25.86 | 8.40 cm | — |
| predicted from geometry | 1.11 | 3.72 cm | (not estimated) |
| offset state, constant | 1.05 | 6.17 cm | 4.8 cm |
| offset state, drifting 2 mm/step | **0.74** | 5.29 cm | 3.3 cm |
| ground-truth offsets removed (ceiling) | 0.74 | 2.63 cm | — |

The calibrated 2-D median reference is 1.386; 0.74 is conservative. Two things worth keeping:

* **The belief becomes conservative without the offsets becoming accurate.** They land ~3.3 cm
  from what they averaged, on quantities of ~5 cm. The position belief no longer claims the
  extreme precision of the baseline because the filter carries offset uncertainty into position.
* **Prediction and state estimation fail in different places.** The predictor is excellent on cameras A, C, D
  (1.8–3.4 cm) and poor on **B (9.4 cm)** — the camera whose silhouette is clipped by shelving
  most often, breaking its assumption that the whole robot is visible. The constant-state model
  is closer to the diagnostic run mean there (3.4 cm). This motivates a geometry-plus-residual model.

Identifiability is a property of the observation graph, not the estimator: the part common
to all four offsets is invisible to the cameras by construction (2.1 cm off), while the
route and odometry connect relative offset constraints (recovered to 2.2 cm here).

## Two animations

Embedded as self-contained players (`to_jshtml`), so they play in the `.ipynb` with no
external files. Both are the real run — no re-enactment.

**The filter running** (end of section 3), 72 steps over the 100 s drive: the camera that
last fired with the detector's actual box on the real frame; the aisle; and the belief
magnified to ±30 cm where the prediction, observation and posterior ellipses are visible.
Under it, that step's arithmetic — innovation, `v'S⁻¹v` against the gate, the 2σ radius, and
the distance from truth. A typical frame reads: *innovation (+0.4, +1.5) cm, v'S⁻¹v 0.03 vs
gate 5.99 → USED, belief 2sd 5.5 cm, off truth by 7.8 cm.* Confidently wrong with every
internal diagnostic content, which is the whole argument in one line.

**The fitting** (section 5), one frame per pass: the four inverse-Wishart posteriors sliding
down and tightening, the fit to the observations, and the median NEES. The two right-hand
panels move in opposite directions as it converges.

Note the middle panel plots `log p(y | R̄)`, which is **not** the objective — the ELBO is,
and it is not computed here. Plug-in evidence has no monotonicity guarantee and indeed
overshoots on pass 1 before settling. The endpoint comparison against the commissioned value
is an in-sample log-fit comparison, not held-out model selection; the honesty panel degrades
throughout regardless.

Sizing, since animations dominate the file: JPEG frames rather than PNG (measured — PNG was
2.11 MB against 1.36 MB even for the line-art one, the opposite of what I expected), 74 dpi,
every 14th grid step. 44 MB → 13 MB total.

## What the detector actually does (notebook section 2)

The notebook shows the perception stage as a population, not one example: a contact sheet
of real crops per camera across the drive, the detection outcome of every logged message,
and where each robot position fell in each image. Two things came out of it.

**Most frames yield nothing.** Per camera over the 100 s route, 352 messages and detection
rates of 0.25 / 0.17 / 0.55 / 0.56 for A / B / C / D.

**Splitting geometry from the detector is essential.** A raw "detection rate against range"
is non-monotone and meaningless here: on one straight traverse, range and whether the robot
is even inside the image move together, so a rate of zero at 7 m means "nothing to detect".
Conditioning on the robot being in frame:

| camera | messages | in frame | detected | rate given in frame | range where detected |
|---|---|---|---|---|---|
| A | 352 | 291 | 88 | 0.30 | 5.3–11.6 m |
| B | 352 | 255 | 61 | 0.24 | 5.4–11.6 m |
| C | 352 | 231 | 195 | 0.84 | 8.7–15.8 m |
| D | 352 | 226 | 195 | 0.86 | 8.7–14.9 m |

The 3× gap is **not** a detector difference — it is the same weights on all four cameras.
The contact sheet shows why: where the robot should be in A's and B's missed frames there is
**shelving**. The failures are dead bands in *space* (A finds nothing between −4 and −2 m
north, B between +2 and +6), which is the signature of passing behind a shelf row, not of
running out of pixels. Since nothing here models occlusion, "in frame and missed" is an
**upper bound** on the detector's own failures. Also logged: 3 detections arrived while the
robot was outside camera D's image — false positives, which is what the χ² gate is for.

## The baseline `R`, and why it is measured here rather than quoted

`commission_observation_noise.py` fits the per-camera observation noise on the three
captures that predate this one (`smoke1`, `smoke2`, `fusion_handover`), with the
**current** parameter-free homography. The notebook capture is held out, so nothing is
scored against a covariance fitted on itself. Earlier values in the repo were fitted
through the retired `projection_calibration_v2` corrections and are not a like-for-like
baseline on today's projection path.

It reports two covariances, and which one is handed over is a modelling decision, not a
detail:

* `R_spread` — covariance about each camera's mean residual. Pure scatter.
* `R_total` — second moment about zero, `E[(y−x)(y−x)ᵀ]`. Scatter **and** offset.

A model that says `y = x + zero-mean noise` has no term for a mean and cannot subtract
one, so commissioning honestly for that model means handing over `R_total`. Measured
(cm, one standard deviation, 1424 observations):

| camera | n | offset | scatter | total | inflation |
|---|---|---|---|---|---|
| A | 125 | 3.08 | 3.03 | 3.72 | 1.23× |
| B | 295 | 5.57 | 2.18 | 4.50 | 2.07× |
| C | 474 | 10.07 | 3.52 | 7.95 | 2.25× |
| D | 530 | 3.52 | 2.89 | 3.81 | 1.32× |

Camera C's 10 cm offset reproduces the known "camera C lean" on the current path.

## Observation rate: what limits it, and the lever

The pipeline emits ~1.1 observations per camera per simulated second against 5 Hz
cameras. What is and is not responsible:

* **Not synchronisation.** The four-way stamp-skew guard dropped only ~18 frames in
  200 s.
* **Not raw inference either — this corrects the earlier reading.**
  `REAL_RUN_FINDINGS_2026-07-21.md` concluded the batch was inference-bound at
  ~805 ms, and derived a ~1 Hz ceiling from it. That 805 ms came from the node's
  **warmup** log line (3 iterations, including cuDNN autotuning and first-call
  allocation), which is not a steady-state figure. Measured properly on real captured
  frames (`benchmark_detector_rate.py`, Quadro P2000):

  | imgsz | ms per four-image batch | implied rate | box-bottom shift vs 960 |
  |---|---|---|---|
  | 960 (deployed) | **77 ms** | 13.0 Hz | — |
  | 768 | 52 ms | 19.1 Hz | 0.16 px median, 0 detections lost |
  | 640 | 38 ms | 26.2 Hz | 0.22 px median, 0 detections lost |

  So inference accounts for ~77 ms of a ~1300 ms observed period. **Roughly 1.2 s per
  batch is spent somewhere else in the node or the pipeline and is not yet accounted
  for** — image transport of four 1280×720 frames, CPU contention with Gazebo's four
  renders, or in-process GIL contention (cf. the single-camera finding that detector
  latency was GIL contention, fixed by a separate process). This is an open lead: if
  that overhead were found, the ceiling would be much higher than anything below.
* Four separate per-camera detectors do not fit: the 4 GiB card holds Gazebo
  (~1255 MiB) plus one detector process (~2180 MiB).
* Note the earlier imgsz finding does not transfer: imgsz 416 was rejected in July for
  collapsing detection to one camera of four, but that was the 640-trained v2
  detector. On the 960-trained v3 used here, 768 and 640 lose **no** detections and
  move the box-bottom pixel by a fifth of a pixel.

The lever that works today, regardless of where that 1.2 s goes. The rate the filter
cares about is observations per **simulated** second, and

```
observations per sim second  =  (batches per wall second) / RTF
```

The detector is wall-bound; the simulation clock is ours to choose. Throttling the
simulator therefore raises the rate the recording sees, capped by the 5 Hz cameras:

| real-time factor | observations per camera |
|---|---|
| 0.68 (as launched) | 1.10 Hz sim |
| 0.25 | **3.6–3.9 Hz sim, all four cameras** |
| ~0.15 | ≈5 Hz sim, the camera ceiling |

This does not alter the data. `max_step_size` stays 0.001 s, so the physics
integration is identical step for step; only wall-clock pacing changes. It is
applied as a runtime `ign service … set_physics` call, so the frozen world file is
untouched. The cost is wall time, which for an offline dataset is free.

`RTF=0.25` is the default in the capture script. Override with `RTF=0.15` for a
denser dataset at ~22 minutes instead of ~15.

## Three traps this study walked into, recorded so nobody repeats them

1. **`pgrep -a "ign gazebo"` never matches a running simulator.** `-a` matches the
   process *name*, and `ign` is a Ruby script, so a live server has `comm=ruby`.
   Three orphaned servers accumulated at 1255 MiB each, filled the card, and caused
   both a genuine CUDA OOM in the detector *and* Gazebo's own fall back to software
   rendering (`libEGL: failed to create dri2 screen`), which dragged the pipeline to
   0.33 Hz. The guard needs `pgrep -f`, and teardown must escalate past SIGTERM and
   verify the VRAM came back. Both are now in `capture_notebook_dataset.sh`.
2. **`/ground_truth_tf` publishes zero header stamps.** Every truth row lands at
   `t=0` and nothing can be joined to it. Clock-stamp at receipt, as
   `record_evaluation_truth.py` does.
3. **`drive_study_route.py`'s sim deadline is really a deadline on absolute sim
   time.** It sets `started_at_s` from the first odometry callback, which fires
   before the node has received `/clock`, so it records `0.0`. After a 130 s startup
   wait (~99 s of sim) only ~32 s of the default 131 s budget remained, and the route
   aborted at 4.2 m of 14.4 m. Pass `--max-sim-runtime-s` explicitly.
