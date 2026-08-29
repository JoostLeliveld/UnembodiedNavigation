# UnembodiedNavigation

ROS 2 + Gazebo thesis repo: external-camera reliability for warehouse robot navigation.

## What this actually does (read this first)

📌 **[`PLAN.md`](PLAN.md) is the plan of record** — the sentences the paper has to earn, what
is done, what is still owed, and which data serves which purpose. Read it before proposing
work. Anything that does not serve one of those sentences belongs in a study README, not in
the paper and not in a status report.

📖 **[`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) is the start-to-finish walkthrough** with the
glossary (belief, NEES, NIS, `R`, `Q`, IPM, bias) and what is settled versus open.

A small robot drives around a simulated warehouse. Its wheel odometry drifts without bound,
so it leans on fixed wall-mounted cameras that can see it on the floor. Cameras are not
equally useful everywhere — shelves block views, accuracy degrades with range and viewing
angle, and a camera can be quietly miscalibrated. The thesis question is therefore **not**
"can the robot see itself" but **"how good is this particular sighting, and does knowing
that change how the robot should drive?"**

The whole pipeline, in order:

```text
wheel odometry (drifts)  ─┐
                          ├─► EKF: predict, then update through a gate chain ─► belief ─► planner
camera image                                                                   (mean +      (route +
  ─► YOLO box                                                                  covariance)   clearance)
  ─► bottom-centre pixel ─► ray ─► floor plane  = position   (plain IPM, zero fitted parameters)
  ─► pixel uncertainty pushed through J         = covariance (R_xy = J R_uv Jᵀ)
  ─► the cameras reconciled by camera_manager
```

**The central object is the *belief*: a position AND a stated uncertainty.** Most of this
thesis is about whether that stated uncertainty is honest, not about whether the position is
accurate. Those are separate claims and a filter can pass one while badly failing the other.

**The core mechanism:** a camera can have a repeated *lean* — a similar error every frame. A
Kalman filter treats N sightings as N independent votes and shrinks its uncertainty like
`1/N`, but the lean does not shrink. No per-frame noise model can represent a constant bias;
it takes a covariance floor that repeated looks cannot shrink.

## The clean sheet (2026-08-25)

Every study before that date is superseded. **Do not reuse any of their numbers.** The
retired control plane (`research/`, `figures/`, `paper_artifacts/` and the old `docs/`) and
the studies it indexed were deleted on 2026-08-28; they are recoverable from git history and
nowhere else. If you need one, read it from history and re-derive, never quote.

`docs/` was rebuilt on 2026-08-29 and holds three contracts, not prose:
[`localization_metrics.md`](docs/localization_metrics.md) (which quantities may be compared
and what makes a drive scoreable), [`reproducibility_inputs.md`](docs/reproducibility_inputs.md)
(the hashed detector and calibration bytes) and [`open_questions.md`](docs/open_questions.md)
(what is unresolved). Read the first before quoting any number.

## Talk and write in plain terms

Say the thing, not the label. Define a term at first use or don't use it.

### Before reporting a number, ask whether it goes in the paper

**Write the sentence it would appear in, in the paper. If you cannot write that sentence, do
not report the number.**

- **Report in centimetres, percentages and seconds.** Not pixels, not unit-less ratios, not
  log-likelihoods. "Half a centimetre of consistent lean", not "0.198 px pooled mean
  residual".
- **One headline number per finding.** A six-row table of variants is a working artifact; the
  reply gets the winner and one sentence on why. The table goes in the study README.
- **Never coin a metric name** or use one without defining it in the same breath.
- **Lead with the verdict, then the mechanism, then the caveat.** Not the derivation.
- **If a result is a null, an artifact, or a correction to something you said earlier, say so
  in the first sentence** — do not bury it after the supporting numbers.
- **Machinery is not a finding.** Conventions, split definitions, parameter counts and
  reproducibility hashes belong in the study README, unless the user asked how something
  works.

The failure mode this exists to stop: reporting the whole chain of reasoning at the same
level of detail as the answer, so the answer is impossible to find.

### Figures must stand alone

Every figure is read by someone who has not read the code. So:

- **The title states the finding**, not the variable names.
- **Axis labels say what the number means** and which direction is good. If a threshold
  matters, draw it and label it in words.
- **No bare arm/condition codes as tick labels.** `F1`/`O2` need their meaning on the axis or
  in a legend, not in a caption only the author will read.
- **Honesty and sharpness always appear together.** A filter that just widens its ellipse
  passes any calibration test and is useless, so never show a calibration score without the
  stated σ and the actual error beside it.
- Say what the data is: how many detections, which world, how many seeds.

## Layout (do not invent new top-level directories)

| dir | contents | rule |
|---|---|---|
| `src/` | ROS 2 runtime packages only: `perception`, `reliability` (the camera manager), `planning`, `state`, `sim`, `experiments`, `unav_common` | needs `colcon build`; **campaigns load from `install/`, not `src/`** |
| `experiments/<study>/` | ONE folder per investigation: code + README saying what question it answers | **new studies go here**, nowhere else |
| `logs/studies/<study>/` | that study's outputs (figures, RESULTS.md, csv) | gitignored |
| `logs/visibility_comparison/` | campaign runs and capture datasets | append-only; never edit or rename an existing run dir |
| `scripts/visibility_comparison/` | the campaign pipeline and its configs; `run_visibility_campaign.py` is the runner | new condition ids must be registered in `CONDITION_PLANNER` or the config is rejected |
| `scripts/perception/` | detector dataset capture, training, and the shape/keypoint model tooling | |
| `scripts/shared/` | THE shared analysis library (`metrics.py`) + `paths.py` (`repo_root()`) | import it; never copy helpers into a script |
| `tests/` | `python3 -m pytest -q` from the repo root | |
| `config/`, `schemas/` | shared configuration and data contracts | |

Root `pyproject.toml` (pytest-only, keeps colcon clean) + `conftest.py` put the `src/*`
packages on the path, so pytest runs from the repo root with no per-file bootstrap.

## The studies that are live

| study | what it produced |
|---|---|
| `experiments/measurement_commissioning/` | the frozen sensor: admission gate, the observation model, the one commissioned pixel-noise number, the offset correction |
| `experiments/fusion_on_fixed_routes/` | the fusion comparison on frozen routes; **`aligned.py` is the only sanctioned loader** for its run CSVs |
| `experiments/learned_measurement_covariance/` | per-camera bias and spread, and whether the stated covariance matches |
| `experiments/unbiased_observation/` | what a stricter admission rule costs in availability and buys in bias |
| `experiments/deck_figures/` | the presentation figure set |
| `experiments/warehouse_v2_sketches/` | the world generator and its freeze manifest |

## Answering questions (read this before searching)

- **Search order: `src/` and `scripts/` first, `tests/` last.** Test names here are long and
  descriptive, so they match nearly every semantic grep. Reading tests to learn what code
  does costs about twice the tokens and gives you the assertion, not the implementation.
- **Answer from a run, not from reasoning.** If a question can be settled by executing
  something, execute it. Prefer, in order: run the thing → read the artifact it produced →
  read the source.
- **Don't trust old data.** Numbers in `RESULTS.md`, memory and prior summaries describe the
  repo *as it was*. Re-measure before quoting, and say which run and date a number came from.
- **The full suite is cheap — just run it.** `python3 -m pytest -q`, about 10 s idle.

### LSP: use it for two things, don't trust it for the rest

| operation | verdict |
|---|---|
| `documentSymbol` | **works** — file outline before reading an unfamiliar module |
| `workspaceSymbol` | **works** — "where is `X` defined" |
| `findReferences` | **broken here** — returned 1 hit where grep found 80 |
| `goToDefinition` / `hover` | **unreliable across packages** — colcon layout, imports report unresolved |

**For "who calls this?" use `grep`, not `findReferences`** — a wrong-but-confident "1
reference" is how you delete something that has 80 callers.

## Before launching ANYTHING (Gazebo, campaign, driver)

```bash
pgrep -a "ros2 launch|ign gazebo|run_visibility_campaign"
```

**Non-empty ⇒ do not launch.** Campaigns run for hours and a second Gazebo collides with the
live one on the same ROS topics and gz partition, corrupting both. A campaign may have been
started by a different session, so "I didn't start one" is not evidence that none is running.
To inspect a live run, read its CSVs; that is always safe and needs no ROS env.

The shell must have `source /opt/ros/humble/setup.bash && source install/setup.bash` —
without it every run dies instantly with `Package 'experiments' not found` and is logged as
`infra_invalid`.

Live-run facts worth not re-deriving:
- Output lands in `logs/visibility_comparison/<campaign>/<task>/<cond>/<seed>/experiment_*/`;
  `run_summary.json` appears only when the run **ends**.
- `perception.csv` is flushed at run end. **0 bytes mid-run is normal, not a bug.**
- `experiment.csv` grows steadily — diff its size to tell "running" from "hung".
- `yaw_error_odom_map_vs_odom_rad` sits at a constant −π/2: a frame-convention offset
  between `odom` and `odom_map`, **not** an estimator error. Don't debug the π/2.

## Hard rules

- **Ground truth (`gt_*`, `eval_*`, oracle labels) and CAD shelf geometry are
  EVALUATION-ONLY** — never a model or deployment input, and never a reason to stop a run.
- **Score each quantity at the instant it describes.** Load fusion-study runs through
  `experiments/fusion_on_fixed_routes/aligned.py`; it fixes both the truth alignment and the
  repeated-reading count, and neither is visible in the output when it is wrong.
- **Never pool run directories by glob or "latest"** — use an explicit frozen manifest of run
  IDs, and fail on a missing or extra run.
- **One drive is not a result.** Aggregate within a run, then across seeds.
- **Never hand-roll brier/logloss/AUC/Spearman/ECE** — import `scripts/shared/metrics.py`.
  An audit once found 15 divergent copies and three different Spearman formulas.
- **Never pick a constant by scanning it against the score it will be judged by.** Measure the
  physical quantity that sets it. A "decision rule written first" that still reads the outcome
  is fitting.
- **No synthetic data.** Real Gazebo, real captures. "An unbiased detector" means refusing
  sightings, not simulating them.
- Temp and scratch files: use the session scratchpad, never the repo.
- Don't add bulk media (frame dumps, videos) to git-tracked paths; put them under `logs/` and
  commit only final figures.
