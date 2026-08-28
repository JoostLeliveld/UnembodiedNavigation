# UnembodiedNavigation

ROS 2 + Gazebo thesis repo: external-camera reliability for warehouse robot navigation.

## What this actually does (read this first)

📌 **[`PLAN.md`](PLAN.md) is the plan of record** — the five sentences the paper has to
earn, what is done, what is still owed, and which data serves which purpose. Read it before
proposing work. Anything that does not serve one of those five sentences belongs in a study
README, not in the paper and not in a status report.


A small robot drives around a simulated warehouse. Its wheel odometry drifts without bound, so
it leans on four fixed wall-mounted cameras that can see it on the floor. Cameras are not
equally useful everywhere — shelves block views, accuracy degrades with range and viewing
angle, and a camera can be quietly miscalibrated. The thesis question is therefore **not** "can
the robot see itself" but **"how good is this particular sighting, and does knowing that change
how the robot should drive?"**

Before quoting any accuracy or uncertainty number, read
[`docs/localization_metrics.md`](docs/localization_metrics.md). It is the canonical boundary
between camera measurements, filter beliefs, honesty statistics, navigation runs, and
historical projection versions.

The whole pipeline, in order:

```text
wheel odometry (drifts)  ─┐
                          ├─► EKF: predict, then update through a gate chain ─► belief ─► planner
camera image                                                                   (mean +      (route +
  ─► YOLO box                                                                  covariance)   clearance)
  ─► bottom-centre pixel ─► ray ─► floor plane  = position   (plain IPM, 0 params, 66.6 mm)
  ─► pixel uncertainty pushed through J         = covariance (R_xy = J R_uv Jᵀ)
  ─► four cameras reconciled by camera_manager
```

**The central object is the *belief*: a position AND a stated uncertainty.** Most of this
thesis is about whether that stated uncertainty is honest, not about whether the position is
accurate. Those are separate claims and a filter can pass one while badly failing the other.

**The core finding:** a camera can have a repeated *lean* — the same error every frame. A
Kalman filter treats N sightings as N independent votes and shrinks its uncertainty like `1/N`,
but the lean does not shrink. In the locked historical-v2 belief study, the robot states a
mean 1.9 cm 1σ while its belief has 5.3 cm RMSE, and the truth escapes its stated 95% ellipse
41.9% of the time. Those magnitudes are mechanism evidence under retired inputs, not current
per-camera accuracy. Fixing the mechanism needs a covariance *floor* that repeated looks
cannot shrink — no per-frame noise model can represent a constant bias.

📖 **[`docs/METHOD.md`](docs/METHOD.md) is the start-to-finish walkthrough with the glossary
(belief, NEES, NIS, `R`, `Q`, IPM, bias) and an explicit list of what is settled vs. open.**
Read it before reading `research/`. If you find yourself reading registry IDs to work out what
the system does, you are in the wrong file.

### Talk and write in plain terms

The `research/` control plane uses short IDs (`EXP-BELIEF`, `C1`, `RQ11`, `WS05`) for
bookkeeping. They are *filing labels, not explanations*. When reporting to the user, in figures,
and in READMEs: say the thing, not the label. "The truth falls outside the stated 95% ellipse
41.9% of the time" — not "EXP-BELIEF NEES exceeds the calibrated reference". Define a term at first use or
don't use it.

### Before reporting a number, ask whether it goes in the paper

Most analysis produces intermediate quantities. Almost none of them belong in a reply.
Apply this test to every number before it reaches the user:

**Write the sentence it would appear in, in the paper. If you cannot write that
sentence, do not report the number.**

Concretely:

- **Report in centimetres, percentages and seconds.** Not pixels, not unit-less ratios,
  not log-likelihoods. Those are working units — convert before speaking. "Half a
  centimetre of consistent lean" not "0.198 px pooled mean residual".
- **One headline number per finding.** A six-row table of model variants is a working
  artifact; the reply gets the winner and one sentence on why. Put the table in the
  study's README where a reader can go looking for it.
- **Never coin a metric name or use one without defining it in the same breath.** If a
  diagnostic needs a symbol to explain, describe what it measures in words instead.
- **Lead with the verdict, then the mechanism, then the caveat.** Not the derivation.
  "The sensor is good enough to use, because X, but Y is still untested."
- **If a result is a null, an artifact, or a correction to something you said earlier,
  say so in the first sentence** — do not bury it after the supporting numbers.
- **Machinery is not a finding.** Conventions, split definitions, parameter counts,
  reproducibility hashes: these belong in the study README, not in a reply, unless the
  user asked how something works or the machinery is itself the answer.

The failure mode this exists to stop: reporting the full chain of reasoning that produced
an answer, at the same level of detail as the answer, so the answer is impossible to find.
When in doubt, say the conclusion in two sentences and offer the detail.

### Figures must stand alone

Every promoted figure is read by someone who has not read the code. So:

- **The title states the finding**, not the variable names. "Under retired v2 inputs, repeated bias makes the filter overconfident:
  what each filter does about it" — not "NEES by arm".
- **Axis labels say what the number means** and which direction is good. If a threshold matters,
  draw it and label it in words ("above the line = overconfident").
- **No bare arm/condition codes as tick labels.** `A0`/`A2` need their meaning on the axis
  ("trust every camera", "per-camera noise model") or in a legend, not in a caption only the
  author will read.
- **Honesty and sharpness always appear together.** A filter that just widens its ellipse passes
  any calibration test and is useless, so never show a calibration score without the stated σ
  and the actual error beside it.
- Say what the data is: how many detections, real or injected fault, which world.

## The five organizing layers (do not invent new top-level directories)

The repo has one axis per row: **contribution** (what it does) · **claim**
(what we assert) · **runtime** (the ROS code) · **investigation** (where work
happens) · **artifact** (locked evidence). A module *points at* the others; it
never copies their code or data.

| dir | layer | contents | rule |
|---|---|---|---|
| `research/` | claim | THE control plane: `registry.yaml` (only machine authority), generated `STATUS.md`, numbered science docs, `papers/`, `workstreams/` | statuses live in `registry.yaml` and nowhere else; never hand-edit `STATUS.md`. Validate with `scripts/research/{validate_registry,build_status,hygiene_check}.py`. |
| `figures/` | decision | EVERY plot decisions are made from, one folder per `registry.yaml` experiment ID, each with a `.provenance.json` | tracked, browsable. Promote with `scripts/research/promote_figures.py`; never hand-copy. Raw plots stay in the ignored `logs/studies/` — this is the copy that survives. |
| `src/` | runtime | ROS 2 runtime packages only | needs `colcon build`; campaign loads from `install/`, not `src/` |
| `scripts/shared/` | — | THE shared analysis library (`metrics.py`) + `paths.py` (`repo_root()`) | import this; never copy helpers into a script. Prefer `repo_root()` over `Path(__file__).parents[N]` in new code. |
| `scripts/visibility_comparison/` | — | campaign pipeline: capture → targets → GP fit → events | stable; `fit_belief_aware_gp.py` is the canonical GP code — import it, don't reimplement. `common.py` here is THE one (the `scripts/shared` twin was deleted). |
| `scripts/geometry_visibility/` | — | geometry/calibration prior module + `campaign_metrics.py` | stable |
| `scripts/{perception,paper_figures,reliability,...}` | — | tooling per topic | extend, don't fork |
| `experiments/<study_name>/` | investigation | ONE folder per investigation study: code + README + REUSE_MAP | **new studies go here**, nowhere else |
| `logs/studies/<study_name>/` | investigation | that study's outputs (figures, RESULTS.md, csv) | **study outputs go here**, one subfolder per experiment |
| `logs/visibility_comparison/` | — | campaign runs + capture datasets (honest_campaign_v1, whitenoise_campaign_v1, …) | append-only; never edit or rename existing run dirs |
| `paper_artifacts/` | artifact | PUBLICATION-LOCKED artifacts only, provenance-sealed | only promote here deliberately; never duplicate its files elsewhere (link/point instead). `docs/paper_vs_current` holds markdown+configs ONLY. The retired IWAI figure bulk (animations, per-figure dumps, legacy archive) was cold-archived 2026-08-06; canonical PDFs stay here. |
| `docs/` | — | reference + contracts + paper-comparison; see `docs/README.md` (categorized index) | kept flat: several files are path-anchored (loaded by tests/generators, referenced by LOCKED ch.00 + workstream-owned ch.08 manifests) — do not move casually |
| `_archive` (at `../_archive/`) | — | path-preserving graveyard for superseded material | move superseded outputs there, don't delete history |

Root `pyproject.toml` (pytest-only, keeps colcon clean) + `conftest.py` put the
`src/*` packages on the path, so `python3 -m pytest tests/` runs from the repo
root with no per-file bootstrap and no `-p no:anyio` flag.

Grandfathered (pre-convention, leave in place): `logs/{paper_figures,perception_models,perception_datasets,multicamera_commissioning_bigwarehouse}`, `experiments/multicamera_*`.

`logs/perception_{models,datasets}` and `local_artifacts/` look misfiled — they are inputs,
not run output — but **do not move them**: 82 files reference those exact paths, including
`research/registry.yaml` and locked campaign configs whose reproduction commands would break.

## Starting a new investigation study

1. `experiments/<study_name>/` — scripts + a README saying what question it answers **and
   which `research/registry.yaml` experiment it serves**; add the experiment entry there,
   setting `study_path` to this folder (that key is what binds figures to the experiment).
2. Outputs to `logs/studies/<study_name>/<expN_name>/` with figures + `RESULTS.md` each,
   then `python3 scripts/research/promote_figures.py` to publish them into `figures/`.
3. Reuse before writing: GP = `fit_belief_aware_gp.py`; scoring = `scripts/shared/metrics.py`;
   log loading = `campaign_metrics.py` (see the campaign-metrics skill — column choice is a
   known trap); camera = `unav_common.camera_model.ObliqueCameraModel`; prisms =
   `unav_common.occlusion_geometry` / `geometry_visibility.prisms_from_json`;
   trust→covariance = `reliability.covariance_mapping` (THE single source of truth;
   the offline `geometry_visibility.trust_to_r_plan` is proven identical to ~1e-9).
4. Template/reference study: `experiments/external_camera_bias_model/` (LOCKED, three evidence
   summaries, 18 promoted figures).

## Answering questions (read this before searching)

- **Search order: `src/` and `scripts/` first, `tests/` last.** Only read `tests/` when the
  question is about coverage, or when src+scripts genuinely didn't answer it. Test names here
  are long and descriptive, so they match nearly every semantic grep — `covariance` hits 44 src
  files and 43 test files, `readiness` hits 3 tests for 1 src file. Reading tests to learn what
  code does costs ~2× the tokens and gives you the assertion, not the implementation.
- **Answer from a run, not from reasoning.** If a question can be settled by executing something,
  execute it. Don't describe what a launch/script "would" do. Prefer, in order: run the thing →
  read the artifact it produced → read the source. Reason from the output you just got.
- **Don't trust old data.** Numbers in `RESULTS.md`, memory, docs, and prior summaries describe
  the repo *as it was*. Re-measure before quoting, and say which run/date a number came from.
  A stale artifact that still parses is the most expensive kind of wrong here.
- **Full suite is cheap — just run it.** `python3 -m pytest -q` from the repo root: 965 tests,
  ~20 s idle / ~90 s while a campaign is using the CPU (both measured 2026-08-06). Never
  speculate about whether something is covered; run it.

### LSP: use it for two things, don't trust it for the rest

Measured on this repo 2026-08-06 (Pylance via `ms-python.vscode-pylance`):

| operation | verdict | use it for |
|---|---|---|
| `documentSymbol` | **works** | file outline — full class/function/constant map of a 300-line module for a fraction of reading it. **Do this before Read** on any unfamiliar file. |
| `workspaceSymbol` | **works** | "where is `X` defined" by name, no grep guessing |
| `findReferences` | **broken here** | — returned **1** hit for `ObliqueCameraModel`; grep found **80** across 30+ files |
| `goToDefinition` / `hover` | **unreliable across packages** | — imports report `could not be resolved` |

**Why:** ROS colcon layout puts packages at `src/<pkg>/<pkg>/`, importable only via
`extraPaths`. `pyrightconfig.json` sets these correctly **but sits in `UnembodiedNavigation/`
while the VSCode workspace root is `/home/joostleliveld/Thesis` one level up**, so Pylance
never loads it. A mirrored `pyrightconfig.json` now exists at the Thesis root (added
2026-08-06, **unverified** — needs an LSP/window reload to take effect).

**Rule: for "who calls / uses this?" use `grep`, not `findReferences`** — a wrong-but-confident
"1 reference" is how you delete something that has 80 callers. Re-test the table above if the
LSP config is ever fixed.

## Before launching ANYTHING (Gazebo, campaign, driver)

```bash
pgrep -a "ros2 launch|ign gazebo|run_visibility_campaign"
```

**Non-empty ⇒ do not launch.** Campaigns run for hours and a second Gazebo collides with the live
one on the same ROS topics + gz partition and corrupts both. Campaigns are often started by a
*different* Claude session (config lives in that session's scratchpad, not the repo) — so "I
didn't start one" is not evidence that none is running. To inspect a live run, read its CSVs
instead; that is always safe and needs no ROS env.

Live-run facts worth not re-deriving (verified 2026-08-06 on `clv2_pilot`):
- Outputs land in `logs/visibility_comparison/<campaign>/<task>/<cond>/<seed>/experiment_*/`;
  `run_summary.json` appears only when the run **ends** (`completion_reason` = `goal`/`stuck`/…).
- `perception.csv` is flushed at run end. **0 bytes mid-run is normal, not a bug.**
- `experiment.csv` grows ~150 KB / 20 s — diff its size to tell "running" from "hung".
- `yaw_error_odom_map_vs_odom_rad` sits at a constant **−π/2 (std 0.0)**: a frame-convention
  offset between `odom` and `odom_map`, **not** an estimator error. `..._vs_belief` ≈ 0.007 rad is
  the one that reflects real heading quality. Don't debug the π/2.

## Hard rules

- **Locked campaigns**: `honest_campaign_v1` and `honest_campaign_v2` are the published
  reference runs — never rerun or modify them.
  (The old **two-world rule**, which confined method development to `warehouse_aws` and
  reserved `warehouse_full_4cam` for frozen methods, was **retired 2026-08-20**. It predated
  `warehouse_v2`, which is now the development world. Choose a world for what it measures and
  say why in the study README.)
- **Never** hand-roll brier/logloss/AUC/Spearman/ECE — import `scripts/shared/metrics.py`
  (an audit found 15 divergent copies; three different Spearman formulas).
- **Never** read `state_x/y` or `truth_x/y` from campaign CSVs as belief/truth — use
  `campaign_metrics.load_run/load_detections` (asserts canonical columns).
- Ground truth (`gt_*`, `eval_*`, oracle labels) and CAD shelf geometry are
  **EVALUATION-ONLY** — never a model/deployment input.
- Temp/scratch files: use the session scratchpad, never the repo.
- Known mismatch: `r_miss_uv` = 40 px in offline tooling vs 120 px runtime default in
  `unicycle_planner_node.py`. The mapping math is reconciled (`reliability.covariance_mapping`,
  identical to ~1e-9); the divergence is only the miss-endpoint constant, and
  `MissEndpointPolicy.require_reconciled()` blocks quoting 40 or 120 until the
  residual-tail measurement is made on real data.
- Don't add bulk media (frame dumps, videos) to git-tracked paths (`docs/`, `src/`);
  put them under `logs/` (gitignored) and commit only final montages/figures.
