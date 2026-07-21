# UnembodiedNavigation — layout rules (read before creating ANY file/folder)

ROS 2 + Gazebo thesis repo: external-camera reliability for warehouse robot navigation.

## The five organizing layers (do not invent new top-level directories)

The repo has one axis per row: **contribution** (what it does) · **claim**
(what we assert) · **runtime** (the ROS code) · **investigation** (where work
happens) · **artifact** (locked evidence). A module *points at* the others; it
never copies their code or data.

| dir | layer | contents | rule |
|---|---|---|---|
| `modules/` | contribution | ONE folder per functional contribution (`NN_name/`), the repo front door; `README.md` is the index | landing pages only: claim + `demos/` (rendered figures) + `baselines/` (external papers reimplemented) + `framings/` (our own paper drafts). Point at `src/`/`experiments/`/`paper_artifacts/`; never own runtime code or data. `modules/07_*` is the multi-camera paper extension. |
| `research_story/` | claim | THE thesis storyline: one folder per chapter (00–11) with claim, gate, `evidence.yaml` manifest | manifests only — point at evidence, never copy it. master plan = `THESIS_PLAN_2026-07-15.md`; statuses = `registry.yaml`; honesty tags = `_shared/honesty_tags.md`. Validated by `tests/research_story/test_manifests.py`. |
| `src/` | runtime | ROS 2 runtime packages only | needs `colcon build`; campaign loads from `install/`, not `src/` |
| `scripts/shared/` | — | THE shared analysis library (`metrics.py`) + `paths.py` (`repo_root()`) | import this; never copy helpers into a script. Prefer `repo_root()` over `Path(__file__).parents[N]` in new code. |
| `scripts/visibility_comparison/` | — | campaign pipeline: capture → targets → GP fit → events | stable; `fit_belief_aware_gp.py` is the canonical GP code — import it, don't reimplement. `common.py` here is THE one (the `scripts/shared` twin was deleted). |
| `scripts/geometry_visibility/` | — | geometry/calibration prior module + `campaign_metrics.py` | stable |
| `scripts/{perception,paper_figures,reliability,...}` | — | tooling per topic | extend, don't fork |
| `experiments/<study_name>/` | investigation | ONE folder per investigation study: code + README + REUSE_MAP | **new studies go here**, nowhere else |
| `logs/studies/<study_name>/` | investigation | that study's outputs (figures, RESULTS.md, csv) | **study outputs go here**, one subfolder per experiment |
| `logs/visibility_comparison/` | — | campaign runs + capture datasets (honest_campaign_v1, whitenoise_campaign_v1, …) | append-only; never edit or rename existing run dirs |
| `paper_artifacts/` | artifact | LOCKED artifacts + ALL locked media — single source of truth (`figures/` root = canonical pdfs+data; `figures/{current_surface,paper_snapshot,explainers,diagnostics}/` = renders) | only promote here deliberately; never duplicate its files elsewhere (link/point instead). `docs/paper_vs_current` holds markdown+configs ONLY (media consolidated 2026-07-15) |
| `docs/` | — | reference + contracts + paper-comparison; see `docs/README.md` (categorized index) | kept flat: several files are path-anchored (loaded by tests/generators, referenced by LOCKED ch.00 + workstream-owned ch.08 manifests) — do not move casually |
| `_archive` (at `../_archive/`) | — | path-preserving graveyard for superseded material | move superseded outputs there, don't delete history |

Root `pyproject.toml` (pytest-only, keeps colcon clean) + `conftest.py` put the
`src/*` packages on the path, so `python3 -m pytest tests/` runs from the repo
root with no per-file bootstrap and no `-p no:anyio` flag.

Grandfathered (pre-convention, leave in place): `logs/{paper_figures,perception_models,perception_datasets,multicamera_commissioning_bigwarehouse,reliability_bigworld_multicamera_story}`, `experiments/multicamera_*`. `logs/optionA_commissioning` and `logs/geometry_visibility_prior` are compat symlinks into `logs/studies/`. The old top-level `yolo/ estimation/ gp/ planning/` demo dirs are now `modules/01_detection`, `02_projection_bev`, `04_reliability_gp`, `08_planning_efe`.

## Starting a new investigation study

1. `experiments/<study_name>/` — scripts + a README saying what question it answers **and
   which `research_story/` chapter it serves**; add the study to that chapter's
   `evidence.yaml` + `research_story/registry.yaml`.
2. Outputs to `logs/studies/<study_name>/<expN_name>/` with figures + `RESULTS.md` each.
3. Reuse before writing: GP = `fit_belief_aware_gp.py`; scoring = `scripts/shared/metrics.py`;
   log loading = `campaign_metrics.py` (see the campaign-metrics skill — column choice is a
   known trap); camera = `unav_common.camera_model.ObliqueCameraModel`; prisms =
   `unav_common.occlusion_geometry` / `geometry_visibility.prisms_from_json`;
   trust→covariance = `reliability.covariance_mapping` (THE single source of truth;
   the offline `geometry_visibility.trust_to_r_plan` is proven identical to ~1e-9).
4. Template/reference study: `experiments/optionA_commissioning/`.

## Hard rules

- **Two-world rule**: method development ONLY in the original warehouse (`warehouse_aws`);
  the full 4-camera world (`warehouse_full_4cam.world.sdf`) evaluates FROZEN methods (scale/handover/fusion). See
  `research_story/README.md`. `honest_campaign_v1` is the locked reference — never rerun/modify.
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
