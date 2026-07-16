# UnembodiedNavigation — layout rules (read before creating ANY file/folder)

ROS 2 + Gazebo thesis repo: external-camera reliability for warehouse robot navigation.

## Repo map (do not invent new top-level directories)

| dir | contents | rule |
|---|---|---|
| `research_story/` | THE thesis storyline: one folder per chapter (00–11) with claim, gate, `evidence.yaml` manifest | manifests only — point at evidence, never copy it; master plan = `research_story/THESIS_PLAN_2026-07-15.md`, statuses = `research_story/registry.yaml` |
| `src/` | ROS 2 runtime packages only | needs `colcon build`; campaign loads from `install/`, not `src/` |
| `scripts/shared/` | THE shared analysis library (`metrics.py`, `common.py`) | import this; never copy helpers into a script |
| `scripts/visibility_comparison/` | campaign pipeline: capture → targets → GP fit → events | stable; `fit_belief_aware_gp.py` is the canonical GP code — import it, don't reimplement |
| `scripts/geometry_visibility/` | geometry/calibration prior module + `campaign_metrics.py` | stable |
| `scripts/{perception,paper_figures,reliability,...}` | tooling per topic | extend, don't fork |
| `experiments/<study_name>/` | ONE folder per investigation study: code + README + REUSE_MAP | **new studies go here**, nowhere else |
| `logs/studies/<study_name>/` | that study's outputs (figures, RESULTS.md, csv) | **study outputs go here**, one subfolder per experiment |
| `logs/visibility_comparison/` | campaign runs + capture datasets (honest_campaign_v1, whitenoise_campaign_v1, …) | append-only; never edit or rename existing run dirs |
| `paper_artifacts/` | LOCKED artifacts + ALL locked media — single source of truth (`figures/` root = canonical pdfs+data; `figures/{current_surface,paper_snapshot,explainers,diagnostics}/` = renders) | only promote here deliberately; never duplicate its files elsewhere (link/point instead). `docs/paper_vs_current` holds markdown+configs ONLY (media consolidated 2026-07-15) |
| `_archive` (at `../_archive/`) | path-preserving graveyard for superseded material | move superseded outputs there, don't delete history |

Grandfathered (pre-convention, leave in place): `logs/{paper_figures,perception_models,perception_datasets,multicamera_commissioning_bigwarehouse,reliability_bigworld_multicamera_story}`, `experiments/multicamera_*`. `logs/optionA_commissioning` and `logs/geometry_visibility_prior` are compat symlinks into `logs/studies/`.

## Starting a new investigation study

1. `experiments/<study_name>/` — scripts + a README saying what question it answers **and
   which `research_story/` chapter it serves**; add the study to that chapter's
   `evidence.yaml` + `research_story/registry.yaml`.
2. Outputs to `logs/studies/<study_name>/<expN_name>/` with figures + `RESULTS.md` each.
3. Reuse before writing: GP = `fit_belief_aware_gp.py`; scoring = `scripts/shared/metrics.py`;
   log loading = `campaign_metrics.py` (see the campaign-metrics skill — column choice is a
   known trap); camera = `unav_common.camera_model.ObliqueCameraModel`; prisms =
   `unav_common.occlusion_geometry` / `geometry_visibility.prisms_from_json`;
   trust→covariance = `geometry_visibility.trust_to_r_plan`.
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
- Known trap: `scripts/shared/common.py` and `scripts/visibility_comparison/common.py` are
  twins — edit the `visibility_comparison` one (consolidation pending). Known mismatch:
  `r_miss_uv` = 40 px in offline tooling vs 120 px runtime default in
  `unicycle_planner_node.py` — reconcile before quoting R_plan numbers.
- Don't add bulk media (frame dumps, videos) to git-tracked paths (`docs/`, `src/`);
  put them under `logs/` (gitignored) and commit only final montages/figures.
