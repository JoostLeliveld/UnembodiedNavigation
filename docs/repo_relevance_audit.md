# Repo Relevance Audit

This audit classifies `/home/joostleliveld/Thesis/UnembodiedNavigation` against the current thesis milestone only:

- External camera
- `perception_backend:=yolo`
- `x,y` from camera homography
- `theta` from odometry
- GP observability field
- Primary planner comparison: `efe1` vs `visibility_unaware_baseline`

The full master table is in `docs/repo_relevance_master_audit.csv`.

## Audit Lens

Evidence priority used for classification:

1. Current `thesis-report` wording and section structure
2. Root README and active docs
3. Main launch surface and runtime nodes
4. Representative run and current GP pipeline
5. Static references from imports, launch wiring, configs, scripts, tests, and docs

Important rule: "used somewhere" was not enough to mark something as core. If a file only supports older backends, broader comparison surfaces, or optional tooling, it was classified accordingly.

## Coverage And Checks

- Master table rows: `266`
- Git-tracked files covered: `247 / 247`
- Missing tracked files in the table: `0`
- Extra tracked-root rows in the table: `archive/`, `docs/`, `scripts/`, `src/`, `tests/`

Category counts from the master table:

| Category | Count |
| --- | ---: |
| Paper-Core | 88 |
| Tracked Artifact, Not Source | 80 |
| Possibly Irrelevant | 29 |
| Paper-Core but Simplify | 26 |
| Legacy Compatibility | 16 |
| Generated Noise | 10 |
| Docs Drift / Story Drift | 8 |
| Overkill / Unnecessarily Complicated | 4 |
| External Dependency Artifact | 3 |
| Local Spillover | 1 |
| Dead Code / Dead File | 1 |

## Working Tree Caveats

The audit was done on the current working tree, not a clean checkout. These local changes affect relevance decisions:

- `docs/README.md` is modified.
- `docs/figures/*` is locally deleted, but those assets are still referenced by docs.
- `src/experiments/config/tasks.yaml` is modified and currently diverges from the representative paper run.
- `docs/repo_relevance_master_audit.csv` is currently untracked.

## Highest-Signal Findings

| Finding | Why it matters for the paper | Evidence | Recommended action |
| --- | --- | --- | --- |
| Launch default must remain YOLO-only | The paper story is YOLO-first, so the first launch path must not expose old detector runs | `src/experiments/launch/warehouse_primary_comparison.launch.py` and `visibility_launch_common.py` | Keep defaults and validation pinned to `yolo` |
| `tasks.yaml` no longer matches the paper run snapshot | The paper methodology now names a concrete logged run, so config drift is dangerous | Current working tree goal differs from the representative `experiment_20260423_111758` snapshot | Decide whether `tasks.yaml` should match the paper benchmark exactly or be treated as a newer benchmark revision |
| World profile defaults still point at packaged legacy GP artifacts | The paper now treats `logs/visibility_comparison/current_gp` as canonical | `world_profiles.yaml` still defaults to packaged `src/experiments/data/visibility_gp/*.npz` and older hyperparameters | Replace defaults with one canonical artifact story and archive the old packaged defaults |
| The repo has three competing artifact locations | It is hard to tell what is "source of truth" for models and GP artifacts | `src/experiments/data/**`, `logs/**`, and root `.pt` checkpoints all coexist | Choose one canonical artifact location per artifact type and archive or externalize the rest |
| Planner and launch surfaces should stay narrow | Extra detector/controller run modes inflate the code story | Runtime surface should focus on `efe1` vs baseline | Keep diagnostics out of the primary launch path |
| Docs still describe older math and older surfaces | This makes the repo look less controlled than the implementation actually is | `docs/planner_method.md`, `src/planning/README.md`, `README.md`, and optional script READMEs all drift | Tighten docs around the exact paper path and archive broader notes |
| `docs/figures` is currently half-present in git history and half-removed locally | This creates broken docs and extra ambiguity about what is maintained | `docs/README.md` still references deleted figure assets | Either restore and keep them consistently, or remove all references and treat them as external outputs |

## Hotspot Subtrees

These are the noisiest areas relative to the current thesis scope.

| Subtree | What it is doing now | Audit signal |
| --- | --- | --- |
| `src/experiments/data/perception_models/**` | Packaged trained-model outputs and Ultralytics run artifacts | Strong artifact noise. Useful for provenance, not source. |
| `src/experiments/data/visibility_gp/**` | Older packaged GP artifacts | Compatibility artifact source, not the current canonical paper source. |
| `archive/visibility_legacy/**` | Historical scripts and docs generators | Clean archive candidate. |
| `docs/figures/**` | Tracked docs assets with current working-tree deletions | Needs a binary keep-or-remove decision. |
| `scripts/visibility_comparison/**` | Mixed core tooling, broad sweeps, and stale READMEs | Keep the core pipeline; demote the broad diagnostics. |
| `scripts/perception/**` | Detector training and dataset tooling | Mostly optional for the current paper path. |

## Focused Code Appendix

### `src/experiments/experiments/core/visibility_launch_common.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 157-178 | `_state_estimator_metadata()` | paper-core | Encodes the thesis state-estimation story shown in logs and manifests | Keep, but make it the single place where the state story is declared |
| 181-363 | `parse_common_launch_config()` | keep but simplify | Central launch config parser, but it carries too many knobs for the current paper | Split paper-default args from optional research args |
| 189-201 within parser | `perception_backend` default path | paper-core | Current thesis is YOLO-first | Keep default and validation pinned to `yolo` |
| 494-767 | `build_shared_nodes()` | keep but simplify | This is the real runtime assembly point for the paper pipeline | Break into smaller builders by concern: perception, state, planner, logging |
| 553-589 | detector node construction | paper-core | `yolo` is the active detector path | Keep this branchless in the paper launch path |
| 770-919 | `build_agent_runtime_actions()` | keep but simplify | Core launch logic for the planner comparison | Narrow the active comparison surface |
| 783-804 | planner whitelist and planner params | paper-core but simplify | Current paper compares `efe1` vs `visibility_unaware_baseline` | Keep non-paper variants out of the primary launch |

### `src/experiments/experiments/nodes/experiment_logger.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 80-703 | `__init__()` | keep but simplify | Creates the experiment record that underpins the paper evidence | Split manifest setup, run CSV schema, perception CSV schema, and subscriptions into helpers |
| 560-640 | main CSV header definition | paper-core | The paper depends on this file being explicit and auditable | Keep, but prune columns that are only for legacy compatibility |
| 823-832 | `_heading_source_name()` | paper-core | Useful because the paper now explicitly discusses `theta` from odometry | Keep |
| 1110-1229 | `_log_perception_sample()` | paper-core | Core to the YOLO-to-world and calibration evidence path | Keep, but remove fields you no longer analyze |
| 1136-1163 within perception logging | homography-based predicted world from selected pixel | paper-core | This still supports the current `x,y` from homography story | Keep and document it clearly as analysis-only logging |
| 1231-1647 | `_log_once()` | keep but simplify | Central run logging path, but currently too monolithic | Split state extraction, planner diagnostics parsing, geometry safety, and CSV writing |
| 1448-1493 | planner diagnostics unpacking by numeric index | overkill workaround | Works, but brittle and hard to read | Replace raw index parsing with a named diagnostics schema helper |
| 1552-1559 | `legacy_x`, `legacy_y`, `legacy_yaw` | legacy compatibility | Clearly retained for older downstream expectations | Remove after downstream readers stop depending on them |
| 1649-1770 | `_finish_run()` | paper-core | Finalizes the exact outputs used for run summaries and reports | Keep |

### `src/planning/planning/planners/base_planner.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 360-390 | visibility shaping and precision-blended `R_plan` | paper-core | This is central to the current paper formulation | Keep and mirror it exactly in docs |
| 400-417 | `planning_visibility_diagnostics()` | paper-core | Drives the planner-side observability summaries that appear in logs | Keep |
| 633-677 | `_simulate_target_tracking_controls_flat()` and `_visibility_target_np()` | overkill workaround | Helpful for optimizer seeding, but hand-crafted and tightly tied to one map family | Keep only if it still materially improves solve robustness |
| former optimizer seeding block | removed from paper runtime | Multistart route-family seeds made the planner harder to interpret than the paper needs | Keep route-family evaluation as an offline diagnostic only |
| 833-915 | `_get_casadi_valgrad()` and cache plumbing | keep but simplify | Important for runtime speed, but a readability hotspot | Refactor cache keying and isolate symbolic backend setup |
| 1065-1258 | `plan()` | keep but simplify | This is the main planner entry point for the paper comparison | Split candidate generation, optimization, candidate scoring, and fallback handling |

### `src/planning/planning/nodes/unicycle_planner_node.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 33-450 | `__init__()` | keep but simplify | Core runtime node, but too many parameters and mode flags are declared here | Separate paper path parameters from optional diagnostics and fallback modes |
| 704-921 | `_apply_pixel_correction()` | paper-core but simplify | This is the bridge between pixel observations and planner belief updates | Keep, but factor out stale-time checks, covariance checks, and yaw update logic |
| 923-1029 | `_resolve_belief_for_planning()` | paper-core | Encodes the current "pixel x,y, odom theta" planning belief story | Keep |
| 977-1023 within belief resolution | no-pixel-correction fallback branch | legacy compatibility | Useful for alternate modes, not for the current paper path | Demote to legacy mode or a separate node path |
| 1134-1171 | `_publish_planner_diagnostics()` | keep but simplify | Logger and offline analysis depend on this payload | Centralize the diagnostics schema so node and logger cannot drift |
| 1173-1279 | `_plan_once()` | paper-core | Main plan loop for the paper | Keep, but reduce logging and failure-handling clutter in the hot path |

### `src/planning/planning/core/casadi_efe.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 161-171 | `expected_visibility_ca()` | paper-core | Encodes belief-level expected observability for the GP field | Keep |
| 184-204 | visibility shaping and precision blending | paper-core | Mirrors the method section directly | Keep and keep docs synchronized |
| 250-262 | `et1_ca()` | paper-core | Current paper path uses ET1 | Keep |
| 263-287 | `et2_ca()` | legacy compatibility | Retained for broader comparisons, not the current primary paper claim | Keep only if you still intend to publish ET2 comparisons |
| 318-388 | `visibility_aware_unicycle_efe_ca()` | paper-core | The core symbolic objective for the paper | Keep |
| 390-469 | `make_efe_valgrad_fn()` | keep but simplify | Necessary, but dense and easy for docs to drift away from | Factor symbolic setup into smaller helpers and document the cache contract |

### `src/perception/perception/nodes/yolo_robot_detector_node.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 46-95 | `__init__()` | paper-core | This is the runtime detector used by the current paper path | Keep |
| 99-107 | `_predict()` | paper-core | Thin, readable wrapper around YOLO inference | Keep |
| 111-176 | `_publish_diagnostics()` | paper-core | Supplies the observability scores and detection metadata used downstream | Keep |
| 178-217 | `_image_cb()` | paper-core | Main runtime observation path | Keep |
| whole file | overall structure | paper-core | This is one of the cleanest files in the current stack | Use it as the style reference for cleanup elsewhere |

### `src/state/state/nodes/pixel_to_bev_state_node.py`

| Lines | Block | Classification | Relevance now | Cleanup note |
| --- | --- | --- | --- | --- |
| 35-114 | `__init__()` | paper-core but simplify | This node defines the current state-estimation story | Keep, but separate camera geometry, trust config, and heading-source config |
| 196-231 | `live_detection_trust()` | paper-core but simplify | This is the local update-trust rule for turning YOLO scores into metric uncertainty | Keep and document the thresholds explicitly |
| 233-351 | `_pixel_callback()` | paper-core | Core `pixel -> BEV state` conversion path | Keep |
| 258-281 within callback | precision-blended update noise from live trust | paper-core | This is the key methodological bridge from detector confidence to measurement covariance | Keep |
| 304-311 within callback | odom heading fallback | paper-core | Current thesis status explicitly uses odometry for `theta` | Keep |
| 313-320 within callback | motion heading fallback | legacy compatibility | Helpful safety net, but not part of the current paper story | Keep only as a guarded fallback or remove if odometry is always available |

## Cleanup Order

### Safe now

- Ignore or prune generated roots from the active repo story: `.venv/`, `.pytest_cache/`, `build/`, `install/`, `log/`
- Move root checkpoint noise out of the repo root: `yolo11n.pt`, `yolo26n.pt`
- Remove the accidental local spillover directory `UnembodiedNavigation/` if it is still empty and accidental
- Archive `archive/visibility_legacy/**` farther away from the active source tree
- Move `scripts/perception/**` under an `optional_training/` or separate training-tools area
- Remove the tracked `.codex` stub file if you do not intentionally use it

### Needs confirmation

- Whether `src/experiments/config/tasks.yaml` should be pinned back to the paper run or kept as a newer benchmark revision
- Whether packaged model artifacts under `src/experiments/data/perception_models/**` should remain versioned in-repo
- Whether `src/experiments/data/visibility_gp/**` should remain as a packaged fallback after switching defaults
- Whether `docs/figures/**` should be restored and maintained or fully removed from docs
- Whether broader diagnostic planner variants still deserve first-class support

### Remove last after replacement

- `legacy_x`, `legacy_y`, `legacy_yaw` and any downstream readers that still expect them
- Raw numeric planner-diagnostics unpacking in the logger once a named schema helper exists
- Motion-heading fallback in `pixel_to_bev_state_node.py` if odometry becomes the only supported heading source
- Expanded multistart seed families and visibility-recovery heuristics if a smaller optimizer surface remains stable
- Optional broad comparison scripts in `scripts/visibility_comparison/` once the paper-core capture/fit/plot path is isolated

## Bottom Line

The main thesis code path is present and coherent, but the active story is diluted by:

- legacy perception backends
- retained planner families
- duplicate artifact sources
- stale docs
- optional training and sweep tooling mixed into the main repo surface

If you want the codebase to feel paper-worthy, the biggest win is not deleting random files first. It is collapsing the active surface so that the first things a reader sees are:

1. one canonical launch path
2. one canonical detector artifact location
3. one canonical GP artifact location
4. one exact benchmark config
5. a tighter planner/runtime/logging surface that matches the paper text
