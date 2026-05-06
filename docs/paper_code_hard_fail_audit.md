# Paper-Code Hard-Fail Audit

This document maps the current codebase into the paper-facing runtime path, diagnostic tooling, and cleanup targets. The guiding rule is simple:

> Paper code should be boring, explicit, reproducible, and hard to misuse.

For this project, the normal runtime behavior is the IWAI/thesis experiment path for visibility-aware active-inference navigation under external-camera observations. Anything outside that path should either be clearly labeled as diagnostic/legacy, moved out of the primary surface, or removed.

## Current Paper-Core Runtime Path

The intended paper runtime path is:

```text
warehouse_primary_comparison.launch.py
-> visibility_launch_common.py
-> Gazebo warehouse world + external camera
-> yolo_robot_detector_node.py
-> pixel_to_bev_state_node.py
-> unicycle_planner_node.py / efe1 planner
-> optional actuation_noise_node.py
-> experiment_logger.py
-> paper metrics and figures
```

Conceptually:

```text
external camera image
-> YOLO segmentation score and pixel point
-> homography x,y state update
-> odometry heading
-> GP observability artifact
-> state-dependent R_eff
-> EFE rollout objective
-> known obstacle map / no-go safety barrier
-> cmd_vel_raw
-> command noise node
-> cmd_vel
-> Gazebo
-> logs
```

This is the path a grader or reviewer should be able to understand without knowing the historical experiments.

## Paper-Core Files

These files should stay first-class and be cleaned for readability.

| Area | Files | Paper role |
| --- | --- | --- |
| Launch | `src/experiments/launch/warehouse_primary_comparison.launch.py` | Main paper launch surface. |
| Launch assembly | `src/experiments/experiments/core/visibility_launch_common.py` | Shared node construction and parameter wiring. |
| Tasks/worlds | `src/experiments/config/tasks.yaml`, `src/experiments/config/world_profiles.yaml` | Benchmark task and world metadata. |
| Detector | `src/perception/perception/nodes/yolo_robot_detector_node.py` | Runtime external-camera detector. |
| State | `src/state/state/nodes/pixel_to_bev_state_node.py` | Pixel-to-BEV conversion and odom-backed heading. |
| Planner node | `src/planning/planning/nodes/unicycle_planner_node.py` | Runtime belief update, planning loop, diagnostics. |
| Planner math | `src/planning/planning/planners/base_planner.py`, `src/planning/planning/core/casadi_efe.py`, `src/planning/planning/core/efe_utils.py` | EFE objective and rollout evaluation. |
| Safety geometry | `src/planning/planning/core/nogo_cost.py`, collision geometry from the world profile | Known obstacle-map barrier shared by all paper conditions. |
| GP map | `src/planning/planning/core/visibility_gp_map.py` | Loaded observability field and planner queries. |
| Actuation noise | `src/sim/sim/actuation_noise_node.py` | Realistic command-space noise/slip for paper experiments. |
| Logger | `src/experiments/experiments/nodes/experiment_logger.py` | Paper evidence: CSVs, manifest, summary. |
| Paper campaign | `scripts/visibility_comparison/run_iwai_campaign.py`, `scripts/visibility_comparison/compute_paper_metrics.py` | Current paper experiment orchestration and metrics. |
| Paper figures | `scripts/paper_figures/make_problem_setting_figure.py` | Paper figure generation. |

## Diagnostic Or Legacy Surfaces

These are useful, but should not be mixed into the paper-core path.

| Area | Examples | Cleanup direction |
| --- | --- | --- |
| Old perception/controller runs | older detector backends and old controller baselines | Do not expose as paper launch conditions. |
| Retained planner variants | broader EFE variants and older sweep profiles | Keep only in diagnostic launch/scripts, not primary paper launch. |
| Rollout probes | `probe_rollout_families.py`, diagnostic videos | Keep as `scripts/diagnostics/`, not as runtime method. |
| Parameter sweeps | broad planner sweeps | Removed from the paper-code surface; rerun only from a separate diagnostic branch if needed. |
| Historical docs/artifacts | `archive/`, old packaged GP/model artifacts | Keep only if explicitly referenced for provenance. |
| One-off figure fallbacks | schematic fallback lines, placeholder panels | Disallow by default for paper figure generation. |

## Hard-Fail Rules For Runtime Behavior

All primary runtime paths should fail early when assumptions are violated. The codebase should not have a separate "paper mode" that behaves more honestly than normal runs; the default behavior should be explicit, reproducible, and hard to misuse.

### Launch-Time Hard Fails

1. `perception_backend` must be `yolo` for paper experiments.
2. `yolo_model` must be provided and must exist.
3. `visibility_artifact_path` must be provided for visibility-aware planners and must exist.
4. The task must be one of the explicit paper tasks, not a legacy/reference task.
5. The world must match the artifact/world metadata.
6. The selected planner must be one of the paper conditions.
7. Command noise settings must be explicitly logged in the manifest.
8. If a planner condition uses constant observation covariance, it must be named as constant-R or visibility-unaware, not "no R".
9. The known obstacle/no-go barrier settings must be explicit in the manifest and paper method section.

### Runtime Hard Fails

1. If the camera model or projection model cannot be initialized, the run should fail.
2. If world bounds or obstacle geometry cannot be loaded, the run should fail.
3. If a YOLO frame is required but the detector cannot load the model, the run should fail.
4. If diagnostics schema fields are missing, logger parsing should fail instead of silently writing NaNs for paper-critical metrics.
5. If TF/world-state truth is unavailable for a paper run, the logger should fail instead of producing incomplete success metrics.
6. If the known obstacle/no-go barrier is enabled, its geometry source and weight must be logged; if it is disabled, the run must be labeled as a no-safety-barrier diagnostic.

### GP / Observability Hard Fails

1. The GP artifact must declare schema version, source target file, world, camera setup, and hyperparameters.
2. The planner must not silently use an old packaged artifact when a paper artifact was expected.
3. The planner should not silently clip out-of-bounds GP queries to the map edge for paper metrics.
4. If clipping is retained as a numerical guard inside symbolic optimization, it must be disclosed and separately logged as a boundary violation risk.
5. The conservative map parameter `beta` must be recorded and reported.

### Plotting / Figure Hard Fails

1. Paper figures must fail if the requested run directory is missing.
2. Paper figures must fail if expected CSVs are missing.
3. Paper figures must fail if they cannot load a real trace, unless an explicit `--allow-schematic` flag is passed.
4. Paper problem-setting figures should require a real Gazebo screenshot for panel (a), unless an explicit `--allow-placeholder-panel-a` flag is passed.
5. Any smoothed or resampled trajectory must be labeled as visualization-only.

### Metrics Hard Fails

1. Collision and penetration fields must exist for any claimed successful run.
2. `timeout_after_first_cmd` must not be counted as goal success unless the goal-distance criterion is also satisfied and the report labels it carefully.
3. "Completed protocol run" and "successful navigation run" must be separate metric concepts.
4. Goal-region entry must be reported separately from completion reason.
5. Stable or idle behavior inside the goal region should terminate as goal-reaching, not as generic `stuck`.
6. Paper metrics scripts should fail if required columns are missing rather than silently outputting NaN.

## Current High-Risk Hidden Behavior

These are the first places to clean because they can quietly change the scientific story.

### 1. GP Boundary Clipping

File: `src/planning/planning/core/visibility_gp_map.py`

Numeric GP queries now reject positions outside the artifact support instead of clipping them to the nearest grid edge. Symbolic optimizer interpolation still uses a clamp as a numerical guard inside CasADi, so selected rollouts must continue to be checked by the numeric diagnostics path.

Required behavior:

```text
numeric query outside support -> hard failure
symbolic clamp -> numerical guard only
selected rollout outside support -> invalid experiment / planner failure, not silent clipping
```

### 2. Implicit Visibility Artifact Fallback

File: `src/experiments/experiments/core/visibility_launch_common.py`

If `visibility_artifact_path` is empty, the launch can fall back to `world_profiles.yaml`. That is convenient for development but bad for paper reproducibility.

Required behavior:

```text
visibility_artifact_path must be explicit
artifact path must exist
artifact metadata must match world/camera/task assumptions
```

### 3. YOLO Paper Path Still Has Legacy Defaults

Files:

```text
src/experiments/launch/warehouse_primary_comparison.launch.py
src/experiments/launch/warehouse_visibility_agent.launch.py
src/experiments/experiments/core/visibility_launch_common.py
```

The current launch surface should not default to non-YOLO perception modes. That makes it too easy to run a non-final method by accident.

Required behavior:

```text
primary paper launch defaults to yolo
legacy perception backends require explicit diagnostic launch
```

### 4. Logger Warnings For Paper-Critical Geometry

File: `src/experiments/experiments/nodes/experiment_logger.py`

Some initialization failures currently warn and continue. Missing camera/world geometry means several metrics and plots can become invalid.

Required behavior:

```text
camera/world/safety geometry failures are fatal
debug runs should use separate scripts, not weaker behavior in the primary launch path
```

### 5. Figure Fallbacks

File: `scripts/paper_figures/make_problem_setting_figure.py`

The script is much closer now, but any fallback schematic or placeholder panel needs an explicit opt-in flag.

Required behavior:

```text
real run required
real CSV trace required
real panel-a screenshot required
no silent schematic trace
```

### 6. Artifact Source Contradiction

Files:

```text
src/experiments/config/world_profiles.yaml
src/experiments/data/visibility_gp/README.md
scripts/visibility_comparison/iwai_campaign_config.yaml
```

The packaged GP artifact location and the current generated GP artifact story are not yet fully aligned. A reader should not have to guess whether `logs/visibility_comparison/current_gp` or `src/experiments/data/visibility_gp` is canonical.

Required behavior:

```text
one canonical artifact path
one provenance README
one schema-verification script
```

### 7. Planner Condition Naming Drift

Files:

```text
scripts/visibility_comparison/run_iwai_campaign.py
src/experiments/launch/warehouse_primary_comparison.launch.py
src/experiments/experiments/core/world_profiles.py
```

Planner names such as `gp_risk_only`, `visibility_unaware_baseline`, and broader retained modes must be consistent across launch validation, campaign config, and report text.

Required behavior:

```text
Condition A: constant-R EFE
Condition B: visibility-aware EFE
optional diagnostic: risk-only visibility ablation
```

### 8. Known Obstacle Barrier Must Be Disclosed

Files:

```text
src/planning/planning/core/nogo_cost.py
src/planning/planning/planners/base_planner.py
src/experiments/experiments/core/visibility_launch_common.py
```

The no-go/obstacle term is not legacy deadwood. It represents the known obstacle map and is a legitimate safety/feasibility component of the navigation problem. The paper risk is that it looks like a hidden route-shaping trick if it is not explicitly described.

Required behavior:

```text
keep obstacle-map barrier active for all main conditions
describe it as a known-map safety term, not as visibility reward
report obstacle/barrier cost separately from EFE risk and ambiguity
log geometry source, enabled flag, weight, and clearance parameters
do not compare methods with different obstacle-barrier settings
```

## Naming Conventions To Enforce

The naming should make method assumptions clear.

### Recommended Paper-Facing Names

| Current / mixed name | Paper-facing name | Why |
| --- | --- | --- |
| `efe1` | `visibility_aware_efe` or `EFE-vis` | Says what the method is. |
| `visibility_unaware_baseline` | `constant_R_efe` | More precise: it still has R, just not state-dependent R. |
| `gp_risk_only` | `risk_only_gp_ablation` | Clearly an ablation, not the main method. |
| `efe_main_fast_soft_risk` | `tuned_visibility_aware_efe` or remove | One-off tuning names should not be paper API. |
| `main_shadow_tradeoff` | `legacy_main_shadow_tradeoff` | Avoid accidental use as current benchmark if it is not. |
| `shadow_tradeoff_a` | `paper_shadow_tradeoff_a` | Explicit paper benchmark. |
| `shadow_tradeoff_b` | `paper_shadow_tradeoff_b` | Explicit paper benchmark. |

### Naming Rules

1. Names should describe method semantics, not tuning history.
2. No run profile should be named after a debugging mood, e.g. `fast_soft_risk`, unless it is archived.
3. Baselines should state what is absent or constant.
4. Ablations should include `ablation`.
5. Legacy tasks and launch files should include `legacy` or move to `archive`.

## Proposed Paper-Facing Structure

This is the target structure for clarity. It does not need to happen in one commit.

```text
UnembodiedNavigation/
  README.md
  docs/
    paper_code_hard_fail_audit.md
    runtime_dataflow.md
    iwai_writer_gpt_context/
  src/
    perception/
    state/
    planning/
    sim/
    experiments/
      config/
        paper_tasks.yaml
        paper_world.yaml
      launch/
        paper_experiment.launch.py
        diagnostic_experiment.launch.py
  scripts/
    paper_campaign/
      run_campaign.py
      compute_metrics.py
      validate_outputs.py
      paper_campaign_config.yaml
    observability/
      capture_samples.py
      build_targets.py
      fit_gp.py
      verify_gp_artifact.py
    paper_figures/
      make_problem_setting_figure.py
      make_result_figures.py
    diagnostics/
      probe_rollouts.py
      make_diagnostic_video.py
  archive/
    legacy_visibility_comparison/
    old_sweeps/
```

The important part is not the exact folder names. The important part is that paper scripts, GP preparation, diagnostics, and legacy sweeps are separated.

## First Cleanup Pass

These changes are high-value and relatively safe.

1. Hard-fail on non-YOLO perception, missing YOLO model, missing GP artifact, invalid task, invalid world, and missing geometry in the primary launch path.
2. Do not add a weaker diagnostic/paper-mode split. If a legacy diagnostic is needed, move it to a separate diagnostic script or launch file.
3. Rename or alias planner conditions to paper-facing names in campaign/report output.
4. Make the paper figure script fail without a real `--run-dir` and real `--panel-a-image`.
5. Split "completed run" from "successful navigation" in metrics names.
6. Add GP artifact verification before campaign runs.
7. Record command noise settings in every paper run manifest and metrics table.
8. Record known obstacle/no-go barrier settings in every paper run manifest and metrics table.
9. Update README with one paper build command, one paper run command, and one paper metrics command.

## Second Cleanup Pass

These are larger and should happen after the primary runtime path is guarded.

1. Move broad comparison and sweep scripts into diagnostics or archive.
2. Move GP capture/target/fit scripts into a clean observability pipeline folder.
3. Remove paper-runtime exposure of old detector/controller runs unless they are part of the final experiment matrix.
4. Replace numeric diagnostics parsing in the logger with a named schema helper.
5. Refactor `experiment_logger.py` into smaller blocks: manifest setup, perception logging, planner logging, safety metrics, summary writing.
6. Refactor `unicycle_planner_node.py` into clearer blocks: belief update, odom yaw anchoring, pixel correction, planning, publishing.
7. Decide whether GP out-of-map queries should be hard invalid rollouts or fatal runtime errors.

## Files To Delete Only After Replacement

Do not delete these casually while experiments are still moving.

| File/type | Why wait |
| --- | --- |
| Old run logs | Needed for comparison and paper figure provenance until final plots are chosen. |
| Packaged GP artifacts | May still be referenced by launch/world defaults. |
| Broad sweep scripts | Useful until final experiment matrix is frozen. |
| Diagnostic rollout/video scripts | Useful for debugging paper failures. |
| Legacy planner modes | Keep until final ablation list is decided. |
| Known obstacle/no-go barrier | Required safety/feasibility term for paper runs; delete only if replaced by a clearer obstacle-map barrier implementation. |

## Grader-Facing Standard

A grader should be able to do this:

```text
1. Read README.
2. Build selected packages.
3. Verify the GP artifact.
4. Run one paper condition.
5. Inspect run_manifest.json.
6. Run metric generation.
7. Regenerate the paper figure from the logged run.
```

At no point should they have to know:

```text
- which old GP artifact was current
- whether the paper run used the required YOLO perception path
- whether a plotted trajectory was schematic
- whether timeout was counted as success
- whether visibility was a reward or a covariance change
```

## Bottom Line

The current codebase contains the right method, but the active surface is still too broad. The cleanup should not start by deleting random files. It should start by making the primary runtime path strict:

```text
explicit inputs
hard failures
single artifact story
paper-facing names
separate diagnostics
no silent fallback plots
```

Once the primary runtime path is strict, redundant code becomes much easier to identify and remove without breaking useful tools.
