# WS03 — validation, statistics, evidence and figures

## Objective

Make the validation logic and evidence maturity honest. Map every reviewer concern and
figure to reproducible evidence, define sampling units/statistics/promotion gates, and design
paper-specific provenance. Do not choose scientific methods or run campaigns.

## Ownership

Writable now:

- `research/04_reviewer_questions.md`
- `research/07_validation_matrix.md`
- `research/08_figures.md`

Writable only after proposing a layout and receiving integration approval:

- `paper_artifacts/correlated_error_icra/`
- `paper_artifacts/reliability_source_comparison/`
- publication wrappers under `scripts/paper_figures/`
- figure-manifest validation tests

Read-only:

- registry, papers, experiments, runtime code and calibration artifacts
- all `logs/studies/**/RESULTS.md`, summaries, tables and existing plots
- old `paper_artifacts/figures/` surfaces

This workstream cannot select calibration v2/v3/v4, launch Gazebo, change algorithms,
retune noise/gates, delete artifacts, or convert demos/pilots into evidence.

## Critical blockers to record

1. The active closed-loop contract says v2 versus v3, but the newer held-out projection
   study says v2/v3 are superseded by v4. No F06 or campaign may proceed until WS05 resolves
   this scientifically.
2. Generated `_clv2.yaml` and `_clv3.yaml` currently contain seed 0 only, not seeds 0-4.
3. `closed_loop_calibration/analyse.py` does not compute promised NEES, NIS,
   acceptance/age, confidence intervals or plots, and does not demand 15 matched pairs.
4. Runtime NIS 9.21 and offline NIS 5.991 are different gates and must not be described as
   one setting.
5. Existing usable-observation README numbers reference absent `dataset_v1`,
   `baselines_v1`, `gp_v1` and `planner_conditions_v1` evidence directories. Narrative
   numbers alone are not evidence of record.
6. F01-F05 are marked `LOCKED` in the registry even though decisive plots live under ignored
   logs and generally lack self-contained provenance. Propose truthful maturity states.

## Statistical requirements

- Independent units are task-seed runs, route/capture blocks or unique commissioning sites;
  never frames/detections when estimating inferential uncertainty.
- Binary closed-loop outcomes use exact paired analysis; continuous outcomes use run-level
  paired bootstrap intervals. Fifteen matched pairs permit a mechanism/null result, not a
  broad safety-superiority claim.
- Each validation row names independent variable, controls, unit, split, primary endpoint,
  diagnostics, paired analysis, promotion/stop rule, figure and allowed claim.
- Separate prediction, failure audit, route discrimination, closed-loop navigation,
  deployment trade-offs and camera-management gates.
- Include distance-only and complete-map oracle/upper-bound geometry baselines.
- Null and negative results are permanent.

## Figure/evidence audit baseline

| Figure | Current reality | Main gap |
|---|---|---|
| F01 | residual, two-DOF and projection-amplification evidence exists | canonical v2/v3/v4 decision panel and provenance |
| F02 | strong offline belief evidence exists | promote canonical panel and inputs |
| F03 | commissioning plots exist | definitive calibration policy unsettled |
| F04 | numeric drift result exists | deterministic figure does not |
| F05 | deterministic precision map exists | recompute if adopted floors change |
| F06 | no campaign evidence | valid arms, complete analysis, paired figure |
| F07 | narrative numbers, missing evidence package | recover verified archive or rebuild |
| F08 | generic planner-branch mechanism only | source-specific route discrimination |
| F09 | partial commissioning evidence | comparable costs for all source families |
| F10 | scattered failures | controlled shared failure gallery |

## Provenance contract

Every promoted figure needs a sidecar with figure/claim/reviewer IDs, caption/scope,
generator and Git state, command/dependencies/seeds, all input paths and SHA-256 values,
operational/evaluation roles, world/detector/calibration/field hashes, capture/split/sample
definitions, result-summary hash, output hashes, timestamp, limitations and archive restore
location.

## Deliverables

1. Two validation matrices: current paper and later benchmark.
2. Reviewer attack rows with real experiment/figure answers or explicit limitations.
3. F01-F10 maturity audit and corrected status proposals.
4. A canonical paper-artifact layout and provenance-sidecar schema proposal.
5. F04 deterministic plot specification and F06 specification without execution.
6. Pre-campaign diagnostic plot checklist.
7. Classification of old single-camera paper surfaces, demos and variants without deletion.
8. Handback with exact blockers and proposed registry changes.

## Acceptance criteria

- No figure is proposed `LOCKED` without canonical output, summary and provenance.
- Existing GP nulls and missing evidence are visible.
- F08 generic planner mechanics are not confused with source-specific route evidence.
- Paper figures, setup diagnostics and demo media are three separate classes.
- No inferential result treats frames as independent replicates.
- Every unanswered reviewer concern remains `OPEN` or `LIMITATION`.

## Paste-ready prompt

```text
Work only on validation, statistics, evidence and figures in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Read research/registry.yaml, research/04_reviewer_questions.md,
research/07_validation_matrix.md, research/08_figures.md, both research/papers files,
experiments/closed_loop_calibration/{README.md,make_configs.py,analyse.py},
experiments/usable_observation/README.md, paper_artifacts/, and all relevant locked
RESULTS.md/summary JSON/plots under logs/studies.

You may edit only:
- research/04_reviewer_questions.md
- research/07_validation_matrix.md
- research/08_figures.md

Do not edit registry/status, algorithms, configs or artifacts. Do not launch Gazebo, select
a calibration arm, retune parameters or delete files. First make a read-only evidence audit,
then update only the three owned research documents.

Record these blockers explicitly: E6 prevents selecting any v2/v3/v4 causal arm until
silhouette, camera, region and yaw are separated;
generated closed-loop configs contain seed 0 only; analyse.py lacks NEES/NIS,
acceptance/age, confidence intervals, plot generation and a 15-pair completeness gate;
runtime NIS 9.21 differs from offline 5.991; usable-observation evidence directories quoted
by its README are absent; F01-F05 are not publication-locked merely because ignored-log
figures exist.

Produce separate decision matrices for the current paper and future source benchmark. Use
task-seed runs, capture/route blocks and unique sites as independent units, never frames.
Map F01-F10 to exact inputs, generators, maturity, gaps, reviewer questions and allowed
claims. Specify a paper-specific artifact layout and provenance sidecar, deterministic F04,
non-executed F06, setup diagnostic plots, and stale/demo artifact classification. Return
proposed registry changes and unresolved scientific decisions.
```
