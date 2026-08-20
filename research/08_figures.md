# Figure, evidence and provenance contract

## Maturity rule

A study can be scientifically `LOCKED` while its paper figure is not. For the epochs
`correlated_error_icra_20260810_v1` and `reliability_source_20260810_v1`, every pre-freeze
capture, run, fit, summary and plot is historical-only. A figure becomes
publication-locked only when a paper-specific directory contains the canonical PDF/PNG,
the frozen result summary, an input manifest and a complete provenance sidecar whose hashes
validate. An ignored plot under `logs/` is source evidence, not a publication artifact.

Audit result: no F01-F10 figure has admissible fresh evidence. F01-F05 have substantive
historical source evidence only; F06 has not been executed; F07-F10 lack the fresh
common-input benchmark evidence their intended claims require. Re-rendering an old summary
does not make it fresh.

## F01-F10 evidence map

| ID | Exact source inputs | Current / proposed generator | Maturity and decisive gap | Reviewer questions | Allowed claim |
|---|---|---|---|---|---|
| F01 Structured projection residuals and identifiability | Fresh epoch raw blocks for all 6 sites x 8 yaws x 2 sessions x cameras A-D; epoch-bound world, detector, CAD silhouette and config hashes | Reuse audited residual/pixel-ground code only after synthetic tests; proposed fresh paper wrapper `scripts/paper_figures/correlated_error_icra/render_f01.py` | **BLOCKED / PREREGISTERED.** All 96 capture blocks and 384 camera-block records are absent. E6 is historical rationale only. | RQ09, RQ14, RQ15 | None until fresh held-out confirmation; then C1 within the tested site/yaw support. |
| F02 Conventional-filter inconsistency and correlated-error floor | Fresh epoch trajectories/detections from complete capture/route blocks; newly selected projection and newly fitted floors; evaluation-only truth; offline NIS policy `5.991` | Existing filter-study code may be reused after epoch binding; proposed `render_f02.py` consumes only the fresh frozen summary | **BLOCKED.** Old three-capture results and promoted plots are historical; fresh block/site replication is absent. | RQ11, RQ13, RQ14 | None until fresh replication; then scoped offline belief-honesty only. |
| F03 Calibration policy and held-out effect | Same fresh 96-block matrix as F01; silhouette-only and silhouette-plus-camera predictions; no imported v2/v3/v4 coefficients | Fresh identifiability analyzer followed by a renderer only if the 10 mm benefit/20 mm harm gate passes | **BLOCKED / PREREGISTERED.** No fresh camera term is identified. | RQ07, RQ09, RQ12, RQ14, RQ15 | Bounded fresh camera-policy effect or an explicit no-calibration null. |
| F04 Calibration drift detection and expiry | Fresh session-2 blocks, the newly selected policy, a before-injection frozen fault ladder and change threshold, and epoch-bound JSON | Existing fault-injection logic may be reused; deterministic `render_f04.py` specified below must read the fresh JSON only | **BLOCKED.** Historical v3 values are demo-only; fresh injection does not exist. | RQ04, RQ07, RQ14 | Fresh controlled-injection detection-versus-harm boundary only; no real-drift latency/false-alarm claim. |
| F05 Achievable precision versus availability | Fresh epoch visibility captures/field, freshly estimated residual floors and fixed manifested rate/speed/odometry-growth assumptions | Existing deterministic map code may be reused, but both field and floors must be rebuilt; proposed `render_f05.py` | **BLOCKED.** Old NPZ, summary, plot and 15.7% value are historical only. | RQ01, RQ09, RQ13 | Fresh simulated steady-state map result only; no route/navigation outcome. |
| F06 Matched closed-loop result | Future approved fresh arm pair; fresh `run_summary.json`, canonical `experiment.csv`, detections/corrections and exact world/detector/calibration/field/config hashes under the current epoch | Current `analyse.py` is partial: it now computes NEES/NIS, acceptance/age, bootstrap intervals, plots and a count gate; it still needs generic arms, duplicate/exact-key/epoch gates and exact paired binary inference | **BLOCKED / NOT EXECUTED.** v4 supersedes planned v2/v3, generated configs contain seed 0 only, no arm/endpoint is approved and no fresh run exists. | RQ07, RQ11, RQ13 | At most the fresh scoped consequence or null of the approved causal contrast; never broad safety superiority. |
| F07 Reliability-source held-out prediction | Fresh `p_use` labels and fresh route/capture/site split; constant, distance-only, FOV/range, operational geometry, GP, hybrid, surveyed CAD baseline and any admitted-DL predictions on identical new blocks | New benchmark builder/analyzer and proposed `render_f07.py`; no recovered old rows/fits | **BLOCKED.** The README-quoted five directories are absent, and even recovered contents would be historical under this epoch. | RQ01, RQ02, RQ03, RQ05, RQ06, RQ10, RQ12, RQ14 | C2/C6 only after a fresh common-input benchmark; CAD is a surveyed reference baseline, not an assumed upper bound. |
| F08 Reliability-source route discrimination | Fresh F07 source fields, freshly frozen route library/tasks/prior/planner and route-level summary | New source-specific analyzer and proposed `render_f08.py`; existing generic branch mechanics are code/mechanism reference only | **PLANNED.** No fresh source-specific route difference exists. | RQ13, RQ14 | C3 only after a prespecified fresh paired route difference. |
| F09 Commissioning and deployment decision matrix | Fresh per-source manifests and comparable collection, setup, storage, runtime, update and transfer costs from the new benchmark | Common fresh cost-table generator and proposed `render_f09.py` | **PLANNED.** Existing commissioning summaries cannot populate the table. | RQ02, RQ03, RQ06, RQ09, RQ10, RQ12 | Fresh tested-simulation trade-offs only; no universal recommendation. |
| F10 Method-specific failure gallery | Fresh common split with prespecified occlusion, unsupported-space, range, stale-layout and fallback fields | New frozen failure-audit table and proposed `render_f10.py`; no manually selected old screenshots | **PLANNED.** Existing demos/scattered failures are hypothesis sources only. | RQ04, RQ05, RQ10, RQ14 | A bounded fresh failure mode and fallback per tested method; no exhaustive ranking. |

## Canonical paper-specific layout (proposal only)

Do not add to or overwrite the old generic `paper_artifacts/figures/` surface. On
integration approval, use:

```text
paper_artifacts/
  correlated_error_icra/
    epochs/correlated_error_icra_20260810_v1/
      MANIFEST.json
      figures/
        F01/F01_persistent_residuals.{pdf,png,provenance.json}
        F02/F02_belief_honesty.{pdf,png,provenance.json}
        F03/F03_calibration_policy.{pdf,png,provenance.json}
        F04/F04_drift_lifecycle.{pdf,png,provenance.json}
        F05/F05_achievable_precision.{pdf,png,provenance.json}
        F06/F06_closed_loop_paired.{pdf,png,provenance.json}
      results/F01/...F06/        # immutable fresh summary/table used by each figure
      inputs/F01/...F06/         # fresh manifests, roles, hashes and restore locations
      diagnostics/setup/         # readiness plots; never manuscript evidence by default
      demos/                     # illustrative media; no inferential claims
      archive/                   # superseded within-epoch renders, never canonical
  reliability_source_comparison/
    epochs/reliability_source_20260810_v1/
      MANIFEST.json
      figures/F07/...F10/
      results/F07/...F10/
      inputs/F07/...F10/
      diagnostics/setup/
      demos/
      archive/
```

`MANIFEST.json` names exactly one canonical sidecar/output set per figure and records the
paper version, epoch ID and preregistration timestamp. Source data need not be duplicated
when immutable/hash-restorable, but every scientific source must live below the matching
fresh raw/derived epoch root and the input manifest must bind its path and restore location.

## Provenance sidecar contract

Every `Fxx_*.provenance.json` must contain:

| Section | Required fields |
|---|---|
| Identity | `schema_version`, `paper_id`, `epoch_id`, `preregistered_at`, `figure_id`, title, caption, scope, maturity, claim IDs and reviewer-question IDs |
| Generator | repository-relative generator path, exact command/arguments, deterministic random seed if any, dependency lock/environment, Git commit, dirty-state patch hash and generation timestamp |
| Inputs | every repository/archive path, SHA-256, byte size, creation/capture timestamp, role (`operational`, `evaluation_only`, `control`, `summary`), freshness decision, and restore location |
| Experimental binding | world/detector/calibration/field/config hashes; task, seed, camera and condition lists; capture/route/site definitions; split manifest and independent-unit definition |
| Statistics | primary endpoint, diagnostics, unit of analysis, pairing key, interval/test method, multiplicity policy, missing/invalid rule and null interpretation |
| Outputs | canonical PDF/PNG/result-summary paths with SHA-256 and byte size; plot dimensions and renderer version |
| Boundaries | limitations, forbidden claims, oracle/evaluation-only uses, superseded inputs and known incompatibilities |

Promotion must fail if an input/output hash differs, the Git state is not recorded, a
required independent unit is absent, an input predates the epoch or lies outside its roots,
or a sidecar omits its allowed-claim boundary.
Existing paired-mechanism and GP provenance files do not meet this schema: most lack
F/claim/RQ IDs, Git state, input/output hashes, split/unit definitions and claim limits.

## Deterministic fresh F04 specification (not rendered)

Input is only a newly generated, epoch-bound and frozen `drift_lifecycle.json` for the
scientifically selected current policy; the renderer must not refit a model, resample
detections or import runtime code. The old v3 JSON and its 0.1/0.25 degree results are
historical/demo material and are prohibited inputs.

1. Panel A plots yaw drift magnitude against raw and stale-current-policy RMS, in metres,
   directly from the frozen fresh `summary`. Draw the reported first change detection and
   first harm values without hard-coding their historical values; label the policy/hash.
2. Panel B plots `change_z` against yaw drift for A-D with the new threshold estimated from
   clean session-1 blocks and frozen in the manifest before any session-2 fault injection;
   add a separate inset/row for translation drift. Camera order and colours are fixed. The
   historical threshold `1.2` must not be imported. The absolute statistic may appear only
   as a grey diagnostic.
3. Include the fresh capture-block and unique-site counts from the manifest and the phrase
   “independent units: capture block and unique camera site; deterministic fault ladder.”
   Do not draw a confidence band over detections.
4. Assert before rendering: `identity_check_max_error_m == 0`; epoch, calibration and
   input hashes match the sidecar; every result key is present; the fixed fault ladder and
   change threshold equal their pre-injection manifest. Any failed assertion stops build.
5. Caption boundary: controlled step faults, no temporal latency or false-alarm rate,
   and no transfer beyond the freshly selected policy and tested sites.

## Non-executed F06 specification

F06 remains an empty contract until WS05 approves arms and one primary endpoint.

- Completeness gate: both arms must contain exactly the same 15 unique `(task, seed)` keys
  (three frozen tasks x seeds `0-4`), no duplicates, valid run summaries and bound hashes.
  Every run must be inside the fresh epoch and match the exact frozen key set. An
  intersection of whatever happens to exist is forbidden.
- Analyzer gate: replace hard-coded v2/v3 names with manifest-provided neutral arm IDs;
  reject duplicate or unexpected keys before loading metrics; verify every epoch/input
  hash; and emit exact discordant-pair inference for each declared binary endpoint. The
  present count gate and continuous bootstrap do not satisfy these conditions alone.
- Panel A: paired primary endpoint by task-seed pair. Binary endpoints show the four paired
  cells and exact discordant-pair result; continuous endpoints show paired differences and
  a task-stratified run-level 95% bootstrap interval.
- Panel B: scoped mechanism summaries per run—median/p95 belief error, median NEES and 95%
  coverage. Frame values are reduced within a run before any interval.
- Panel C: NIS and correction process—runtime gate `9.21`, acceptance/rejection, correction
  age and relevant-camera exposure. Offline `5.991` may be shown only as a separately
  labelled historical diagnostic, never as the runtime setting.
- Panel D or supplement: breach/contact, path length, elapsed time, goal distance, invalid
  run ledger and all outcome flips.
- The figure and summary must present a full null without changing endpoints, arms, tasks,
  seeds or exclusions. No figure exists and none was generated in this audit.

### Required analyzer acceptance tests before F06 collection

1. A synthetic exact 3-task x 5-seed matrix passes; one missing, extra, duplicated,
   invalid or mismatched key exits nonzero before any result/plot is written.
2. Any input with the wrong epoch, a pre-freeze timestamp, a path outside the epoch root,
   or a missing/mismatched hash exits nonzero.
3. Arm labels and delta direction come from the frozen manifest; no v2/v3 name or output
   filename is hard-coded.
4. Hand-computed NEES with correlated covariance, logged NIS, acceptance and age fixtures
   match exactly; runtime `9.21` and offline `5.991` remain distinct labelled policies.
5. Binary fixtures exercise zero, one-sided and two-sided discordant pairs and verify the
   exact paired result and risk difference. Continuous fixtures verify the task-stratified
   run-level interval and that frames cannot become bootstrap leaves.
6. Identical inputs, dependency environment and plot seed produce identical summary and
   figure hashes. A failed completeness/provenance test produces no promotable artifact.

## Setup diagnostic plots (not manuscript evidence)

Produce these from fresh readiness inputs under the active epoch's `diagnostics/setup/`
only after an arm decision and before campaign execution. They are fail-closed readiness
checks, not a tuning surface; old readiness plots cannot be copied forward.

1. Matrix-completeness tile for every task, seed and arm, plus duplicate/invalid reasons.
2. Planned route, truth route, belief route, no-go geometry and contact locations for the
   readiness run; axes and world hash shown.
3. Camera exposure and selected/accepted corrections by route segment, including the
   cameras actually affected by the chosen calibration contrast.
4. Measurement/correction age, stale fraction, detector cadence and accepted/rejected
   counts by camera and task.
5. Per-run NEES/coverage and NIS summaries with separate reference lines for runtime
   `9.21` and offline `5.991`; never pool frames as replicates.
6. Noise-seed verification, command/encoder-noise activation, elapsed time, stuck flags,
   path length and final goal distance by task-seed run.
7. Artifact compatibility panel: world, detector, calibration, field, config and code
   hashes, with any unbound July field shown as a hard failure rather than a warning.

Diagnostics may invalidate readiness. They may not be used to select the calibration arm,
retune parameters, choose a favourable seed/task, or become F06 post hoc.

## Existing artifact classification — preserve, do not delete

| Class | Paths / examples | Treatment |
|---|---|---|
| Locked historical source evidence, not epoch-admissible | Relevant `logs/studies/**/{RESULTS.md,summary JSON,plots}` for F01-F05 and tracked E6 | Preserve summaries, tables, plots and generators. They may motivate preregistration and synthetic regression tests but may not be promoted, copied, refitted or re-rendered into either fresh epoch. |
| Prior-paper/current-surface artifacts | `paper_artifacts/figures/current_surface/`, `paper_snapshot/`, top-level paired-mechanism PDFs/data, `campaigns/robustness_campaign_headline/`, `gp/warehouse_visibility_gp_v1/`, `evidence/single_camera_uigp/` | Canonical only for their documented older single-camera/GP surface. They are stale for F01-F10 and cannot answer the current reviewer questions. |
| Explicitly stale/superseded/archive | `figures/archive/`, `figures/_archive_nonpaper_20260612/`, `gp/archive/aws_gp_v7b_superseded/`, `campaigns/archive/`, `metrics/archive/`, `paired_mechanism_taskA_smoke_PREFIX_OLD.pdf` | Retain for traceability, exclude from all canonical manifests. The nonpaper README itself points to a now-absent `paired_mechanism_taskA.pdf`, confirming that its canonical pointer is stale. |
| Diagnostics | Existing `paper_artifacts/figures/diagnostics/` and solve/path/drift diagnostic images | Historical readiness/debug only. New setup diagnostics must be generated inside the epoch and still carry no captioned scientific claim or status promotion. |
| Demo / explainer / setup | `paper_artifacts/figures/explainers/`, `problem_setup_camera.png`, `problem_setup_snapshots.png`, `localization_pathway.png`, `yolo_training_clarification.png`, `logs/studies/multicamera_commissioning_bigwarehouse/demos/` and `four_camera_showcase/` | May illustrate system/setup with an explicit `DEMO` or `SETUP` label. Never count as F01-F10 evidence or an independent replicate. |
| Variant/smoke paired media | Top-level `paired_mechanism_*current*`, `*lowlat*`, `*fixed*`, `*seed2*`, `*smoke*` and their copied data | Preserve as historical variants. Names such as “current” do not establish scientific or publication maturity; exclude unless a future paper manifest deliberately and hash-verifiably promotes one. |

No artifact is reclassified by moving or deleting it in this workstream; classification is
metadata and claim discipline only.
