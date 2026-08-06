# Figure, evidence and provenance contract

## Maturity rule

A study can be scientifically `LOCKED` while its paper figure is not. A figure becomes
publication-locked only when a paper-specific directory contains the canonical PDF/PNG,
the frozen result summary, an input manifest and a complete provenance sidecar whose hashes
validate. An ignored plot under `logs/` is source evidence, not a publication artifact.

Audit result: F01-F05 have substantive source evidence but are not publication-locked;
F06 has not been executed; F07-F10 lack the common-input benchmark evidence their intended
claims require.

## F01-F10 evidence map

| ID | Exact source inputs | Current / proposed generator | Maturity and decisive gap | Reviewer questions | Allowed claim |
|---|---|---|---|---|---|
| F01 Structured projection residuals and identifiability | Existing capture roots, v2/v3/v4 calibration artifacts and residual/projection summaries, plus tracked E6 evidence under `paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/` | Existing residual and pixel-ground generators. Proposed paper wrapper: `scripts/paper_figures/correlated_error_icra/render_f01.py`. | **READY, not publication-locked.** E6 shows that the prior C/D camera-bias interpretation is confounded by object silhouette and fixed route yaw. The canonical panel must show that challenge rather than selecting a preferred calibration. | RQ09, RQ14 | C1 only: structured, repeated projection residuals exist and projection amplification varies spatially; their decomposition into robot-geometry and camera-specific terms remains unresolved. |
| F02 Conventional-filter inconsistency and correlated-error floor | `logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting/{summary.json,fig_s1_honesty_and_sharpness.*,fig_s2_where_the_trust_goes.*}`; `logs/studies/bayesian_filter_showcase/exp2_does_it_generalize/{summary.json,fig_v1_generalization.*}`; `logs/studies/operational_residual_rcond/exp{2_operational_rcond,3_two_dof_rcond}/operational_rcond.json`; the same three capture roots as F01. Offline NIS is `5.991`. | Existing: `experiments/bayesian_filter_showcase/{exp1_graceful_vs_trusting.py,exp2_does_it_generalize.py}`. Proposed: `scripts/paper_figures/correlated_error_icra/render_f02.py`, consuming summaries rather than refitting. | **READY, not publication-locked.** Strong offline result and leave-one-capture-out diagnostic exist; no paper-specific canonical output/sidecar. | RQ11, RQ13, RQ14 | C1/C3: repeated biased measurements make the offline belief overconfident, and per-camera flooring restores honest uncertainty on these recordings without material RMSE loss. No navigation claim. |
| F03 Calibration policy and held-out effect | Existing residual, commissioning and v2/v3/v4 artifacts plus the tracked E6 summary | Existing calibration and commissioning generators; future identifiability study required before a paper renderer. | **BLOCKED.** E6 explains away the C/D gate signal under the object model while A/B become confounded outliers. A policy panel is prohibited until camera, region and yaw are independently varied. | RQ07, RQ09, RQ12, RQ14 | Historical result only: fitted terms can help or harm on their original split. No identified camera-calibration policy claim yet. |
| F04 Calibration drift detection and expiry | `logs/studies/calibration_drift_lifecycle/exp1_stale_correction/drift_lifecycle.json` (SHA-256 `eecc912a1bb387da1afc2c00e1e00d96ff776827746d95be23e4fb69c28c2b4a`); `logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data`; `logs/studies/multicamera_commissioning_bigwarehouse/projection_calibration_v{2,3}/projection_calibration.json` | Existing `experiments/calibration_drift_lifecycle/drift_lifecycle.py` generates JSON only. Proposed deterministic `scripts/paper_figures/correlated_error_icra/render_f04.py`, specified below. | **BLOCKED.** Numeric controlled-fault evidence exists; no plot exists, and it tests a historical v3 term whose interpretation is unresolved after E6. | RQ04, RQ07, RQ14 | In the explicit v3 controlled-injection scope, the change statistic detects camera-C yaw drift at 0.1 degrees before first harm at 0.25 degrees. No real-drift latency, false-alarm or current-policy claim. |
| F05 Achievable precision versus availability | `paper_artifacts/gp/warehouse_full_4cam_fused_v1/fused_planner_four_camera.npz`; fixed residual floors in `experiments/achievable_precision_map/exp1_precision_vs_coverage.py`; `logs/studies/achievable_precision_map/exp1_precision_vs_coverage/summary.json` (SHA-256 `c0e2f5427e28ccc5132c3898db7b30a9e968c2dd897cead756a2a6d802d6a6d8`) | Existing: `experiments/achievable_precision_map/exp1_precision_vs_coverage.py`. Proposed: `scripts/paper_figures/correlated_error_icra/render_f05.py` or promote an exact deterministic regeneration after floor reconciliation. | **READY, not publication-locked.** Deterministic plot exists in ignored logs. It must be regenerated if adopted floors or the field change. | RQ01, RQ09, RQ13 | C1/C3/C5: under the stated steady-state model, availability and achievable precision select different cameras on 15.7% of reachable floor area. No route or navigation outcome. |
| F06 Matched closed-loop result | Current but unapproved inputs `scripts/visibility_comparison/{_clv2.yaml,_clv3.yaml,warehouse_full_4cam_missions.yaml}` and v2/v3 calibration JSON; future approved campaign `run_summary.json`, canonical `experiment.csv` and detection/correction logs; world/detector/calibration/field manifests and hashes. There is no `logs/studies/closed_loop_calibration` evidence. | Existing `experiments/closed_loop_calibration/analyse.py` writes only `v2_vs_v3.json` and prints text. Proposed analyzer completion plus `scripts/paper_figures/correlated_error_icra/render_f06.py` only after WS05 approval. | **BLOCKED / NOT EXECUTED.** E6 makes the calibration terms unidentifiable on the current data; configs contain seed 0 only; analyzer lacks NEES/NIS, acceptance/age, intervals, plots and a 15-pair gate. | RQ07, RQ11, RQ13 | At most the scoped closed-loop consequence or null of the selected causal contrast. A calibration-only pair cannot validate the full correlation-floor/LOO method or broad safety superiority. |
| F07 Reliability-source held-out prediction | Required future package: frozen `p_use` labels; route/capture/site split manifest; constant, distance-only, FOV/range, operational geometry, GP, hybrid, complete-map oracle and admitted-DL predictions. The README-quoted `logs/studies/usable_observation/{dataset_v1,baselines_v1,gp_v1,confidence_v1,planner_conditions_v1}` directories are absent. | No verified generator at the cited evidence paths. Proposed future `scripts/paper_figures/reliability_source_comparison/render_f07.py` must consume a recovered/rebuilt frozen benchmark summary. | **BLOCKED.** Narrative Brier numbers are not evidence of record. Older single-camera GP artifacts are not substitutes. | RQ01, RQ02, RQ03, RQ05, RQ06, RQ10, RQ12, RQ14 | C2/C6 only after recovery/rebuild: relative held-out predictive calibration on prespecified simulated blocks. Oracle remains an upper bound. |
| F08 Reliability-source route discrimination | Required source-specific field artifacts, frozen route library/tasks/prior/planner and a route-level summary. Existing `logs/studies/planner_covariance_branching/exp1_scaled_vs_branch/{summary.json,fig_p1_relative_trace_error.*,fig_p2_equivalent_covariance.*,fig_p3_acceptable_region.*}` is analytic mechanism evidence only. | Existing generic generator: `experiments/planner_covariance_branching/exp1_scaled_vs_branch.py`. Proposed source-specific benchmark analyzer and `scripts/paper_figures/reliability_source_comparison/render_f08.py`; neither exists at the quoted usable-observation paths. | **PLANNED.** Generic branch mechanics do not show that any source changes a route. | RQ13, RQ14 | C3 only after a prespecified paired route difference using frozen source fields. |
| F09 Commissioning and deployment decision matrix | Partial input `logs/studies/network_commissioning_realism/exp1_gate_without_truth/summary.json`; required but absent source-specific prediction summaries, manifests and comparable collection/setup/runtime/transfer costs | Existing partial generator: `experiments/network_commissioning_realism/exp1_gate_without_truth.py`. Proposed table generator `scripts/paper_figures/reliability_source_comparison/render_f09.py`. | **PLANNED / PARTIAL.** No comparable cost surface across source families; tracked GP/campaign artifacts belong to other scopes. | RQ02, RQ03, RQ06, RQ09, RQ10, RQ12 | C2/C6 trade-offs in the tested simulation regime; no hardware deployment or universal recommendation. |
| F10 Method-specific failure gallery | Required common split with prespecified occlusion, unsupported-space, range, stale-layout and fallback fields. Current failures are scattered across bias, drift, GP, fusion and demo studies. | No shared generator. Proposed `render_f10.py` from the frozen failure-audit table; no manually selected screenshots as primary evidence. | **PLANNED.** Existing demos/scattered failures cannot establish a controlled source comparison. | RQ04, RQ05, RQ10, RQ14 | A bounded failure mode and fallback per tested method; no claim that the gallery is exhaustive. |

## Canonical paper-specific layout (proposal only)

Do not add to or overwrite the old generic `paper_artifacts/figures/` surface. On
integration approval, use:

```text
paper_artifacts/
  correlated_error_icra/
    MANIFEST.json
    figures/
      F01/F01_persistent_residuals.{pdf,png,provenance.json}
      F02/F02_belief_honesty.{pdf,png,provenance.json}
      F03/F03_calibration_policy.{pdf,png,provenance.json}
      F04/F04_drift_lifecycle.{pdf,png,provenance.json}
      F05/F05_achievable_precision.{pdf,png,provenance.json}
      F06/F06_closed_loop_paired.{pdf,png,provenance.json}
    results/F01/...F06/          # immutable summary/table used by each figure
    inputs/F01/...F06/           # manifests of source paths, roles, hashes and restore locations
    diagnostics/setup/           # readiness plots; never manuscript evidence by default
    demos/                       # illustrative media; no inferential claims
    archive/                     # superseded paper-specific renderings, never canonical
  reliability_source_comparison/
    MANIFEST.json
    figures/F07/...F10/
    results/F07/...F10/
    inputs/F07/...F10/
    diagnostics/setup/
    demos/
    archive/
```

`MANIFEST.json` names exactly one canonical sidecar/output set per figure and records the
paper version. Source data need not be duplicated when immutable/hash-restorable; the input
manifest must still bind each path and archive restore location.

## Provenance sidecar contract

Every `Fxx_*.provenance.json` must contain:

| Section | Required fields |
|---|---|
| Identity | `schema_version`, `paper_id`, `figure_id`, title, caption, scope, maturity, claim IDs and reviewer-question IDs |
| Generator | repository-relative generator path, exact command/arguments, deterministic random seed if any, dependency lock/environment, Git commit, dirty-state patch hash and generation timestamp |
| Inputs | every repository/archive path, SHA-256, byte size, role (`operational`, `evaluation_only`, `control`, `summary`), and restore location |
| Experimental binding | world/detector/calibration/field/config hashes; task, seed, camera and condition lists; capture/route/site definitions; split manifest and independent-unit definition |
| Statistics | primary endpoint, diagnostics, unit of analysis, pairing key, interval/test method, multiplicity policy, missing/invalid rule and null interpretation |
| Outputs | canonical PDF/PNG/result-summary paths with SHA-256 and byte size; plot dimensions and renderer version |
| Boundaries | limitations, forbidden claims, oracle/evaluation-only uses, superseded inputs and known incompatibilities |

Promotion must fail if an input/output hash differs, the Git state is not recorded, a
required independent unit is absent, or a sidecar omits its allowed-claim boundary.
Existing paired-mechanism and GP provenance files do not meet this schema: most lack
F/claim/RQ IDs, Git state, input/output hashes, split/unit definitions and claim limits.

## Deterministic F04 specification (not rendered)

Input is only the audited `drift_lifecycle.json`; the renderer must not refit a model,
resample detections or import runtime code.

1. Panel A plots camera-C yaw drift magnitude against raw and stale-v3 RMS, in metres,
   directly from `summary`. Draw the first change detection at `0.1 deg` and first harm at
   `0.25 deg`; label the v3 scope in the panel title.
2. Panel B plots `change_z` against yaw drift for A-D with the fixed threshold `1.2` and a
   separate inset/row for translation drift. Camera order and colours are fixed. The
   absolute statistic may appear only as a grey diagnostic because it is non-monotone and
   fires at rest for marginal cameras.
3. Include the facts “one held-out capture,” “230 detections within the capture,” and
   “independent unit: unique camera site; deterministic fault ladder.” Do not draw a
   confidence band over detections.
4. Assert before rendering: `identity_check_max_error_m == 0`, camera-C change detection
   is `0.1`, camera-C first harm is `0.25`, and calibration path/hash are the audited v3
   inputs. Any failed assertion stops the build.
5. Caption boundary: controlled step faults, no temporal latency or false-alarm rate,
   historical v3 unless the underlying study is explicitly regenerated after a separate
   scientific decision.

## Non-executed F06 specification

F06 remains an empty contract until WS05 approves arms and one primary endpoint.

- Completeness gate: both arms must contain exactly the same 15 unique `(task, seed)` keys
  (three frozen tasks x seeds `0-4`), no duplicates, valid run summaries and bound hashes.
  An intersection of whatever happens to exist is forbidden.
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

## Setup diagnostic plots (not manuscript evidence)

Produce these under the paper's `diagnostics/setup/` only after an arm decision and before
campaign execution. They are fail-closed readiness checks, not a tuning surface.

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
| Locked scientific source evidence, not paper-locked | Relevant `logs/studies/**/{RESULTS.md,summary JSON,plots}` for F01-F05 | Preserve summaries, tables, plots and generators. Promote only through the paper-specific layout with hashes. |
| Prior-paper/current-surface artifacts | `paper_artifacts/figures/current_surface/`, `paper_snapshot/`, top-level paired-mechanism PDFs/data, `campaigns/robustness_campaign_headline/`, `gp/warehouse_visibility_gp_v1/`, `evidence/single_camera_uigp/` | Canonical only for their documented older single-camera/GP surface. They are stale for F01-F10 and cannot answer the current reviewer questions. |
| Explicitly stale/superseded/archive | `figures/archive/`, `figures/_archive_nonpaper_20260612/`, `gp/archive/aws_gp_v7b_superseded/`, `campaigns/archive/`, `metrics/archive/`, `paired_mechanism_taskA_smoke_PREFIX_OLD.pdf` | Retain for traceability, exclude from all canonical manifests. The nonpaper README itself points to a now-absent `paired_mechanism_taskA.pdf`, confirming that its canonical pointer is stale. |
| Diagnostics | `paper_artifacts/figures/diagnostics/`, solve/path/drift diagnostic images and future `diagnostics/setup/` | Readiness/debug only; no captioned scientific claim and no status promotion. |
| Demo / explainer / setup | `paper_artifacts/figures/explainers/`, `problem_setup_camera.png`, `problem_setup_snapshots.png`, `localization_pathway.png`, `yolo_training_clarification.png`, `logs/studies/multicamera_commissioning_bigwarehouse/demos/` and `four_camera_showcase/` | May illustrate system/setup with an explicit `DEMO` or `SETUP` label. Never count as F01-F10 evidence or an independent replicate. |
| Variant/smoke paired media | Top-level `paired_mechanism_*current*`, `*lowlat*`, `*fixed*`, `*seed2*`, `*smoke*` and their copied data | Preserve as historical variants. Names such as “current” do not establish scientific or publication maturity; exclude unless a future paper manifest deliberately and hash-verifiably promotes one. |

No artifact is reclassified by moving or deleting it in this workstream; classification is
metadata and claim discipline only.
