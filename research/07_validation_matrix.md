# Validation and decision matrices

## Evidence-audit verdict — 2026-08-06

The frozen study narratives and summaries support a strong offline mechanism package, but
the publication package and both planned campaigns are not ready.

1. E6 shows that the C/D cross-bearing signal disappears after the robot silhouette is
   modelled on external deployed logs. Camera, route region and yaw are confounded, so v2,
   v3 and v4 are preserved historical artifacts but none is an approved causal F06 arm.
2. Generated `_clv2.yaml` and `_clv3.yaml` contain seed `0` only for each of three tasks:
   six runs, not the documented 30 and not 15 matched pairs.
3. `experiments/closed_loop_calibration/analyse.py` omits promised NEES, NIS,
   correction acceptance/age, confidence intervals and plot generation. It accepts any
   intersection of run keys instead of requiring exactly 15 complete matched pairs.
4. Runtime NIS `9.21` and offline NIS `5.991` are different chi-square policies (99% and
   95% for 2 DOF respectively); neither may be presented as the other's validation.
5. The usable-observation README's `dataset_v1`, `baselines_v1`, `gp_v1`,
   `confidence_v1` and `planner_conditions_v1` directories are absent. Its numerical
   narrative is not a recoverable evidence package at the stated paths.
6. F01-F05 are scientifically useful source results, but ignored `logs/` summaries and
   plots are not publication-locked artifacts. Canonical paper outputs, frozen summaries
   and complete provenance sidecars are absent.

## Statistical contract

- Independent units are matched `(task, seed)` runs, route/capture blocks, or unique
  commissioning sites. Frames, detections and grid interpolation samples are never used
  as independent replicates for uncertainty or significance.
- A frame-level statistic is first reduced within its independent unit (for example,
  median NEES per task-seed run or Brier score per held-out route). Intervals are then
  computed across units.
- Binary paired outcomes use an exact paired test on discordant task-seed pairs and report
  the paired risk difference with an exact interval. Continuous paired outcomes use a
  run-level paired bootstrap interval, stratified by task when the campaign has three
  tasks. No frame-level confidence interval is allowed.
- Spatial maps report an area-weighted fraction over unique evaluated sites. Map cells are
  a domain quadrature, not stochastic replicates; no pseudo-confidence interval over cells
  is reported.
- Fifteen matched task-seed pairs can support a scoped mechanism result or a documented
  null. They cannot support broad safety superiority.
- Every gate is fail-closed. Missing pairs, duplicate keys, invalid runs, unbound artifact
  hashes or forbidden evaluation inputs stop promotion. Null and negative results remain
  permanent.

## Decision matrix A — current correlated-error paper

| Stage / figure | Independent variable | Frozen controls and exact inputs | Independent unit / split | Primary endpoint | Diagnostics | Analysis and promotion / stop rule | Allowed claim |
|---|---|---|---|---|---|---|---|
| Projection mechanism / F01 | Projection component: raw, deployed v2, floor plane, gated cross-bearing and CAD-silhouette alternatives | Existing capture directories plus the tracked E6 summary; world and v2/v3/v4 artifacts | Capture/route block and unique camera site; future grouped yaw/region holdouts | Held-out residual bias and RMS | Projection amplification, object-model residual, coverage and fold disagreement | Retain nulls and camera-A instability. Promote only a panel that exposes the current identifiability failure. | C1: persistent structured residuals and geometric amplification exist; camera-specific attribution remains open. |
| Belief honesty / F02 | Offline filter treatment: trust, NIS gate, conditional covariance, LOO check, per-camera correlation floor and ablations | Same three recorded capture blocks; v2 projection; fixed process/noise constants in `exp1_graceful_vs_trusting.py`; offline NIS `5.991` | Capture block; leave-one-capture-out for floor transfer. Detections are within-block samples. | Truth outside stated 95% ellipse, with RMSE and stated sigma jointly reported | Median NEES, floor sensitivity, per-camera/pooled ablations | Descriptive pooled panel must travel with capture-block generalization panel. Stop any navigation inference. | C1/C3: repeated biased measurements make the offline belief overconfident; per-camera flooring restores honesty on these captures. |
| Calibration policy / F03 | Per-camera leave-raw versus fitted correction after a fixed object model | Existing residual/commissioning summaries, tracked E6 result and historical artifacts | Unique site/capture block with future grouped yaw and region holdouts | Held-out RMS/bias and stability of the CALIBRATE/RAW decision | Bias-to-scatter, interactions, sample efficiency and harm | Stop until WS05 identifies a camera term independently of silhouette, route and yaw. Preserve a no-calibration null. | Historical policy behavior only; no current calibration-benefit claim. |
| Drift lifecycle / F04 | Controlled yaw or translation perturbation magnitude; raw versus stale v3 correction | `drift_lifecycle.json`, held-out `fusion_handover_20260721`, v2/v3 JSON, fixed change threshold 1.2 | Unique camera site within one held-out capture; deterministic fault ladder, no frame-level inference | Fault magnitude at detection versus fault magnitude at first harm | Absolute-versus-change statistic, raw/stale RMS, identity re-projection check | Existing result is v3-specific and descriptive. Promote deterministic F04 only with explicit v3 scope, or regenerate the locked result after the calibration policy is scientifically selected. | RQ07 only: controlled change detection preceded camera-C harm in this v3 experiment; no field false-alarm or latency claim. |
| Achievable precision / F05 | Select maximum availability versus minimum steady-state achievable sigma | `fused_planner_four_camera.npz`; fixed per-camera residual floors; 3 Hz, 0.3 m/s and odometry growth specified in generator | Unique reachable floor site; cells used only for area integration | Area fraction where selected cameras differ | Sigma penalty, per-camera territory, sensitivity to adopted floors | Deterministic recomputation required if the adopted residual floors/field change. No route or outcome promotion. | C1/C3/C5: availability and modeled achievable precision disagree over a measured portion of this simulated floor. |
| Matched closed loop / F06 | One scientifically approved calibration/belief mechanism contrast; not selected here | Chosen generated pair, same tasks/seeds/planner/world/detector/noise; frozen artifact hashes; canonical run and detection loaders | Exactly 15 matched `(task, seed)` pairs: 3 tasks x 5 seeds. Frames are within-run only. | One endpoint must be prespecified by WS05: clean goal, run-level belief calibration, or p95 localization error | Breach/contact, NEES/NIS, 95% coverage, correction acceptance/age, calibrated-camera exposure, path/time and invalid-run ledger | Require both arms to have the identical 15 keys, no duplicates, and all validity fields. Binary exact paired analysis; task-stratified paired bootstrap for continuous run summaries. Stop on any incomplete matrix or protocol/hash mismatch; publish the null. | At most C4 for the selected causal contrast. A calibration-only contrast cannot establish benefit of the complete correlation-floor/LOO package or broad safety superiority. |

### F06 readiness contract (not executed)

Before any run, WS05/WS06 must provide: an approved arm decision consistent with v4
evidence; one primary endpoint; seeds `0-4` in every task for both arms; exact world,
detector, calibration and field hashes; a declared invalid-run/rerun rule; and an analyzer
that fails unless the 15 pair keys are complete. Its output must include run-level tables,
exact paired binary results, paired bootstrap intervals, NEES/NIS and correction
acceptance/age summaries, figures and a null interpretation. The existing analyzer meets
none of these completeness requirements beyond pairing the intersection.

## Decision matrix B — future reliability-source benchmark

This benchmark begins only after the current paper package closes. All arms predict the
same `p_use`; evaluation truth and current detector outcome are forbidden operational
features. Constant, distance-only, calibrated FOV/range and complete-map oracle/upper-bound
geometry are mandatory baselines. The operational geometry arm must separately identify
sensed depth/raycast provenance; GP and hybrid are challengers; DL is admitted only after
the same legality and calibration gates.

| Stage / figure | Independent variable | Frozen controls and required inputs | Independent unit / split | Primary endpoint | Diagnostics | Promotion / stop rule | Allowed claim |
|---|---|---|---|---|---|---|---|
| Evidence recovery | Recovered archive versus deterministic rebuild | Frozen gate definition, label semantics, manifest, world/detector hashes and route list | Dataset/capture manifest | Hash-verified completeness | Row counts by route/camera, forbidden-feature audit | Stop while the five README-referenced evidence directories remain absent. Narrative numbers are never imported as results. | No scientific claim; provenance restoration only. |
| Feature legality | Reliability source family | Candidate pose, camera geometry and source-specific operational map/model only | Source implementation and manifest | Zero forbidden evaluation inputs | Availability at future poses, missing-feature fallback, timing and age | Any truth leakage, current detector outcome or unavailable future feature retires the arm. | Operational-input feasibility only. |
| Held-out prediction / F07 | Constant; distance-only; FOV/range; operational sensed-depth/raycast; GP; hybrid; complete-map oracle; admitted DL | Identical labels, route/capture splits, detector, candidate sites and commissioning budget | Route/capture block, with unique sites nested within block; leave complete routes out, plus a site/layout holdout | Mean held-out-route Brier score; calibration error secondary | PR-AUC, reliability curve, log loss, support/range/occlusion strata and block intervals | Beat or explain the simple distance/FOV-range baseline on prespecified blocks. Oracle is an upper bound and cannot win deployment. Retain GP/hybrid nulls. | C2/C6: relative predictive calibration on tested simulated blocks only. |
| Failure audit / F10 | Same source arms under prespecified occlusion, unsupported-space, stale-layout and range regimes | Same fitted models, split and sample budget | Route/capture block and unique failure site | Worst prespecified regime Brier/calibration gap | False-safe rate, support distance, fallback invocation, stale-age response | Every arm needs a reproducible failure and conservative fallback. No post-hoc cherry-picked gallery. | Method-specific failure modes, not universal ranking. |
| Offline route discrimination / F08 | Frozen `p_use` field only | Same planner, prior belief, route library, costs and hit/miss update; generic planner-branch study is a mechanism check, not source evidence | Prespecified task-route block; paired route alternatives | Correct ranking / terminal expected belief difference on tasks chosen before seeing source outcomes | Route flip count, effect size, field exposure and null tasks | Only a source that produces a nontrivial paired route difference proceeds. Generic `planner_covariance_branching` alone cannot pass this gate. | C3: the frozen source field changes predicted belief/route ranking on prespecified tasks. |
| Closed-loop source test | Winning frozen source field versus simple baseline | Same controller, tasks, seeds, world, detector and runtime; only field artifact differs | Matched task-seed run | One prespecified navigation endpoint | Realized observability, breach/contact, calibration, path/time | Exact complete-pair gate and run-level paired analysis; preserve null. | Navigation consequence of the frozen field only. |
| Deployment trade-off / F09 | Source family | Common reporting template | Unique commissioned camera site plus route/capture block | Held-out performance at fixed commissioning budget | Samples, collection/setup time, storage, inference latency, map age/update cost, transfer and fallback | No arm is recommended without comparable cost and operational provenance. | C2/C6 trade-off within tested simulation regime. |
| Camera management (separate) | Nearest, max availability, achievable precision, hysteretic selection, conservative fusion | Winning reliability fields frozen once; estimator not refit per policy | Matched task-seed run | One prespecified localization/navigation endpoint | Switch rate, correlated exposure, NEES/coverage, availability and runtime | Starts only after F07/F08 freeze fields. No independent-fusion claim. | C5: how fixed quality fields should be consumed, separate from how they were estimated. |

## Registry decisions applied on 2026-08-06

| Registry item | Status | Reason |
|---|---|---|
| F01 | READY | Scientific inputs and an E6 tracked summary exist; no canonical paper panel yet. |
| F02 | READY | Strong offline result exists; canonical paper promotion and provenance are missing. |
| F03 | BLOCKED | Calibration terms are not identifiable after E6; v4 supersession alone no longer resolves the policy. |
| F04 | BLOCKED | Numeric v3 fault ladder exists, but no deterministic canonical figure and current-policy applicability is unresolved. |
| F05 | READY | Deterministic map exists; publication wrapper/provenance are missing and adopted floors may require recomputation. |
| F06 | BLOCKED | No approved causal arms; configs are seed-incomplete and analysis is incomplete. |
| F07 | BLOCKED | README-referenced evidence package is absent. |
| F08 | PLANNED | Generic planner mechanics exist, but source-specific route discrimination does not. |
| F09 | PLANNED | Partial commissioning evidence only; no common source cost matrix. |
| F10 | PLANNED | Failures are scattered across non-common studies/demos. |
| EXP-CL-CAL | BLOCKED current focus | Identifiability work may continue; confirmatory execution may not. |
| EXP-USABLE | BLOCKED | Evidence directories named as completed are absent; split package must be recovered or rebuilt. |
| EXP-PIXEL-GROUND | LOCKED supporting evidence | Candidate code is experiment-local; E6 is preserved and no runtime promotion is authorized. |

Experiment-level `LOCKED` status for the existing offline studies may remain: that status
means their source evidence is frozen, not that F01-F05 are publication-ready.

## Unresolved scientific decisions

1. WS05 must first design a yaw-diverse, route/region-disjoint identifiability study that
   separates object-silhouette error from camera calibration. Only a passing result may open
   selection of a causal F06 contrast.
2. After that gate, WS05 must select exactly one F06 primary endpoint and accept the corresponding claim
   boundary and limited power of 15 pairs.
3. WS06 must decide whether the current world is compatible with July fields whose
   manifests do not bind a world hash; shared exposure alone does not establish external
   interpretation.
4. The current paper must decide whether F04 remains an explicitly historical v3
   lifecycle demonstration or is regenerated under the scientifically selected current
   calibration policy.
5. The residual floors used by F05 must be reconciled with the selected minimal projection
   pipeline before publication.
6. The future benchmark must recover a hash-verifiable usable-observation archive or
   preregister a rebuild, then freeze route/capture/site splits and the exact `p_use` label.
7. The operational geometry provenance (sensed depth, commissioned static map, or raycast)
   and the separate complete-map oracle must be chosen and labelled before F07.
