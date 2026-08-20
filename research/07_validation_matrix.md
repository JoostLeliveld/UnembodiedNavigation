# Validation and decision matrices

## Evidence-audit verdict — 2026-08-10

The frozen study narratives and summaries support a useful historical offline mechanism
package, but they are outside the fresh-evidence epoch declared below. They may motivate
the preregistered questions and failure strata; they may not supply a paper estimate,
confidence interval, figure, threshold, fitted artifact or claim. The publication package
and both planned campaigns are not ready.

1. E6 shows that the C/D cross-bearing signal disappears after the robot silhouette is
   modelled on external deployed logs. Camera, route region and yaw are confounded, so v2,
   v3 and v4 are preserved historical artifacts but none is an approved causal F06 arm.
2. Generated `_clv2.yaml` and `_clv3.yaml` contain seed `0` only for each of three tasks:
   six runs, not the documented 30 and not 15 matched pairs.
3. The originally audited `experiments/closed_loop_calibration/analyse.py` lacked NEES,
   NIS, correction acceptance/age, confidence intervals, plot generation and a 15-pair
   completeness gate. The current checkout has subsequently added those calculations and
   a count gate, but remains non-reportable: it is hard-coded to superseded v2/v3 arms,
   silently overwrites duplicate `(task, seed)` keys, does not verify the exact frozen key
   set or evidence epoch, and does not emit the preregistered exact paired binary result.
4. Runtime NIS `9.21` and offline NIS `5.991` are different chi-square policies (99% and
   95% for 2 DOF respectively); neither may be presented as the other's validation.
5. The usable-observation README's `dataset_v1`, `baselines_v1`, `gp_v1`,
   `confidence_v1` and `planner_conditions_v1` directories are absent. Its numerical
   narrative is not a recoverable evidence package at the stated paths.
6. F01-F05 are scientifically useful historical source results, but ignored `logs/`
   summaries and plots are not publication-locked artifacts. Under the fresh-evidence rule
   they are not admissible inputs to new paper figures even if copied or re-rendered.
   Canonical fresh outputs, frozen summaries and complete provenance sidecars are absent.

## Fresh-evidence epoch — preregistered, no runs executed

| Field | Frozen value / rule |
|---|---|
| Current-paper epoch | `correlated_error_icra_20260810_v1` |
| Future-benchmark epoch | `reliability_source_20260810_v1` |
| Preregistration time | `2026-08-10T12:02:26+02:00` |
| Earliest admissible capture | At or after the preregistration time, with a monotonic capture ledger and UTC offset |
| Code basis at declaration | Git commit `2662a97f870c29e36b7cd508e9ce6a07dc184ef1`; every execution must additionally bind its actual commit and dirty-patch SHA-256 |
| Fresh raw roots | `logs/evidence_epochs/<epoch_id>/raw/<collection_id>/` only |
| Fresh derived roots | `logs/evidence_epochs/<epoch_id>/derived/<analysis_id>/` only |
| Canonical paper root | `paper_artifacts/<paper_id>/epochs/<epoch_id>/` only |
| Prohibited scientific inputs | Every capture, run, summary, fitted artifact and plot created before the epoch; all v2/v3/v4 fitted outputs; all existing `logs/studies/**` and generic `paper_artifacts/**` results |
| Permitted historical use | Question formation, failure-stratum definition, code tests with synthetic fixtures, and explicitly labelled historical/demo comparison outside manuscript estimates |
| Independent units | Task-seed run, capture/route block and unique site. Frames/detections are repeated observations within a unit. |
| Promotion state | `PREREGISTERED`; no F01-F10 figure is fresh, executed or promotion-eligible |

An admissible input manifest must bind `epoch_id`, collection start/end, source path,
SHA-256, byte size, creation timestamp, world/detector/config/calibration/field hashes,
camera/site/region/yaw/session identifiers, operational versus evaluation-only role, Git
commit and dirty-patch hash. Analysis must fail if a path is outside the epoch roots, a
timestamp predates the freeze, a hash is absent or mismatched, a required block/site is
missing, or a pre-epoch derived artifact appears anywhere in the dependency graph. Merely
rerunning a renderer over an old summary is not fresh evidence.

The normative machine-readable manifest, once creation outside the owned documents is
authorized, must implement at least this schema and must not replace placeholders with
historical values:

```yaml
schema_version: 1
epoch_id: correlated_error_icra_20260810_v1
status: PREREGISTERED
preregistered_at: '2026-08-10T12:02:26+02:00'
not_before: '2026-08-10T12:02:26+02:00'
fresh_inputs_only: true
roots:
  raw: logs/evidence_epochs/correlated_error_icra_20260810_v1/raw
  derived: logs/evidence_epochs/correlated_error_icra_20260810_v1/derived
  canonical: paper_artifacts/correlated_error_icra/epochs/correlated_error_icra_20260810_v1
required_bindings: [git_commit, dirty_patch_sha256, world_sha256, detector_sha256,
  config_sha256, calibration_sha256, field_sha256]
independent_units: [task_seed_run, capture_or_route_block, unique_site]
forbidden_unit: frame
reject_pre_epoch_dependencies: true
```

`reliability_source_20260810_v1` uses the same keys with its own epoch and canonical roots.
The validator and registry integration are proposed work, not implemented in this
documentation-only change.

## Preregistered identifiability collection matrix

This collection identifies robot silhouette, camera, region and yaw terms before any
closed-loop arm or endpoint is selected. It does not launch or authorize Gazebo here.

| Factor | Frozen levels |
|---|---|
| World | `warehouse_full_4cam.world.sdf` |
| Cameras | A, B, C and D recorded simultaneously at every pose block |
| Regions | south, central and north |
| Unique sites | `S_W=(-1.0,-4.0)`, `S_E=(1.0,-4.0)`, `C_W=(-1.2,0.6)`, `C_E=(1.2,0.6)`, `N_W=(-1.0,4.0)`, `N_E=(1.0,4.0)` metres |
| Robot yaw | `0, 45, 90, 135, 180, 225, 270, 315` degrees |
| Sessions | Two separately started capture sessions; session 2 remains untouched until the model form and gates are frozen from session 1 |
| Planned blocks | `6 sites x 8 yaws x 2 sessions = 96` capture blocks; four simultaneous camera records per block = 384 camera-block records |
| Within-block sampling | A fixed acquisition window and cadence, recorded in the manifest; frames are QC/repeated measurements only |
| Evaluation split | Leave-one-unique-site-out analysis on session 1, followed by one frozen confirmation on all session-2 blocks; no frame-random split |

The model ladder is fixed as: point/floor projection; CAD silhouette with recorded yaw;
CAD silhouette plus camera main effect; then the prespecified camera-by-region and
camera-by-yaw diagnostics. The primary contrast is fresh held-out block RMSE for
`silhouette + camera` minus `silhouette only`. A camera-calibration term is identifiable
only if the site-clustered 95% interval has an upper bound below zero, the mean improvement
is at least 0.010 m, its direction is stable in every held-out region, and no prespecified
yaw band is harmed by more than 0.020 m. Failure of any condition preserves a
no-camera-calibration null and keeps F03/F06 blocked. Detection yield is a support
diagnostic, not permission to drop a camera/site/yaw cell post hoc.

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
- Every estimate, interval and plot is regenerated from admissible epoch inputs.
  Deterministic code reuse is allowed; data, fitted artifacts, summaries and rendered
  outputs from before the epoch are not.

## Decision matrix A — current correlated-error paper

| Stage / figure | Independent variable | Frozen controls and exact inputs | Independent unit / split | Primary endpoint | Diagnostics | Analysis and promotion / stop rule | Allowed claim |
|---|---|---|---|---|---|---|---|
| Projection mechanism / F01 | Projection component: floor point, CAD silhouette and any subsequently identified camera term | Fresh 96-block identifiability capture and epoch-bound world/detector/CAD/config inputs; old E6 and v2/v3/v4 are rationale only | Capture block and unique site; leave-one-site-out plus untouched session 2 | Held-out residual bias and RMSE | Projection amplification, detection support, region/yaw interactions and fold disagreement | Promote only after all 96 blocks validate and the frozen session-2 confirmation is reported, including a null. | C1 only after fresh confirmation: structured residuals and spatial amplification on the tested design; camera attribution only if its separate gate passes. |
| Belief honesty / F02 | Offline filter treatment: trust, NIS gate, conditional covariance, correlation floor and prespecified ablations | Fresh epoch detections/trajectories only; projection model selected by F01/F03; freshly frozen process/noise constants; offline NIS `5.991` | Fresh capture/route block with grouped site holdout; detections are within-block samples | Truth outside stated 95% ellipse, with RMSE and stated sigma jointly reported | Median NEES, floor sensitivity, per-camera/pooled ablations | Descriptive pooled panel travels with block/site generalization. No old floor estimate may be reused. | C1/C3 only after fresh replication; no navigation claim. |
| Calibration policy / F03 | `silhouette only` versus `silhouette + camera`, followed by leave-raw versus fitted only if the identifiability gate passes | Same fresh 96-block matrix; no v2/v3/v4 coefficient imported | Unique site and capture block; session-2 confirmation untouched during development | Held-out RMSE improvement and stability of the camera term | Bias-to-scatter, camera-region/yaw interactions, sample efficiency and harm | Apply the exact 0.010 m/clustered-interval/region/yaw gate above. Preserve and publish a no-calibration null. | A bounded fresh calibration-policy claim only if identified; otherwise silhouette-only result. |
| Drift lifecycle / F04 | Controlled yaw or translation perturbation of the newly selected policy | Fresh session-2 capture and newly frozen calibration; fresh fixed change threshold declared before injection | Unique camera site and capture block; deterministic fault ladder, no frame-level inference | Fault magnitude at detection versus fault magnitude at first harm | Absolute-versus-change statistic and identity re-projection check | Historical v3 numbers are demo-only. Render deterministically from a new frozen fault JSON without refitting. | RQ07 only after fresh execution; no real-drift latency or false-alarm claim. |
| Achievable precision / F05 | Select maximum fresh availability versus minimum fresh steady-state achievable sigma | Fresh visibility field and residual floors derived inside the epoch; fixed rate/speed/odometry model bound in the manifest | Unique reachable floor site; cells used only for area integration | Area fraction where selected cameras differ | Sigma penalty, per-camera territory and floor sensitivity | Rebuild field and floors from fresh inputs; old NPZ and 15.7% value are historical only. | C1/C3/C5 only for the fresh simulated field; no route or navigation outcome. |
| Matched closed loop / F06 | One scientifically approved calibration/belief mechanism contrast; not selected here | Chosen generated pair, same tasks/seeds/planner/world/detector/noise; frozen artifact hashes; canonical run and detection loaders | Exactly 15 matched `(task, seed)` pairs: 3 tasks x 5 seeds. Frames are within-run only. | One endpoint must be prespecified by WS05: clean goal, run-level belief calibration, or p95 localization error | Breach/contact, NEES/NIS, 95% coverage, correction acceptance/age, calibrated-camera exposure, path/time and invalid-run ledger | Require both arms to have the identical 15 keys, no duplicates, and all validity fields. Binary exact paired analysis; task-stratified paired bootstrap for continuous run summaries. Stop on any incomplete matrix or protocol/hash mismatch; publish the null. | At most C4 for the selected causal contrast. A calibration-only contrast cannot establish benefit of the complete correlation-floor/LOO package or broad safety superiority. |

### F06 readiness contract (not executed)

Before any run, WS05/WS06 must provide: a passed fresh identifiability gate and an approved
arm decision (v4 supersedes the planned v2/v3 pipeline but is itself not pre-approved);
one primary endpoint; seeds `0-4` in every task for both arms; exact world,
detector, calibration and field hashes; a declared invalid-run/rerun rule; and an analyzer
that fails unless the 15 pair keys are complete. Its output must include run-level tables,
exact paired binary results, paired bootstrap intervals, NEES/NIS and correction
acceptance/age summaries, figures and a null interpretation. The current analyzer is a
useful partial implementation, but it must become arm-generic, reject duplicate and wrong
keys, require the epoch manifest, and add exact paired binary inference before use.

## Decision matrix B — future reliability-source benchmark

Method investigation, interface work and explicitly exploratory failure probes may proceed
before the current paper package closes. The frozen confirmatory benchmark and any
source-ranking claim begin only afterward. All arms predict the same `p_use`; evaluation
truth and current detector outcome are forbidden operational features. Constant,
distance-only, calibrated FOV/range and complete-map oracle/upper-bound geometry are
mandatory baselines. The operational geometry arm must separately identify sensed
depth/raycast provenance; GP and hybrid are challengers; DL is admitted only after the same
legality and calibration gates.

| Stage / figure | Independent variable | Frozen controls and required inputs | Independent unit / split | Primary endpoint | Diagnostics | Promotion / stop rule | Allowed claim |
|---|---|---|---|---|---|---|---|
| Fresh evidence build | Deterministic new collection and rebuild only | Freshly frozen gate definition, label semantics, manifest, world/detector hashes and route list; no recovered old rows or fitted artifacts | New route/capture block and unique site manifest | Hash-verified fresh completeness | Row counts by route/camera/site, timestamp/epoch and forbidden-feature audits | Stop until a complete post-freeze package exists. The five absent README directories and all narrative numbers remain historical, not recoverable paper inputs. | No scientific claim; provenance establishment only. |
| Feature legality | Reliability source family | Candidate pose, camera geometry and source-specific operational map/model only | Source implementation and manifest | Zero forbidden evaluation inputs | Availability at future poses, missing-feature fallback, timing and age | Any truth leakage, current detector outcome or unavailable future feature retires the arm. | Operational-input feasibility only. |
| Held-out prediction / F07 | Constant; distance-only; FOV/range; operational sensed-depth/raycast; GP; hybrid; complete-map oracle; admitted DL | Identical labels, route/capture splits, detector, candidate sites and commissioning budget | Route/capture block, with unique sites nested within block; leave complete routes out, plus a site/layout holdout | Mean held-out-route Brier score; calibration error secondary | PR-AUC, reliability curve, log loss, support/range/occlusion strata and block intervals | Beat or explain the simple distance/FOV-range baseline on prespecified blocks. Oracle is an upper bound and cannot win deployment. Retain GP/hybrid nulls. | C2/C6: relative predictive calibration on tested simulated blocks only. |
| Failure audit / F10 | Same source arms under prespecified occlusion, unsupported-space, stale-layout and range regimes | Same fitted models, split and sample budget | Route/capture block and unique failure site | Worst prespecified regime Brier/calibration gap | False-safe rate, support distance, fallback invocation, stale-age response | Every arm needs a reproducible failure and conservative fallback. No post-hoc cherry-picked gallery. | Method-specific failure modes, not universal ranking. |
| Offline route discrimination / F08 | Frozen `p_use` field only | Same planner, prior belief, route library, costs and hit/miss update; generic planner-branch study is a mechanism check, not source evidence | Prespecified task-route block; paired route alternatives | Correct ranking / terminal expected belief difference on tasks chosen before seeing source outcomes | Route flip count, effect size, field exposure and null tasks | Only a source that produces a nontrivial paired route difference proceeds. Generic `planner_covariance_branching` alone cannot pass this gate. | C3: the frozen source field changes predicted belief/route ranking on prespecified tasks. |
| Closed-loop source test | Winning frozen source field versus simple baseline | Same controller, tasks, seeds, world, detector and runtime; only field artifact differs | Matched task-seed run | One prespecified navigation endpoint | Realized observability, breach/contact, calibration, path/time | Exact complete-pair gate and run-level paired analysis; preserve null. | Navigation consequence of the frozen field only. |
| Deployment trade-off / F09 | Source family | Common reporting template | Unique commissioned camera site plus route/capture block | Held-out performance at fixed commissioning budget | Samples, collection/setup time, storage, inference latency, map age/update cost, transfer and fallback | No arm is recommended without comparable cost and operational provenance. | C2/C6 trade-off within tested simulation regime. |
| Camera management (separate) | Nearest, max availability, achievable precision, hysteretic selection, conservative fusion | Winning reliability fields frozen once; estimator not refit per policy | Matched task-seed run | One prespecified localization/navigation endpoint | Switch rate, correlated exposure, NEES/coverage, availability and runtime | Starts only after F07/F08 freeze fields. No independent-fusion claim. | C5: how fixed quality fields should be consumed, separate from how they were estimated. |

## Proposed registry changes — not applied

| Registry item | Status | Reason |
|---|---|---|
| F01 | BLOCKED | Historical inputs motivate the design but are prohibited by the fresh epoch; the 96-block matrix is not collected. |
| F02 | BLOCKED | Historical filter results are not admissible; fresh block/site replication is absent. |
| F03 | BLOCKED | Calibration terms are not identifiable after E6; v4 supersession alone no longer resolves the policy. |
| F04 | BLOCKED | Numeric v3 fault ladder is historical only; fresh selected-policy injection and deterministic rendering are absent. |
| F05 | BLOCKED | Old field/floors/map are prohibited; a fresh deterministic rebuild is absent. |
| F06 | BLOCKED | No approved causal arms; generated configs remain seed-0-only; analysis is only partially complete and not epoch-bound. |
| F07 | BLOCKED | README-referenced evidence package is absent. |
| F08 | PLANNED | Generic planner mechanics exist, but source-specific route discrimination does not. |
| F09 | PLANNED | Partial commissioning evidence only; no common source cost matrix. |
| F10 | PLANNED | Failures are scattered across non-common studies/demos. |
| EXP-CL-CAL | BLOCKED current focus | Identifiability work may continue; confirmatory execution may not. |
| EXP-USABLE | BLOCKED | Evidence directories named as completed are absent; the fresh rule requires a new preregistered collection/rebuild rather than recovery for scientific use. |
| EXP-PIXEL-GROUND | LOCKED supporting evidence | Candidate code is experiment-local; E6 is preserved and no runtime promotion is authorized. |
| RQ07 | OPEN | Historical v3 drift evidence is outside the epoch; fresh selected-policy injection is required. |
| RQ11 | OPEN | Historical filter-honesty evidence is outside the epoch; fresh block/site replication is required. |
| RQ15 | OPEN | The preregistered matrix exists but has not been collected or analyzed. |

Experiment-level `LOCKED` status for existing offline studies may remain only as historical
source preservation. It does not make any F01-F10 estimate admissible in either fresh
epoch. Proposed registry fields for both paper experiments are `evidence_epoch`,
`not_before`, `fresh_inputs_only: true`, `historical_inputs_allowed_for_claims: false` and
`independent_units`; promotion should validate them against the manifest.

## Unresolved scientific decisions

1. WS05 must approve or amend the exact six-site, eight-yaw, two-session identifiability
   matrix and its 10 mm benefit/20 mm harm thresholds before collection. Only a passing
   fresh result may open selection of a causal F06 contrast.
2. After that gate, WS05 must select exactly one F06 primary endpoint and accept the corresponding claim
   boundary and limited power of 15 pairs.
3. WS06 must freshly collect/rebuild each adopted field against the execution-world hash;
   July fields are prohibited rather than candidates for compatibility arguments.
4. The current paper must regenerate F04 under the scientifically selected current policy
   or omit it; the historical v3 lifecycle demonstration is not admissible under this epoch.
5. The residual floors used by F05 must be reconciled with the selected minimal projection
   pipeline before publication.
6. The future benchmark must preregister a wholly fresh collection/rebuild, then freeze
   route/capture/site splits and the exact `p_use` label; archive recovery is provenance
   work only and cannot populate the new benchmark.
7. The operational geometry provenance (sensed depth, commissioned static map, or raycast)
   and the separate complete-map oracle must be chosen and labelled before F07.
