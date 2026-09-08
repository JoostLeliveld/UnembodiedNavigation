# Localization metrics contract

## Visibility speed commissioning (2026-09-08)

`visibility_speed_1mps_commissioning_20260908` records a separate user-requested
1 m/s operating configuration. V1 Uniform timed out at the clearance guard.
V2 Uniform passed navigation; IWAI stopped near the first bend and timed out
following starting-belief-footprint rejections. Commissioned was not launched.
All completed v2 attempts passed source/crop, ledger, runtime/handoff replay and
own-time support checks. Uniform's nominal 95% containment was 41.5%.
The final-campaign gate failed. These commissioning attempts establish neither
calibrated uncertainty, speedup nor replicated superiority. Exact selections and
results are in the registry and [commissioning record](visibility_speed_commissioning.md).
Keep both failed attempts; do not pool this operating point with the frozen pilot.

## One-task visibility pilot (2026-09-08)

`one_task_visibility_pilot_20260908` selects three completed west-to-east drives,
one per Uniform, IWAI score-GP and Commissioned availability-GP planning interface,
seed 1200. Joint runtime fusion is identical. Future readings are independent in
all three planning arms. All three reach belief goals and finish within 10 cm
of the physical goal with positive mapped clearance. All select the shorter
global corridor and retain feasible seeds under the fixed optimizer budget.
The mean goal cost dominates the evaluated candidate ranking. The fitted spatial
fields change the forecast but do not establish an EFE route-choice benefit.
All source/crop, terminal-ledger, runtime and handoff replay checks pass. All
28,161 scored belief timestamps have own-time native reference support.
Nominal 95% containment is 61.9%, 60.6%, 70.6%. These single drives do not
establish superiority or calibrated uncertainty. The earlier long-route and
60-drive selections remain separate. Original and amended preparation candidates
are retained. IWAI is a network adaptation, not a literal published reproduction.

## Four warehouse crossings (2026-09-08)

`long_warehouse_routes_v1_20260908` selects four completed attempts, one per
long task under JointJoint. Three traverse 23.49–31.25 m and reach belief goals.
Two meet the physical 10 cm and mapped-clearance criterion. The southwest
attempt remains stationary through a 600 s timeout because the uncertainty
clearance guard blocks departure. All four source/crop identities, terminal
ledgers, source hashes, runtime replays, global-to-local queries and own-time
reference support pass. The moving drives contain the reference in their
nominal 95% ellipses on 49.3%, 62.7% and 72.6% of scored timestamps. Keep the
stationary attempt's containment separate from navigation uncertainty.
This selection establishes long-route integration diagnostics only. There is
one attempt per different task, and no replicated EFE benefit or calibrated
uncertainty claim. The global solutions retained map-derived seeds. Their
objective is undiscounted over the mission; local discount remains 0.98.
The GP underrepresents the geometric corridor visibility contrasts. The
previous 60-drive comparison remains a separate short-task selection.

## Provisional thesis navigation comparison (2026-09-08)

`provisional_thesis_navigation_20260908` selects 60 completed attempts, three
arms and five paired seeds on four development-selected tasks. This is a
method-development comparison, not the final campaign. All native
source/crop identities, received terminal ledgers, source/data hashes and
exact delivery-order replays passed. All belief timestamps have own-time
native reference support. The 45 core and 15 stress attempts remain separate.
All core attempts met the navigation criterion. Long-aisle success is I/I 5/5,
J/I 0/5 and J/J 2/5. Three J/I stress timeouts and four stress mapped-clearance
failures remain included. Navigation success does not certify calibration.
All three methods undercovered on the long aisle. No warehouse-wide
calibration, unseen-region confirmation or global obstacle-fork claim follows.
Native publication tails remain separate from received and assimilated frames.
The three preparation pilots are not final replicates. No method or parameter
set is frozen by this selection. It does not change the invalidated legacy
fixed-route fusion boundary below.

## Development route iteration (2026-09-08)

`route_iteration_v1_20260908` selects five completed development drives with
new route shapes, an original-case retest and a four-metre aisle extension.
The method changes world-XY goal preference, XY entropy, CasADi representation
and path feedback together. It retains the frozen camera model. All five own-time
evaluations, source ledgers and exact delivery-order replays passed. These are
different trajectories without independent replication; do not attribute gains
to one change or call the observation uncertainty calibrated. The long-aisle
run completes while its belief remains biased and undercovers. Keep navigation
completion and uncertainty consistency separate.

## Overnight GP/joint forecast selection (2026-09-08)

`gp_joint_forecast_navigation_20260908` now selects three diagnostic drives on
segment 0, seed 850, one per arm. The user canceled the remaining campaign.
One extra segment-6 drive completed while the stop took effect and is preserved
separately in `scope_change.json`. The three selected drives all timed out;
source ledgers and delivery-order replay passed. There is no replicated result.
Its model is a hypothesis under test: the development pilot on segment 0 timed
out and undercovered. Report all attempts, including timeouts and integrity
failures. The campaign does not certify calibration or global route choice.
Use the dedicated exact source-frame ledger and own-time native reference
evaluator, then verify delivery-order runtime replay. Its expected-belief EFE
objective extension and sequential covariance approximation must be named.
The legacy fixed-route selection remains invalidated.

This file defines which quantities may be compared and which runtime evidence makes a
drive scoreable. The original contract is dated 2026-08-29; the versioned identity
clarification below describes the 2026-09-07 repair and does not upgrade historical runs.

## Evidence boundary

- Studies before 2026-08-25 are superseded.
- Fusion drives with `logging_schema_version < 4` are diagnostic only.
- There is currently no frozen paper-facing fusion selection. Scoring must stop until
  `logs/studies/fusion_on_fixed_routes/frozen_runs.json` explicitly names a complete,
  provenance-homogeneous schema-4 set.
- A run directory, `RESULTS.md`, modification time, or `latest` name is never evidence
  selection.

## Three quantities that must stay separate

1. A camera reading is scored against ground truth at `obs_stamp`.
2. A fused correction is scored against ground truth at `fused_stamp`.
3. A planner belief is scored against ground truth at `planner_belief_stamp`.

State the layer, statistic, reference, run set, seed count and sample count. Never compare a
camera-reading RMSE with a belief median or call either one simply "localization error".

## Event identity and weighting

- For `camera_batch_outcome.v2`, `source_batch_id` identifies a logical camera cycle.
  Each native detector call has a distinct `invocation_id`; observation members retain
  it as `detector_invocation_id`. A cycle can contain several calls when inference is
  chunked. Each received image has `source_frame_id`, binding producer epoch, camera,
  integer capture timestamp and image-content digest. This is received-image identity,
  not an unobserved hardware capture sequence. Every call and member must join explicitly
  to its cycle and outcome; neither logger ticks nor held messages create new events.
  The earlier statement that every `source_batch_id` was one physical invocation is
  insufficient for chunked inference. Historical records lacking these fields retain
  that evidence limitation; no missing invocation identities are inferred.
- Camera capture timestamps inside one batch must span less than 0.20 seconds; the repaired
  campaign enforces 0.05 seconds at both detector and manager boundaries.
- Every published fused correction must have exactly one row in
  `correction_assimilations.csv` with the same `source_batch_id`.
  Schema-8 logging supplies an actual correction-publication ledger and raw delivery
  evidence. Validate the complete publication/terminal ledger before reference filtering.
  Exact integer timestamps must be preserved when supplied. A terminal belief epoch
  identifies recursive state; a source epoch identifies the correction producer. These
  independent epochs must not be equated. Canonical ledger epoch comparisons use the
  publication's source epoch and the terminal's explicit `source_epoch`.
- Only `accepted`, `accepted_bootstrap`, or `reanchored` assimilation rows are belief-update
  events. NIS rejections and dropped corrections are not post-correction beliefs.
- A run is invalid when a correction is **unaccounted for**: a missing assimilation, a
  duplicate one, an extra one, an unclassifiable status, or a refusal with no recorded
  reason. The five valid statuses are `accepted`, `accepted_bootstrap`, `reanchored`,
  `rejected`, `dropped`.
- A **refusal that records its reason does not invalidate the run.** The filter declining a
  measurement it cannot causally bridge is a gate decision, the same class of event as a NIS
  rejection, which has never invalidated a run. The commonest cause is a camera outage longer
  than the replay cap — a property of the warehouse, not a fault in the drive. Measured on
  the 2026-08-29 drives, 90% of runs contain such an outage and the median longest one is
  13–17 s on three of the four routes; failing the run on it would discard those drives *by
  coverage*, keeping only the well-covered route and deleting the comparison.
- **Report `correction_dropped_fraction` and `longest_correction_gap_s` beside the accuracy.**
  How much of a drive the cameras actually carried, and how blind its worst stretch was, are
  results — not preconditions.
- Aggregate within each drive first, then compare the five paired seeds. Logger ticks and
  detector frames are not independent replicates.

## Accuracy and consistency

- Position errors are Euclidean errors in metres internally and reported in centimetres.
- Report median and 95th percentile position error; label means and RMSE explicitly when used.
- Report uncertainty consistency beside accuracy. For planar NEES, the mean target is 2,
  the median target is `2 ln 2`, and 95% ellipse coverage uses chi-square threshold 5.991.
- Heading error is reported separately in degrees or radians and scored at its own stamp.

## Ground-truth firewall

`gt_*` fields are offline references. They may score estimates and physical outcomes but
may not enter the online estimator, admission gate, planner, goal decision or stuck decision.
Wheel odometry is diagnostic input and must never be named ground truth.

The executable implementation of alignment and event selection is
`experiments/fusion_on_fixed_routes/aligned.py`. If this prose and that loader disagree,
stop and repair the contract before reporting a number.

## Dedicated joint-state navigation comparison (2026-09-07)

The registry entry `joint_bayesian_navigation_v2` names a separate, frozen,
20-drive short-range comparison. Its estimator maintains one joint robot/error
state; it does not publish a separate fused correction. The dedicated evaluator
therefore verifies exact received camera-frame membership, unique accepted
members and one reasoned terminal batch classification, then uses
`aligned.TruthSeries` to score each recorded belief at its own timestamp.
Source/model hashes, deployed robot identity and exact runtime replay are checked
separately. Five paired seeds are compared within each segment.

Only this registered selection is covered by the dedicated schema. It does not
change the legacy fixed-route fusion evidence boundary above. Its scope is current
belief estimation and one-metre closed-loop navigation. The planner still uses its
legacy future-observation forecast, so these runs cannot establish a benefit from
forecasting correlated observations, gradual occlusion or global route choice.

## Conservative stochastic-model confirmation (2026-09-07)

The registry entry `conservative_stochastic_model_20260907` selects eight new
out-and-back captures on four segments, with seeds 721/722. The evaluated
13-state artifact was frozen before scoring that comparison. Its manifest
records a structural amendment during model-free acquisition. These are
capture-time localization replays on the same trajectories, not closed-loop
trials controlled by that artifact.
The dedicated capture loader uses bounded `aligned.TruthSeries` support and
unique source frames. Average each drive's statistic before combining seeds.
Report nominal ellipse containment beside ellipse size. Conservative containment
does not establish precise conditional 95% calibration.

The separate exact-delivery replay of the 20 registered v2 navigation paths
compares counterfactual beliefs. It does not replace their original closed-loop
outcomes. Earlier seed-720 confirmation was used to revise that covariance
and is development evidence. The binary availability model was selected after
inspecting the newer recordings. Its path-conditioned horizon scores remain
development evidence, separate from the camera-error model freeze. Neither
comparison establishes a new EFE planning benefit or promotes the legacy fusion
selection.
