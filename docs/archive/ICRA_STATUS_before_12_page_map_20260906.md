> Historical account, superseded by ../ICRA_STATUS.md after the 12-page AIES format clarification.

# Thesis and ICRA: current framing and evidence

Updated 2026-09-06. **Focus: a clean master's thesis with verified results. Thesis writing
was explicitly resumed by the user; a separate ICRA manuscript remains conditional on evidence.**
This is the current status account. Earlier execution/protocol documents retain historical
proposals; the metrics registry still does not authorize a replicated paper-facing fusion result.

Open [`../../papers/master_thesis/thesis.pdf`](../../papers/master_thesis/thesis.pdf).
Exact commands and measurement contract: `experiments/icra_commissioning/README.md`.

## Active thesis and completed pilot

The active editable thesis is `../../papers/master_thesis/main.tex`. It separates Kouw's
preceding sensor-ambiguity work, the published IWAI navigation contribution, the new
commissioning experiments, and the additional claim a possible ICRA paper would need.
The camera-ready IWAI archive remains unchanged. The thesis contains methods, protocol,
generated tables and vector figures, results, discussion, conclusions and reproducibility
details. Its institutional template and author contribution declaration remain for author review.

The predeclared six-run pilot is complete: three overlap-rich executions reached the goal;
three network-traverse executions recorded physical collisions. Every run is retained and
passes the current selection/observation accounting checks. These are collection outcomes
under the unchanged runtime system, not a comparison of commissioned navigation methods.
The canonical selection is
`logs/studies/icra_commissioning_20260905/thesis_evidence/selection.json`.
It lists exact execution IDs and hashes; no expected run is pending or invalid at this revision.

The original seed-110 and seed-111 overlap runs were recovered by explicit identities from
their completed artifacts. Their empty campaign ledger is preserved. The remaining four
runs used `field_pilot_remaining_cpu.yaml`, with the same scientific settings and only the
remaining predeclared task/seed entries. Ledger writes now use an atomic replacement; the
failed-replacement check confirms that the previous complete ledger survives. This repair
is separate from all sensor-model claims.

In all six capture-time replay comparisons, the calibrated-score covariance lowers median
position error relative to the fitted full constant covariance. Its nominal 95% position
ellipse coverage remains below nominal: 85.6--88.5% on overlap and 77.7--80.1% on traverse.
These are paired per-run pilot findings, not a statistical-significance claim. The exploratory
bias states improve coverage but do not establish calibrated fusion across both routes.
GP availability improves Brier score relative to the fitted geometry baseline in these runs;
future covariance/error agreement is not established by that availability gain.

**Critical interpretation:** replay applies measurements at capture time, without live
processing delay or the live delayed-update budget. It is not an arrival-time-equivalent
implementation of the runtime estimator. Live failures must not be attributed to covariance
models evaluated only offline. Traverse forecasts cover prefixes terminated by collision;
they do not validate complete-route ranking against a completed overlap route.

Regenerate selected evidence in the robot repository with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/thesis_evidence.py --static`.
From the workspace root, run `python3 papers/master_thesis/build_evidence.py`, then
`python3 papers/master_thesis/build.py`. Figure hashes, input hashes and selected execution
IDs are recorded in `papers/master_thesis/generated/evidence_manifest.json`.

The immediate scientific next action is delayed-arrival replay with the runtime refusal
policy, followed by independently validated future predictions. A new navigation-improvement
claim still requires the matched planner-only comparison. No new RGB network or planner
is a dependency for completing this scoped thesis.

## Thesis and second-paper decision, 2026-09-06

The accepted IWAI source was inspected at
`../../Thesis_Cold_Archive/camera_ready_IWAI2026/source/main.tex`, especially sections
03--07. Its contribution is single-camera EFE route choice using a spatial GP of detector
scores and a designed precision-to-covariance mapping. It explicitly identifies the lack
of direct metric-error calibration as a limitation. Its archived results describe the
original paper and are not newly validated current-repository fusion results.

**Recommendation:** retain the system and narrow the second-paper claim to whether
commissioning can predict the localization value of a camera network on held-out routes.
Multi-camera EKF fusion, GP interpolation, and separating availability from conditional
quality are established components, not standalone novelty claims. A second paper needs a
specific validated commissioning method or a reproducible mechanism explaining when the
usual information prediction fails and how a small correction restores useful prediction.
The current evidence does not yet establish that paper.

Keep the existing mean and robot Q fixed. First replicate full constant, geometry and
residual-calibrated confidence covariance on identical observations. Retain the manually
parameterized persistent-bias filter as a diagnostic only. Then test future uncertainty
and route ranking using measured held-out perception; finish a compact matched navigation
comparison with the estimator fixed. Defer new RGB networks, a new planner, and a general
correlated-fusion framework. Keep GP availability as a candidate; its interpolation
uncertainty is not camera noise or robot belief covariance.

The static field study is grouped development evidence, not installation transfer. The
pilot recovery and completion are recorded above. All selected driving evidence is loaded
through `aligned.py`. The cause of the original empty ledger was not established; low
disk space was observed, but must not be asserted as the cause without evidence.

Next decision sequence:

1. Pilot accounting and collection are complete. Use its run-level variability to set
   replication beyond the pilot; first resolve delayed-arrival/runtime equivalence.
2. Establish whether accuracy and coverage gains survive across routes and seeds. Check
   temporal persistence before interpreting repeated frames as independent information.
3. Evaluate availability separately from conditional quality, then compare forecast
   covariance and route ranking with held-out driving observations. Match update cadence.
4. Run a small planner-only matched comparison if forecasts support it. Freeze the
   estimator, objective and controller. Report failures and null cases.
5. Pursue a second submission only when a distinct methodological claim and its evidence
   survive these checks. The official ICRA 2027 contributed-paper deadline is September 15,
   2026: https://2027.ieee-icra.org/announcements/call-for-technical-papers/ .

The thesis remains one investigation: learned external-camera reliability for navigation
(IWAI), extension to a camera network, commissioning of metric measurement error, and
validation of future information. Clearly separate original published work from new
extensions. A second accepted paper is an ambition, not the condition for this research
to form a coherent thesis; degree requirements remain those agreed with the supervisors.

## Approved execution plan and thesis figures

The user approved the system framing and requested a clear plan with reusable block diagrams
on 2026-09-05. This section is the current plan; older execution and cascade protocols remain
historical rationale. The empirical results below retain their development-only status.

**Question:** What must commissioning learn for a fixed-camera network to predict its
localization value now and along a future route?

Generalize the observation interface, with experimental claims scoped to the tested external
camera installation. Preserve the existing metric-reference NN, detector, camera geometry,
robot dynamics and Q, planner objective, and controller. Fit uncertainty for the corrected
ground-position residual. Keep systematic correction, conditional scatter, availability,
temporal dependence and robot belief covariance separate.

The thesis-ready conceptual figures and plan are in
[`camera_information_plan.pdf`](../logs/studies/icra_commissioning_20260905/thesis_plan/camera_information_plan.pdf).
Each of five figures also has vector PDF, editable SVG and 300 dpi PNG versions; captions
and LaTeX inclusion examples are in
[`FIGURE_INDEX.md`](../logs/studies/icra_commissioning_20260905/thesis_plan/FIGURE_INDEX.md).
These are proposed architecture diagrams, not measured-result figures.

| Figure | Purpose |
|---|---|
| 01_system | Shared commissioning model from current perception to future belief and navigation |
| 02_perception | Existing bbox-to-metric-reference mean, separate corrected-residual covariance, complete opportunity contract |
| 03_fusion | Individual fresh camera evidence, capture-time updates, fixed robot Q, optional justified persistent errors |
| 04_gp | GP availability, separate quality model, joint outcomes, and belief prediction |
| 05_studies | Measurement -> identical-log fusion -> held-out forecasts -> matched closed-loop navigation |

Regenerate all diagrams and the combined plan with:

```bash
MPLCONFIGDIR=/tmp/icra_mpl python3 experiments/icra_commissioning/thesis_plan.py
```

### Dependency order and evidence gates

| Stage | Concrete work and control | Evidence required to advance |
|---|---|---|
| 1. Measurement contract and collection | Extend detector/manager/logger records to retain every opportunity and synchronized crop references; install frozen NN residual offset and ground covariance; verify live/replay agreement, identity, delay and measured-odometry handling | Reproducible baseline, complete accounting, matching mean/R at each capture; identify upstream Q provenance without retuning main comparisons |
| 2. Measurement calibration | Separate grouped mean training, covariance fitting, selection and final test; tune full constant, geometry and selected contextual models on identical accepted readings | Useful held-out calibration AND sharpness; covariance-only arms share the same mean and acceptance; retain the simple model for a null result |
| 3. Sequential fusion | Use individual camera measurements in one robot filter; retain current robust network fusion as a baseline; compare each camera and their combination on identical synchronized logs | Run-level accuracy/consistency, complementarity and dropout; residual dependence diagnosed within runs and synchronized camera pairs; any added bias state or joint covariance fitted on development runs |
| 4. Future prediction | Compare constant, geometry, local empirical and GP availability with the same conditional quality; align capture cadence, support, gating semantics and dependence with runtime | Independently collected fixed routes test forecast magnitude and route ordering; separate availability errors from conditional quality errors |
| 5. Navigation | Cross baseline/selected estimator with baseline/selected future model in a small 2 x 2 study; matched initial conditions, environment and nuisance seeds | Aggregate independent runs: success/failure, position and heading error, uncertainty sharpness/coverage, travel cost/time, correction gaps and latency; keep all failures |

Freeze scenario families before final outcomes: ordinary clear coverage, complementary views,
camera transitions, partial occlusion and dropout. Begin with independent pilot executions on
overlap-rich and network-traverse routes. Use run-level variability and a stated precision target
to choose the final number of independent runs; a fixed small pilot is not the final sample-size
justification. Use final seeds/conditions that have not been inspected for model choice.

Static repeats must introduce declared nuisance/state variation if they are used to estimate
deployment dispersion. Reprocessing an identical render is not independent sensor noise.
Retain original mean-model training cost in commissioning budgets. Add a new-installation
evaluation only with a separate explicit claim about whether new commissioning labels are allowed.

### GP decision and implementation boundaries

**Use the GP as a candidate spatial availability model first.** Its output is
`q_i(s) = P(usable camera observation | future configuration s)` for each camera. Train from
all opportunities using the frozen operational usability definition. If usability is
deterministic at exact scene state, the map represents support or probability marginalized
over explicitly declared position/nuisance variation. Do not invent independent Bernoulli
randomness at a fixed rendered state.

Reuse these existing components:

- `src/reliability/reliability/observation_gp.py`: `ObservabilityGP` and conditional
  `TwoStageGP`; spatial grid aggregation, Beta smoothing, latent logit regression.
- `scripts/visibility_comparison/fit_belief_aware_gp.py`: canonical GP fitting and an
  expected-kernel variant for uncertain inputs; reuse rather than create another GP engine.
- `src/reliability/reliability/observation_planner_artifact.py` and
  `scripts/reliability/build_planner_p_use_artifacts.py`: grid export and provenance pattern.
- `src/planning/planning/core/visibility_gp_map.py`: existing planner map adapter.

Verified code limitations to resolve before the new GP comparison:

1. `ObservabilityGP.predict_proba` returns sigmoid of latent mean and discards latent
   standard deviation. This is an approximate smoothed-logit regressor, not a Bernoulli GP
   classifier or integrated posterior-predictive probability. Validate it as such; compare
   a Bernoulli/binomial treatment if the approximation limits calibration.
2. The current planner artifact writer sets `P_conservative_plan_map = P_mean_map` and
   writes placeholder `F_std_map = 0.05`. These fields do not demonstrate conservative
   probability or quantified model uncertainty. Export real model uncertainty if used;
   do not reuse the placeholder as measured evidence.
3. Numerical map queries reject points outside the grid; the CasADi adapter clamps them.
   Align operational support semantics before route comparisons. Grid bounds alone do
   not establish observational support within the grid.
4. Current wrapper inputs are XY. Add periodic heading features only if the data support
   them. Explicitly distinguish surveyed/reference commissioning coordinates from noisy
   runtime belief coordinates; integrate future position uncertainty consistently.
5. Marginal per-camera GPs do not define joint availability. Keep the existing joint-outcome
   empirical reference; use independent outcomes only if synchronized opportunities support
   that approximation. If innovation gating depends on prior belief, model it in the rollout
   or keep a fixed pre-innovation usability definition and report the later gate separately.

Hold the conditional quality model fixed while testing availability. Future-image-dependent
quality must be marginalized through commissioned feature/quality regimes. A planner cannot
query tomorrow's detector confidence. Do not equate inverse average information to average
posterior covariance; use explicit hit/miss or joint-outcome averaging as the reference.

A **later** covariance GP can model a positive scale:

```text
R_i(s) = exp(g_i(s)) R_geometry,i(s),    g_i ~ GP.
```

Fit the scale from frozen-mean residuals using an appropriate residual likelihood and grouped
selection. A scalar scale cannot change ellipse shape; a Cholesky model is an extension only
if directional structure and sample support justify it. The GP posterior variance describes
uncertainty about the learned function; it is not automatically measurement covariance R.
A static GP map also does not establish a temporal white-noise model. Persistent spatial
function uncertainty, if propagated, is shared across repeated route queries rather than
redrawn independently at every step.

A GP stays in the method only if it improves held-out probability/calibration or future
prediction at acceptable cost over geometry and local empirical outcomes. If not, the
conclusion is that the simpler commissioned representation suffices for the tested domain.
No GP benefit, transfer result, or navigation improvement is presumed.

Primary methodological reference: Rasmussen and Williams, *Gaussian Processes for Machine
Learning*, [regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf) and
[classification](https://gaussianprocess.org/gpml/chapters/RW3.pdf), MIT Press, 2006.

### Empirical figures to earn

| Study | Thesis figure | Required interpretation |
|---|---|---|
| A | Raw/corrected spatial residual fields with heading panels | Call single-sample arrows residuals; claim conditional bias only with supported averaging |
| A | Predicted/empirical ellipses and support | Same reference, measurement space, accepted population and directional units |
| A | Coverage curves beside sharpness and commissioning budget | Grouped evaluation; include mean-training cost and independent subset repetition |
| B | Individual and fused trajectories in complementarity/dropout | Same observations, odometry, Q, initialization, update rate and acceptance |
| B | Within-run temporal and synchronized cross-camera residual correlation | Condition scale/mean appropriately; reference-error caveat; no lag pairs across runs |
| C | GP/geometry availability maps and held-out probability calibration | Misses retained, support and sample counts shown; no future-image inputs |
| C | Predicted versus realized route uncertainty and route ordering | Independent held-out executions; prediction windows are not independent replicates |
| D | Matched trajectories, belief ellipses, camera geometry and outcomes | Separate estimator and planner effects; report failed runs, time/cost and gaps |

All future numerical panels must follow `localization_metrics.md` and use explicit manifests
through `aligned.py` for driving comparisons. The conceptual figures above are usable now;
existing numerical panels remain marked development-only until their corresponding evidence
gate is met.

## Framing supported now

**Commissioning a useful per-frame camera model is not sufficient to predict sequential
localization quality: persistent errors and update cadence matter.** Keep the scientific
chain from commissioning through multi-camera fusion and future belief to navigation.
Do not frame this as a new covariance NN or an established navigation improvement.

- Existing metric-reference NN correction is retained; no new image mean network is needed.
- Full constant covariance receives the same selection-set scale/shape tuning as competitors.
- Confidence conditioning beats the tested geometric, spatial and crop-statistics alternatives
  on the current static development score. This does not establish superiority over image encoders.
- The temporal-bias ablation is not a consistent accuracy winner. Retain it as a dependence
  diagnostic; do not declare a calibrated final model from one drive.

## Measured results (not final-test or physical claims)

Static development: 1,172 corrected readings on 14 held-out tiles, with 1,172 covariance-fit
and 819 selection readings. Existing NN training costs 3,249 boxes. Camera-reading median
error before the extra per-camera offset is about 30.1 cm raw and 3.3 cm NN-corrected;
all covariance arms use the same additional offset. Second capture: 6,884 returned readings
on previously examined dense configurations in the same installation.

Capture-time belief replay, full rate; each row uses identical observations/odometry/Q/mean
within its drive. Reference: simulator GT at each evaluated odometry timestamp.

| Model | Sept 3 drive median / 95% coverage | New CPU drive median / 95% coverage |
|---|---:|---:|
| Full constant | 3.95 cm / 93.7% | 2.82 cm / 99.6% |
| Geometry | 3.46 cm / 97.8% | 2.62 cm / 99.4% |
| Confidence | 2.92 cm / 80.9% | 1.89 cm / 98.8% |
| Confidence + 2 s bias state | 3.45 cm / 99.9% | 2.58 cm / 99.9% |

Sept 3: 1,559 timestamps. New execution: 835 timestamps, ending at collision. Both nominal
seed 10; different executions, hardware cadence and realized paths. These are **two separate
diagnostic comparisons**, not independent-seed replication or an aggregate significance test.

The new **live baseline** is separate from those replays: collision after 79.84 s, median
planner-belief position error 2.79 cm, p95 33.86 cm over 788 unique in-drive belief timestamps.
All 246 published correction batches have terminal outcomes; 18 (7.3%) were dropped, and
the longest accepted-correction gap was 17.60 s. Contact was with `obs_Cs_p0`. This is a valid
failed execution; the earlier 720 s wall-time-interrupted CPU run remains invalid.

Future prediction: one recorded route, 1/3/5 s forecast horizons, joint outcome averaging versus
expected information. Predictions materially depend on cadence. No independently replicated
route-ranking or closed-loop planning benefit is established.

## Implemented and checked

New calibration, manifests, common static records with misses, image-statistics probe,
same-installation configuration application, fixed-Q replay, per-camera comparisons, temporal
and cross-camera residual diagnostics, stationary camera-bias ablation, and joint-outcome
forecast reference. Models, JSON tables, vector PDFs and PNGs are on disk. Focused checks:
79 passed, 3 skipped. Runtime repairs are separate from sensor-method changes.

The working deployed path is `batched_four_camera_yolo_node.py` → `CameraObservation` →
`camera_manager_node.py`/`LearnedBoxCorrection` → `fusion.py` → `FusedMapMeasurementSource`
/`belief_correction.py` → unicycle prediction → preselected-route turn-then-go controller.
Mean inference is truth-free. `/odom_noisy` drives prediction; the runtime can fall back to
commands when odometry history is absent. Q=0.01/0.02 is the existing configuration; the
original process-noise identification artifact remains unverified. Legacy fusion logs do
not retain all refused camera opportunities or synchronized RGB crops. No perception cascade
is used in the main replay. New covariance and future models are not yet installed online.

## Exact next experimental work

1. Collect three independent seeds on at least overlap-rich and network-traverse routes,
   retaining every camera contract and synchronized RGB crop. Use the CPU baseline command
   in the README with >=1100 s wall allowance; choose scenario/seed manifest before runs.
   This enables run-level comparison, temporal-model selection and image-feature replay.
2. Add the frozen full-constant and confidence model artifacts to the manager after NN
   correction, including the same per-camera mean offset and artifact hashes. Replay-check
   each live observation before comparing estimator arms. Freeze acceptance, Q and controller.
3. Extend the existing planner's observation interface with joint-outcome averaging and a
   dependence/cadence model justified on those drives. Compare route ranking on two fixed
   held-out routes before matched estimator/planner factorial navigation trials.
4. Audit and finish `warehouse_v2_icra_p1b_geometry_20260904` before treating it as new evidence;
   its manifest was still `running` with 6,015 rows when inspected. A new installation and
   physical validation remain separate missing datasets, not claims from the dense-line capture.

Literature anchors verified from primary sources: Bultmann et al. (arXiv:2303.03797),
Rabiee & Biswas (2306.16698), Russell & Reale (1910.14215), Zhang et al.
(10.36227/techrxiv.11663871.v5). The intended difference is validated future sensor prediction
from commissioning, not external-camera localization or learned covariance alone. The accepted
IWAI score-reliability/planning contribution remains separate.
