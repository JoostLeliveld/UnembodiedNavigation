# IWAI, the 12-page AIES thesis, and a possible ICRA paper

Updated 2026-09-06 during the matched network-navigation pilots, after full-route
preflight and runtime residual-calibration integration. This is the single current
framing and execution account. The preceding account is preserved in
[archive/ICRA_STATUS_before_12_page_map_20260906.md](archive/ICRA_STATUS_before_12_page_map_20260906.md).
The localization metrics registry continues to control scientific claims.

## Decision and format

**Recommended thesis:** extend the IWAI navigation mechanism to a fixed-camera network,
then use a compact commissioning study to explain and test the sensor assumptions.
A second conference paper is optional. A calibrated full navigation system is a stronger
claim than the minimum thesis needs, and has not yet been established.

The user specifies **12 pages, two columns, TU/e Artificial Intelligence and Engineering
Systems**. The official guide and 2026–2027 procedure require a conference/journal-style
paper suitable for submission and a mandatory AIES cover. **Appendices are allowed but
are not assessed.** Essential theory, evidence and limitations must fit in the main paper.
The documents do not state a universal 12-page limit; we follow the user's constraint and
provisionally include one page of references. Cover/reference counting remains unconfirmed.
The [verified requirements and rubric mapping](../../papers/master_thesis/planning/aies_requirements.md)
links the official documents, versions and preserved downloads. Five assessment categories
have equal weight; conference acceptance is not stated as a graduation requirement.

The [49-page report](../../papers/master_thesis/thesis.pdf) is now a source bank. Preserve
its measured figures, method explanations and provenance for reuse; do not compress it
indiscriminately into tiny text. First settle the argument and main evidence, then write the
12-page paper. The accepted IWAI source and its results remain their own published record.

Suggested working question:

> When do learned camera-reliability fields improve navigation through a camera network,
> and which localization assumptions determine whether those decisions are trustworthy?

This question supports useful positive, null and failure results. It does not require a
novel planner, a new detector, or identification of every sensor-noise mechanism.

**Full-route planner:** the uniform, geometry and GP fields all select the upper corridor
on the launch-resolved warehouse task. All selected paths pass the runtime feasibility
gate and a 2 cm physical-clearance walk; goal gaps are 6.7–9.6 cm. All reached the
80-iteration limit. Offline solves took 53–65 s. This is an implementation result and a
null route-choice result for this setup, not a navigation improvement. All candidates,
including cheaper incomplete ones, remain in `network_planner/full_route_v1`.

**Camera-level runtime match:** `commissioned_reference_r` applies the frozen residual
offset after the existing NN and uses the same full camera R as the planner artifacts.
Checkpoint identity, observation reference, frame, units and SPD are validated at load.
The actual manager agrees with the frozen offline model on 100 recorded test observations;
all nine completed original/tracking/corrected runs agree exactly on 6,190 logged camera
means/R. The corrected baseline packet passes 150 tests; the subsequent command/recovery
packet passes 159, with three archived pixel traces absent in each.
Robust `joint_network` fusion still differs from independent
precision addition; this is not a claim of a calibrated shared posterior predictor.

**The first two integration pilots failed, and their failures exposed runtime defects.**
All three original runs stopped at the northern turn. The ideal-motion check reproduces
corner cutting with 1 m waypoints and a 35 cm arrival radius. The separate tracking pilot
uses 20 cm/10 cm: it passes that corner but ends with one stuck run and two collisions.
In its P0 camera-return event, the filter discarded available odometry over most of a
26.4 s gap, including a turn. This is a verified estimator defect, not a covariance result.
The [runtime integrity audit](runtime_integrity_audit.md) records the event, source snapshots,
repairs and remaining policy problems. Repairs also cover correction ownership, publication
timestamps, command races, angular limits, same-time camera updates and a command watchdog.
`network_navigation_runtime_pilot.yaml` finished with P0 stuck near the final waypoint and
P1/P2 reaching the goal without recorded contact. The
[complete diagnostic table](../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/runtime_results.md)
retains all failures and aligned position/heading/coverage results. P2's 57.32 cm p95 belief
position error and 73.195 s accepted-correction gap show why success alone is insufficient.
The gap contains many detector sightings but very few fused candidates; per-camera terminal
admission reasons are not retained. One isolated fused candidate is refused by the total-gap
rule before NIS. These are system diagnostics, not a GP effect estimate.

The separate P0 `network_navigation_recovery_pilot.yaml` is collecting with checked rotation
recovery through the unchanged local clearance gate. Its source also guards fatal-stop/clock
rewind command publication and clears stopped execution diagnostics. Q, NN, residual offset
and camera R remain fixed. All attempts are retained; one seed per arm is descriptive.

The [complete paper map](../../papers/master_thesis/planning/paper_map.md),
[code implementation plan](../experiments/icra_commissioning/planner_implementation_plan.md),
[planner block diagram](../../papers/master_thesis/planning/planner_setup.svg) and
[paper evidence diagram](../../papers/master_thesis/planning/paper_evidence_map.svg)
are the current reviewable plan. The stronger q/R forecast remains separate from the
IWAI score cost. The existing IWAI rollout does not recursively apply hypothetical
camera updates through its horizon. Camera-level equivalence has been checked; matching
the fused posterior and validating future routes remain separate scientific gates.

## Work map: what exists and how it belongs in the thesis

Status language: **published** describes the IWAI record; **verified development/pilot**
describes the current scoped evidence; **implemented** means code/artifacts were inspected
without promoting their earlier metrics; **provisional** requires validation; **missing**
means the claimed experiment has not been completed.

| Work | Verified implementation / artifact | Evidence status | Thesis role |
|---|---|---|---|
| Kouw's analytical sensor/ambiguity work | Prior paper cited in the IWAI and thesis bibliography | Established prior work, not a new contribution by this thesis | Short conceptual starting point |
| IWAI learned-score GP and EFE route choice | `../../Thesis_Cold_Archive/camera_ready_IWAI2026/source/sections/03_learned_reliability.tex`, `04_efe_planning.tex`, `05_experiments.tex` | Published single-camera simulation study; archived campaign is separate from current runs | Main foundation and clearly attributed prior result |
| ROS/Gazebo navigation, detector, geometry, estimator and controller | `src/perception/perception/nodes/batched_four_camera_yolo_node.py`; manager, fusion and planning modules below; `field_pilot_cpu.yaml` | End-to-end collection demonstrated with five cameras; the detector filename is historical | Common platform and experimental controls |
| Fresh-frame measurement contract and logging | `src/reliability/reliability/contracts.py`, `src/experiments/experiments/core/camera_opportunity_log.py`, `experiments/fusion_on_fixed_routes/aligned.py` | Current pilot has capture identity, retained misses and terminal assimilation accounting | Essential methods paragraph; detailed schema in appendix |
| Raw bbox, fixed offset and projected-hull interpretation | `experiments/camera_observation_characterization/derive_interpretations.py`, `fair_hull_comparison.py`; `silhouette_observation.py` | Implemented characterization and fair-hull tools; raw bbox is not a ground-reference point | Explain the observation choice; compact ablation or appendix |
| Static commissioning and heading/view fields | `logs/perception_datasets/warehouse_v2_bbox_characterization_20260831/`; capture/derive scripts | Frozen grouped development data. Static reference is commanded pose without independent settled-pose verification | Bias/tail and availability evidence; not independent noise repeats |
| Existing bbox-feature NN correction | `src/reliability/reliability/learned_box_correction.py`; `logs/perception_models/box_feature_bias_correction_20260831/models.joblib` | Runtime/offline agreement reproduced; large median improvement with substantial residual tail | Keep this mean for the main controlled comparison |
| Full constant, diagonal, isotropic, geometric, spatial and calibrated-score covariance | `experiments/icra_commissioning/model.py`, `study.py`; `logs/studies/icra_commissioning_20260905/models.joblib` | Frozen mean, separate fitting/selection tiles, verified development comparison | Select a compact ladder; put all six alternatives in appendix |
| Multi-camera selection and fusion | `src/reliability/reliability/fusion.py`, `nodes/camera_manager_node.py`; `experiments/icra_commissioning/replay.py` | Runtime fusion exists; covariance alternatives have controlled six-run capture-time replay | New thesis extension; standard fusion itself is not novelty |
| Explicit camera-subset pilot | `experiments/icra_commissioning/network_replay.py`; `thesis_network_pilot/protocol.json` under the current study root | All 5 singles, 10 pairs and the full network, 2 covariance models, 6 existing runs; common prediction and scoring grid; exploratory | Tests what the network adds; exposes typical-error versus containment/tail tradeoffs |
| Availability fields, including GP | `observation_gp.py`, `experiments/icra_commissioning/commissioned_field.py`, `field_study.py`; `field_study/field.joblib` | Hits and misses retained; GP tested on grouped development and the six drives | Natural bridge from IWAI score proxy to an interpretable commissioning target |
| Joint observation outcomes and future belief | `experiments/icra_commissioning/field_driving.py`, `future.py`; `planning_covariance.py` | Forecast diagnostics exist; incoming belief/cadence/dependence matter; no independent complete-route ranking result | Limited diagnostic now; main result if validated |
| Persistent camera errors / bias states | `experiments/icra_commissioning/replay.py`, within-run diagnostics in `field_driving.py` | Diagnostic two-second bias state improves containment but is not an identified final model | One mechanism figure or appendix; possible focused research extension |
| Dense spatial sampling and perception-side KF/smoother | `experiments/perception_filter_cascade/`; `logs/studies/perception_filter_cascade_20260902/` | Implemented with dense-line artifacts. Teleported spatial sequences are not synchronized driving; no validated cascade-to-robot interface | Explain why temporal smoothing is an optional ablation, not a thesis dependency |
| Noise estimation from camera disagreements | `experiments/learned_measurement_covariance/estimate_r.py`; `logs/studies/learned_measurement_covariance/recommissioned.json` | Implemented diagnostic; shared bias is unidentifiable, independence/overlap support need checking | Optional appendix or separate future recommissioning direction |
| Image-conditioned uncertainty probe | `experiments/icra_commissioning/image_uncertainty.py`; `image_models.joblib`, `image_results.json` | Five handcrafted crop statistics were tested; this is not a trained RGB encoder; driving RGB crops are absent | Inexpensive ablation; do not call image-based uncertainty solved or disproven |
| Alternative RGB/keypoint/shape correction | `logs/perception_models/warehouse_v2_center_keypoint_20260828_r1/manifest.json`; geometry-residual/shape model directories | Provisional checkpoints exist. Centre keypoint targets z=0.35 m, not the floor reference. Earlier full-residual variants are explicitly invalid | Recover only for a focused tail/heading hypothesis; not the main method |
| Additional configurations | `generalization.py`; `logs/perception_datasets/warehouse_v2_generalization_20260902/`; `generalization_manifest.json` | Previously inspected dense configurations in the same installation, not transfer; P1b capture requires completion audit | Supporting application check with scope stated |
| Process noise and heading diagnostics | `experiments/gate0_process_noise/validate_q.py`; `coupled_heading.py`, `docs/open_questions.md` | Diagnostic code/artifacts exist; original supplied-Q identification provenance remains unverified | Freeze Q across sensor arms; investigate model mismatch without attributing it to R |
| Multi-camera planner integration | `planning/core/camera_network.py`, `base_planner.py`, `casadi_efe.py`; `export_network_planner.py`, `network_route_probe.py` | Three full-route solutions use the same corridor. Two failed three-arm pilots and a corrected pilot (one stuck, two goals) are frozen; controller follow-up collecting. All remain diagnostic | Main thesis extension is implemented; repeated robust execution and independent field-effect evidence remain |
| Physical validation / new installation | No selected physical study or untouched installation-transfer evaluation | Missing | State the simulation scope; not necessary to invent a hardware project |

Some old file comments and result documents claim that GP or a particular noise model wins
or loses. Those are tied to their old dataset and target. They do not override the frozen
current selection. Pre-2026-08-25 new-study metrics remain superseded; the IWAI results can
be described as the original published study, not as newly validated fusion evidence.

## What has succeeded, with the correct limits

1. **IWAI established a route-choice mechanism.** A GP of heading-averaged detector scores,
   with misses entered as zero, supplied a designed camera precision blend. The same EFE
   objective and route initialization could then prefer a better-observed route. This is
   a published single-camera simulation result, not a calibrated measurement-noise claim.
2. **Mean correction produces a substantial typical-reading improvement.** Recomputed on
   the same 1,172 accepted evaluation readings in 14 development tiles, against commanded
   simulator reference XY: raw ground-projection median error is 30.11 cm and existing-NN
   median is 3.27 cm, before the new shared per-camera offset. The corresponding 95th
   percentiles are 44.29 and 41.18 cm. The large tail remains; do not report the median alone.
3. **Covariance choices affect the same trajectory evidence.** All six selected pilot
   runs use fixed mean, observations, initialization, Q and score timestamps within each
   replay comparison. Calibrated-score covariance has lower median position error than
   fitted full constant covariance in all six. Its nominal 95% position-ellipse containment
   remains 77.7--88.5% across runs. Geometry improves median on all overlap runs and worsens
   it on all traverse runs. Context is not automatically better.
4. **Availability is learnable in this installation.** The GP has better held-out
   development/pilot probability scores than the fitted geometry baseline. That target is
   a valid detector output with finite projection, before the robot innovation gate. It
   is distinct from the IWAI mean-score field and from final estimator acceptance.
5. **The new pilot is complete and includes failures.** Three overlap runs reached the
   goal and three traverse runs recorded contact. Every run remains selected. These were
   fixed-runtime collection trials; they are not a comparison of new navigation methods.
6. **The evidence can now test specific assumptions.** Fresh-frame logging, fixed-input
   replay, directional covariance, availability, temporal correlation and future forecast
   diagnostics exist. The earlier focused software check passed 34 tests. Software
   correctness does not itself establish sensor calibration.
7. **The network needs an accuracy-and-consistency evaluation.** The new exhaustive subset
   pilot compares all singles, all pairs and the five-camera network on a common propagation
   grid. With score-conditioned R on the three traverse recordings, the full network lowers
   median position error relative to camera B alone, but increases p95 error and reduces
   nominal 95% containment. With constant R in overlap, the network has a larger p95 than E
   alone in all three recordings. These are exploratory within-run observations, not a
   selected best-camera benchmark or proof of the error mechanism. See the complete
   [pilot interpretation and per-run tables](../../papers/master_thesis/planning/network_pilot.md)
   and the [exhaustive figure](../logs/studies/icra_commissioning_20260905/thesis_network_pilot/camera_subsets.svg).

The exact current selection is
`logs/studies/icra_commissioning_20260905/thesis_evidence/selection.json`.
Each selected run was loaded again through `field_driving.load_run` and `aligned.py` for
this map. The six replay result files, input hashes and figures are recorded in the thesis
[verification record](../../papers/master_thesis/generated/verification.json).

**Critical limit:** covariance replay applies camera readings at capture time and does not
reproduce live processing latency/refusal. The traverse forecasts end at collision and
cover route prefixes. Neither offline accuracy nor a good availability map establishes
new navigation success or complete-route ranking. Undercoverage can involve remaining
mean error, timing, robot-model/reference mismatch and dependence; persistence is not yet
proven to be the unique cause.

## Three thesis routes and a fallback

| Route | Central claim | Additional work needed | Assessment |
|---|---|---|---|
| A. IWAI extended to a network | Study how a spatial camera-reliability model changes belief-aware route decisions with several cameras | Coherent network observation adapter; matched planner-field and camera-subset comparisons | Smallest navigation-centred extension; recommended foundation |
| B. A plus focused commissioning | Test when calibrated measurement quality and availability improve on the IWAI score proxy | Freeze constant/contextual R and availability target; same-data replay; field comparison | Recommended thesis scope: A plus the parts of B already supported |
| C. Commissioning as the main contribution | Predict useful localization information over a route from installation data | Calibrated sequential application, latency equivalence, independent future routes, then navigation | Stronger potential ICRA direction; larger evidence burden |
| Fallback if live extension remains unreliable | IWAI mechanism plus a controlled multi-camera sensor-model investigation | Finish replay/reference audit; present the recorded failure and forecast limits accurately | Coherent thesis argument without claiming a new closed-loop gain; assess scope with supervisors |

Do not make the thesis depend on C succeeding. Choose one main method and a small baseline
ladder. Additional implementations can establish an informed design decision in an appendix;
they do not all need their own main-text method or positive result.

## Smallest complete IWAI-to-network extension

Retain the detector, metric-reference NN, robot process model, Q, EFE objective, candidate
route initialization and controller. Verify one network observation function and units at
both filter and planner. The reference-position contract uses h(x)=[x,y]; the hull contract
has a different function and cannot be switched underneath a covariance comparison.

For each camera, an IWAI-style score field rho_i(s) can remain a **reliability proxy**.
Map it to the same chosen per-camera precision regime in every arm. Aggregate full state
information H_i^T Lambda_i H_i for the cameras expected at that state. State that the
independent-camera and expected-information approximations are modelling choices. Avoid
replacing the network by max(score) or probability(any camera): those discard direction,
measurement quality, and the difference between one useful view and several useful views.

Actual updates use only fresh delivered readings; misses do not create measurements. The
planner uses future configuration and commissioned fields, never future detector scores.
A score is not a hit probability, and GP interpolation variance is not camera R or robot P.

The existing `GPVisibilityMapModel` remains the legacy scalar adapter. The new opt-in
`CameraNetworkModel` is connected to both NumPy and CasADi paths in the actual planner.
It uses one fixed cost camera, full metric precision addition and the existing goal and
EFE terms. Its finite miss endpoint is a designed cost proxy, not camera noise. The
separate one-step q/R reference performs no update on a miss. Current robust runtime
fusion has different covariance algebra; the fitted mean offset and camera R are now
installed and checked online. Neither that equivalence nor a multicamera launch file
establishes calibrated network navigation.

Use two controlled comparisons, not one arm changing everything:

- **Network effect:** replay one predeclared camera, a predeclared pair, and all cameras
  through the same estimator; then use matched deployment conditions for selected live
  comparisons. Camera masks must reach both the estimator and the future model. Include
  a dropout/transition case; all cameras need not beat the best single camera everywhere.
- **Planning effect:** keep the network estimator fixed and change only the future field:
  P0 uniform per-camera score; P1 geometry-conditioned score; P2 IWAI-style
  learned score field. Add P3 commissioned availability plus calibrated quality only when
  its forecast adapter is ready. Freeze per-arm settings before the final route trials.

Pick a compact scenario set before inspecting new outcomes: ordinary overlap, a transition
or dropout, and a short poorly observed route versus a longer better-observed alternative.
Use the pilot's run-level variability to set replication; do not label three seeds a
powered final study. Freeze nuisance seeds and report failures, travel cost, heading and
position accuracy, uncertainty coverage, correction gaps and processing cost. A null result
in well-observed conditions is useful evidence about when the field matters.

## Ways to incorporate commissioning without overloading the thesis

**Minimum:** one observation-geometry diagram, one raw/corrected error CDF or field, and one
constant-versus-calibrated covariance comparison. These explain why the chosen observation
is defensible and why detector score should not be called physical covariance.

**Preferred addition:** separate q_i(s), the probability of a valid projected detection,
from R_i(c), the error model of an actual reading. Compare the score proxy against this
separation in future belief forecasts with identical conditional quality when isolating q.
Use explicit manageable joint outcomes as the reference for expected-information blending.
Its multi-step branch compression and dependence assumptions still require validation.

**Supplementary mechanisms:** heading-conditioned residuals; diagonal versus full covariance;
commissioning budget; dense-line smoothing; camera-disagreement noise estimation; handcrafted
image features. Include a mechanism only if its assumptions and exact sample population
are understandable. Do not present unrelated scripts as one coherent validated method.

## Highest-value candidates for better results

Order is a recommendation based on current evidence, not a promise of improvement.

1. **Match the measured runtime.** Arrival-time replay, detector-to-update delay, refusal
   accounting, odometry history and timestamp alignment come first. Preserve the existing
   fixes (frame identity, scaled Joseph update, campaign ledger) as correctness work.
   Gains caused by repairs cannot be called a commissioning-method gain.
2. **Deploy the strongest small covariance model with the frozen mean.** Full constant
   is the essential baseline; calibrated confidence is the current accuracy candidate;
   geometry is a meaningful comparator, not a presumed winner. Apply the same fitted
   mean offset online. Test coverage and tails alongside median error before choosing.
3. **Predict whether a view will arrive.** Retain the existing GP availability model as
   a candidate, with geometry and local interpolation baselines. Match its label to the
   deployed support/admission rule. Improving q can change route preference even if R is
   constant, so this is a useful low-complexity extension.
4. **Account for accumulated persistent error when the diagnostics justify it.** Compare
   full constant, a simple conservative covariance calibration, common-rate sampling,
   and a small fitted bias state on the same separate development runs. Freeze the
   chosen persistence parameters and propagate the same model into forecasts. Widening
   ellipses alone is insufficient; report sharpness, point error and prediction utility.
5. **Address the remaining tail with observable evidence.** Probe bbox truncation,
   confidence, view geometry and existing encoder/crop features for residual mean or
   unusability. Evaluate acceptance fraction and full trajectories. Do not reject using
   GT occlusion or error, or claim a crop-statistics null disproves image information.
6. **Add orientation-sensitive observations only for a demonstrated heading bottleneck.**
   Existing centre-keypoint and shape checkpoints can seed an investigation, but a single
   centre point does not directly measure yaw. Multiple distinguishable keypoints or a
   validated hull function would change the observation contract and need new calibration.
7. **Optional later: spatial residual mean or positive covariance-scale GP.** A GP of
   exp(g_i(s)) times a reference covariance is a compact candidate if geometry/score
   conditioning leaves a supported spatial gap. It cannot rotate ellipses or identify
   temporal independence. Use a full SPD conditional model only with sufficient data.

Camera disagreements are another possible recommissioning tool, but they cannot recover
an error shared by every camera. With only two independent views, their difference gives
R_1+R_2, not the two covariances separately. More overlap helps only if the resulting
parameter system is identifiable and cross-camera dependence is handled. This is a
separate research hypothesis, not the shortest path to finishing the thesis.

## What could become ICRA

The most promising emphasis is **commissioning a camera network to predict useful future
localization information**, with a compact correction for a demonstrated failure of the
usual approximation. Candidate mechanisms are persistent perception error and observation
availability/quality outcomes. The evidence must distinguish them from timing, mean-model
and robot-process errors.

Minimum persuasive chain: frozen commissioning model -> held-out multi-camera estimation
accuracy AND consistency -> independently tested forecast magnitude and complete-route
ranking -> matched navigation consequence. Add commissioning cost and a second configuration
with a clear recommissioning policy. If simulation-only, strengthen scene/nuisance diversity;
do not imply physical validation. A validated, bounded failure and remedy could support a
paper even if average accuracy gains are modest, but no acceptance judgement is possible now.

Multiple cameras, GP regression, a learned covariance head and EKF fusion are already
established tools. The closest sources make that explicit:
[Bultmann et al.](https://arxiv.org/abs/2303.03797),
[Rabiee and Biswas](https://arxiv.org/abs/2306.16698),
[Russell and Reale](https://arxiv.org/abs/1910.14215).
The [IWAI publication](https://wmkouw.github.io/publication/externalnav-gpvis/) establishes
this project's earlier score-based route-planning contribution. The new submission must
clearly distinguish its data, method and claim from that contribution.

The official [ICRA 2027 call](https://2027.ieee-icra.org/announcements/call-for-technical-papers/)
currently lists September 15, 2026. That near deadline is not a reason to relabel the pilot
as a complete study. The AIES thesis should remain independently finishable.

## Twelve-page allocation and figure selection

This is a design allocation, not a rendered page count or an official template requirement.

| Main-paper material | Pages | Main job |
|---|---:|---|
| Title, abstract, introduction and contribution | 1.25 | One navigation question and the new-versus-IWAI boundary |
| Background, closest work and concise IWAI foundation | 1.00 | Explain EFE/sensor-model role without reproducing the whole paper |
| System, reference point and measurement contract | 1.00 | Make the perception/estimation interface inspectable |
| Camera-network method and selected commissioning addition | 2.00 | Explain what the field predicts and how fusion/planning use it |
| Experimental design and baselines | 1.25 | Same logs versus matched live trials; fixed controls and grouped splits |
| Results | 3.25 | Prior IWAI mechanism, new network result, selected calibration and failure evidence |
| Discussion and conclusion | 1.25 | Answer the question with tested scope and null cases |
| References | 1.00 | Provisional in-limit reservation |
| Total | 12.00 | Appendices are permitted but not assessed; cover counting unconfirmed |

Aim for five or six main figures: (1) two-interface system/perception diagram; (2) camera
layout, field and route alternatives; (3) raw/corrected readings with a visible tail;
(4) paired network accuracy and coverage; (5) matched navigation outcomes and trajectories;
(6) forecast-versus-realized quality if ready, otherwise a clearly labelled limitation
figure. A published IWAI panel needs its own caption/attribution and must not be pooled with
new runs. Do not fill a missing main result with a synthetic illustration labelled empirical.

Appendices are unassessed and can use the long-form source bank: detailed EFE derivation, transforms and
Jacobians, covariance calibration and splits, all-camera/heading plots, temporal and
cross-camera diagnostics, commissioning cost, unsuccessful candidates, exact configurations,
latency accounting and software verification. Evidence essential to the conclusion stays in
the main 12 pages; appendices do not hide failed assumptions.

## Exact next actions

1. **Scope now chosen:** IWAI network extension with a compact commissioning audit.
   Requirements and equal-weight rubric are verified; preserve the accepted IWAI contribution
   boundary. The exhaustive six-run subset diagnostic is complete. Select a small prospective
   set using this development study, then evaluate it on new recordings.
2. **Complete-route preflight is done.** The three fields produce feasible upper-corridor
   solutions and no different route choice in this setup. Inspect all candidate results
   and the iteration limit in `network_navigation_evidence/preflight_results.json`.
3. **Runtime-equivalence evidence:** implement delayed-arrival replay using the selected
   `camera_opportunities.jsonl` and `correction_assimilations.csv`; compare baseline events
   against the live estimator before a new covariance result is attributed to R.
4. **Finish the separate guarded-controller P0 follow-up.** The three original, three
   tracking and three corrected baseline runs are complete and frozen. Analyze the new
   follow-up with its own config/root and verify source/model/event identity. Preserve
   every stop and collision. The next sensor-policy priority is supported-history
   admission: the P2 isolated opportunity demonstrates why dropping a first return is
   not equivalent to merely delaying one camera period. Add terminal accounting from
   detector opportunity through manager admission and fusion before assigning missing
   updates to a specific gate. The [runtime audit](runtime_integrity_audit.md) and separately
   source-pinned module audits specify remaining recovery, command and fusion policies.
5. **Collect/validate complete candidate routes, then execute matched route-choice trials.**
   Predetermine scenario families and use run-level variability for replication. Retain every
   collision or refusal. New routes and raw RGB collection are separate evidence from old logs.
6. **Use the evidence to settle the 12-page thesis argument and figures.** Writing is
   secondary to the user's current runtime and results priority. Preserve the IWAI/network/
   commissioning boundary and clearly identify missing navigation evidence. Pursue ICRA
   only if the stronger chain is supported independently of the thesis deadline.

Existing reproduction commands, run from the robot repository unless stated otherwise:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/thesis_evidence.py --static
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_replay.py
```

From the workspace root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 papers/master_thesis/build_evidence.py
python3 papers/master_thesis/build.py
python3 papers/master_thesis/planning/build_maps.py
python3 papers/master_thesis/planning/summarize_network_pilot.py
```

The first command reanalyzes the frozen diagnostic selection; it does not launch a new
navigation comparison. `build.py` builds the existing source bank, not the future 12-page
article. The [planner implementation plan](../experiments/icra_commissioning/planner_implementation_plan.md)
now contains exact export, optimizer-probe, test, dry-run and live-pilot commands. Recorded
planner outputs are under `logs/studies/icra_commissioning_20260905/network_planner/`.
Use `probe_reviewed/` for current plots; `probe/` preserves the initial clearance setting.
