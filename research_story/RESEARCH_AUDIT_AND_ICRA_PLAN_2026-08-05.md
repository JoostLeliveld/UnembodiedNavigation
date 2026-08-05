# Research audit and ICRA plan — 2026-08-05

## Executive decision

The repository contains enough work for a strong paper, but not for all of the paper
stories currently documented. The work should be consolidated around one finding:

> Fixed infrastructure cameras leave temporally correlated, camera-specific systematic
> errors. A conventional filter repeatedly counts that error as fresh evidence and becomes
> confidently wrong. A gated calibration policy removes only resolvable outliers; a
> per-camera residual floor and network-held-out cross-check keep the belief honest when
> calibration cannot remove the error; a change-based monitor detects when a correction
> has expired.

This is stronger than the uncertain-input-GP, factorised-`R_cond`, or “more cameras improve
localisation” stories because it is supported by several independent measurements and by
informative negative results. It also has one precise missing experiment: the closed-loop
consequence of honest versus overconfident belief.

The recommended ICRA paper is therefore a **measurement-model and belief-honesty paper
for assistive infrastructure-camera localisation**, with navigation as the consequence.
It is not a GP paper and not a generic multi-camera fusion paper.

No files were deleted during this audit.

## Repository-level diagnosis

There are three distinct deliverables and they should stop competing for authority:

| Repository | Proper role | Decision |
|---|---|---|
| `RobotControlExternalCamera` | Frozen/revised single-camera paper implementation and its 60-run C0/C1/C2 campaign | Preserve as Paper 1 evidence; maintenance only |
| `thesis-report` | Historical submitted-paper source | Freeze; do not turn it into the new ICRA paper |
| `UnembodiedNavigation` | Active multi-camera, calibration, belief-honesty, health, and planning research platform | Sole active research repository |

The active repository is about 13 GB, has roughly 102k lines of Python across source,
scripts, and experiments, and has accumulated multiple overlapping narratives. Its unit
and contract suite currently reports **891 passed and 2 failed**. Both failures concern
the four-camera detector runtime contract: a source-shape assertion for the batched node
and a missing `output_wall_hz` field in the v6 detector configuration. The core scientific
library is therefore well tested, but the detector-runtime evidence contract is not fully
green and should be repaired before the final campaign.

The present `research_story/registry.yaml` is no longer a reliable source of record. It is
dated 2026-07-15 and still calls uncertain-input GP the primary contribution, says the
factorised observation model is unimplemented, and describes several August results as
future work. The August framing and result files contain the more current decisions.

## Implemented module map

The ten public module folders are useful for navigation, but the paper-facing maturity is
better represented by the following functional map.

| Functional module | Main implementation | Evidence/maturity | Paper role | Decision |
|---|---|---|---|---|
| Simulation and world contracts | `src/sim`, `src/unav_common` | AWS, four-camera, and Meerhoven worlds; route and Gazebo-version tests | Experimental infrastructure | Keep; only AWS and full-4cam are core evidence worlds |
| External-camera detector | `src/perception`, `scripts/perception` | Single-camera detector is established; four-camera runtime exists but two contract tests fail; several successor configs remain diagnostic-only | Supporting measurement source | Keep one frozen evidence runtime; retire successor churn |
| Projection and calibration | `src/state`, `reliability.projection` | v2 along-bearing correction audited; gated v3 cross-bearing correction implemented and held-out tested | Core method/mechanism | Keep and centre |
| Operational logging and GT firewall | `src/experiments`, `reliability.contracts`, `reliability.firewall`, event builders | Operational/evaluation split, manifests, residual schema, campaign metrics | Credibility backbone | Keep; mandatory for every promoted result |
| Belief estimator and uncertainty accounting | planner correction code, `bayesian_filter_showcase` | Conventional filter, NIS gate, per-camera `R_cond`, leave-one-out cross-check, correlation floor, ablations | Core method | Keep and centre |
| Spatial availability/observability | `observation_gp`, geometry/FOV/range priors, legacy GP maps | Spatial field is real; learned GP ties or loses to simple geometry/distance on held-out routes | Baseline/support | Keep simplest geometry/FOV model; GP becomes ablation |
| Factorised observation model | `reliability.observation_model`, hit/miss posterior, CasADi parity | Structurally implemented and tested; `p_qual` is saturated in single-camera data; per-camera `R_cond` ties pooled constant | Correctness/support | Keep code; retire as headline |
| Multi-camera selection/handover/fusion | `reliability.fusion`, `handover`, `camera_manager`, Toro baseline | Extensive unit-tested plumbing; one real traverse; naive all-camera fusion loses to best single; confirmatory campaign absent | Secondary systems evaluation | Keep selection and Toro baseline; do not claim fusion superiority |
| Health and calibration lifecycle | `health_ewma`, operational residuals, drift lifecycle | GT-free change statistic detects injected drift before stale correction becomes harmful | Core supporting contribution | Keep and centre |
| EFE and belief-space planning | `src/planning`, lane graph, hit/miss mixture | Strong single-camera 60-run predecessor; analytic covariance result; new multi-camera closed-loop evidence absent | Consequence/evaluation | Keep, but run only the discriminating campaign |
| Achievable-precision map | `experiments/achievable_precision_map` | Composes measured availability, odometry growth, and residual floors; coverage selects the wrong camera on 15.7% of floor | Bridge from method to planning | Keep; use as one main figure |
| Active commissioning | Module 10/plans only | No implementation or evidence | Future work | Park outside the paper |
| Meerhoven 12-camera world | world, layout, capture experiments | Built and visually verified; detector/provenance chain incomplete | Exploratory external-validity asset | Keep outside critical path and main claims |

## Results that are genuinely promising

### 1. Correlated systematic error is the binding mechanism

The deployed calibration leaves up to **78 mm cross-bearing bias**. Projection geometry
alone produces a roughly **4.1×** change in ground error over a camera footprint, but the
held-out studies show bias transfer dominates the remaining variance model: one camera's
90% coverage falls to **13%**, and train-to-test bias transfer costs up to **47 nats**.
This is a clear, physical failure mechanism rather than an arbitrary learned score.

### 2. Conventional filtering is confidently wrong

Across three recorded four-camera captures (~1400 detections), trusting all observations
gives median NEES **4.22** instead of the calibrated 1.39, states 1.9 cm uncertainty while
delivering 5.3 cm RMSE, and places truth outside the stated 95% ellipse **41.9%** of the
time. This is an excellent paper problem because the failure is both measurable and
important downstream.

### 3. Standard remedies fail for different, explanatory reasons

- A chi-square innovation gate rejects only **0.2%** of updates and leaves overconfidence
  unchanged.
- Sharper per-camera `R_cond` worsens NEES from 4.22 to **5.11**.
- Comparing a camera against the belief it helped create produces self-confirmation; one
  camera's error is understated **4.2×**.
- Uniform four-camera fusion loses to the best single camera on the real traverse.

These are not failed experiments to hide. Together they are the strongest section of the
paper because they rule out obvious reviewer objections.

### 4. The positive method restores honesty without sacrificing accuracy

The per-camera correlation floor plus leave-one-camera-out reference reduces unearned
confidence from 41.9% to **3.3%**, states 5.1 cm versus an actual 5.0 cm, and leaves RMSE
essentially unchanged. The floor is the primary mechanism: without the cross-check it
still reaches 6.9%; a pooled floor reaches only 19.3%, showing why camera identity matters.
Leave-one-capture-out evaluation remains substantially better than the baseline on all
three folds (1.1%, 3.5%, and 11.4% outside the 95% ellipse versus 35.8–46.4%).

### 5. Calibration has a defensible, bounded role

One additional gated cross-bearing parameter reduces camera C's bias from 77 mm to about
4 mm and reduces held-out belief NEES from **8.51 to 1.06**. Applying the correction
without the gate harms camera A by 27 mm. The result supports a memorable policy:
**correct resolvable outliers; leave marginal cameras raw**.

### 6. The lifecycle result closes the calibration story

The beneficial camera-C correction becomes harmful at **0.25° yaw drift**. Reusing the
absolute commissioning gate as a health monitor fails because it can trigger at rest and
can be masked by bias cancellation. The change-based statistic detects all tested yaw
faults at 0.1° and translations at 0.025–0.05 m, before the harmful crossover. This turns
calibration from a one-shot fit into a complete commission–monitor–expire lifecycle.

### 7. The planning model has a clean analytic correction

Replacing hit/miss uncertainty by a single inflated `R` understates posterior position
trace by about **90%** at the runtime operating point. The equivalent covariance changes
by **6.6–10.8×** with the prior, so a cached position-only `R_plan(s)` cannot be generally
correct. This is a valuable correctness result, but it is not yet navigation evidence.

### 8. Achievable precision is more useful than coverage alone

On **15.7%** of the reachable four-camera floor, the most available camera is not the most
informative. Camera C owns 25% of the floor by symmetric coverage but only **14.8%** under
the achievable-precision criterion. This is the best bridge from the belief method to a
planner-facing quantity.

### 9. Preserve the single-camera navigation anchor

The revised `RobotControlExternalCamera` campaign is a clean 60-run anchor: C0 geometry
and C1 constant-R each reach 15/20 goals, while C2 reaches **20/20 with zero contacts**.
This establishes that observation-aware planning can matter. It should be cited as the
predecessor result, not mixed numerically with the new multi-camera campaign or presented
as evidence for correlated-bias handling.

## Stories to retire, park, or demote

### Retire as paper headlines

1. **Uncertain-input GP as the primary novelty.** Methods tie at real belief uncertainty;
   the operational learned map does not beat geometry on the robust arbiter; the simple
   distance/FOV baseline matches the GP on held-out-route Brier. Keep the implementation
   and null result as an ablation or thesis chapter.
2. **A learned GP beats geometry.** In the present worlds, 95% of misses are out-of-FOV.
   Geometry generalises for free and is the selected model under the repository's own
   gate.
3. **Factorised availability plus spatial `R_cond` changes route choice.** `p_qual` is
   nearly saturated, per-camera `R_cond` ties a pooled constant, and no closed-loop route
   result exists. Keep the factorisation as correct bookkeeping.
4. **More cameras or naive fusion improves localisation.** The real traverse shows the
   opposite: all-camera uniform fusion loses to the best single camera, and the best
   selected subset wins.
5. **The hit/miss mixture changes route choice.** The analytic result is strong; the route
   claim is untested and previously designated a deferral.
6. **Formal safety or deployment guarantees.** The evidence is Gazebo-only, one robot,
   2-D position, controlled faults, and empirical operating-envelope characterization.

### Park outside this ICRA paper

- Active commissioning and information-gain routes.
- Large-fleet, multi-target association, duty-cycled onboard sensing, energy saving, and
  throughput claims.
- Real-image transfer and real-hardware deployment.
- Spatially varying `R_cond(x)` with the current 1426 detections.
- The Meerhoven world as a required experiment. It is useful only if the core campaign is
  complete and time remains.
- Further detector architecture/runtime successors unless needed to make the frozen
  campaign run reliably.

## Cleanup and deletion plan

Deletion should be separated from scientific retirement. Null-result evidence is valuable
and must be archived; generated build products and failed captures are disposable.

### Delete now: reproducible generated material

- `build/`, `install/`, and `log/` after recording the build command. Together they are
  about 547 MB and are regenerated by `colcon build`.
- `.pytest_cache/`, all `__pycache__/`, `rec_launch.log`, and the ignored generic
  `yolo26n.pt` checkpoint.
- Duplicate local model copies in `local_artifacts/` after confirming the canonical
  checkpoint and hash in `logs/perception_models/`.
- Failed and explicitly diagnostic/smoke dataset directories, especially paths matching
  `*.failed_*`, `_smoke_*`, `*diag*`, and obsolete detector `last.pt` copies. Preserve one
  frozen `best.pt`/runtime artifact plus its training manifest and metrics.

### Archive, then remove from the active workspace

- The many `logs/visibility_comparison/mc_*`, `rsweep_*`, `*smoke*`, and one-off
  scheduling/rate probes. Keep a compact index containing the command, outcome, and reason
  for retirement; retain raw data only for runs cited by a promoted `RESULTS.md`.
- Retired-world captures such as `full2cam_*`, `warehouse_big_zeroshot_*`, and other data
  that cannot be paper evidence under the two-world rule. These account for several GB.
- Superseded detector datasets/checkpoints. The active four-camera dataset alone is about
  3.2 GB; the earlier capture and smoke variants add over 1 GB. Move raw images to cold
  storage and keep manifests, split lists, audit contact sheets, and the chosen model.
- Exploratory presentation exports, videos, and duplicate PDFs once a final figure is
  promoted to `paper_artifacts`.
- The top-level `_archive` directory (about 3.7 GB) should live outside the active thesis
  workspace or in compressed cold storage. Do not allow archived code to appear in search
  results during normal development.

### Deprecate in code before deleting

- The three trust-to-`R_plan` implementations must become one. Preserve the deployed
  precision blend as a named **legacy baseline**, route all new code through the hit/miss
  observation model, then remove the divergent 40 px offline copy and the unresolved
  40-versus-120 miss endpoint.
- Keep only one canonical calibration loader and one active calibration ID per campaign
  arm. v2 and v3 remain immutable evidence artifacts, not mutable defaults.
- Reduce detector configs to: one frozen evidence runtime, one clearly named development
  runtime if still needed, and archived predecessors.

### Never delete

- `honest_campaign_v1` and the revised 60-run `honest_campaign_v2` evidence.
- Raw captures used by the bias, filter, calibration, drift, or final closed-loop results,
  unless copied to verified cold storage with hashes.
- Any `RESULTS.md`, manifest, preregistration, analysis script, summary JSON/CSV, or figure
  that supports a quoted paper number.
- Null-result studies: uncertain-input GP, geometry-vs-GP, per-camera `R_cond`, naive
  fusion, and route-choice nulls. They justify the chosen method and protect against
  hindsight bias.

## Recommended ICRA paper shape

### Working thesis

> An assistive infrastructure-camera localisation service is useful only when it is
> honest about persistent camera-specific error. Correlated residual bias defeats
> conventional covariance, gating, and naive fusion; a gated calibration policy plus
> residual-aware belief flooring restores calibrated uncertainty and exposes a spatial
> achievable-precision contract to the navigation system.

### Recommended title

**You Cannot Calibrate Your Way Out: Honest Belief for Infrastructure-Camera Robot
Navigation**

The current subtitle “Correlated Camera Bias and Belief Honesty” is also strong, but the
word “navigation” should remain only if the final closed-loop campaign is completed.

### Contributions to claim

1. A measurement and decomposition of persistent camera-specific systematic error in an
   infrastructure-camera network, including its consequence for belief consistency.
2. An experimental demonstration that innovation gating, sharper per-camera covariance,
   self-referenced health monitoring, and naive fusion do not address the correlated
   component.
3. A gated per-camera calibration policy with a measured resolvability/data requirement
   and an operational change monitor that detects calibration expiry before harm.
4. A residual-aware belief update policy—per-camera correlation floor plus held-out-camera
   cross-check—that restores calibrated uncertainty without reducing positional accuracy.
5. If and only if the final campaign succeeds: evidence that honest uncertainty improves
   closed-loop navigation safety/progress over the matched overconfident condition.

The hit/miss posterior and achievable-precision field support Contributions 4–5; they
should not become separate headline contributions.

### Section order

1. **Introduction:** assistive infrastructure localisation needs an uncertainty contract.
2. **Related work:** infrastructure-based AMRs, multi-camera localisation/fusion, robust
   filtering, camera calibration and drift, belief-space navigation.
3. **Observation and belief model:** separate availability, conditional stochastic noise,
   persistent systematic error, timing, and hit/miss update.
4. **What the cameras actually do:** residual audit, geometry amplification, correlated
   bias, and failure of zero-mean assumptions.
5. **Why standard remedies fail:** NIS gate, sharper `R`, self-confirmation, naive fusion.
6. **Commission, bound, and monitor:** gated v3 calibration, sample requirement, stale
   correction, change detector.
7. **Honest belief:** correlation floor, leave-one-out reference, ablations,
   leave-one-capture-out validation.
8. **Navigation consequence:** achievable-precision map, analytic hit/miss result, and
   final matched closed-loop campaign.
9. **Limitations and conclusion:** simulation, one robot, 2-D state, reference
   commissioning, controlled drift, no formal guarantee.

### Minimum figure/table set

1. System/assist-contract diagram: camera network → observation service → belief → planner.
2. Residual decomposition: per-camera bias directions and temporal correlation.
3. Failure-of-remedies figure/table: baseline, NIS gate, sharp `R`, self-reference.
4. Gated calibration and lifecycle figure: who gets corrected, benefit, drift crossover,
   detection threshold.
5. Belief honesty figure: stated sigma vs RMSE, NEES/coverage, and ablations.
6. Achievable precision versus coverage map.
7. Closed-loop trajectories/outcomes if the campaign is completed.

Everything else belongs in supplementary material. In particular, do not spend main-paper
space on YOLO training curves, uncertain-input GP derivations, detector runtime history,
world-design iterations, or all fusion modes.

## Critical research plan

### Phase 0 — freeze the story and repository (1–2 focused days)

1. Declare this audit or its condensed successor the story source of record.
2. Update `research_story/registry.yaml` to reflect August evidence and mark retired
   headlines explicitly.
3. Create a new ICRA paper directory/repository; do not repurpose `thesis-report`.
4. Fix the two detector-runtime contract tests and obtain a fully green suite.
5. Freeze the exact world, detector, camera manager, planner, task definitions,
   calibration v2/v3 artifacts, seeds, and metrics code by hash.

### Phase 1 — cheap discriminating check before machine time

Run the route-level prediction already motivated by the achievable-precision map. Choose
routes where camera C versus another camera changes the predicted precision by enough to
affect a boundary or handover. The output is a pre-registered set of tasks, not a tuned
result. If no route differs materially, ship the localisation/belief-honesty paper and do
not manufacture a navigation story.

### Phase 2 — the one required campaign

Use `experiments/closed_loop_calibration` as the base design:

- two matched arms: v2 deployed calibration versus gated v3 calibration;
- same C2 planner, three tasks, five seeds: 30 runs;
- differ in exactly one checked config key;
- primary outcomes: GT no-go breaches, contacts, clean goals, final distance, NEES/NIS,
  covariance exposure, accepted/rejected corrections, and correction age;
- analyse per matched task/seed, not per frame;
- report a null honestly if a 7 cm improvement never changes an outcome.

However, calibration alone is not identical to the full honest-belief method. Before the
run, decide whether the paper's causal comparison is:

- **calibration consequence:** v2 versus v3, validating Contribution 3; or
- **belief-honesty consequence:** conventional filter versus correlation-floor filter
  under the same calibration, validating Contribution 4.

The strongest paper ultimately needs the second comparison. If machine time permits only
one 30-run campaign, prefer the arm that isolates the proposed belief policy rather than
bundling calibration, floor, cross-check, and planning changes. Calibration already has
strong offline evidence; belief honesty is the paper's headline mechanism.

### Phase 3 — analysis and stop rule

Proceed with a navigation headline only if the proposed arm improves belief calibration
and produces a matched improvement in at least one meaningful navigation outcome without
global covariance inflation or a condition-specific route cost. A pure NEES/coverage win
is still a valid localisation paper. A route difference without calibrated belief is not
acceptable evidence.

### Phase 4 — optional external-validity work

Only after the core paper tables are frozen:

- repeat on a second capture/session or route family;
- add the Toro baseline on identical detections;
- add Meerhoven as a qualitative or small quantitative stress test;
- run a drift ramp to measure detection latency in time rather than only drift threshold.

## Immediate next decisions

1. Make **belief honesty under correlated per-camera bias** the ICRA headline.
2. Treat GP/geometry, factorisation, mixture, and fusion as baselines or supporting
   analyses, not parallel contributions.
3. Freeze Paper 1 and the historical report; start a clean ICRA manuscript.
4. Pre-register one causal closed-loop comparison before starting further exploratory
   experiments.
5. Stop building new worlds, detector successors, GP variants, or fusion rules until that
   campaign and its paper figures are complete.

