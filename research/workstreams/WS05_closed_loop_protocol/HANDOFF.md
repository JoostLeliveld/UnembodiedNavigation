# WS05 — calibration identifiability before closed-loop protocol

## Objective

Resolve whether the residual terms currently called *camera calibration* are identifiable
separately from robot-silhouette geometry, route region and robot yaw. Only after that gate
passes may this workstream select a causal closed-loop arm pair, endpoint and matrix.

This replaces the earlier task of choosing directly among v2, v3 and v4. Those artifacts are
preserved historical implementations; none is currently an approved confirmatory arm.

## Why the start gate changed

Tracked E6 evidence is under:

```text
paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/
```

On the 1,195 external-log rows with recorded yaw, the CAD silhouette model reduced mean
projection error from 143.9 mm to 34.7 mm and mean along-bearing error from -134.2 mm to
+2.5 mm. The cross-bearing gate that fired for cameras C and D no longer fired after the
model. Cameras A and B then crossed the heuristic gate, but camera/region/heading are
confounded.

Allowed conclusion: the existing C/D constants are not identifiable as camera calibration
on these logs once silhouette geometry is modelled.

Forbidden conclusion: no per-camera lateral correction is ever warranted.

The current captures cannot settle the issue: two are straight routes at fixed, different
yaws; the third repeats one route and lacks recorded yaw. A held-out capture is not an
independent calibration test when route and yaw move together.

## Two-stage workstream

### Stage 1 — identifiability protocol

Design, but do not run, one bounded offline capture and analysis that independently varies:

- camera identity;
- spatial region/site;
- robot yaw, including 45-degree diagonals;
- range and image position within supported bounds.

The design must compare one fixed observation definition at a time. Do not mix the deployed
bottom-centre/contact-plane path with the candidate box-centre/0.085 m path or its covariance.
The candidate implementation remains experiment-local in
`experiments/pixel_ground_path/box_projection.py`.

The minimum decomposition is:

```text
observed projection residual
  = object/silhouette term
  + camera-specific repeatable term
  + camera x region interaction
  + yaw/appearance interaction
  + remaining noise
```

The exact statistical form may be a prespecified mixed-effects model or an equivalent
grouped held-out comparison. It must not estimate and evaluate a camera constant on the same
camera-region-yaw cells.

Required held-out tests:

1. leave-yaw-band-out;
2. leave-spatial-block-out;
3. leave-route-out if routes are used to collect the grid;
4. camera-specific transfer across at least two regions and multiple yaws;
5. an explicit no-correction null and the CAD-only model.

Independent units are unique commanded sites or capture blocks. Frames/detections repeated
at one site are within-unit observations, not replicates. Sample budgets count unique sites
(A18), and grouping is chosen for the estimand (A19).

### Stage 1 decision gate

A per-camera correction becomes an eligible causal arm only if all are true:

- its residual remains directionally stable after the object model;
- it transfers to held-out yaw and spatial blocks;
- its held-out effect exceeds a preregistered practical margin;
- it does not materially harm cameras below the correction gate;
- its statistic, inversion plane and calibration artifact are bound as one measurement
  definition (A20);
- the result summary, split manifest and provenance are tracked.

If the correction fails, preserve the null and redefine Paper A around structured correlated
projection error and belief honesty without a camera-calibration-benefit claim.

### Stage 2 — closed-loop protocol

Open only after Stage 1 approves one identified contrast. Then select exactly one arm pair
that differs in the identified mechanism and freeze:

- one primary endpoint;
- three tasks and seeds 0-4, giving 15 matched task-seed pairs;
- binary breach/contact diagnostics;
- run-level localization error, NEES/coverage, NIS/acceptance/correction age, relevant-camera
  exposure, path length and elapsed time;
- invalid-run and rerun rules;
- exact paired analysis for binary outcomes and task-stratified paired bootstrap intervals
  for continuous run summaries;
- a full null interpretation.

Fifteen pairs can support a scoped mechanism result or documented null, not broad safety
superiority. A calibration-only contrast cannot establish benefit for the complete
correlation-floor plus leave-one-out package.

## Controls and blockers that remain

- Runtime detector threshold is 0.05 while the offline quality gate uses 0.25.
- Runtime NIS is 9.21 while historical offline analysis uses 5.991.
- The historical miss endpoint differs between 40 and 120 px, although the explicit
  hit/miss mixture needs no Gaussian miss endpoint.
- Existing generated closed-loop configs contain seed 0 only.
- The analyzer does not enforce a complete 15-pair matrix and lacks several preregistered
  outputs.
- The current world postdates July GP fields whose manifests contain no world hash.
- No confirmatory run may use a dirty worktree or an unbound artifact.

## Ownership

Stage 1 writable after the integration design review:

- `experiments/closed_loop_calibration/identifiability/`
- `tests/experiments/test_calibration_identifiability.py`
- a protocol section in `research/papers/correlated_error_icra.md`

Stage 2 writable only after Stage 1 passes:

- `experiments/closed_loop_calibration/`
- generated closed-loop configs under `scripts/visibility_comparison/`
- `research/papers/correlated_error_icra.md`

Read-only throughout:

- runtime planner, perception and reliability packages;
- world, detector and calibration artifacts;
- existing evidence and raw captures;
- `research/registry.yaml`, `STATUS.md` and numbered research documents.

Do not launch Gazebo, collect data, refit calibration, wire the candidate box-centre path,
or edit shared runtime code in WS05.

## Deliverables

1. A decision memo explaining why the existing captures cannot identify camera calibration.
2. A frozen Stage 1 factorial capture/split plan and minimum practical effect.
3. An analysis specification that separates object, camera, region and yaw terms and fails
   on leakage or incomplete blocks.
4. A deterministic capture manifest/config generator and unit tests, but no capture run.
5. Pass/fail rules mapping Stage 1 outcomes to the allowable Paper A claim.
6. A Stage 2 protocol only if the identifiability gate passes; otherwise a documented null
   and revised paper scope.
7. A precise WS06 readiness contract if and only if Stage 2 opens.

## Acceptance criteria

- Camera, region and yaw are not aliased by design.
- The primary result is evaluated on held-out groups and unique-site units.
- CAD-only and no-correction nulls are mandatory.
- Candidate measurement definitions remain separate and internally coupled.
- E6's primary recorded-yaw stratum and weaker tangent-derived stratum are not pooled as
  equivalent evidence.
- No closed-loop arm is selected from the current v2/v3/v4 results alone.
- Null outcomes narrow the paper rather than disappearing.

## Paste-ready prompt

```text
Work on WS05 in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Start in protocol-design mode. Do not run Gazebo or collect data. Read this handoff, the
tracked E6 evidence under
paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/, the pixel-ground README,
research/{02_claims,03_assumptions,07_validation_matrix,08_figures}.md, and
research/papers/correlated_error_icra.md.

The previous v2/v3/v4 arm-selection task is obsolete. E6 shows that the C/D correction signal
disappears after modelling the robot silhouette on external logs, while A/B become confounded
outliers. Design a bounded yaw-diverse, route/region-disjoint identifiability study that can
separate object/silhouette error, camera-specific residual, camera-region interaction and
yaw/appearance interaction. Use unique sites/capture blocks as independent units; repeated
frames are not replicates. Include CAD-only and no-correction nulls and held-out yaw, spatial
block and route tests. Do not mix pixel statistic/plane/calibration definitions.

First return a decision memo, exact factorial capture and split design, minimum practical
effect, analysis model, leakage/completeness checks and pass/fail mapping to the paper claim.
After integration approval, implement only deterministic protocol/config/analysis skeletons
under experiments/closed_loop_calibration/identifiability/ and their tests. Do not edit the
registry, numbered research documents, runtime code, worlds or calibration artifacts. Do not
select or generate a closed-loop arm pair unless the identifiability gate has actually passed.
```
