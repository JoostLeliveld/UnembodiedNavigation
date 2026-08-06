# WS07 — reliability-source benchmark

## Objective

Design and later implement the thesis chapter comparing operational sources for future
usable-observation probability: constant, distance, FOV/range, operational depth/raycast,
GP, hybrid and gated DL. Use one target, legal future-pose inputs, identical splits and
budgets, and offline promotion gates before any navigation campaign.

## Start gate

Design may start after WS01/WS02. Implementation needs the WS04 measurement-interface
decision. Gazebo execution waits until the current correlated-error paper package is closed.

## Ownership

Writable:

- `research/papers/reliability_source_comparison.md`
- `experiments/usable_observation/`
- new benchmark-specific shared source/tests only after proposing their namespace

Read-only:

- registry/status and WS01-WS03 research documents
- frozen planner, `R_cond`, bias/correlation floor and expected hit/miss posterior
- camera-management experiment
- existing single-camera usable-observation scripts/evidence and cold archive

Do not refit the observation representation per source, change camera policies, launch
closed-loop runs before offline gates, or present perfect simulator geometry as an
operational sensor.

## Evidence inherited, not sufficient

- Existing single-camera foundation reports distance `0.0546`, FOV/range `0.0557`, GP
  `0.0584` Brier on its hard route. Preserve this GP null if recovered/reproduced.
- Existing four-camera spawn-grid GP validation also loses to prior-only fields, but events
  come effectively from one grid campaign and are not route-independent replication.
- The old dataset is in-FOV-only and has `p_qual` about `0.997`, so it mostly estimates
  detection probability.
- README-referenced evidence directories are currently absent. Recover via a verified cold
  archive or rebuild with frozen splits; prose numbers alone are not evidence.
- Existing “depth” prototypes mix complete SDF, perfect/degraded Gazebo depth, monocular
  depth and sensed height maps. None is a frozen primary operational arm yet.

## Common target and representation

- Primary target: `p_use,c(s)`, usable-observation probability for camera `c` at candidate
  pose `s`; decide explicitly whether `s=(x,y)` or `(x,y,heading)`.
- Preserve availability and conditional accuracy separately.
- All source arms share frozen `R_cond`, persistent bias/correlation floor, freshness rules
  and the expected hit/miss posterior:

  `E[P+] = p_use P_hit + (1-p_use) P-`.

- Operational interfaces cannot contain evaluation truth, current detector result,
  instantaneous confidence or future rendered imagery.

## Required arms

1. Fixed conservative constant; optionally train-prevalence constant, labelled as requiring
   data.
2. Distance-only.
3. Analytic cold-start FOV/range; keep label-fitted logistic geometry separately labelled.
4. Operational depth/raycast with exact provenance and missing-cell fallback.
5. GP with epistemic support and prior fallback.
6. Hybrid, preregistered preferably as geometric-prior logit plus GP residual with fallback.
7. DL challenger only after legal inputs, calibration and offline route discrimination.
8. Complete SDF/CAD raycast only as a clearly labelled map/oracle upper bound unless a full
   deployment map is an accepted assumption.

## Gate sequence

| Gate | Required result |
|---|---|
| B0 legality/data | in-FOV and out-of-FOV opportunities, all four cameras, no truth leakage |
| B1 nominal prediction | held-out Brier primary; calibration and false-high-trust diagnostics |
| B2 failure audit | error by FOV, occlusion, range, support, camera and layout |
| B3 commissioning curve | performance versus unique sites, collection time and runtime |
| B4 transfer/staleness | paired nominal/changed layout; stale/rescanned/frozen/updated states |
| B5 offline planning | prespecified route ranking and expected terminal-belief difference |
| B6 closed loop | only B5 survivors plus simple references; complete paired result or null |

WS08 camera management is a separate B7 and uses frozen winning fields.

## Frozen comparison contract

- Same opportunity labels, detector/threshold, candidate-pose set, train/calibration/test
  splits, commissioning budget, probability-calibration allowance and paired seeds/sites.
- Report cold-start and commissioned variants honestly.
- Primary nominal run adds no synthetic noise; use WS02's accepted OFAT sensitivities only.
- Do not run the Cartesian product.
- Include grouped route, spatial block, changed-layout and optional second-world holdouts.
- Before planning, quantify route length, `p_use` exposure, terminal belief, clearance,
  cameras and handovers. Include a uniformly-good negative control.

## Outputs

- F07: held-out prediction/calibration and error strata.
- F08: offline route discrimination, including null routes.
- F09: commissioning/runtime/transfer/adaptation decision matrix.
- F10: one comparable failure and fallback per method.
- Permanent results/provenance/null summaries; raw generated outputs remain ignored.

## Decisions requiring explicit approval

- Whether `p_use` includes heading.
- Primary operational depth provenance.
- Exact hybrid equation and fallback.
- Legal DL input that is scientifically distinct from a spatial MLP/GP.
- Local/global layout-change definitions and second-world requirement.
- Minimum meaningful Brier, route-belief and navigation effects.

## Acceptance criteria

- Every arm predicts the same target and uses legal operational information.
- Distance-only, analytic FOV/range and complete-map upper bound are distinct.
- Sample budgets count unique sites, not repeated frames.
- No method reaches Gazebo before held-out calibration and route discrimination.
- Each promoted method has a documented failure and fallback.
- Negative GP evidence is preserved.

## Paste-ready prompt

```text
Work on the reliability-source benchmark design in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Read completed WS01/WS02/WS04 handbacks, research/registry.yaml,
research/papers/reliability_source_comparison.md,
experiments/usable_observation/README.md, docs/usable_observation/, existing source scripts,
locked/archived RESULTS.md files and planner hit/miss evidence.

You may initially edit only:
- research/papers/reliability_source_comparison.md
- experiments/usable_observation/README.md
- new design/protocol files inside experiments/usable_observation/

Do not edit registry/status, planner, camera management or current-paper files. Do not run
Gazebo. Do not implement source models until the design and data contracts are reviewed.

Define one future-pose target p_use,c(s) and a legal operational feature contract. Keep
frozen R_cond, persistent bias/correlation floor and expected hit/miss posterior across all
arms. Separate fixed/train-prevalence constants, distance-only, analytic and fitted
FOV/range, operational depth/raycast, GP, preregistered hybrid and gated DL. Treat complete
SDF raycast as a map/oracle upper bound unless a full deployment map is declared. Define
missing-depth and epistemic fallbacks.

Build B0 legality/data, B1 held-out prediction, B2 failure, B3 commissioning, B4 layout
transfer, B5 offline route and B6 closed-loop gates. Use grouped route/spatial/layout splits,
all four cameras, both in-FOV and out-of-FOV opportunities, identical unique-site budgets and
no nominal synthetic noise. Preserve existing GP nulls and flag absent README-referenced
evidence directories for verified recovery/rebuild. Return exact subquestions, requirements,
assumptions, metrics, promotion gates, route archetypes, noise sensitivities, F07-F10 outputs
and unresolved approval decisions.
```
