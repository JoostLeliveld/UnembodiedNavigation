# WS05 — correlated-error closed-loop protocol

## Objective

Define the one scientifically valid closed-loop experiment needed by the current paper.
Resolve the v2/v3/v4 contradiction, state exactly which mechanism the arms isolate, freeze
outcomes and sampling units, and only then prepare the experiment module. Do not run it.

## Start gate

Begin after WS01-WS04 hand back. This workstream must consume their accepted claim boundary,
assumptions, evidence maturity and projection-path decision. If those are unresolved, remain
in protocol-design mode.

## Ownership

Writable after the design decision is approved:

- `research/papers/correlated_error_icra.md`
- `experiments/closed_loop_calibration/`
- generated closed-loop configs matching the chosen arms under
  `scripts/visibility_comparison/`

Read-only:

- `scripts/visibility_comparison/warehouse_full_4cam_missions.yaml`
- `scripts/visibility_comparison/run_visibility_campaign.py`
- planner, perception, reliability, world and camera-manager runtime
- v2/v3/v4 calibration artifacts and all source evidence
- registry/status and WS01-WS04-owned research documents

Shared/runtime defects are reported to integration; do not fix them here.

## Immediate scientific conflict

The current contract compares:

- v2: 5 cm contact plane plus along-bearing correction;
- v3: v2 plus gated cross-bearing correction.

Newer held-out evidence in
`logs/studies/external_camera_bias_model/exp3_projection_pipeline_minimal/RESULTS.md`
concludes the 5 cm plane costs about 9 cm, the along-bearing fit does not generalize, and v4
(floor plane plus gated cross-bearing constants) supersedes v2/v3. Current hashes:

- v2 `6b8b15890f694fb7c0055b06739ab80a37c9078f12cc22b17d5ee41bc352c583`
- v3 `c3fab37d01790ff7751e2b1938c4d5bf822cf6891479ac6385dd6d514897bd0d`
- v4 `31d60b8523c5e6dbe7987d74abffa9bbd5f89d8329a363fc1b4d135f23559d99`

Evaluate explicitly:

1. v2 versus v4: deployed incumbent versus corrected minimal pipeline;
2. raw/floor projection versus v4: isolates gated cross-bearing correction;
3. v2 versus v3: narrow historical mechanism study, while admitting both are superseded;
4. a small offline three-arm comparison followed by one prespecified closed-loop pair.

Do not choose based on which result is likely to look best.

## Other blockers

- `_clv2.yaml` and `_clv3.yaml` currently contain only seed 0; they are six runs, not 30.
- The current world hash postdates the July GP fields, whose manifests contain no world
  hash. Shared exposure may preserve the contrast but not external interpretation; WS06 must
  fail closed until compatibility is resolved.
- The existing analyzer omits NEES, NIS, correction acceptance/age, intervals, figures and a
  complete-pair gate.
- Fifteen matched pairs permit a mechanism/null conclusion, not a strong safety-superiority
  claim.
- A calibration-only contrast cannot establish that correlation flooring, LOO validation or
  the entire belief-honesty package improves navigation.
- Runtime NIS 9.21 and offline 5.991 are different policies.

## Required protocol decisions

- One primary endpoint: clean goal, belief calibration, or p95 localization error. Recommend
  one and explain the consequence; obtain integration/supervisor approval.
- Binary diagnostics: breach and contact, using exact paired analysis.
- Continuous/mechanism diagnostics: run-level median/p95 belief error, NEES/coverage,
  NIS/acceptance/correction age, calibrated-camera exposure, path length and elapsed time.
- Independent unit: matched `(task, seed)` run. Frames/detections are within-run samples.
- Paired bootstrap by run for continuous intervals; no frame-level significance.
- Complete-matrix and invalid-run/rerun rules fixed before data.
- Null interpretation and exact claim boundary written before launch.
- Seed count justified against the intended claim strength.

## Deliverables

1. A decision memo comparing the four arm designs above and selecting one causal contrast.
2. Revised paper scope and experiment README with allowed/non-allowed claims.
3. Frozen protocol: arms, tasks, seeds, controls, endpoints, analysis and null handling.
4. Deterministic config generation that proves the chosen arms differ only in declared keys.
5. Analysis specification and tests; implementation may follow only after protocol approval.
6. A handoff contract for WS06 listing exact artifacts, matrix and readiness thresholds.

## Acceptance criteria

- The selected arms are consistent with the newest held-out calibration evidence.
- The causal mechanism is named narrowly and all other runtime inputs are frozen.
- Sample size matches claim strength.
- The planned analysis can reject incomplete/unmatched runs and produce all preregistered
  outcomes and a full null.
- No campaign, runtime pilot or parameter tuning occurs.

## Paste-ready prompt

```text
Design—but do not run—the correlated-error closed-loop protocol in:
/home/joostleliveld/Thesis/UnembodiedNavigation

First read the completed handoffs/results from research/workstreams/WS01 through WS04,
research/papers/correlated_error_icra.md,
experiments/closed_loop_calibration/{README.md,make_configs.py,analyse.py},
scripts/visibility_comparison/warehouse_full_4cam_missions.yaml, and the held-out calibration
evidence, especially
logs/studies/external_camera_bias_model/exp3_projection_pipeline_minimal/RESULTS.md and
projection_calibration_v2/v3/v4.

Critical conflict: the active v2-vs-v3 design uses a 5 cm contact plane and along-bearing
fit, while newer held-out evidence says both are superseded and v4 should use the floor plane
plus gated cross-bearing constants. Compare v2-v4, raw/floor-v4, narrow historical v2-v3,
and offline-three-arm-then-one-pair designs. Select the scientifically cleanest causal
contrast, not the most favorable expected outcome.

Also account for: current generated configs have seed 0 only; 15 pairs support a mechanism
or null report rather than broad safety superiority; analyse.py lacks completeness, NEES,
NIS, acceptance/age, intervals and figures; runtime NIS 9.21 differs from offline 5.991; and
a calibration-only contrast does not test the whole correlation-floor method.

During stage 1, make no edits: return a decision memo, recommended primary endpoint, claim
boundary, sample-size interpretation, frozen controls and WS06 readiness contract. After
integration approval, edit only research/papers/correlated_error_icra.md,
experiments/closed_loop_calibration/, and generated configs for the approved arms. Never
edit registry/status or shared runtime code. Do not launch Gazebo or any campaign.
```
