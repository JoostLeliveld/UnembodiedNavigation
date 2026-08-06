# WS08 — camera selection and conservative fusion

## Objective

After WS07 freezes observation-quality fields, compare how those fields should drive online
camera choice and localization trust. Keep estimator quality fixed so policy effects are not
confounded with refitting.

## Start gate

Do not start implementation until WS07 has frozen the winning/reference fields, their
conditional covariance, bias/correlation floor, support and health semantics.

## Ownership

Writable:

- `experiments/multicamera_fusion_extension/`
- camera-management-specific source/tests only after integration approves exact paths

Read-only:

- all WS07 field artifacts and prediction models
- planner, projection, detector and calibration contracts
- registry/status and paper scopes

Do not refit `p_use`, change source features, create a new observation representation, or
use ground truth operationally.

## Required separation

There are two decisions:

1. **Selection:** which camera(s) should be asked/trusted now?
2. **Measurement update:** how are selected measurements weighted/fused into localization?

High GP value alone is not localization precision. Trust must account for conditional
covariance, persistent bias/correlation floor, epistemic support, freshness/health and
expected update value. Distance is a baseline policy, not automatically a covariance.

## Policy arms

- nearest camera;
- maximum predicted `p_use`;
- maximum achievable precision / minimum expected posterior criterion;
- hysteretic version of the winning selector;
- conservative multi-camera fusion with shared/correlated-error control;
- optional all-eligible naive independent fusion as an explicit failure baseline.

Keep association, detector, fields, calibration, update gates, controller, routes and seeds
identical. Do not train a different field per policy.

## Questions

- When does nearest differ from maximum availability or achievable precision?
- Does hysteresis reduce handover chatter without retaining a stale/poor camera too long?
- When does fusion improve accuracy, and when does correlated evidence make it overconfident?
- How do dropout, latency, stale health and overlap change selection?
- Which policy fails safely in camera-poor unavoidable routes?

## Evaluation sequence

1. Deterministic offline decision maps and disagreement regions.
2. Recorded-data replay for covariance calibration, handovers, rejection and chatter.
3. OFAT health/latency/dropout sensitivities with paired events.
4. Offline route/belief consequence.
5. Closed-loop campaign only for policies surviving all earlier gates.

Primary localization metrics: run/event-level error, NEES/coverage and false-high-trust
events. Management metrics: switch count, dwell time, stale selection, handover failures,
eligible-camera utilization and runtime. Navigation remains a downstream paired outcome.

## Failure cases that must remain

- nearest camera is occluded;
- maximum `p_use` chooses a camera with poor conditional precision;
- precision selector chooses an unsupported/stale estimate;
- hysteresis clings to a degraded camera;
- naive fusion counts correlated measurements as independent;
- conservative fusion rejects useful diversity or becomes too diffuse.

## Acceptance criteria

- Fields are byte-identical across policies.
- Selection and measurement fusion are reported separately.
- Evaluation truth is used only for scoring.
- At least one failure and fallback exists per promoted policy.
- No closed-loop time is allocated before replay and route gates.

## Paste-ready prompt

```text
Design the downstream camera-management study in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Start only after WS07 has frozen p_use fields, R_cond, persistent bias/correlation floor,
epistemic support and health/freshness semantics. Read those artifacts and
experiments/multicamera_fusion_extension/, but do not refit or alter any source estimator.

Initially edit only design/protocol files inside experiments/multicamera_fusion_extension/.
Do not edit registry/status, current paper, projection, detector, planner or WS07 artifacts.
Do not run Gazebo during design.

Separate camera selection from localization measurement fusion. Compare nearest,
maximum-p_use, maximum-achievable-precision/expected-posterior, hysteretic selection,
conservative fusion and naive-independent-fusion failure baseline. High GP value is not by
itself precision: trust must include R_cond, persistent bias/correlation floor, support,
freshness and health. Freeze fields, detector, calibration, routes, seeds, association and
controller across policies.

Specify offline decision maps, replay, OFAT health/latency/dropout, route discrimination and
only then closed-loop gates. Define localization calibration, handover/chatter, runtime and
navigation metrics, one failure/fallback per policy, and operational/evaluation interface
separation. Return subquestions, assumptions, required files, promotion gates and figures;
do not implement until reviewed.
```
