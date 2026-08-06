# WS06 — campaign setup, reproducibility and readiness

## Objective

Build one fail-closed release gate around the frozen WS05 protocol: deterministic matrix,
counterbalanced schedule, artifact/environment lock, source/install parity, static checks,
setup plots and unit-tested runtime readiness. This workstream must expose blockers, not make
scientific choices or run the campaign.

## Start gate

Do not begin implementation until WS05's calibration-identifiability stage has passed and
WS05 then supplies approved arms, primary/secondary outcomes, tasks, seeds, invalid-run
policy and expected matrix. E6 means v2/v3/v4 alone cannot satisfy this gate. Until then,
this handoff is planning context only.

## Ownership

Writable:

- `experiments/closed_loop_calibration/readiness/`
- `tests/experiments/test_closed_loop_calibration_readiness.py`
- `tests/experiments/test_closed_loop_calibration_schedule.py`
- optional `scripts/research/capture_environment.py`
- optional `tests/research/test_capture_environment.py`

Generated output:

- ignored `logs/studies/closed_loop_calibration/preflight/<preflight_id>/`

Read-only:

- frozen WS05 protocol, configs, generator and analyzer
- shared campaign runner/base config
- world, detector, calibration, GP/field and runtime code
- existing multicamera preflight/readiness tools and tests
- registry/status and all research science documents

Do not modify runtime/planner/perception/reliability/world/artifacts, the chosen protocol, or
the current WS04 dirty paths. Report discovered defects as separate blockers.

## Existing tools to compose, not duplicate

- Host/static gate: `experiments/multicamera_commissioning_bigwarehouse/tools/experiment_preflight.py`
- Live barrier: `experiments/multicamera_commissioning_bigwarehouse/tools/runtime_readiness.py`
- Detector live identity: `src/perception/perception/core/four_camera_runtime_contract.py`
- Evaluation firewall: `src/reliability/reliability/firewall.py`
- Route/collision tests: `tests/planning/test_lane_graph_routes.py`
- Config parity: `tests/experiments/test_campaign_config_parity.py`
- Gazebo contract: `tests/sim/test_gazebo_version_contract.py`

The commissioning preflight cannot be used unchanged; its default detector differs from the
current four-camera campaign detector.

## Known fail-closed conditions

- A dirty confirmatory worktree, including the current WS04 changes.
- Generated configs not matching the approved seeds/matrix or declared arm differences.
- Current world/artifact compatibility not earned. The world changed after the July GP
  fields and their manifests lack a world hash.
- Missing source-versus-`install/` parity.
- Missing/invalid detector, calibration, GP/field, config or environment hashes.
- Stale ROS/Gazebo/campaign processes, insufficient disk/RAM, active swap or unavailable
  required GPU/driver.
- Any operational/evaluation firewall violation.
- Ambiguous existing partial evidence or overwrite attempt.

## Required outputs

1. Deterministic, immutable and counterbalanced run schedule with stable row IDs and no
   all-arm-A-then-arm-B ordering.
2. Artifact lock covering world, model, calibrations, fields, configs, source and environment.
3. Environment capture: OS/kernel, Python/packages, ROS/Gazebo, CUDA/driver/GPU, colcon,
   Git state, CPU/RAM, locale and relevant ROS/Gazebo variables.
4. Static preflight JSON with atomic no-overwrite semantics and all failures reported.
5. Source-versus-installed parity check because campaigns run from `install/`.
6. Unit-tested runtime barrier for retained detector identity, advancing clock/odom/noisy
   odom, GT heartbeat without values, four fresh cameras, planner belief, commands/noise,
   finite timing and positive real-time factor.
7. Setup plots: world/routes/no-go/cameras; field/world compatibility; declared calibration
   delta; schedule counterbalancing; noise/seed contract.

These plots are diagnostics, never paper figures.

## Test requirements

- Exact approved matrix and one-to-one matched pairs.
- Deterministic generation and declared-only arm differences.
- Counterbalanced order and duplicate/missing-row rejection.
- Artifact mutation changes the lock; missing artifacts fail.
- Dirty confirmatory worktree fails.
- World mismatch fails without a versioned compatibility record.
- Firewall, unsafe route and source/install mismatch fail.
- Reports, ledgers and outputs cannot be overwritten.
- Runtime readiness discards pre-contract samples and rejects identity drift.
- Unit suite does not launch ROS or Gazebo.

## Acceptance criteria

- All unit/static checks pass, while real unresolved host/scientific issues remain visible as
  failures.
- Registry and hygiene checks still pass.
- One immutable lock binds environment, artifacts, configs and source to every row.
- Runtime readiness is implemented and tested but not executed.
- No campaign logs, runtime pilots or scientific artifact changes are created.

## Paste-ready prompt

```text
Implement only campaign-readiness tooling in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Do not start until the integration-approved WS05 protocol identifies exact arms, tasks,
seeds, outcomes and invalid-run policy. Do not assume v2-v3 if WS05 selected another causal
contrast.

You may edit only:
- experiments/closed_loop_calibration/readiness/
- tests/experiments/test_closed_loop_calibration_readiness.py
- tests/experiments/test_closed_loop_calibration_schedule.py
- optionally scripts/research/capture_environment.py and its test

Treat the frozen protocol/configs, campaign runner, runtime code, world, detector,
calibrations and GP/fields as read-only. Do not edit registry/status. Do not touch the
integrated pixel-ground supporting study or the separately reviewed malformed-box runtime fix.

Compose existing multicamera experiment_preflight.py, runtime_readiness.py, detector runtime
contract, leakage firewall, route/collision checks, config parity and Gazebo-version checks
into one fail-closed package. Produce a deterministic counterbalanced schedule, immutable
artifact/environment lock, atomic static report, source/install parity gate, and static setup
plots for routes, field/world compatibility, calibration delta, schedule and noise. Generated
outputs go only under ignored logs/studies/closed_loop_calibration/preflight/<id>/.

Known blockers must fail rather than be waived: dirty worktree; generated-matrix mismatch;
world changed after GP fields whose manifests lack world hash; missing install parity; stale
processes/resources/GPU failures; firewall leakage; artifact/schema/hash mismatch; ambiguous
partial evidence. Implement and unit-test live readiness, but do not run it. Do not launch
ROS, Gazebo, a pilot or the campaign.
```
