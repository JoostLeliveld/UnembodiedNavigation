# closed_loop_calibration — does the bias fix change closed-loop safety?

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-CL-CAL
status: BLOCKED
claim_ids:
- C3
- C4
assumption_ids:
- A01
- A03
- A08
- A10
- A11
- A12
- A13
- A14
- A15
- A19
- A20
reviewer_question_ids:
- RQ07
- RQ11
- RQ13
- RQ15
figure_ids:
- F06
dependencies:
- ASSET-RUNTIME
- ASSET-PLANNER
- ASSET-CAMPAIGN
operational_inputs:
- camera_measurements
- odometry_belief
- frozen_calibration
- camera_residual_floor
evaluation_only_inputs:
- ground_truth_pose
- contact_events
- nees
primary_metric: Held-out identifiability of silhouette geometry versus camera calibration
  before any closed-loop endpoint is selected
promotion_gate: First identify the calibration components under yaw-diverse route-disjoint
  evidence; then freeze one causal arm pair and analysis, pass readiness, and complete
  the matched campaign or retain a documented null.
evidence_paths:
- experiments/closed_loop_calibration/README.md
archive_rule: Preserve every completed run ledger summary provenance and null result.
next_action: Redesign WS05 around camera region and yaw identifiability using E6;
  no v2-v3-v4 arm or closed-loop endpoint may be selected before that gate passes.
```

<!-- RESEARCH-METADATA:END -->


**Question.** Every observation-model result in this workstream is **offline**. C1 — the
safe operating envelope — is a closed-loop claim. Does the gated 2-DOF per-camera bias
correction change breach rate, contacts, goal completion, or belief calibration when the
robot is actually driving on it?

**Claims served.** C3 and C4; this is the active gate in
[`research/papers/correlated_error_icra.md`](../../research/papers/correlated_error_icra.md).

**Register status.** Runway item **#2** — the last genuinely open item.

## Design: calibration is the independent variable

Two arms, matched seeds, one changed key:

| | arm `clv2` | arm `clv3` |
|---|---|---|
| `manager_projection_calibration` | `projection_calibration_v2` (deployed today) | `projection_calibration_v3` (2-DOF, gated) |
| everything else | identical | identical |

v2 is an **arm**, not a baseline that gets reopened — so no v2-locked comparison is
invalidated by running this, and the comparison is matched-seed rather than
across-campaign.

Both configs are **generated** from `warehouse_full_4cam_missions.yaml` by
`make_configs.py`, which fails the build unless the two differ in exactly one key. That
turns "matched conditions" from a claim in the write-up into a checked property.

```bash
python3 experiments/closed_loop_calibration/make_configs.py --seeds 0 1 2 3 4
```

Design: 3 tasks (`mc_central_ns`, `mc_south_we`, `mc_north_we`) × condition **C2** (the
deployed visibility-aware planner) × 5 seeds × 2 arms = **30 runs**. The planner is held
fixed because the calibration is the variable; C1 remains available as a robustness check
but doubles machine time for a question nobody asked.

Seeds vary the actuation/encoder noise realisation, which is the stochastic input the
campaign already models (`use_command_noise` / `use_encoder_noise`).

## What v3 actually changes

Along-bearing constants are **bit-identical to v2** (the frozen deployed term). Only the
cross-bearing constants differ, and only on the two cameras the commissioning gate said
`CALIBRATE`:

| camera | gate | cross intercept (m) | cross slope (per m) |
|---|---|---|---|
| A | RAW | 0.0 | 0.0 |
| B | RAW | 0.0 | 0.0 |
| C | CALIBRATE | −0.1292 | +0.00444 |
| D | CALIBRATE | +0.0156 | +0.00167 |

So the expected closed-loop effect is confined to segments observed by C and D, and its
size is the measured bias removal: camera C 77 → 4 mm, D 33 → 2 mm.

**This bounds the claim honestly**: a ~7 cm belief improvement on C-observed segments only
changes an outcome where 7 cm decides a breach. The campaign either finds that regime or
it does not, and "no closed-loop difference" is a publishable result — it would say the
bias fix is a belief-calibration result, not a safety result, and C1 would have to rest on
the coverage/health arms instead.

## Run

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/_clv2.yaml \
  --log-root logs/visibility_comparison/clv2
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/_clv3.yaml \
  --log-root logs/visibility_comparison/clv3
python3 experiments/closed_loop_calibration/analyse.py
```

Real Gazebo runs. Gotchas that bite here: the campaign loads from `install/`, not `src/`
(verified 2026-08-04 that the build tree serves the 2-DOF projection); `setsid` detaches
the runner so the shell returns immediately; and stale `drive_study_route` processes
publish `/cmd_vel` — `pgrep` before launching.

## Primary outcomes

Scored per matched (task, seed) pair, GT for evaluation only:

- GT no-go breaches (`inside_no_go`) and physics contacts (`collision_any`)
- clean goal completion (`goal_region_success`, `completed`), final goal distance
- belief calibration: NEES / NIS at detection instants, `tr(P)` exposure
- accepted vs rejected corrections, correction-age

## Reuse map

| need | reused from |
|---|---|
| campaign runner | `scripts/visibility_comparison/run_visibility_campaign.py` |
| base config | `scripts/visibility_comparison/warehouse_full_4cam_missions.yaml` |
| run/detection loading | `campaign_metrics.load_run` / `load_detections` (never raw `state_x`/`truth_x`) |
| scoring | `scripts/shared/metrics.py` |
| calibration artifacts | `projection_calibration_v2` / `_v3` (loaded, never refitted) |
