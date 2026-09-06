# Planner implementation and experiment plan

Scope: complete the IWAI camera-network extension first, then add the smallest commissioning
analysis that explains its navigation behavior. [The paper map](../../../papers/master_thesis/planning/paper_map.md)
assigns the evidence to the 12-page thesis. [ICRA_STATUS.md](../../docs/ICRA_STATUS.md)
is the single current decision/status account. This file specifies interfaces, code and gates.

## 1. Method now implemented

The network is an opt-in adapter inside `UnicyclePlannerBase`; the legacy path stays the
default. `camera_network_artifact_path` replaces the single-camera visibility artifact
for that plan. Both artifacts together are rejected. The adapter changes the covariance
proxy supplied to the existing IWAI objective; it does not replace the objective/controller.

| Component | Code | Contract |
|---|---|---|
| Frozen runtime artifact | `src/planning/planning/core/camera_network.py` | NPZ without pickle; explicit camera IDs; XY grid; separate score and availability fields; full SPD metric R and designed miss endpoint; schema, reference, units and source hashes |
| Numpy / differentiable network query | Same module | Same bilinear fields and five XY sigma points in NumPy/CasADi; outside grid returns zero field values; preserves full ellipse shape |
| Existing EFE integration | `src/planning/planning/core/casadi_efe.py`, `planners/base_planner.py` | Add full precision matrices, transform once into the fixed camera-A cost chart; single goal preference, same EFE terms and weights, same frozen Q |
| Forecast reference | `CameraNetworkModel.forecast_posterior` | Separate q/R model; at most five cameras/32 hit-miss outcomes; Joseph updates; no update on a miss. Compare expected-information approximation without calling either empirical calibration |
| Node / launch | `nodes/unicycle_planner_node.py`; `src/experiments/launch/warehouse_primary_comparison.launch.py`; `core/visibility_launch_common.py` | Explicit artifact argument; disable field in the ordinary local tracker; reject pixel correction/proxy reuse, preselected-route mislabelling and incompatible hit/miss switch |
| Logging / campaign | `experiment_logger.py`; `scripts/visibility_comparison/run_visibility_campaign.py` | Record artifact path and SHA, identify proxy semantics, refuse resumed runs with another artifact. P0/P1/P2 all actually select `visibility_aware_efe` |
| Export | `export_network_planner.py` | Verify frozen source hashes; fit scores on mean-training and covariance-fitting roles only; average headings at each position; preserve misses; reuse separately fitted q and full constant R |
| Integration probe | `planner_probe.py` | Optimize the same short warehouse-lane problem for three frozen artifacts; save inputs, source hashes, controls, model forecasts, figures and solver results |

The score GP reuses the repository's canonical latent RBF fitting code. Its interpolation
uncertainty is **not** used as camera R. R remains residual-calibrated and identical in
P0/P1/P2. The q GP has binary detection/projection labels and is stored separately.

The exported grids are fitted models for this installation, not an asserted optimal
selection. Score GP length 1 m, latent noise variance 0.05 and geometry ridge penalty 1
are fixed implementation-pilot settings. Selection and final evaluation remain separate.
The 0.5 m export grid and 2 m nearest-training support are explicit approximations; grid
interpolation softens that support boundary. Heading is marginalized over commissioned
headings, so heading-sensitive failure is a specific test, not a solved feature.

## 2. Controlled arms

| Arm | Planner field | Current-image estimator | What the comparison isolates |
|---|---|---|---|
| P0 | Per-camera uniform mean detector score | Same fixed estimator in every arm | Whether spatial information is useful |
| P1 | Fitted geometry/view score, using the same fitting positions | Same | Whether learned spatial detail adds beyond geometry |
| P2 | Commissioned XY GP of heading-averaged score, misses zero | Same | Whether the IWAI-style learned field helps network route decisions |
| C0 / C1 reference | Existing shortest-path / legacy constant-R EFE | Same | Practical conventional and current-system reference; separate representation ablations |
| P3, conditional later | Validated q and conditional quality with compatible temporal model | Frozen selected estimator | A distinct stronger prediction model; implement only after the reference agrees sufficiently with actual perception |

P0 uses the **same network representation** as P1/P2. It is not the existing single-camera
`constant_R_efe` arm. Include C0/C1 as contextual baselines, but attribute spatial-field
effects within P0/P1/P2. Add an IWAI-equivalent single-camera run only with its observation
representation and calibration explicitly frozen; do not claim published reproduction
from a new matrix algebra test.

The executed `network_navigation_pilot.yaml` holds the NN, post-NN residual offset,
full constant per-camera R and `joint_network` robust fusion fixed. The versioned
`ReferenceCalibration` wrapper matches the frozen offline camera model exactly.
Robust fused covariance still differs from the future independent precision sum.
Consequently this tests **planner proxy utility**, with no shared calibrated posterior claim.

## 3. Work order and completion gates

### A. Planner setup — implemented and probed

- Validate matrix SPD, full directional information, camera-order invariance and
  single-camera algebra; a forecast with all q=0 must preserve P.
- Match NumPy/CasADi values at interior, boundary and outside queries; finite-difference
  the actual planner control gradient. Keep the legacy golden checks passing.
- Verify node, campaign, launch and logger routing. A declared network artifact must
  reach an actually solved global EFE plan and must not become camera measurement noise.
- Save a real optimizer probe with source hashes. It is neither a physical drive nor a
  statistical localization experiment.

Artifacts: `logs/studies/icra_commissioning_20260905/network_planner/{uniform,geometry,gp}.npz`,
`manifest.json`, and `probe_reviewed/{protocol,results}.json`. The earlier `probe/` is
preserved: it used the inherited width-only 0.275 m collision disc. The reviewed probe
uses the current URDF body's circumscribed radius `hypot(0.400,0.275)` and a 0.55 m lane
clearance. This is a configuration correctness correction, not a sensor-method gain.

### B. Complete-route planning — completed as a software/geometry preflight

`network_route_probe.py` preserves the frozen short probe and resolves the actual launch
parameters. All three 200-step solves select the upper corridor; all hit the 80-iteration
limit. Their goal gaps are 6.7–9.6 cm, and dense 2 cm body-clearance checks pass. All
optimized and seed candidates remain in `network_planner/full_route_v1`. The
`network_navigation_analysis.py preflight` command generates the comparison figure and
candidate audit. These checks alone do not establish local trackability.

1. Run every seed through the actual global objective; record all candidates, selected
   route, optimizer status, component costs, terminal distance and computation time.
2. Check the continuous sampled route against the declared driveable map and true robot
   footprint; optimizer success alone is not route feasibility. Freeze a sufficient horizon
   and identical stopping/terminal selection policy. Audit the fixed camera chart's projective
   denominator and linearized covariance conversion over every reachable candidate.
3. Check that the map can discriminate route candidates under fixed settings; if all fields
   choose the same route, retain that as an ordinary case and use the predeclared occlusion
   family to test the mechanism. Do not choose only favorable final scenarios.

Gate: complete feasible planned routes with traceable cost decomposition. A shorter probe
is not this gate. Absolute objective values across different sensor models are not accuracy.

### C. Match the estimator before claiming calibrated information

Code seams: `reliability/learned_box_correction.py`, `nodes/camera_manager_node.py`,
`planning/core/belief_correction.py`, `experiments/icra_commissioning/replay.py`.

1. **Implemented and tested:** an explicit versioned wrapper for the extra per-camera mean offset and frozen
   constant R; apply the offset once after the existing metric NN. Use the same frozen
   artifact in replay, runtime and the future model. The old NN checkpoint is preserved.
   `export_reference_calibration.py` exports the frozen parameters without fitting;
   `reference_calibration.py` checks NN hash, units, reference and SPD. The actual camera
   manager matches offline outputs on 100 recorded samples. Completed live runs are audited
   against their opportunity contracts and logged pre-fusion means/R as well.
2. Reproduce arrival-time acceptance from `camera_opportunities.jsonl` and
   `correction_assimilations.csv`, including delay, repeated messages, refusals and odometry
   history. Capture-time ideal replay remains a separate diagnostic.
3. Compare current robust aggregation with the existing direct per-camera robot-update mode.
   The current Huber/sandwich `joint_network` covariance is not the inverse sum of independent
   precisions; equal repeated readings do not give the same covariance as independent fusion.
   Select and validate one interface, rather than declaring this mismatch solved by a field.
4. Keep Q, mean, initialization, update cadence and acceptance fixed when comparing R.
   Analyze within-run temporal correlation and synchronized reference residual dependence.
   Add a small persistence correction only for an observed consistency failure.

Gate: baseline events and measurement means agree; scoped sequential accuracy and containment
are reported on a frozen independent selection. Do not reuse scores from a stale manifest
after these source changes; create a successor manifest and retain the earlier pilot.

### D. Validate future localization quality

Reuse `commissioned_field.py`, `future.py` and `field_driving.py`; the new network's one-step
reference is an implementation cross-check. Preserve misses; distinguish pre-gate q from
final estimator acceptance. Future images and simulator visibility labels are not inputs.

Collect both candidate routes with the same frozen perception model. Compare predicted
versus observed availability, conditional reading covariance, belief scale and route
ranking at actual camera cadence. Use separate held-out drives, not samples drawn from
the fitted q/R model. Estimate uncertainty over independent runs/blocks.

For multistep prediction, keep the incoming belief and motion model fixed and validate
branch compression, timing and dependence explicitly. Compare existing expected-information
blending with branch averaging. If a recursive q/R planner is introduced, both its recursion
and its changed cost must be named as P3 and ablated against the IWAI proxy.

Gate for a calibrated-information claim: acceptable held-out forecast magnitude/containment
and useful route ranking, with failure cases exposed. If only ranking is useful, state that
narrower result and continue the thesis navigation study.

### E. Matched navigation

Use the existing campaign runner and controller. `network_planner_pilot.yaml` is the
earlier unexecuted dry-run starter. The executed sequence is `network_navigation_pilot.yaml`,
`network_navigation_tracking_pilot.yaml`, then `network_navigation_runtime_pilot.yaml`.
Each uses one seed and three arms, geometry-only lane-graph seeds, a 200 s global horizon
and a 0.22 m/s speed limit. Each is an integration pilot, not the final study.
That bounds possible travel to 44 m before turning time; gate B must verify sufficient reach.

Predeclare three families before final collection: ordinary overlap, partial occlusion/
camera transition, and one lost or degraded view. Include a changed stock configuration
with an explicit no-recommissioning or recommissioning policy. Do not call stock changes
new camera-installation transfer. Choose final paired repetitions from pilot variability;
three fields or thousands of frames are not independent runs.

Hold the estimator fixed for P0/P1/P2. Report success, all failures/contact, path length,
travel time, reading/fused/belief quantities separately, nominal ellipse containment,
refusal fraction, longest camera gap and cost of planning/perception. A later estimator ×
planner factorial separates estimation gains from route-choice gains.

Gate: matched trials complete, frozen exact run list passes the metrics loader, main
figures answer RQ1/RQ2/RQ3. Retain methods that fail and conditions where methods tie.

### Recorded tracking failure and separate remedy pilot

The original P0 trial stopped at waypoint 16. The same controller on its saved global
path also stops with ideal motion: 1 m waypoint spacing and 35 cm arrival tolerance
permit a corner cut that the local clearance gate refuses. Spacing 20 cm and arrival
10 cm complete the ideal-motion check in 169.75 s, with 0.374 m minimum modeled body
clearance. This is a tracker/configuration diagnosis, not a perception result.
`tracker_preflight.py` records all three tested settings, including the original failure.
The separate `network_navigation_tracking_pilot.yaml` keeps the global planner and sensor
models identical and changes only those two tracking settings. It finished with one stuck
run and two collisions. Its P0 camera-return event exposed loss of a recorded turn during
outage recovery. The earlier ideal-motion check also used the old hard-coded 1.5 rad/s
controller limit; current declared-limit checks are separately preserved in the tracking
evidence's `diagnostics/postfix_tracker.json`.

### Runtime correctness gate — separate repaired baseline

The [runtime audit](../../docs/runtime_integrity_audit.md) gives evidence and scope for
the fixes. `network_navigation_runtime_pilot.yaml` freezes those sources before collection.
It retains Q, mean, camera R, fusion and the first-return camera-gap refusal. This separates
software repairs from later statistical/admission policy changes. Tests include a blind
turn with full measured odometry, missing-history support, simultaneous cameras, duplicate
events, state/command interleavings, covariance/frame validation and the actuator watchdog.
The focused packet passes 150 tests; three absent archived pixel traces are skipped.

Analyze full-pose and heading errors as well as planar coverage. Log a reason for every
refusal and stop. A correct refusal can reveal a planner/controller or support mismatch;
disable no gate solely to obtain a successful drive. Keep both failed pilot selections.

## 4. Reproduction commands

From `UnembodiedNavigation` (outputs already exist; use new output directories for repeats):

```bash
# Fit and validate the three exported fields from the frozen commissioning sources.
python3 experiments/icra_commissioning/export_network_planner.py --out logs/studies/icra_commissioning_20260905/network_planner_repeat

# Run the short optimization/reference probe with the existing frozen artifacts.
python3 experiments/icra_commissioning/planner_probe.py --out logs/studies/icra_commissioning_20260905/network_planner/probe_repeat

# Validate the prospective live command matrix without starting the simulator.
python3 scripts/visibility_comparison/run_visibility_campaign.py --config experiments/icra_commissioning/network_planner_pilot.yaml --log-root /tmp/network_planner_dryrun --dry-run

# Execute the corrected baseline in a fresh directory. Host ROS/Gazebo is required.
source install/setup.bash
python3 scripts/visibility_comparison/run_visibility_campaign.py --config experiments/icra_commissioning/network_navigation_runtime_pilot.yaml --log-root /tmp/network_runtime_reproduction --first-cmd-timeout 400 --run-timeout 1800

# Focused numerical, legacy, node-routing, campaign and logging checks.
MPLCONFIGDIR=/tmp/icra_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/planning/test_camera_network.py tests/planning/test_efe_hit_miss_mixture.py tests/visibility_comparison/test_network_planner_config.py tests/visibility_comparison/test_campaign_ledger.py tests/experiments/test_logger_schema.py tests/planning/test_planner_node_per_camera_correction.py tests/planning/test_planner_node_state_correction.py tests/experiments/test_logger_time_alignment.py

# Verify every selected export/probe input and figure hash, without re-running a drive.
python3 experiments/icra_commissioning/verify_network_planner.py
```

Do not promote the integration pilot to final evidence. Before a large campaign, budget
disk for retained logs and required crops: this workspace had only about 0.3 GB free during
setup. Existing outputs are not disposable. GPU/physical hardware access is not assumed.

## 5. Deliberately deferred code

No new RGB mean model, GP covariance head, fully general correlated filter, perception-KF
cascade, Q/R joint identification, or new planner objective is required for the first
thesis result. A GP over log residual scale is a later compact option only if the frozen
constant/geometry models leave a measured spatial gap. Full covariance flexibility needs
directional evidence and support. The GP's latent variance never substitutes for R or P.

Current full-route / live-calibration commands (run from the robot repository):

```bash
source install/setup.bash
# Choose a fresh --out; recorded preflights are immutable.
ROS_LOG_DIR=/tmp/network_route_ros OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_route_probe.py --out /tmp/network_route_reproduction
# The exported reference calibration is already frozen; this refuses overwrite.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/export_reference_calibration.py --out /tmp/reference_calibration_reproduction.json
python3 scripts/visibility_comparison/run_visibility_campaign.py --config experiments/icra_commissioning/network_navigation_pilot.yaml --log-root /tmp/network_navigation_reproduction --first-cmd-timeout 300 --run-timeout 1400
python3 scripts/visibility_comparison/run_visibility_campaign.py --config experiments/icra_commissioning/network_navigation_tracking_pilot.yaml --log-root /tmp/network_tracking_reproduction --first-cmd-timeout 400 --run-timeout 1800
# Analyze the recorded explicit ledger; no lookup by directory recency.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_navigation_analysis.py navigation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_navigation_analysis.py navigation --config experiments/icra_commissioning/network_navigation_tracking_pilot.yaml --campaign logs/studies/icra_commissioning_20260905/network_navigation_tracking_pilot --out logs/studies/icra_commissioning_20260905/network_navigation_tracking_evidence
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_navigation_analysis.py navigation --config experiments/icra_commissioning/network_navigation_runtime_pilot.yaml --campaign logs/studies/icra_commissioning_20260905/network_navigation_runtime_pilot --out logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence
```

Launching an older YAML against current source does not reproduce the old runtime.
Use the source snapshots identified in each protocol to reproduce a historical pilot.
The analysis commands consume recorded immutable runs; they do not change the simulator.

The new campaigns use separate ROS domains, unique Gazebo transport partitions, and an
inherited process marker for cleanup. They do not kill unrelated simulator/ROS processes.
Live collection needs host ROS/Gazebo permissions. CPU perception and simulation share
resources; report solver and inference wall time alongside simulated travel time.
