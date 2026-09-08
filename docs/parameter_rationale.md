# Navigation parameter audit — 2026-09-07

**Body-model update:** the later [oriented rectangular collision model](rectangular_collision_model.md) supersedes the circular planner/tracker checks described below. The earlier audit is retained as change history.

The shared launch still carried a small-robot collision radius and coarse waypoint
defaults. Those defaults are repaired. The 0.22 m/s speed is defensible as an explicit
slow experimental control, but it is not a commissioned operating speed for the AMR.
A separate 0.44 m/s development configuration now makes that assumption testable.
The tracker's renewable 5 mm boundary-penetration allowance is also removed. No live
campaign was executed for this audit and no navigation improvement is claimed.

## Scope and sources

Inspected the current shared launch, primary launch, campaign resolver, planner node,
tracker, AMR URDF and the registered runtime/recovery campaign configurations. This is
an audit of configuration and code, not a reanalysis of historical drives. The metrics
contract and registry remain authoritative. Existing source edits and frozen campaigns
were preserved; default changes affect future runs that omit explicit overrides.

Source paths:

- `src/experiments/experiments/core/visibility_launch_common.py`
- `src/experiments/launch/warehouse_primary_comparison.launch.py`
- `src/planning/planning/nodes/unicycle_planner_node.py`
- `src/planning/planning/nodes/efe_agent_node.py`
- `src/sim/launch/bringup_sim.launch.py`
- `src/sim/robot_description/urdf/warehouse_amr.urdf.xacro`
- `experiments/icra_commissioning/network_navigation_runtime_pilot.yaml`
- `docs/module_audits/09_tracking_routes_termination.md`
- `docs/module_audits/13_configuration_provenance_campaigns.md`

## Decisions

| Setting | Decision and defence | What remains to establish |
|---|---|---|
| Shared collision radius: 0.125 m | **Changed to 0.48541219597369 m.** The spawned AMR has an 0.80 × 0.55 m body; `hypot(0.80/2, 0.55/2)` encloses its corners under rotation. The old disc did not. Primary launch and shared fallback agree. | This is a planar body bound, not a stopping-distance allowance or proof that the scene includes every obstacle. Explicit old overrides remain explicit. |
| Lane standoff: 0.35 m | **Changed primary/shared fallback to 0.55 m.** Matches the current pilot's conservative lane constraint and exceeds the body half-diagonal by about 6.46 cm. | Lane standoff and obstacle-disc clearance have different geometry semantics. The extra distance is a chosen margin, not a calibrated uncertainty quantile. Narrow routes may become infeasible. |
| Waypoint spacing: 1.0 m | **Changed to 0.20 m** in primary/shared/node defaults and shared fallback reads. Matches the repaired pilot and limits coarse route handoff. | Spacing is a selection threshold; it does not prove corner preservation or that all generated gaps are exactly 0.20 m. |
| Waypoint arrival radius: 0.35 m | **Changed to 0.10 m.** Matches the repaired pilot and avoids accepting intermediate targets 35 cm away. | Still a controller tolerance, not a bound on physical tracking error. |
| Speed: 0.22 m/s | **Retained as explicit slow baseline; new candidate is 0.44 m/s.** The project documents the old speed as inherited. Doubling it provides an interpretable perturbation without asserting a manufacturer's operating limit. | Candidate is unexecuted. No speed is validated by body size alone. |
| Camera/manager nominal rate: 5 Hz | **Retained.** At maximum speed the nominal capture-opportunity spacing is 4.4 cm at 0.22 m/s and 8.8 cm at 0.44 m/s (`v/f`). | Nominal opportunities are not delivered corrections. Measure actual capture rate, batch latency, missed detections and gaps. Increasing the configured rate does not establish detector throughput. |
| Command publication: 10 Hz | **Retained.** Nominal maximum travel per publication is 2.2/4.4 cm at baseline/candidate speed. | The noise adapter samples per command message; changing this rate also changes the stochastic experiment. Do not use it as a free performance knob. |
| Local planning rate: 4 Hz | **Retained.** It matches the 0.25 s local discretization in the current protocol. | Actual scheduling and solver latency can be slower. This is not a real-time guarantee. |
| Local `dt`: 0.25 s | **Retained.** Maximum straight step is 5.5/11 cm at baseline/candidate speed, below the 20 cm arrival-disc diameter. | That inequality is a scale check, not a guarantee of waypoint capture or collision avoidance between discrete samples. |
| Local horizon: 12 | **Retained.** Gives a 3 s modeled local tape. | Execution may accept only a safe prefix; it must not be interpreted as three seconds of guaranteed motion. |
| Global horizon/dt: 200 × 1 s | **Retained in speed candidate.** Upper straight-line reach is 44/88 m at baseline/candidate speed. Holding these parameters fixed isolates the speed parameter, including its planning consequences. | Effective physical lookahead changes with speed. A later fixed-spatial-grid experiment is a different study; it should not silently rescale horizon and discount. |
| Angular bound: ±1 rad/s | **Retained.** Current planner declarations and actuator clipping agree. | Launch still hardcodes the actuator limits; this is not a vehicle specification. Any angular-limit tuning must wire planner and actuator together and test both. |
| Tracker yaw gate: 0.6 rad | **Retained as heuristic.** Translation is suppressed when misaligned by more than about 34.4 degrees. | No evidence establishes this as optimal. Test turns and boundary approach before changing it. |
| Controller: `turn_then_go` | **Retained in matched speed candidate.** A speed comparison should not simultaneously change the controller. | Recovery is a separate policy. Audit 09 reports unresolved recovery/boundary issues; do not treat a speed increase as their repair. |
| Goal radius/hold: 0.35 m / 2 s | **Retained in the pilot configuration.** Defines operational goal acceptance and rejects a momentary entry. It is separate from the intermediate waypoint radius. | Goal acceptance uses belief, not physical truth. Changing this changes the success criterion; validity/staleness concerns remain in audit 09. |
| Run timeout: 600 s | **Retained across speeds.** Provides the same censoring budget, including turns, stops and computation. | Straight-line minimum time alone cannot justify a route timeout; report timed-out runs. The shared 75 s legacy default is unsuitable as a general warehouse-route budget. |
| Stuck window: 8 s; displacement 0.08 m | **Retained, flagged.** These define an operational stall heuristic. | Endpoint XY motion can misclassify rotation, waiting or stale belief. Fix phase/validity semantics before merely enlarging the timeout. |
| Watchdog: 0.5 s | **Retained, flagged.** Bounds nominal silence before the adapter requests zero. At 0.44 m/s, `v*t` is 22 cm before timer overshoot and braking. | This is not a physical stopping-distance bound. Validate command loss and stopping at each adopted speed. |
| Process noise XY/theta: 0.01 / 0.02 | **Retained, uncommissioned here.** The meanings come from the implemented propagation; neither may be treated casually as a measurement standard deviation. | Estimate against the actual command/encoder noise and time dependence. Validate consistency on a separately selected dataset; do not tune to make a preferred arm win. |
| Encoder/command noise and correlation | **Retained across arms and speeds.** They define the simulated disturbance model. | They lack a new physical calibration in this audit. Message-rate dependence prevents claiming physical realism from their names. |
| NIS threshold: 9.21 | **Retained.** Conventional approximately 99% threshold for a two-dimensional Gaussian innovation. | That interpretation requires a valid innovation covariance and model; it does not establish a 1% rejection rate in this system. |
| Rejection inflation: 0.05 m² | **Retained, flagged as heuristic.** It permits uncertainty to grow after refusal. | Repeated additions depend on rejection frequency and can weaken later gating. Needs an explicit time/event model and replay comparison, not an arbitrary smaller replacement. |
| Prediction bridge cap: 1.5 s | **Retained as refusal policy.** A cap makes replay support finite. | It is not an expected outage duration; first-return refusals and accounting must be reported. |
| Bootstrap min cameras/disagreement: 2 / 0.91 m | **Retained, flagged as heuristic.** Requires corroboration for initialization. | Two agreeing biased cameras can still be wrong. No new justification for 0.91 m is established here. |
| Fusion disagreement gate: 0.60 m | **Retained, flagged as heuristic.** Supplies a physical rejection scale. | Requires held-out false-accept/refusal analysis and explicit covariance semantics. It must not be presented as commissioned simply because it is fixed. |
| Batch timestamp spread: 0.05 s | **Retained.** Nominal robot travel over this spread is 1.1/2.2 cm at baseline/candidate speed. | Timestamp compensation and actual capture identity still matter. This is not an observed camera-reading error. |
| Detector confidence 0.25, IoU 0.45, image size 960 | **Retained.** Changing these changes the observation stream and may invalidate fitted camera artifacts. | Select using held-out detection/admission objectives, including empty frames; runtime convenience is not enough. |
| Camera NN, offsets and full constant R | **Retained.** These are shared observation inputs across the compared conditions. | Use the commissioning/metrics contracts for any replacement; this audit does not fit a new sensor model. |
| Optimizer 80 iterations, 500 function calls | **Retained.** Explicit computational budgets, shared across conditions. | Budget exhaustion is not convergence. Require accepted feasibility and inspect solver status before changing budgets. |
| Objective weights, discount 0.98, uncertainty/no-go modes | **Retained.** Changing them along with the observation model confounds the intended comparison. | These are modeling choices, not physical constants. Sensitivity work needs predeclared shared settings and separate conclusions. |
| Low-CPU physics/contact rates | **Not substituted into speed candidate.** Candidate uses the runtime pilot's original world. | A faster simulation world changes another experimental factor and needs separate physical acceptance checks. |

## Executable protection

`unav_common/navigation_parameters.py` rejects nonfinite numbers, booleans used as
numbers, nonpositive speeds/rates/radii/steps, fractional horizons, negative noise
scales and discounts outside `(0, 1]`. The campaign runner applies it to the campaign
and each effective route/condition/task cell. Documented zero sentinels, including
`global_dt=0` and disabled NIS gating, remain valid. This is intentionally a domain
check, not a complete configuration schema or a stability certificate. Direct ROS
launches bypass this campaign check.

The geometry regression reads dimensions from the URDF and checks that the shared
disc encloses the chassis. A separate regression verifies that the speed candidate
changes only speed, execution domain and descriptive metadata from the runtime pilot.

## Adoption boundary

Use `experiments/icra_commissioning/network_navigation_speed_candidate.yaml` only as
a separate development protocol. It contains the same three arms and single seed as
the runtime pilot, so it cannot establish a replicated speed effect. Run a fresh
0.22 m/s baseline on the same source and simulator before comparing. Then predeclare
paired seeds and speed/rate conditions for replication. Validate stopping, turns,
tracking, actual capture/delivery timing, terminal accounting and aligned estimator
metrics. Preserve run/config/source identity and use the required alignment loader.
Do not pool these runs with any registered historical pilot.

The remaining heuristics above have not been defended by evidence merely by being
listed. Their replacement requires the stated measurement or implementation repair.

## Planner and no-go follow-up

The user explicitly requested planner and no-go tuning as well. The following checks
trace `core/nogo_cost.py`, `core/casadi_efe.py`, `planners/base_planner.py` and the
tracker gate. Reproduce the model arithmetic with
`python3 scripts/visibility_comparison/probe_parameter_scales.py`.
The retained output is [parameter_scales.json](parameter_scales.json).
These are deterministic model values, not route performance or camera error.

| Parameter or coupling | Finding and decision |
|---|---|
| Hidden `RECOVERY_EPS=0.005` | **Removed.** The gate previously set each new floor to `min(start_clearance, 0) - 0.005`. Replanning renewed the allowance. It now uses `min(start_clearance, 0)`, permitting holding/recovery but rejecting deeper penetration. The change applies to obstacle and driveable-lane checks. Tests exercise repeated outward attempts, an initially valid pose, an initially invalid pose, holding and recovery. A stricter gate can stop a run that formerly crept outwards; this is an intended correctness change. |
| `nogo_mode` | `keep_in` means stay inside the union of driveable rectangles; `keep_out` means stay outside obstacle rectangles. **Keep `keep_in` in the network protocol.** Treating these as interchangeable changes the map's meaning. Campaign validation now rejects unknown modes. |
| Margin versus robot radius | In keep-in mode, clearance is distance inside the lane boundary **minus 0.55 m**. The no-go model does not add the collision radius again. A separate obstacle model applies the robot disc. Adding radius to the no-go standoff a second time would over-erode the lane. |
| Lane feasibility | A straight lane narrower than 1.10 m has no nonnegative centre clearance under the 0.55 m standoff. A 1.10 m lane has only a zero-clearance centreline; a 1.20 m lane reaches the 0.05 m warning-band edge. This is a cross-section calculation, not a connectivity guarantee at corners. |
| `nogo_warning_band=0.05` | **Retain.** Cost is zero when remaining clearance is at least 5 cm. Combined with the standoff, a straight lane needs width 1.20 m for a zero-cost centre. The 5 cm band is a smoothing/design choice, not measured stopping distance. |
| `nogo_near_weight=50` | **Retain pending objective-scale sensitivity.** The warning term is `50*log(1+((0.05-c)/0.05)^2)` below the band. At clearance zero it contributes about 34.66, and at 2.5 cm positive clearance about 11.16. It does bias routes within the warning band, even though it is zero farther inside. |
| `nogo_weight=40`, `nogo_logbarrier_eps=0.001` | **Retain together pending solver/objective testing.** Despite its name, epsilon divides a quadratic violation, not a true infinite log barrier. Violation stiffness is `weight/eps² = 40,000,000` per square metre. Doubling epsilon quarters that stiffness. At 1 cm penetration, total cost is about 4,044.60; at 5 cm it is about 100,080.47. Do not interpret 40 independently of epsilon. |
| Weight zero | The current model's `enabled` property requires positive violation weight and nonempty geometry. Weight zero also disables its clearance check and warning term, even if `near_weight` remains positive. **Keep weight 40.** Using zero as a harmless objective ablation would remove more than an objective term. Decoupling geometry enforcement from penalty enablement would be a separate API/policy change. |
| Invalid no-go scales | **Now rejected** in the model constructor and before base-planner clamps, as well as in campaign cells. Negative weights, nonfinite scales and zero band/epsilon must not silently become another configuration. Existing small positive numerical floors in the implementation remain. |
| Belief no-go and `kappa=1` | **Keep disabled in the current mean-geometry protocol.** For keep-in, enabled mode subtracts `kappa*sigma_max` from clearance; for keep-out it uses weighted sigma-point costs. Those are different computations. Kappa is not universally a coverage percentage, and enabling the flag changes the objective. |
| `discount_gamma=0.98` | **Retain as a declared step-based choice.** Weight halves after about 34.31 steps: 8.58 s at local dt 0.25, but 34.31 s at global dt 1.0. At step 200 the weight is about 0.0176. The nominal 200 s horizon does not weight its far end equally. A time-based discount would need `gamma(dt)=exp(-dt/tau)` and a declared tau, applied consistently; there is no evidence here choosing tau. |
| Horizon normalization | CasADi divides all accumulated terms by `sum(gamma**t)`. This prevents a trivial sum-length rescaling, but changing horizon still changes reachable endpoints, relative time weighting and the goal-prior schedule. **Do not shorten the horizon just to reduce computation without checking these effects.** |
| Legacy goal-prior schedule: 80 to 4, power 0.45, 90 steps | **Retain only for `legacy_pixel_chart` reproduction.** The widths are in the fixed camera cost chart, not metres. The schedule uses step progress; its physical duration depends on dt. These parameters cannot be defended as localization precision or a physical goal radius. |
| Metric network goal width: `network_goal_std_m=0.15` | **Development value, not frozen.** This is the standard deviation of the world-XY Gaussian goal preference in `metric_expected_belief`; it is not the success radius or a localization-accuracy claim. Validate it jointly with objective weights before a final campaign. |
| `control_weight=0` | **Retain for matched protocols, explicitly acknowledge no direct quadratic effort penalty.** Bounds and the tracker still constrain commands; there is no objective-based energy/smoothness preference from this term. A positive value needs normalized units or a stated tradeoff between linear and angular command costs. |
| Observation-risk and ambiguity weights | **Retain all arms' common values.** Their influence depends on cost-chart covariance scales and term magnitudes. A larger numeric weight does not alone imply a larger contribution. A defensible sensitivity study must report decomposed costs and route changes using the same start, belief, geometry and artifacts. |
| `optimizer_terminal_goal_tolerance_m=0.35` | **Retain aligned with the protocol's goal radius.** It affects candidate ranking, not only numerical convergence: terminal candidates can outrank incomplete ones before EFE comparison. It does not prove the complete trajectory is feasible. |
| `ftol=1e-6`, `gtol=1e-5`, 80 iterations/500 calls | **Retain pilot values; validate domains.** Negative/nonfinite tolerances and fractional/nonpositive iteration budgets are rejected in campaigns. Tight tolerances do not ensure convergence of a heavily penalized objective; inspect feasibility and termination reason. Shared generic gtol remains 1e-4, so the pilot's tighter value must stay explicit. |
| Lane-graph seeds, multistart, no direct seed | **Retain across the three network arms.** These restrict which local optima are offered. Different seeds or candidate routes across arms would change more than the camera field. No-go weight tuning cannot establish global optimality. |

The no-go repair is a fresh source change. Any later live result needs new source
identity and a fresh matched baseline; old pilot outcomes are not evidence for it.
Current checks: **115 tests passed**, covering configuration, tracker boundaries,
command installation/transactions, network planning and optimizer cache behavior.
No live stopping test or full-route planner sensitivity campaign was performed.

## Body-collision follow-up

The radius change alone was insufficient: the local tracker called
`collision_signed_distance_state_np`, which measures centre-to-obstacle distance.
It now calls `collision_clearance_state_np`, subtracting the configured body radius,
as the global trajectory diagnostics already do. The clearance helper now preserves
NaN and negative infinity for rejection instead of converting them into clear space.
A wall regression rejects a 1 cm step when the centre remains clear but the body disc
would overlap; a farther wall still permits the step. The updated suite passes 119 tests.

This establishes a consistent conservative planar body check for represented obstacles,
not complete collision safety. The world profile still lists only `warehouse_shell`
and `warehouse_v2_occluders` for collision parsing; audit 14 identifies omitted physical
props and incomplete contact instrumentation. The local guard checks rollout samples,
not a continuous swept volume. Global keep-in validity also differs from the local
standoff gate. These limitations remain unresolved and require separate repairs.
