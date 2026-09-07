# 14 — Simulator, world and physical outcomes

2026-09-06. **The current active world file and its 46 declared planner boxes match the geometry freeze, but that is not the complete physical scene.** Five collision-bearing included props are outside the planner collision JSON and outside contact instrumentation. One of them, the pallet jack beside the selected start, violates the configured circular clearance of the registered plan and intersects the exact robot rectangle for some in-place headings. Separately, private Gazebo Sim 6 probes establish that an acknowledged `reset.all` is not a fresh physical epoch, and that losing the clock bridge or command adapter can leave a previously applied command moving the robot.

These are infrastructure and evidence-contract findings. I did not change the world, robot, collision geometry, rates, noise, planner, selected runs, or production source. All live probes used fresh Gazebo partitions, unique world names and unique ROS names/topics. They signalled only their own process groups. I did not inspect, stop or reuse another chat's simulator. The sandbox cannot certify that the host had no simulator outside its process namespace.

## Scope and evidence boundary

I read the repository contracts first: `AGENTS.md`, `PLAN.md`, `docs/localization_metrics.md`, `docs/localization_metrics_registry.json`, `docs/open_questions.md`, `docs/runtime_integrity_audit.md`, and the completed module audits. The registered corrected-runtime reference is [network_navigation_runtime_pilot.yaml](../../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml), its exact [protocol](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/protocol.json), and [selection](../../logs/studies/icra_commissioning_20260905/network_navigation_runtime_evidence/selection.json). Tracking evidence is taken only from its registered [selection](../../logs/studies/icra_commissioning_20260905/network_navigation_tracking_evidence/selection.json). No run was chosen by modification time or a `RESULTS.md` file, and no localization metric was recomputed.

The source tree changed concurrently during this audit. Each baseline callback result records source SHA-256 values. The clock-throttle baseline was `29777d…`; the current repaired file is `a7ef82…`. `visibility_launch_common.py` changed from `de61a0…` to `f8c9b2…` when audit 04 added detector/manager fail-fast shutdown. Frozen observations are not relabelled as current behavior.

Probe evidence is under [simulator_world_20260906](../../experiments/runtime_integrity/simulator_world_20260906/):

- [asset_results.json](../../experiments/runtime_integrity/simulator_world_20260906/asset_results.json) evaluates the exact campaign command through the launch constructors, resolves source/install paths and dependencies, hashes selected files, and applies an independent Shapely/SciPy geometry oracle.
- [prop_geometry_results.json](../../experiments/runtime_integrity/simulator_world_20260906/prop_geometry_results.json) inventories collision-bearing included props and measures their conservative footprints against the driveable union, selected plan and robot body.
- [physics_results.json](../../experiments/runtime_integrity/simulator_world_20260906/physics_results.json) and [native_summary.json](../../experiments/runtime_integrity/simulator_world_20260906/native_summary.json) retain known-clear and known-contact private physics evidence, contact identities, rates and raw-file hashes.
- [transport_results.json](../../experiments/runtime_integrity/simulator_world_20260906/transport_results.json) retains native odometry/TF/clock traffic and an acknowledged pause/reset sequence.
- [watchdog_results.json](../../experiments/runtime_integrity/simulator_world_20260906/watchdog_results.json) and [watchdog_native_summary.json](../../experiments/runtime_integrity/simulator_world_20260906/watchdog_native_summary.json) use the installed ROS bridges, current clock throttle and actual command-noise node with active parameters, joined offline to native world poses.
- [callback_results.json](../../experiments/runtime_integrity/simulator_world_20260906/callback_results.json) is the frozen initial callback snapshot; [callback_current_results.json](../../experiments/runtime_integrity/simulator_world_20260906/callback_current_results.json) rechecks the concurrent clock repair. These probes use real message/time types with fake transport; they establish callback behavior, not physical outcomes.

## Ranked findings

| Rank | ID | Classification | Finding and active reachability |
|---:|---|---|---|
| 1 | S14-01 | **P0 confirmed scene-contract bug** | Five floor props have physical collisions but are absent from `collision_geometry_json` and all contact sensors. The active start/plan is within the pallet jack's omitted clearance. Reached in every warehouse-v2 pilot. |
| 2 | S14-02 | **P0 confirmed reset/epoch bug plus simulator limitation** | `reset.all` can rewind time without restoring pose, wheel odometry or the DiffDrive command target. ROS buffers and noise/estimator state have no coordinated epoch reset. Active when `reset_world=true` or any in-place reset is attempted; registered runtime pilots set it false. |
| 3 | S14-03 | **P1 confirmed command-safety gap** | The only command watchdog needs the ROS simulation clock and its own process. Private live failures moved the robot about 0.18 m during clock-bridge loss and 0.25 m after adapter restart, without a new command. Active failure boundary. |
| 4 | S14-04 | **P1 confirmed bridge/TF bugs** | Launch bridges a nonexistent `/model/turtlebot3/odometry_tf`; Sim 6 publishes `/model/turtlebot3/tf`. Joint states are also advertised in launch but not produced. Repairing TF naively creates two parents for `base_link`. Active, although the planner consumes numeric odometry directly. |
| 5 | S14-05 | **P1 confirmed contact-evidence ambiguity** | Contact sensors publish only nonempty evidence in this backend; a healthy clear scene is silent. Publisher count proves ROS bridge presence, not upstream liveness. Logger identity, epoch and finalization checks are incomplete. Active in all registered runs. |
| 6 | S14-06 | **P1 confirmed startup admission bug** | Clock/odom readiness accepts one unvalidated message and launch exit handlers ignore return status. A failed reset or timed-out odom gate can still start downstream work. Active startup path. |
| 7 | S14-07 | **P1 confirmed provenance gap; current assets agree** | The freeze and selected manifests do not attest the complete resolved physical/rendering scene or expanded robot. Current source/install assets agree, but historical runs cannot prove that fact from their manifests alone. Active evidence limitation. |
| 8 | S14-08 | **P2 confirmed robot-conversion bug and engine limitation** | Caster friction written inside URDF `<collision>` disappears during URDF→SDF conversion. The requested ODE physics declaration actually loaded DART in Sim 6. Active dynamics, with no attribution of a navigation failure to either item. |
| 9 | S14-09 | **P2 confirmed camera-frame contract gap** | The online manager reads all five SDF poses directly and uses matching fixed intrinsics, but TF publishes only camera A, names a non-optical camera frame, and disagrees by 90° with the image optical axis. Active only for TF-based image consumers, not the active manager projection. |
| 10 | S14-10 | **P2 confirmed outcome conflation** | Physical contact, offline circular geometry overlap and mission failure remain partially collapsed into `crashed`/`collision`, while their clocks and identities differ. Active reporting path; ground truth does not drive the active online stop decision. |
| 11 | S14-11 | **P2 confirmed isolation hazard** | Registered campaigns use isolated partitions/domains, but domains are arithmetic rather than leased, all ROS topics are global inside a domain, and the warehouse sketch helper kills simulator processes globally. Conditional on parallel/manual use. |
| 12 | S14-12 | **P2 measured limitation/hypothesis boundary** | Headless and GUI expansion use the same assets/plugins, but render cadence parity was not established. Declared sensor rates do not prove delivered/inferred rates or latency. No live GUI comparison was run. |

## Detailed findings

### S14-01 — The planner and contact system omit physical props

**Path and trigger.** The world generator deliberately adds a parked forklift, pallet jack, bin and two loaded pallets as `<include>` models at [make_world.py:675](../../experiments/warehouse_v2_sketches/make_world.py:675). The active file contains them at [warehouse_v2.world.sdf:76](../../src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf:76). Their model files contain 13 floor-height collision geometries. The profile restricts collision parsing to `warehouse_shell` and `warehouse_v2_occluders` at [world_profiles.yaml:166](../../src/experiments/config/world_profiles.yaml:166); `resolve_world_setup` serializes only those selected models at [visibility_launch_common.py:964](../../src/experiments/experiments/core/visibility_launch_common.py:964). `_make_contact_bridge` discovers only direct world/model/link sensors at [bringup_sim.launch.py:50](../../src/sim/launch/bringup_sim.launch.py:50). None of the five prop models has a contact sensor.

**Expected.** Every collision body that can meet the robot is represented by the planner/clearance contract and has an attributable physical contact path, or a checked enclosing non-driveable region makes the omission explicit and guarantees clearance for the configured body.

**Observed.** The 46 direct shell/occluder boxes match the planner scene exactly, but that equality omits the props. The independent prop probe found:

| Prop | Collision parts at AMR height | Area intruding into declared driveable union | Selected plan centreline clearance | Clearance after 0.485412 m circle |
|---|---:|---:|---:|---:|
| pallet jack | 1 mesh | 0.1300 m² | 0.45696 m | **−0.02846 m** |
| dock-office bin | 1 mesh | 0.3289 m² | 0.82058 m | 0.33516 m |
| forklift | 6 boxes | 0 | 9.5109 m | 9.0255 m |
| loose pallet 1 | 2 boxes | 0 | 2.3877 m | 1.9023 m |
| loose pallet 2 | 2 boxes | 0 | 10.7991 m | 10.3137 m |

At task start `(-7.9,-8.7,0.05,yaw=0)`, the exact 0.80×0.55 m body is 0.04975 m from the pallet-jack collision mesh. The probe clips every DAE collision triangle to the robot's vertical slab, projects the surviving triangles into the floor plane and applies the include pose. A 0.5° sweep over a full in-place rotation contains intersecting headings. This reproduces the requested tight-turn failure mode. It does not prove the registered run physically hit the mesh; it proves the active circular clearance contract is violated and the included prop is absent from online avoidance and contact evidence.

**Consequence.** An omitted-prop collision can be a physical mission failure with no `/world_contacts` event. Conversely, the route may remain physically clear for its actual body heading while violating the circular planner proxy. That is precisely why planner clearance, physical collision and online failure must remain separate.

**Smallest repair.** Extend resolved-world parsing to included models and supported mesh/box/cylinder collision geometries, then fail launch unless each robot-height physical collider is either in the collision scene/contact ledger or inside a validated exclusion region with configured-body clearance. Add contact sensors/aggregation for the five props without changing their geometry. Replan only if this existing world contract rejects the registered route; that would be a new experimental environment/version, not a favorable tuning fix.

### S14-02 — `reset.all` does not establish a new epoch

**Path and trigger.** `reset_world.py` sends only `world_control.reset.all` at [reset_world.py:35](../../src/sim/sim/reset_world.py:35). A response with `success=false` is logged but still sets `done=true`, and `main` returns 0 at [reset_world.py:41](../../src/sim/sim/reset_world.py:41). Bringup then spawns on any reset-process exit at [bringup_sim.launch.py:341](../../src/sim/launch/bringup_sim.launch.py:341).

**Physical reproduction.** In a private Sim 6 world, all control calls returned `data: true`. Immediately before reset, source-stamped ground truth was `(−7.548854,−8.700000)` at 3.222 s and wheel odometry was `x=0.3530`, `v=0.2` at 3.220 s. After `reset.all`, time restarted at 0, but ground truth was `(−7.545254,−8.700000)` at 0.017 s and odometry was `x=0.3536`, `v=0.17999` at 0. The robot then continued to `(−7.304054,−8.700000)` by 1.223 s from the single pre-reset Twist. Gazebo logged time-rewind warnings for Physics, DiffDrive, Contact and LogRecord. This is a Sim 6 system limitation exposed by an over-strong repository reset contract.

The encoder adapter compounds the epoch problem. It updates `_last_stamp` before rejecting `dt<=0` at [encoder_noise_node.py:262](../../src/sim/sim/encoder_noise_node.py:262) and retains pose, covariance, correlated noise and Jacobians. A deterministic sequence `t=100→100.1→0.1→0.2` produced `x=5.04`; a fresh instance produced `x=0.02`. Audit 04's current producer/manager repair now treats a clock rewind as fatal coordinated-restart failure; it does not implement hot reset.

**Smallest repair.** Remove in-place reset from supported experimental operation. Define a run epoch as newly started simulator, bridges, producers, estimator, planner, adapters and logger. If hot reset remains a feature, add a session/epoch ID, explicit stop acknowledgement, delete/recreate or verified pose reset, monotonic clock progression, fresh odom/camera/contact/TF sequences, and reset hooks for every stateful buffer. Treat reset refusal and readiness timeout as launch failure.

### S14-03 — A stopped clock bridge or adapter can leave physical motion

**Path and trigger.** Commands flow `/cmd_vel_raw` → command-noise adapter → `/cmd_vel` → bridge → Gazebo DiffDrive. The adapter uses unstamped Twist receipt and a 0.5 simulated-second timeout at [actuation_noise_node.py:106](../../src/sim/sim/actuation_noise_node.py:106) and [actuation_noise_node.py:190](../../src/sim/sim/actuation_noise_node.py:190). It has no downstream acknowledgement and publishes no zero on startup or destruction at [actuation_noise_node.py:205](../../src/sim/sim/actuation_noise_node.py:205).

**Observed.** The private live probe used the installed bridge, current throttle and adapter with seed 210 and active noise defaults:

- Ordinary source silence stopped the robot at the configured timeout.
- A real physics pause held the simulation clock; after unpause, the watchdog elapsed and stopped the robot.
- Suspending only the owned clock bridge froze ROS clock at 2.860 s while native odometry advanced to 3.860 s at 0.17908 m/s. Ground truth moved **0.17997 m** before the bridge returned and the watchdog could emit zero.
- Killing and restarting only the adapter after a command left the Gazebo target active. With no new input, ground truth moved **0.25122 m in 1.428 simulated seconds**; the fresh adapter emitted nothing. An explicit final zero stopped it.
- A reset with no intervening receipt exercised the existing negative-age guard and did emit zero. The callback probe separately shows that a new receipt before the next watchdog tick overwrites the old epoch and hides the rewind.

**Smallest repair.** Put an independent wall-time command-expiry system at the final actuator boundary, with stamped sequence/epoch, source liveness and applied-command acknowledgement. Publish and verify zero before adapter shutdown/restart and launch teardown. A simulator pause needs an explicit policy so wall time does not incorrectly expire a deliberately paused run; a missing clock bridge while physics advances must always fail safe.

### S14-04 — TF and joint-state bridge contracts are disconnected

The DiffDrive URDF publishes odometry on `/model/turtlebot3/odometry`, child `base_link`, at 50 Hz and enables TF at [warehouse_amr.urdf.xacro:269](../../src/sim/robot_description/urdf/warehouse_amr.urdf.xacro:269). No `tf_topic` is configured. Bringup instead bridges `/model/turtlebot3/odometry_tf` at [bringup_sim.launch.py:394](../../src/sim/launch/bringup_sim.launch.py:394). The private native topic inventory found 223 odometry messages and 223 messages on `/model/turtlebot3/tf`, with zero on the configured topic. This matches the Sim 6 DiffDrive default documented in the [upstream implementation](https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/src/systems/diff_drive/DiffDrive.cc).

`/model/turtlebot3/joint_states` was not advertised and produced zero messages: the robot has no JointStatePublisher plugin, while launch bridges that topic at [bringup_sim.launch.py:401](../../src/sim/launch/bringup_sim.launch.py:401).

Changing only the TF topic is unsafe. Robot state publisher supplies static `base_footprint→base_link` with z=0.010 at [warehouse_amr.urdf.xacro:61](../../src/sim/robot_description/urdf/warehouse_amr.urdf.xacro:61), while DiffDrive supplies `odom→base_link`. An actual `tf2_ros.Buffer` reproduction then cannot connect `map_bev` to `base_footprint` because `base_link` has two parents.

**Smallest repair.** Configure DiffDrive and the bridge to publish `odom→base_footprint`, or revise the canonical robot frame consistently; test one connected, single-parent graph from `map_bev` to every robot/image frame. Add a real joint-state publisher or remove the unused bridge/readiness claim.

### S14-05 — Silence is not “no collision,” and contact records are incomplete

The active world has 46 contact sensors: 10 shell and 36 stock/occluder links. A private known-clear placement at the selected start produced **zero** contact messages. A 2 cm intersection with unchanged `obs_A2b3e` produced **2,500 messages from 0.001 through 2.500 s**, exactly 1 kHz, despite each SDF sensor declaring 60 Hz. The first payload identified collision entity 73, `warehouse_v2_occluders::obs_A2b3e::collision`, and entity 159, `turtlebot3::base_footprint::base_footprint_fixed_joint_lump__base_link_collision`, with four world contact points. The Sim 6 Contact implementation publishes only nonempty samples and updates with physics here; see the [upstream Contact system](https://github.com/gazebosim/gz-sim/blob/ign-gazebo6/src/systems/contact/Contact.cc). The 60 Hz declaration is therefore not the delivered contact rate in this backend.

The bridge merges 46 native topics into `/world_contacts` and does not retain which source sensor supplied a message. Logger checks substring membership for `turtlebot3`, ignores any other name containing `ground_plane`, accepts missing obstacle identity as `unknown`, and accepts zero/future/old-epoch stamps at [experiment_logger.py:1995](../../src/experiments/experiments/nodes/experiment_logger.py:1995). Callback reproductions show `turtlebot3_clone` is accepted and `not_ground_plane_rack` is ignored. Empty message and no message both leave the same collision flag. Publisher liveness is checked only once, at first command, at [experiment_logger.py:3231](../../src/experiments/experiments/nodes/experiment_logger.py:3231).

The exact selected runtime P0/P1 summaries report 46 publishers, zero messages and no contact. That means **no recorded contact**, not verified collision-free motion. Selected tracking P1 records physical contact with `obs_A2b3e` at 165.184 s. Its summary says one contact, but its final raw CSV row at 165.200 says 17 and still carries command `(0.1862,−0.3103)`, proving the finalization race described by audit 10. The complete contact payload, robot link, source topic, receipt time and exact contact-time command/state were not retained.

**Smallest repair.** Add a per-epoch contact-channel health record using source topic discovery plus a known positive-control fixture before evidence collection. Preserve raw payload/hash, source sensor, exact entity IDs/names, source and receipt stamps, epoch and sequence in an append-only event ledger. Keep “healthy and no contact,” “contact,” “telemetry absent,” and “malformed/stale contact” as distinct states. Do not synthesize periodic empty contacts as physical evidence.

### S14-06 — Startup sequencing treats process exit as readiness

`wait_for_clock` accepts any first message, including stamp zero, with no progress test at [wait_for_clock.py:29](../../src/sim/sim/wait_for_clock.py:29). Bringup gives it no timeout at [bringup_sim.launch.py:285](../../src/sim/launch/bringup_sim.launch.py:285). `wait_for_odom` defaults to one message and validates neither frame, stamp, finite values, epoch nor uniqueness at [wait_for_odom.py:60](../../src/sim/sim/wait_for_odom.py:60). The active common launch uses one `/odom` message and no pose match at [visibility_launch_common.py:1141](../../src/experiments/experiments/core/visibility_launch_common.py:1141). A synthetic foreign-frame future-stamped wrong pose passes; three identical old messages also pass the optional three-message pose gate.

Both bringup and the primary launch use `OnProcessExit` without return-code gates at [bringup_sim.launch.py:333](../../src/sim/launch/bringup_sim.launch.py:333) and [visibility_launch_common.py:2073](../../src/experiments/experiments/core/visibility_launch_common.py:2073). The robot spawn also omits an explicit `-world` argument at [bringup_sim.launch.py:307](../../src/sim/launch/bringup_sim.launch.py:307), relying on a single discoverable world and not verifying entity identity/pose. The active primary resolver does validate internal world name, but direct bringup remains vulnerable to filename/internal-name mismatch.

**Smallest repair.** Make each gate emit an explicit success token and have failure trigger launch shutdown. Require monotonic clock progress, current-epoch finite odometry in the expected frame, exact entity spawn result and verified source-stamped pose. Pass the selected world to spawn explicitly.

### S14-07 — Current assets resolve correctly, historical manifests do not prove it

The evaluated active launch resolves the installed world symlink to `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf`, internal world `warehouse_v2`, SHA-256 `cb872fd438bd3275ac75da44c12013c01c91c95287ce29edaf6bd0f09073f3e3`. `world_freeze.py` passes `WHV2-GEOMETRY-V2`; `verify_world.py` passes 211 peak, 177 shipout and 8 paired checks. All resolved model/material/mesh URIs examined by the asset probe exist. Headless and GUI expansion resolve the same world, resource paths and expanded robot SHA-256 `f590fca5…`; only their server/render flags differ.

The other active artifacts are the detector at `logs/perception_models/warehouse_v2_yolo_detect_halfopen_20260825_r1/model.pt`, learned box-feature correction at `logs/perception_models/box_feature_bias_correction_20260831/models.joblib`, commissioned covariance/bias at `logs/studies/icra_commissioning_20260905/network_planner/reference_calibration.json`, and P0 field at `logs/studies/icra_commissioning_20260905/network_planner/uniform.npz`. Their selected-manifest hashes were verified by the asset probe.

The profile still labels the same world `WHV2-GEOMETRY-V1` at [world_profiles.yaml:148](../../src/experiments/config/world_profiles.yaml:148), a metadata defect. More materially, the freeze's `SOURCE_PATHS` at [world_freeze.py:25](../../experiments/warehouse_v2_sketches/world_freeze.py:25) excludes the camera model SDFs, expanded URDF and resolved third-party mesh/material bytes. Selected run manifests include hashes for planner collision JSON, model weights and calibration artifacts but no `world_sha256` or `robot_description_sha256`. The working installation currently points to source; that cannot attest what a historical process loaded.

**Smallest repair.** At launch, record and validate the resolved canonical paths, hashes and versions for the world, expanded robot, every included model/collision/render asset, camera SDF/intrinsics, Gazebo/physics plugin and relevant environment. Bind them to run ID before the first command. Correct the profile freeze ID. Refuse an unresolved or changed dependency instead of silently refreezing after results exist.

### S14-08 — Requested caster friction is discarded; the active engine is DART

The caster macro places `<surface><friction><ode>…` inside URDF `<collision>` at [warehouse_amr.urdf.xacro:176](../../src/sim/robot_description/urdf/warehouse_amr.urdf.xacro:176). `xacro` followed by `ign sdf -p` emits both caster collision spheres with no `<surface>` and no warning. The intended `mu=mu2=0.05` is therefore not part of the executed robot; backend defaults apply.

The world declares `<physics type="ode">` at [warehouse_v2.world.sdf:9](../../src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf:9), but both private Sim 6 runs logged `dartsim` from ign-physics 5. Thus the declaration is not evidence that ODE was executed on this installation.

**Smallest repair.** Express friction through a supported Gazebo URDF extension or explicit SDF and assert the converted SDF. Record the loaded engine/plugin from runtime. Friction/engine corrections change contact dynamics and require a new environment/version and new runs; they must not be applied merely to obtain better navigation.

### S14-09 — Online camera geometry is coherent; the public TF contract is not

All five active cameras load their pose from the exact world SDF, and their camera sensors declare 1280×720, horizontal FOV 1.5708 and 5 simulated Hz. The manager constructs each `ObliqueCameraModel` directly from those includes at [camera_manager_node.py:685](../../src/reliability/reliability/nodes/camera_manager_node.py:685). Its intrinsics are independently hard-coded to those same values at [projection.py:67](../../src/reliability/reliability/projection.py:67); camera-info is bridged but not used for this model. Audit 05 covers the validated direct projection and visual-hull observation model.

The exact localization-camera include poses `(x,y,z,roll,pitch,yaw)` in metres/radians are A `(-11.45,-9.45,5,0,0.7679,0.7854)`, B `(-1.5,-9.72,5,0,0.8378,2.0071)`, C `(-6.95,9.45,5,0,0.6632,-1.0472)`, D `(11.45,7.2,5,0,0.6632,-2.4435)`, and E `(11.45,-9.45,5,0,0.7330,2.3038)`. The pinhole matrix implied by the manager/SDF contract is `fx=fy≈640`, `cx=640`, `cy=360` pixels. The world also loads presentation overview RGB at 1 Hz, 1280×720, and plan-view RGB at 1 Hz, 1600×1200; neither is bridged by the registered runtime launch. Each localization model also declares depth at 5 Hz and segmentation at 1 Hz with `always_on=0`; those streams are not bridged or consumed in the active runtime.

The TF launch publishes only camera A as `map_bev→external_camera/link/camera` at [tf_static.launch.py:60](../../src/perception/launch/tf_static.launch.py:60). It uses the SDF link quaternion whose forward axis is +X, whereas a ROS optical frame points +Z; the probe measures a 90° axis disagreement. No B–E TF frames are published. This is not an active manager projection error, but it makes TF-based image use ambiguous and leaves calibration duplicated rather than bound to each image stream.

**Smallest repair.** Publish distinct link and optical frames for A–E, verify each optical ray against its image/camera-info, and create the manager model from the same validated calibration record. Record calibration identity with each stream/run. Do not change the commissioned extrinsics to repair naming.

### S14-10 — Physical and analytical outcomes are not one event type

The logger correctly refuses `terminate_on_geom_collision=true` because geometry uses ground truth at [experiment_logger.py:369](../../src/experiments/experiments/nodes/experiment_logger.py:369). Inspection found no ground-truth subscription in the active planner, camera manager, mission or stop decisions. Contacts may explicitly stop the online run; `gt_*` remains offline reference.

Reporting still collapses contact and offline circular overlap into `crashed` at [experiment_logger.py:3381](../../src/experiments/experiments/nodes/experiment_logger.py:3381), and the runner maps any `crashed` summary to `collision`. `_record_collision_event` shares one first stamp and overwritable reason for contact and geometry. Geometry is evaluated from latest ground truth but stamped with odom-map/log time at [experiment_logger.py:2987](../../src/experiments/experiments/nodes/experiment_logger.py:2987). Mission failure, stuck, planner-clearance violation, exact-body intersection and backend contact are therefore different evidence and must not inherit each other's name or clock.

**Smallest repair.** Store independent immutable records for physical contact, exact-body/offline clearance, planner proxy violation, operational termination and process/infrastructure failure. Each needs source, epoch, source/receipt time, identity and evidence validity. Derive presentation labels only after those fields are preserved.

### S14-11 — Isolation is configured, but not allocated

The active runner's isolated mode creates per-run `IGN_PARTITION`/`GZ_PARTITION` and uses ROS domain bases. The original, tracking, runtime and recovery configurations currently use distinct bases 191, 201, 211 and 214. However, a second invocation of the same config computes the same domains, and all topics (`/clock`, `/cmd_vel`, `/odom`, cameras, contacts) lack a run namespace. There is no lease/lock proving exclusive ownership.

The manual warehouse helper preflight matches only selected process command lines and its trap runs blanket `pkill -9` for Gazebo, bridges, clock throttle and robot-state publisher at [sim_up.sh:18](../../experiments/warehouse_v2_sketches/sim_up.sh:18). It was not executed in this audit because it could terminate another chat's work.

**Smallest repair.** Atomically lease ROS domains and log destinations, generate random simulator partitions and run namespaces, record ownership, and terminate only owned process groups. Reject reuse before launch. Remove global teardown from helper and legacy paths.

### S14-12 — Rates and headless parity have narrower evidence than configuration

The active physics step is 0.001 s; native clock advanced at 1 kHz. DiffDrive odometry was exactly 50 Hz. Native SceneBroadcaster ground truth had a median positive interval of 0.017 s in the private trace, not the logger comment's assumed 1 kHz; individual bridged transforms have zero stamps, so the logger records receipt simulation time at [experiment_logger.py:1653](../../src/experiments/experiments/nodes/experiment_logger.py:1653). Interarrival spacing cannot bound the unknown source-to-receipt delay.

RGB sensors declare 5 simulated Hz, but the detector has synchronous callbacks with depth-one subscriptions at [batched_four_camera_yolo_node.py:528](../../src/perception/perception/nodes/batched_four_camera_yolo_node.py:528). Manager decisions are 5 Hz; belief/command/log sampling are 10 Hz; local planning is 4 Hz. A declared 5 Hz camera does not promise 5 Hz inference on CPU, and current ledgers do not account for every physics-applied command or source acquisition. Audit 04's repair now bounds/labels detector and manager queues, but those new outcome topics are not yet in the experiment CSV.

Headless and GUI launch expansion use identical assets, Gazebo version 6, resource variables and NVIDIA offload environment. Headless adds `-s --headless-rendering`; GUI does not at [gazebo.launch.py:98](../../src/sim/launch/gazebo.launch.py:98). No GUI process was started, so driver choice, rendered pixels, delivery rates and timing parity remain hypotheses requiring a dedicated paired capture.

## Exact active path and signal trace

| Stage | Executed source/artifact | Signal, frame and units | Time/rate | Online role |
|---|---|---|---|---|
| World | resolved installed symlink → `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf`, SHA `cb872f…`, world `warehouse_v2` | metres/radians; 0.001 s physics step; Sim 6.17.1 loaded DART | nominal real-time factor 1 | physics/rendering |
| Robot | expanded `warehouse_amr.urdf.xacro`, SHA `f590fc…`; spawned entity `turtlebot3` | collision body 0.800×0.550×0.270 m; bottom z≈0.070, top≈0.340; visual bounds x −0.406…0.407, y ±0.281, z 0…0.400 | spawn after first clock; active task pose `(−7.9,−8.7,0.05,0)` | physical actor |
| Map | profile `known_2d_regions`, 14 traversable rectangles; collision JSON from shell/occluder models | `map_bev`, metres; circle radius 0.485412 m; route lane clearance 0.55 m | static | global/local planning and offline geometry |
| Commands | planner `/cmd_vel_raw` → noise adapter `/cmd_vel` → native `/model/turtlebot3/cmd_vel` | unstamped Twist, m/s and rad/s; active bounds `[0,0.22]`, `[−1,1]` | publication 10 sim Hz; callback-driven noise; 0.5 sim-s watchdog | physical actuation |
| Clean odom | DiffDrive wheel-joint positions → native `/model/turtlebot3/odometry` → ROS `/odom` | parent `odom`, child `base_link`; metres/radians, m/s/rad/s | measured 50 sim Hz | startup, logger; not active filter prediction input |
| Noisy odom | `/odom` → encoder-noise integration → `/odom_noisy` | copies frames/stamp, forces pose z=0; covariance from noise propagation | callback-driven, nominal 50 Hz | active estimator prediction |
| Robot TF | expected native `/model/turtlebot3/tf`; launch listens to `/odometry_tf` | intended `odom→base_link`, conflicts with RSP `base_footprint→base_link` | native count matched 50 Hz odom; ROS bridge count 0 | currently disconnected |
| Cameras A–E | resolved external-camera model SDFs, hashes in asset results | RGB 1280×720, HFOV 1.5708; SDF +X ray converted directly to model optical coordinates | declared 5 sim Hz; ROS bridge per camera; detector depth 1 | online observations |
| Ground truth | native `/world/warehouse_v2/dynamic_pose/info` → `/ground_truth_tf` | world pose of entity `turtlebot3`; bridged child transform stamp zero | native observed median 0.017 s; logger uses receipt sim clock | **offline logger/reference only** |
| Contacts | 46 per-link native sensor topics → merged `/world_contacts` | source-stamped collision entity/link IDs and world points | measured nonempty at 1 kHz; healthy clear is silent | explicit online physical termination + evidence |
| LiDAR/IMU/joints | active config disables LiDAR and scan bridge; URDF has an IMU link but no active sensor; no joint-state publisher | no active sensor signals | none | none |

The camera manager reads `/odom_noisy` and direct SDF camera models. Clean `/odom` is used for startup and diagnostic logging; it is not also applied to the active filter. Gazebo DiffDrive odometry is wheel-position integration, not ground truth. Ground truth is allowed only for offline reference/geometry logging and is explicitly rejected as an online geometry-termination source.

At the selected spawn, DiffDrive odometry begins near local `(0,0,0)`. The static `map_bev→odom` transform carries the task start translation/yaw, so the registered manifest's first frame-sanity record maps that origin to `(-7.9,-8.7,0)` within floating-point error. This frame convention is coherent for numeric odometry even though the robot TF branch is currently disconnected.

## Reset, restart and shutdown state

| Operation | State actually cleared | State not established as cleared |
|---|---|---|
| `reset.all` | simulation time rewinds | robot pose, wheel odometry origin, DiffDrive target; ROS/DDS queues; command/noise/filter/planner/camera/contact/TF/logger buffers |
| Full process restart | fresh Python node state and random/process session IDs where implemented | historical run identity unless assets/epoch are bound; final physical zero unless acknowledged before teardown |
| Physics pause | simulation time and watchdog time hold | previously applied DiffDrive target; wall queues/transport state |
| Clock-bridge loss | ROS simulation clock stops | physics and previous command continue |
| Adapter restart | new adapter starts disarmed | downstream Gazebo target continues until another command reaches it |
| Logger completion | summary write and delayed launch shutdown requested | producer quiescence, final rows/contact count, physical rest; audit 10 owns full finalization repair |
| Detector/manager exit, current tree | audit 04 added launch `Shutdown` handlers | hard failure delivery ledger and acknowledged physical stop remain open |

## Prior regressions and repair status

- **Planner/actuator angular bounds:** audit 03 documents the prior tracker ±1.5 versus adapter ±1.0 mismatch and its repair. The evaluated active path now resolves both planner/controller and adapter to ±1.0 rad/s. This does not repair final-actuator liveness.
- **Contact liveness:** publisher count and message count were documented repairs, and registered summaries contain both. The known-clear/known-collision probe shows why publisher count plus silence still cannot distinguish a healthy empty sensor from a dead upstream channel.
- **Ideal-motion tracking failure:** audits 03/09 reproduce the controller/tracking problem under ideal motion, so it cannot be attributed to encoder or command noise. This audit did not change noise to improve the outcome.
- **Source/install mismatch:** the current install resolves to the inspected source and all selected file hashes verified. Run manifests still lack complete physical/render asset attestation, so current agreement does not repair historical provenance.
- **Clock duplicate traffic:** during this audit, a concurrent repair changed the throttle to publish changed timestamps at 50 wall Hz and repeat an unchanged timestamp only once per second. The current callback probe outputs `[100,100,1]` rather than the frozen baseline's repeated `[100,100,100,100,1]`; `tests/sim/test_clock_throttle_traffic.py` passes. Cached-source freshness and reset epoch remain unresolved.
- **Camera clock rewind:** audit 04's current producer/manager source fails the process on a backward clock jump and launch shuts down. It deliberately supports coordinated restart rather than hot reset. New event topics are not retroactive evidence for selected runs.

## Verification performed

- Existing requested simulator/geometry tests plus command watchdog: **16 passed in 3.62 s**; exact command/output in [existing_tests.txt](../../experiments/runtime_integrity/simulator_world_20260906/existing_tests.txt).
- Current clock-throttle follow-up: **1 passed in 0.10 s** in [clock_followup_tests.txt](../../experiments/runtime_integrity/simulator_world_20260906/clock_followup_tests.txt).
- `verify_world.py`: peak **211**, shipout **177**, paired **8** checks, all pass in [world_verify.txt](../../experiments/runtime_integrity/simulator_world_20260906/world_verify.txt).
- `world_freeze.py`: `WHV2-GEOMETRY-V2` pass in [world_freeze_verify.txt](../../experiments/runtime_integrity/simulator_world_20260906/world_freeze_verify.txt).
- Independent geometry: all 46 selected direct collision-prism bounds agree exactly; selected start exact-body clearance to those prisms is 0.58 m and its full sampled turn remains clear. The later included-prop probe supersedes any inference that these 46 boxes are the whole physical scene.
- Private physics: one clear and one known-contact fixture, 2,500 steps each; native pause/reset/TF topic probe; real ROS/native watchdog failure probe. Raw native streams are retained and SHA-256 bound in their summaries.

## Repair order and ownership

1. **14 + 13:** attest the complete resolved collision scene, add included props to collision/contact contracts, and gate launch on world/robot/camera/runtime identity.
2. **03 + 14:** establish an acknowledged final-actuator stop with independent liveness and a defined pause policy.
3. **02 + 04 + 14:** replace hot reset with coordinated epoch restart, or carry an epoch through every buffer and event.
4. **10 + 14:** add raw, source-identified contact/liveness/applied-command records and an atomic final drain.
5. **09 + 10:** keep contact, exact-body clearance, circular planner clearance, operational termination and physical rest as separate outcomes.
6. **13 + 14:** lease partitions/domains/log destinations and eliminate global cleanup.

No broad experiment should begin until ranks 1–6 have either been repaired and source-frozen or explicitly excluded by a protocol that restarts the complete stack and verifies the physical/contact scene before motion.

## Acceptance follow-up — 2026-09-07 profile-hash mismatch

The later full-suite failure in `tests/test_icra_geometry.py` is a **real provenance refusal caused by unrelated live-registry drift, not evidence that the 2026-08-31 camera geometry changed**. The frozen capture manifest records `world_profiles_sha256=3559f0a78b562bcad7cb7cbb476acfcc48442e14eedb017e8002fb1ae3f4f5fb`. That hash exactly matches the repository `HEAD` profile. The current installed/source symlink resolves to modified bytes with SHA-256 `742cb32c69ea9f48ecdf1ce899e9bf8cd45949be863312369b85cb67a79405cd`.

The only YAML diff is a new `warehouse_v2_low_cpu.world.sdf` entry inheriting the existing `warehouse_v2` anchor and setting `evidence_role: performance_development_only`. Parsed comparisons establish that global camera intrinsics, `warehouse_v2.world.sdf`, and `warehouse_v2_shipout.world.sdf` are semantically identical before and after the edit. Their canonical JSON hashes remain `0a3f9774…` and `1bac9d0c…`. The active physical world still has SHA-256 `cb872fd4…`, and the capture manifest's five poses, image dimensions and capture index are unchanged. Loading the exact profile blob whose hash is frozen reconstructs all five cameras and `fx=fy=639.997649…` successfully.

The new low-CPU SDF intentionally changes physics 1000→200 Hz and contact declarations 60→20 Hz while retaining all other world text. Its focused contract tests and the ordinary warehouse world tests pass, **6/6**. The geometry test fails before its round-trip assertions because `derive_interpretations.camera_models` compares the hash of the entire current multi-world registry at [derive_interpretations.py:62](../../experiments/camera_observation_characterization/derive_interpretations.py:62). Therefore an unrelated added profile makes historical reconstruction unavailable even when every consumed capture field is unchanged. This is strict behavior by design and must not be bypassed by refreshing the frozen manifest.

**Smallest sound reconciliation:** preserve the manifest and its hash; archive the exact `3559f0…` profile bytes as a content-addressed capture-provenance file, and make historical reconstruction resolve and verify that immutable blob. Keep a separate `live_source_changed` result so integrity audits still report current-registry drift. Future capture manifests should embed or hash a scoped immutable camera/world contract containing the exact consumed intrinsics, poses, driveable regions and world identity, rather than hash a mutable registry containing unrelated worlds. If the low-CPU work is not retained, simply removing its uncommitted profile entry also restores this test, but that is not a general provenance repair and would break its current fast campaign configuration.

## Correctness repair follow-up — 2026-09-07

The following current-source repairs supersede the corresponding open findings above. They do not change the frozen world geometry, camera extrinsics, detector/model settings, physics rate, declared contact rate or noise parameters.

- **Final simulator-side command guard (S14-02/S14-03).** A new Gazebo Sim 6 system plugin in [command_guard_system.cc](../../src/sim_command_guard/src/command_guard_system.cc) is the sole publisher to the DiffDrive input. ROS commands enter `/model/turtlebot3/cmd_vel_guard_input`; the guard forwards the newest fresh command to `/model/turtlebot3/cmd_vel_applied`, emits startup/timeout/reset zeros, uses advancing `UpdateInfo.simTime` for the 0.5 s watchdog, rejects a command retained across a longer wall-clock pause, and latches at zero after a simulation-time rewind until full process restart. Invalid timeout configuration now fails plugin startup. Its JSON outcome explicitly says `forwarded`/`forwarded_zero` and `physical_application_verified:false`; it is not mislabeled as wheel application. The state machine and five regression cases are in [test_command_guard_core.cc](../../src/sim_command_guard/test/test_command_guard_core.cc).
- **Command-adapter lifecycle (S14-02/S14-03).** [actuation_noise_node.py](../../src/sim/sim/actuation_noise_node.py) publishes exact zero at startup and shutdown, clears correlated noise on time rewind, and latches at zero after rewind. The native guard remains authoritative if the adapter, ROS clock bridge or planner dies while physics continues.
- **TF/joint and readiness contracts (S14-04/S14-05).** The active AMR now configures DiffDrive `odom→base_footprint`, explicitly publishes native `/model/turtlebot3/tf`, and publishes wheel joint state at 50 Hz in [warehouse_amr.urdf.xacro](../../src/sim/robot_description/urdf/warehouse_amr.urdf.xacro). [bringup_sim.launch.py](../../src/sim/launch/bringup_sim.launch.py) bridges those exact topics. The clock gate requires three strictly advancing stamps, the odometry gate requires advancing stamps, finite state, normalized-enough quaternion and exact `odom`/`base_footprint` frames, and a failed gate no longer starts the next action.
- **Reset policy (S14-03/S14-04).** `reset_world:=true` now aborts before simulator startup with the documented Sim 6 state-retention reason. A fresh launch is the supported epoch boundary, and ordinary bringup no longer exposes the unsafe ROS `ControlWorld` reset bridge. The standalone reset client now returns failure for a negative service response or exception instead of reporting success when it is used with a separately configured service bridge.
- **Physical collision-scene closure (S14-01).** [occlusion_geometry.py](../../src/unav_common/unav_common/occlusion_geometry.py) resolves explicitly registered `model://` includes, transforms box/cylinder/DAE collision assets, and filters collision prisms to the AMR body-height slab. Module 13 keeps the exact five prop instance names in a runtime registry outside the frozen warehouse profile and passes them into collision serialization. This adds the forklift, pallet jack, bin and two loose pallets to planner clearance without changing their physical assets or the frozen profile hash.
- **Contact identity and silence (S14-01/S14-06).** The active AMR has a 60 Hz body contact sensor on its converted chassis collision. It supplies contact evidence against included props that have no world-side sensor. The bridge now registers 47 exact native source topics, including the robot sensor. [contact_evidence_node.py](../../src/sim/sim/contact_evidence_node.py) publishes `/sim/contact_channel_status` with epoch, event ID, configured source identities, delivery counts and source/receipt times. Before a positive delivery its state is `configured_silent_requires_positive_control` and `silence_is_no_contact=false`; silence is never promoted to evidence of a clear run.

The isolated active-world command probe used ROS domain 217 and Gazebo partition `module14_guard_measure`. Six 0.2 m/s commands ended at simulation stamp 16.299 s; the native timeout zero was forwarded at 16.800 s, exactly 0.501 simulation seconds later. Odometry moved 0.0864 m from the final-command vicinity to the terminal sample, below the 0.1 m distance implied by 0.2 m/s for 0.5 s. This is separate physical stopping evidence; the guard record alone only proves forwarding.

The known-contact probe used domain 219 and partition `module14_contact_known`, spawning the AMR at `(11.8,0,0.05,0)` against the east wall. A raw event at 14.342 s identified collision entity 204 as `warehouse_shell::wall_east::collision` and entity 999 as `turtlebot3::base_footprint::base_footprint_fixed_joint_lump__base_link_collision`, with four world contact positions. The concurrent liveness record named all 47 configured sources and reported `silence_is_no_contact=false`. Exact results are retained in [repair_probe_results.json](../../experiments/runtime_integrity/simulator_world_20260906/repair_probe_results.json).

Final requested simulator/world/geometry validation is **33 passed**. The native package builds successfully and reports **5/5 guard cases passed**. Headless active-world runtime advertised the corrected guard input/output/outcome, odometry, TF and joint-state native topics. The original GUI/headless rendered-pixel parity hypothesis and the camera A–E link/optical TF limitation remain simulator limitations requiring paired rendering/calibration work; no commissioned camera extrinsic was changed here. Gazebo Fortress still publishes nonempty contact sensors at physics cadence rather than the declared 60 Hz, so raw contact volume remains a measured backend limitation rather than a reason to alter the frozen rate.
