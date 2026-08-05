# Meerhoven new-world commissioning plan

Status: **commissioning in progress; exploratory until all gates pass**  
World: `src/sim/gazebo_worlds/worlds/warehouse_meerhoven.world.sdf`  
Initial SHA-256: `6f133ca74b4d007d57e3c9692a7d56e36dcf1181dff06696d06663f5816dac62`  
Accepted post-camera-D-correction SHA-256: `648da844b574c7ff83a46e33a357c4eb5faa1f11b53a58a60c645c465433ebc8`  
Code baseline: `fc0c670cfb93fa8d29ce82e8e77a1b0d9a4a9f95`

## Scientific role

Meerhoven is a **second, exploratory evaluation world**. It does not replace the compact
`warehouse_aws.world.sdf` B1 benchmark, which remains the paper core. Meerhoven tests the
broader infrastructure-localization story: a brownfield logistics facility, an inherited
and heterogeneous camera network, real 3-D occlusion, blind aisle segments, overlap,
handover, and per-camera commissioning at larger scale.

No Meerhoven result is paper-ready until the complete world -> detector -> projection
calibration -> learned reliability -> frozen configuration -> seeded logs -> figures ->
wording chain below passes. Old `warehouse_full_4cam` detector/calibration/GP artifacts
remain valid only for that old world and must never be reused as Meerhoven evidence.

## Dependency-ordered plan and gates

### 1. Freeze the physical contract

- Treat `exp4_meerhoven_hub.py` as the layout source and `exp5_build_world.py` as the SDF
  generator. Do not hand-edit the generated SDF.
- Record hashes for the generated world, all included camera models, robot model, launch
  files, and any AWS meshes used during capture.
- Freeze camera names, A--L IDs, poses, intrinsics, image sizes, rates, and topic prefixes.
- Keep 2-D traversability, 3-D visibility/occlusion, and learned observation reliability
  as separate layers.

Gate: SDF regeneration is deterministic; plan-vs-SDF geometry agrees; all collision and
visual assets resolve locally.

### 2. Fix and verify the simulator contract

- Use the ROS Humble-compatible Gazebo Fortress path (`ign gazebo`, Gazebo Sim 6), not the
  separately installed Harmonic `gz sim` binary. A Harmonic server can advertise topics
  while the Humble bridge receives no images.
- Export both Fortress-era `IGN_GAZEBO_RESOURCE_PATH` and the forward-compatible
  `GZ_SIM_RESOURCE_PATH` to the same frozen model/world roots.
- Launch through `ros_gz_sim`, verify `/clock`, set-pose/control services, odometry, ground
  truth, and contact bridges, then spawn the TurtleBot at a collision-free pose.
- Fail preflight if a wrong simulator generation is selected or another Gazebo/ROS domain
  can leak frames into capture.

Gate: the expected Fortress server owns the world; ROS receives live clock and fresh
camera messages; robot spawn/motion/contact behavior is valid.

### 3. Commission the twelve camera transports and geometry

- Extend the launch bridge, semantic capture bridge, detector camera registry, camera
  manager, and logger metadata from A--D to A--L.
- Confirm every RGB, camera-info, segmentation-label, and optional depth topic is unique.
  Use sensor-data/BEST_EFFORT QoS for images.
- Verify every camera through the shared projection parser, including heterogeneous height
  and pitch. Check optical axis, horizon, projected ground footprint, nearby self-occlusion,
  rack/column/mezzanine interference, and that the camera is not embedded in geometry.
- Capture a labeled 12-panel sheet from one running world. Reject/re-aim cameras whose
  operational area is mostly obstruction, sky/wall, or out of frame.

Gate: all 12 RGB streams publish fresh frames and each accepted camera view agrees with
its intended operational role.

### 4. Add the world-facing runtime configuration

- Add a Meerhoven `world_profiles.yaml` entry with exact bounds, camera list, direct-goal
  spawn/goal pads, traversable regions, forbidden regions, and occlusion annotations.
- Generate/validate the driveable layer from the same layout geometry, including walls,
  racks, columns, stacks, cage, machines, conveyor, mezzanine posts, guardrail posts, and
  robot clearance.
- Add camera A--L parameters to detector scheduling and manager launch paths. Require a
  complete matching calibration and GP set; do not silently fall back to old four-camera
  or constant artifacts.
- Build a clearly branded geometry-only A--L scheduling prior to break the pre-GP
  bootstrap dependency. It may select which camera to infer during commissioning, but it
  is not learned detector reliability and is forbidden as the final planner GP.
- Design fair C1/C2 tasks with one start and one goal. No mission waypoint chain or
  condition-specific route initializer may choose the route for the planner. Multistart,
  if needed, is identical across conditions and reported only as basin handling.

Gate: configuration parity tests pass, direct start/goal paths are collision-free, and
both conditions differ only in the intended observation-reliability model.

### 5. Capture a Meerhoven detector dataset

- Sample collision-free poses over the driveable union, stratified by camera, range,
  bearing, yaw, facility zone, and clear/partial-occlusion state.
- Synchronize RGB and simulator semantic labels after every teleport/settle; reject stale,
  duplicate, truncated, or projection-inconsistent samples.
- Include one geometry-certified, duplicate-guarded background negative per fixed camera
  (twelve camera-distinct backgrounds after pooling). Multiple off-screen robot poses in a
  static view are not independent images and must not be counted as additional negatives.
  Split by spatial pose block so headings from
  one position cannot leak across train/validation; reserve operational routes for a
  separate held-out evaluation.
- Audit counts and acceptance reasons per camera/range/zone before training. Small-footprint
  and structurally different cameras must not disappear in a pooled count.

Gate: minimum samples and acceptance rates pass for **each** commissioned camera, route
holdouts remain untouched, duplicate fraction is below threshold, and the manifest binds
the data to the frozen world/camera assets.

### 6. Fine-tune and gate YOLO

- Fine-tune from the existing clean four-camera detector because robot appearance is
  unchanged; first measure one epoch, then select the smallest justified 3--6 epoch run.
- Retain training config, base-model hash, data manifest, software versions, best/last
  weights, curves, and wall-clock/GPU runtime.
- Evaluate bottom-centre localization and detection rate per camera, range bin, facility
  zone, and occlusion state on spatially and route-held-out data. Also benchmark scheduled
  single-image inference at the deployed resolution/rate.

Predeclared detector gate (fixed before training results are inspected): at confidence
0.25, every camera must achieve at least 0.75 mask-opportunity recall on its spatial
validation split; every camera/range cell with at least eight validation opportunities
must achieve at least 0.65 recall; matched detections must have median/p90 bottom-point
error no worse than 20/60 px; and the twelve geometry-certified camera-background frames
must produce zero false positives. The deployed single-image scheduler must sustain at
least 2 Hz with p95 correction age no worse than 0.50 s on the commissioning host.

Gate: those per-camera, range-conditioned, localization, background and timing thresholds
pass. If any camera fails, improve its dataset/view or exclude it explicitly; do not
continue to GP fitting with an undocumented weak detector.

### 7. Refit projection calibration

- Record commissioning traverses with synchronized detections, camera IDs, raw pixels,
  timestamps, odometry/belief, and an evaluation-only reference trajectory.
- Fit along-bearing and gated cross-bearing constants per camera using the canonical
  projection implementation. Never fit calibration online or expose truth to runtime.
- Validate on a held-out traverse: signed residuals, median/RMS point error, bias transfer,
  cross-bearing gate outcome, NIS/NEES, and empirical covariance coverage.

Gate: every deployed correction is supported by held-out improvement and the gate; cameras
with unresolved or harmful corrections stay raw by default.

### 8. Refit learned observation reliability and planning covariance

- Build observation opportunities and hit/miss events from the accepted detector and new
  calibration. Separate availability, conditional quality, systematic residual, timing,
  and health terms.
- Fit one spatial model per camera and a conservative fused planning artifact using the
  canonical `fit_belief_aware_gp.py`; include `P_conservative_plan_map` and exact camera IDs.
- Use spatially blocked/route-held-out validation and report calibration, Brier/log loss,
  AUROC, coverage and sparse/out-of-bounds behavior with shared metrics only.
- Confirm that no `gt_*`, `eval_*`, CAD visibility truth, or future frame evidence enters
  the runtime provider.

Gate: artifact audit passes, every map matches world bounds/camera IDs, and conservative
planning reliability is empirically justified rather than copied from geometric coverage.

### 9. Offline planner and optimizer gates

- Plot known forbidden zones, camera geometry, learned reliability, planner-facing
  covariance, ambiguity, goal, control, boundary, and total EFE terms separately.
- Run geometry checks and open-loop rollouts from identical starts for C1/C2. Check horizon,
  covariance propagation, stopping behavior, local optima, multistart neutrality, and
  command timing before starting Gazebo campaigns.
- Choose tasks that can reveal visible detours, delayed commitment, handover, or a safe
  stop without demanding a particular outcome.

Gate: no route forcing, no collision in offline rollout, no hidden fallback, acceptable
solver timing, and an interpretable observation-model effect exists before expensive runs.

### 10. Fresh seeded Meerhoven runs

- Freeze campaign YAML with world/detector/calibration/GP hashes, camera set, code commit,
  task, seed, condition, solver settings, launch arguments, and transport environment.
- Run matched C1/C2 seeds. Record detections, selected camera/handover, belief, truth,
  covariance, EFE terms, commands, contacts, runtime resources, and terminal status.
- Compute navigation/localization metrics only at or after the first non-trivial velocity
  command. Report launch wait, warm-up, and global-solve time separately.

Gate: repeated seeded runs finish without stale topics, hidden fallbacks, route scripts, or
missing provenance; metrics and logs agree on completion/collision/localization status.

### 11. Figures, framing, and evidence promotion

- Produce the 12-camera sheet, world/driveability/coverage overview, detector gate panels,
  calibration residuals, GP/uncertainty maps, C1/C2 trajectories, observability/belief
  error, camera handovers, and decomposed EFE terms.
- Use Meerhoven to support the assistive-infrastructure framing: heterogeneous inherited
  cameras need an honest spatial service contract and explicit handover/calibration. Do not
  claim battery savings, real-warehouse deployment, multi-robot coordination, or universal
  route-choice improvement.
- Keep results tagged exploratory until all gates above pass. Only then update the evidence
  registry and prepare paper-ready wording/figures. TeX remains untouched unless explicitly
  authorized.

## World-coupled artifacts that must not be reused silently

| Dependency | Why it changes with a new world/camera rig |
|---|---|
| SDF, model assets, robot spawn and collision map | define the physical/rendered scene |
| camera IDs, poses, intrinsics, image rates and topics | define pixels and projection |
| semantic captures and train/validation split | define the detector distribution |
| YOLO weights and thresholds | depend on viewpoint, range, scale and background |
| projection correction and conditional covariance | depend on camera pose and residuals |
| opportunity/event tables and GP maps | depend on geometry, detector and calibration |
| driveable/forbidden layer and planner grid | depend on physical obstacles and clearance |
| tasks, starts/goals, horizon and optimizer diagnostics | depend on scale/connectivity |
| campaign YAML, launch arguments and logger schema | bind the runtime evidence chain |
| figures, metrics, captions and claims | must describe the exact artifacts above |
