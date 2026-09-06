# 05 — Observation geometry and camera calibration

Reviewed 2026-09-06 against the shared working tree. **The configured NN reference observation agrees with its frozen offline implementation, including the actual floor projection and full camera covariance. The main current defect is insufficient validation of the incoming observation's geometric meaning. Several additional defects are confined to optional paths or unsupported geometry.** No nominal recorded-box correction-order mismatch was found. This is a software audit, not evidence of empirical calibration quality or navigation improvement.

## Scope, provenance and reachability

Read first: repository `AGENTS.md`, `PLAN.md`, `docs/localization_metrics.md`, the registry, `open_questions.md` and `runtime_integrity_audit.md`; then `ICRA_STATUS.md`, the module investigation map and existing state review. The September 6 update supersedes PLAN's preserved “only active stage” text. Pre-August-25 results are not reused.

The anchor is [network_navigation_runtime_pilot.yaml](../../experiments/icra_commissioning/network_navigation_runtime_pilot.yaml): `warehouse_v2`, five cameras, native strict YOLO, original images 1280×720, inference size 960, masks off; `observation_model=learned_nn`, `covariance_profile=commissioned_reference_r`, fixed offset zero, `use_pixel_correction=false`. Launch sets `map_bev` and constructs camera/include pairs by ID: [visibility_launch_common.py:1776](../../src/experiments/experiments/core/visibility_launch_common.py#L1776). The manager disables silhouette correction for non-hull models at [camera_manager_node.py:1117](../../src/reliability/reliability/nodes/camera_manager_node.py#L1117). Hull **admission** remains enabled. Compiled detection-only inference is excluded by this launch at line 1690.

The working tree contains substantial other work. No runtime code, launch settings, weights, calibration, manifests or experiment outputs were edited. Audit 04 confirmed ownership of detector scheduling and supplied its [report](04_camera_acquisition_and_batching.md). Only uniquely named `05_*` audit files were created. Process inspection in this execution environment did not establish host-wide live-process state; no processes were launched, stopped or cleaned up. Source and artifact hashes are in [probe results](05_geometry_probe_results.json); the live check names an exact registered run and validates its selected input-file hashes.

## Ranked findings

### G01 — P1: current input-contract defect; malformed semantics reach active observation construction

**Files:** [contracts.py:138](../../src/reliability/reliability/contracts.py#L138), [contracts.py:489](../../src/reliability/reliability/contracts.py#L489), [manager:1178](../../src/reliability/reliability/nodes/camera_manager_node.py#L1178), [manager:1311](../../src/reliability/reliability/nodes/camera_manager_node.py#L1311), [learned_box_correction.py:97](../../src/reliability/reliability/learned_box_correction.py#L97).

**Trigger:** a finite, schema-valid message has a foreign `image_frame_id`/`calibration_id`, or `pixel_uv` differs from its declared bbox bottom centre. Degenerate/reversed boxes also pass the shape-only contract. Projection consumes `pixel_uv`; NN feature construction independently computes `(u,v)` from `bbox_xyxy`. Thus one observation can combine two different pixels without an error.

**Expected:** reject incompatible image/calibration identity and inconsistent selected-pixel semantics before projection. In bbox mode require positive ordered extents and a declared image geometry. **Observed:** using the actual `_map_observations`, active NN/R settings, admission enabled and no prior, a 20-pixel horizontal inconsistency produces a valid metric observation; foreign frame and calibration labels are ignored. Zero-size and reversed boxes also produce metric observations in the synthetic probe. The malformed-box examples additionally retain the original pixel, deliberately demonstrating that neither relation is enforced. Their output is not claimed to survive network quorum.

**Reachability/consequence:** the callback checks schema and topic camera ID but not these semantic identities. Native selection rejects ordinary reversed/zero boxes; its normal producer constructs consistent fields. No malformed labels/pixel pairs were found among the 1,112 joined logged readings checked below. This is a confirmed current boundary defect under malformed/misconfigured input, **not** a diagnosed nominal pilot failure. With a prior, hull admission rejects many bad boxes; at startup geometry validation is deferred to quorum, which is a different property. Multiple mutually compatible malformed readings can still look like good measurements.

**Smallest repair:** validate camera-specific image/calibration identity and selected-source consistency at ingestion, with explicit legacy exceptions only for legacy modes. Require positive box dimensions and half-open coordinate bounds; attach/check original image dimensions or a validated calibration digest. Do this independently of the admission policy. Add mutation regressions for each field and ensure no `MapObservation` is constructed. Coordinate manager edits with 04/06/07; do not alter quorum thresholds to mask invalid inputs.

### G02 — P1: current configuration-integrity gap; artifacts are not bound to projection geometry

**Files:** [manager:794](../../src/reliability/reliability/nodes/camera_manager_node.py#L794), [learned loader:57](../../src/reliability/reliability/learned_box_correction.py#L57), [reference_calibration.py:12](../../src/reliability/reliability/reference_calibration.py#L12), [projection.py:49](../../src/reliability/reliability/projection.py#L49).

**Trigger:** camera include/extrinsics change while camera ID and NN/calibration bytes remain unchanged. The manager checks only equal list lengths before zipping IDs/includes. The NN uses its embedded camera XY/yaw/dimensions; raw IPM uses the separately constructed world camera. ReferenceCalibration validates the NN hash, semantic reference/frame/order and full SPD R, but not the world, K/R/t, image geometry or robot geometry. The learned loader stores the free-text `target` without validating it.

**Expected:** fail at startup if the commissioned feature/projection geometry or required physical reference changes. **Observed:** replacing camera B's projection model with camera A while retaining B's frozen NN and R yields a finite measurement shifted by **8.370707 m** for the same recorded box, with no error in actual observation mapping. This is a deterministic configuration mutation, not a measured camera error. Altered camera lists of the same length are not intrinsically safe.

**Boundary:** the selected launch already derives include names from an ID map and rejects unknown requested IDs. All 120 processing-order permutations preserved the tested reading exactly; the recorded five-camera projection matches the frozen table. Therefore this is a missing startup equivalence check, not evidence that the current launch accidentally swaps cameras. Audit 04 separately reproduced malformed per-chunk detector result counts shifting camera association; fixing only manager mapping cannot detect a wrongly labelled box upstream.

**Smallest repair:** validate canonical per-ID projection geometry and original image dimensions against frozen commissioning metadata; bind calibration to a projection/robot-reference digest, and validate target semantics. Use the current artifacts unchanged for current experiments. A future metadata export can record identity without refitting. Camera order is data, not calibration identity; retain keyed selection. Add startup tests for shifted camera, altered height/pitch, FOV/resolution and duplicate IDs.

### G03 — P2: confirmed library defect for unsupported rays; no nominal active-image trigger established

**Files:** [camera_model.py:63](../../src/unav_common/unav_common/camera_model.py#L63), [projection.py:83](../../src/reliability/reliability/projection.py#L83), [projection.py:113](../../src/reliability/reliability/projection.py#L113).

**Trigger:** a pixel above the floor horizon, nonfinite pixel, or ray arbitrarily close to the horizon. `pixel_to_world` checks only the homogeneous denominator's absolute magnitude, never forward ray distance, finiteness or image support. Its wrapper promises rejection behind the camera but delegates to this method.

**Reproduction:** for camera A, horizon is v≈−257.983 at u=640. Ten pixels above it returns approximately (−452.111, −450.112) m; `pixel_to_world_at_z(...,0)` correctly returns None. One micro-pixel below it returns coordinates around 4.37 billion metres. NaN returns a tuple of NaNs in both helpers. The input contract blocks NaN on the active route, and the selected downward cameras have the horizon above the image: these are unsupported-input/library defects, not normal 1280×720 detections.

**Consequence:** plausible finite coordinates can escape a utility advertised as physically valid; changing camera tilt or accepting out-of-image pixels makes the risk operational. Near-horizon numerical Jacobians can also become meaningless.

**Smallest repair:** use one finite, forward ray/plane intersection with explicit support checks; keep image support and physical ray validity separately named. Reject unsupported geometry with a reason before covariance/NN inference. Define an explicit numerical conditioning bound; do not introduce an empirically tuned range gate under the name of numerical repair.

### G04 — P2: confirmed optional single-camera covariance-axis error

**Files:** [pixel_to_bev.py:35](../../src/state/state/core/pixel_to_bev.py#L35), [pixel_to_bev_state_node.py:307](../../src/state/state/nodes/pixel_to_bev_state_node.py#L307), [node:368](../../src/state/state/nodes/pixel_to_bev_state_node.py#L368).

**Trigger:** oblique camera with both pixel axes contributing to world X/Y. The helper labels the **norms of Jacobian columns** as world-axis standard deviations. Actual world marginal variances require row norms of `J`, and generally a nonzero XY covariance. With transform noise enabled, three independent perturbed camera draws are used for centre/u/v rather than one coherent geometry.

**Reproduction:** camera A at (640,400), pixel sigma 0.001: helper returns (1.05634e−5,1.42828e−5) m; independent world marginal standard deviations are (1.25615e−5,1.25615e−5) m. The small sigma isolates the derivative error from nonlinear finite steps. The node uses these values in its covariance publication.

**Reachability:** optional single-camera state transformer, replaced by manager in the active multicamera launch. The active projection correctly computes full `J R_uv Jᵀ` and retains cross terms.

**Smallest repair:** return full world covariance from the same geometry/Jacobian; preserve cross terms in the ROS covariance. Treat calibration uncertainty as a separate coherent perturbation/Jacobian contribution rather than resampling while differentiating pixel noise. Do not change active constant R.

### G05 — P2: confirmed optional hull condition-limit implementation mismatch

**Files:** [silhouette_observation.py:126](../../src/reliability/reliability/silhouette_observation.py#L126), [silhouette_observation.py:222](../../src/reliability/reliability/silhouette_observation.py#L222).

**Trigger:** anisotropic observation Jacobian. `_inverse_2x2` compares `||H||_F²/|det H|` against `2*max_condition`, despite claiming it only rejects more conservatively than a spectral condition test. Since that expression is κ+1/κ, a stated bound 20 accepts κ approaching 40.

**Reproduction:** diagonal H=(30,1) is accepted at `max_condition=20`. **Expected:** refuse κ=30. **Consequence:** optional hull inversion can amplify noise beyond its declared limit. Actual current NN observation does not invert H. No active hull pose with this conditioning was established.

**Smallest repair:** check the actual 2-norm condition number or compare κ+1/κ with `limit+1/limit`. Preserve the declared limit. The optional hull's fallback also retains raw silhouette landing as a centre measurement if correction fails ([manager:1414](../../src/reliability/reliability/nodes/camera_manager_node.py#L1414)); refuse or explicitly type that observation rather than silently mixing reference points. That fallback is documented behaviour, but physically incompatible with an unconditional centre interpretation.

### G06 — P2: confirmed optional compiled-backend half-open mismatch

**File:** [batched_four_camera_yolo_node.py:710](../../src/perception/perception/nodes/batched_four_camera_yolo_node.py#L710).

Compiled detection-only clipping uses [0,width−1] and [0,height−1] for both endpoints. A half-open full-image box [0,0,1280,720] becomes [0,0,1279,719]; its bottom moves by one pixel and centre by half a pixel. Native selection uses returned continuous xyxy coordinates and bottom v=y1 ([yolo_selection.py:130](../../src/perception/perception/core/yolo_selection.py#L130)).

The active launch explicitly refuses this compiled backend, so this is **not** the current pipeline's resize bug. It is established by endpoint arithmetic/source inspection, not an executed compiled inference comparison. Smallest repair: clip exclusive maxima to width/height, and regression-test inverse letterboxing with boundary boxes against the frozen native convention before enabling this backend. Audit 04 owns backend execution tests. No detector rerun or NN retraining was performed.

### G07 — P3: numerical floor can still fail at extreme scale; different from the repaired bias floor

**File:** [projection.py:179](../../src/reliability/reliability/projection.py#L179).

`_floor_spd_2x2` adds absolute jitter using a cancellation-prone analytic minimum eigenvalue. For `[[1e8,1e8],[1e8,1e8]]` and floor=1e−12, the addition rounds away; returned eigenvalues are (0,2e8). Expected an SPD result or explicit refusal, not a singular matrix advertised as floored.

The active profile replaces this projected covariance with frozen SPD R, though it still invokes projection first. Normal image geometry in the probes does not produce this extreme matrix. Treat as numerical robustness outside established support, not recurrence of the previously repaired one-zero-axis bias-floor configuration bug. Smallest repair: finite/conditioning checks and a representable relative numerical floor with a Cholesky postcondition, or explicit refusal. Do not inflate empirical camera R to fix it.

## One unchanged live detector box, end to end

The trace comes from registered `fusion_network_traverse__P0__seed210`, run `experiment_20260906_213015`, selected by the runtime-pilot selection, not by directory age. [Live trace results](05_live_trace_results.json) record the full path, selection hash, batch ID and exact values. It is a logged candidate, not a claim that quorum/fusion used it. The run was opened through `aligned.rows`; no truth/error/NEES score was computed.

Camera B has SDF position (−1.5,−9.72,5) m and (roll,pitch,yaw)=(0,0.8378,2.0071) rad. Its optical axes are +X image-right, +Y image-down, +Z forward; SDF sensor forward is local +X. K uses f=640/tan(1.5708/2), principal point (640,360), square pixels, no distortion. Plane z=0 is `base_footprint` floor reference. World XY is labelled `map_bev`; there is no further world→map transform in this selected setup.

| Quantity and owner | Physical reference, frame and units | Required inputs and time |
|---|---|---|
| Detector xyxy | Silhouette rectangle, original camera-B image; [1.519744873,371.282958984,128.508926392,459.311767578] px | Captured RGB, frozen YOLO and selected score; capture 3.400 s, not publication time |
| Pixel interpretation | Bbox bottom-centre (65.014335632,459.311767578) px; no single material point on the robot is guaranteed | Same xyxy, masks disabled; `calibration_id=warehouse_v2_camera_B`, image frame `camera_B` |
| Raw IPM | (−7.688169242,−8.998425588) m, world/map_bev floor **ray landing**, not robot centre | B camera K/R/t, z=0 and selected pixel; same capture stamp |
| NN features | Range and inverse range: m and m⁻¹; box fractions/aspect, normalized u/v, cos/sin of bearing, score and camera one-hot: dimensionless | Raw landing relative to embedded B XY; embedded B yaw in radians and 1280×720 dimensions; no robot pose/yaw/truth |
| Frozen NN correction | Two offsets in metres, along camera→raw ray and its left normal; added once to raw world XY | Packaged StandardScaler followed by MLP; frozen feature order and camera ID; no separate runtime normalization |
| NN output | (−7.907289159,−8.672935429) m, estimated robot ground reference XY | Training target is `robot_x/y − raw_xy`, resolved in the raw ray basis; runtime adds that correction, not subtracts it |
| Residual subtraction | b_B=(−0.020673684,−0.039925005) m in map_bev; z=NN−b=(−7.886615475,−8.633010424) m | Frozen B entry bound to NN checkpoint hash; applied exactly once after NN |
| Camera covariance | R_B=[[0.033313104,0.012617475],[0.012617475,0.279090819]] m²; residual covariance of the reference observation, with XY cross term | Frozen per-B constant full matrix. Replaces provisional projected pixel R; is not added to it or pushed through an NN Jacobian |
| `MapObservation` | Camera B, same reference XY/R, timestamp 3.400 s | Manager construction retains contract timestamp; later common-time alignment/fusion belongs to 07 |

Image receipt=3.427 s; inference start/finish and detector publication are logged as 3.476 s; logger receipt=3.693 s. Equal logged inference stamps do not establish zero wall-time inference: scheduling/log-clock meaning is investigation 04. Every observation-derived geometric value above describes the capture instant. Calibration/artifacts are static and have no per-frame timestamp. NN range is the raw landing's range; the diagnostic `range_m` in manager line 1499 uses corrected XY instead, so it must not be reused as the NN feature without recomputation.

The optional hull function instead predicts silhouette bottom-centre from `(x,y,yaw)` and CAD vertices, backprojects it to `h_ground`, differentiates at that same supplied pose and fixed yaw, and applies `z_eq=x_prior+H⁻¹(z−h)`, `R_eq=H⁻¹ R H⁻ᵀ`. This is a local 2D equivalence, not a full three-state heading model. The selected NN mean/R does not use this Jacobian. Admission obtains the nearest operational belief within a time tolerance ([manager:225](../../src/reliability/reliability/nodes/camera_manager_node.py#L225)), which may be newer than capture and is not interpolated to capture. This affects admission in the active path, and hull mean/Jacobian in optional mode. Both hull quantities use the same supplied pose, but it need not describe the observation instant. Recovery/admission policy and causal replay handling belong to 02/06; no Gazebo truth fallback was found in operational observation construction.

## Independent checks and evidence limits

Reproduce from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/05_geometry_probe.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/module_audits/05_live_trace.py
```

[Geometry probe](05_geometry_probe.py) uses actual projection and actual manager mapping, only bypassing the unrelated quality provider and ROS startup. It compares all 6,412 valid frozen table boxes: A=1,294; B=1,235; C=877; D=1,339; E=1,667. These are implementation-equality counts, not independent statistical replicates. Raw XY difference is exactly zero, maximum feature difference 3.55e−15, NN and manager XY difference 3.55e−15 m, and full R difference zero. Unlike the previous manager regression, projection is not mocked. No fitting function is called.

An independent SDF Euler-rotation/ray-plane oracle at four pixels per camera, including image corners and half-open outer endpoints, agrees within 2.85e−14 m. Analytic homography quotient derivatives agree with central differences within 1.60e−11 m/px. Full correlated pixel covariance propagation agrees within 2.40e−11 m² using a 0.001 px derivative step. This verifies Jacobian direction and XY cross terms. Optional hull step-size comparison at one smooth pose differs by 1.26e−9; inverse-Jacobian covariance agrees with NumPy matrix algebra within 1.39e−17 m². It does not test all piecewise silhouette switches.

[Live probe](05_live_trace.py) joins camera opportunities to pre-fusion rows by physical batch and camera, deduplicates and verifies raw projection, NN−bias and R against the actual logged outputs. On the one exact selected diagnostic run: **1,112 joined camera readings, zero raw/mean/R difference and zero checked pixel/calibration/frame inconsistencies.** It does not claim all delivered/rejected opportunities were logged or that admission/fusion/timing is equivalent; those boundaries have separate owners.

The fixed SDF camera poses all have zero roll, and the camera sensor assets inspected specify 1280×720 and horizontal FOV 1.5708. `camera_model_from_world` ignores roll and hard-codes those intrinsics; it does not parse nested sensor transforms or pose-relative frames. This is a supported-configuration limitation today and becomes a bug if reused for a rolled/different camera without rejection. Camera constructor also lacks explicit validation of coincident look-at, parallel up vector or invalid FOV/dimensions. Prefer explicit refusal for unsupported inputs before claiming general camera support.

`robot_hull.py` includes the base joint's 0.010 m once, wheel bottoms at z=0, skirt at 0.090 m, 0.35 m deck and 0.40 m rear cabinet. These match the named warehouse AMR xacro dimensions; `BODY_PRISM` is a deliberately different ablation. The capture defaults `robot_z=0` and training target uses the commanded robot XY, matching the base-footprint ground-reference XY in this setup. Physical settling, mesh/facet silhouette approximation, occlusion and whether learned targets transfer to real surveyed calibration remain model/commissioning questions. Sampled wheels/casters and partially behind-camera hull clipping are not exact silhouette geometry. No claim of physical-camera calibration accuracy follows from source equality.

## Regression history, unresolved limits and handoffs

- **Documented bootstrap repair:** active config uses 0.91 m; `test_camera_manager_node.py:321` reproduces two readings 0.60 m apart being rejected at 0.30 and admitted at 0.91. The original rationale concerns silhouette landing offsets. Current NN observations target the ground reference, so that rationale does not independently validate the active threshold. Preserve the configured value; 06 owns policy evaluation.
- **Documented singular bias-floor repair:** manager lines 935–946 reject exactly one positive slope at startup; `test_bias_floor.py:269` retains the singular example. Both slopes are off in the selected setup. This is distinct from G07's numerical projection floor.
- **Documented envelope repair:** strict reference calibration and receiver checks reject invalid frame/schema/asymmetric/indefinite/nonfinite R. Reference tests pass. They do not imply that upstream pixel/calibration semantics in G01 are validated.
- **Documented runtime/offline equivalence repair:** reference schema binds physical reference, frame, units, operation order and checkpoint hash; export preserves full R and refuses overwrite. The previous manager test mocked raw projection and the live audit started at logged raw XY. This report extends their scope to real projection and the unchanged logged box.
- **Verification:** [retained combined test output](05_existing_regressions.txt) reports **75 passed, 3 failed**. The three failures are command transaction regressions (fatal-stop publication, clock rewind retaining tape, stop diagnostic clearing), handed to 03; no geometry repair was attempted for them. The initial focused reference/NN/hull run passed 26 tests. Nonfatal installed pandas optional-dependency warnings were emitted. No missing-artifact skips occurred in that focused run.
- **Optional covariance semantics:** `covariance_mapping.py` is a pixel trust-to-R/planning mapping, not the selected constant metric residual R. Its unresolved 40/120 px miss endpoint is not an active camera-R inconsistency. Combining learned NN with projected raw pixel R in another profile would require the Jacobian of the complete corrected observation or separately fitted corrected residual R; this active reference profile already takes the latter approach.
- **Model limitations:** frozen R is conditional residual scatter from a static development population, not proof of sequential consistency, Gaussian tails, calibration under admission or independent camera errors. No R tuning, NN retraining, new observation method or ground-truth-derived online fallback was introduced.
- **Handoffs:** 04 owns detector association/letterboxing and source capture identity; 06 owns belief-conditioned admission, bootstrap and recovery; 07 owns common-time propagation, fusion and shared cross-camera covariance; 12/13 own future geometry/reference artifact binding and commissioning metadata. G01/G02 need coordinated small interface repairs before broad policy experiments. G03–G07 should not be presented as causes of the current pilot without their stated reachability conditions.
