# Assumptions register and frozen-control contracts

This document explains the scientific content of each assumption and freezes the
experimental controls. `registry.yaml` remains the only authority for assumption *state*:
where this audit proposes a different state the row reads `registry → proposed` and the
change is listed in §8 for the registry owner. Nothing here changes runtime behaviour, and
no value below may be quoted as frozen while it is marked `PENDING`.

**State vocabulary.** `ACCEPTED` — scoped and argued, deliberately not measured; converts to
a stated limitation in writing. `TESTED` — a locked experiment measured the assumption or a
controlled violation of it, and the result is on file. `DEFERRED` — knowingly outside both
contracts below.

**Two contracts.** The controls are frozen separately because the two studies have different
independent variables.

- **Contract A** (§5) — the current correlated-error / belief-honesty closed-loop paper,
  `EXP-CL-CAL`. Independent variable: the projection/calibration arm. Everything else,
  including the reliability fields, is frozen.
- **Contract B** (§6) — the later reliability-source benchmark, `EXP-USABLE`. Independent
  variable: the source of `p_use,c(s)` (constant, distance, FOV/range, depth/raycast, GP,
  hybrid, gated DL). Everything else, including the belief representation and the planner,
  is frozen.

An assumption can be a cheap scope statement in A and load-bearing in B. The `A`/`B` columns
in §1 record which.

---

## 1. Register at a glance

| ID | One-line statement | A | B | Registry state | Integration decision |
|---|---|:-:|:-:|---|---|
| A01 | Commissioned calibration exists; drift is bounded or monitored | ● | ● | TESTED | keep |
| A02 | The operational floor is locally planar | ● | ● | ACCEPTED | keep + non-claim |
| A03 | Intrinsics, optics, resolution and timing are fixed within an arm | ● | ● | ACCEPTED | keep |
| A04 | Geometry provenance is declared and distinct | ○ | ● | ACCEPTED | keep, B-blocking |
| A05 | Unknown depth invokes an explicit conservative fallback | ○ | ● | ACCEPTED | keep |
| A06 | Static and dynamic occlusion are separate regimes | ● | ● | DEFERRED | keep |
| A07 | Lighting and real-image transfer are outside current claims | ● | ● | DEFERRED | keep |
| A08 | Detector weights and threshold are frozen within comparisons | ● | ● | ACCEPTED | keep + U6 |
| A09 | Robot dimensions and appearance are fixed | ● | ● | ACCEPTED | keep |
| A10 | One robot; no association ambiguity | ● | ● | ACCEPTED | keep |
| A11 | Heading is odometry-backed; cameras correct 2-D position | ● | ● | ACCEPTED | keep |
| A12 | Synchronization and latency are measured operational inputs | ● | ● | TESTED | keep |
| A13 | Errors persist within and correlate across cameras | ● | ● | TESTED | keep |
| A14 | Evidence is simulation-only | ● | ● | ACCEPTED | keep |
| A15 | Evaluation truth is unavailable operationally | ● | ● | ACCEPTED | keep + caveat |
| A16 | Cameras vary geometrically and in residual structure, not optically | ● | ● | ACCEPTED | keep |
| A17 | The candidate-pose grid and target height are frozen and shared | ● | ● | *new* | ACCEPTED |
| A18 | Sample budgets count unique sites, not frames | ○ | ● | *new* | ACCEPTED |
| A19 | Splits use preregistered grouping appropriate to the estimand | ○ | ● | *new* | ACCEPTED |
| A20 | Pixel statistic and inversion plane are one coupled choice | ● | ● | *new* | ACCEPTED |

● load-bearing · ○ not exercised by that contract.

---

## 2. Assumption detail

### A01 — Commissioned camera calibration is available; drift is bounded or monitored

- **Need.** Every camera→ground measurement is a pixel ray intersected with a plane using a
  fixed extrinsic/intrinsic set. Without a commissioning step there is no measurement model
  and no residual to floor.
- **Plausibility.** High for wall-mounted infrastructure surveyed once at install; degrades
  over months through thermal cycling, vibration and knocks. The assumption is about the
  *monitor*, not about the cameras never moving.
- **Sensitivity / justification.** Measured on a controlled yaw/translation ladder. The
  *change* form of the commissioning statistic detects at 0.1° yaw; a stale correction turns
  harmful at 0.25° yaw. Detection therefore precedes harm by one rung. The commissioning gate
  itself cannot double as the in-service monitor — it fires at rest on raw cameras and is
  maskable and non-monotone.
- **Consequence if violated.** Projection bias grows with range; a stale correction becomes
  worse than no correction; the belief is confidently wrong rather than merely inaccurate.
- **Evidence.** `logs/studies/calibration_drift_lifecycle/exp1_stale_correction/drift_lifecycle.json`
  (held-out capture `fusion_handover_20260721`); `EXP-NET-COMMISSION`.
- **State.** `TESTED` — keep. Scope caveat to carry into writing: faults are *injected*, not
  observed; no real drift process has been measured (RQ07 is ANSWERED with that caveat).

### A02 — The operational floor is locally planar for ground-point projection

- **Need.** The monocular measurement is depth-unobservable without a known plane.
- **Plausibility.** High for a poured warehouse floor. In evidence terms it is *exactly*
  satisfied: both worlds use a Gazebo ground plane, so the assumption is untested, not
  confirmed.
- **Sensitivity / justification.** The dominant sensitivity is *which* plane, not floor
  flatness. On 1844 scored detections: box bottom at the **floor plane = 66.6 mm** mean error
  (deployed since 2026-08-07); at the formerly deployed 0.05 m contact plane = 110.2 mm; box
  centre at z\* = 0.085 m with a mesh model = 50.4 mm (measured, not adopted). Plane error
  enters ground error through the mount obliquity (6.10 m, 0.92 rad).
- **Consequence if violated.** A tilted or stepped floor biases every ground point in a
  range-dependent way that a constant per-camera bias cannot absorb, and conditional
  covariance stays too tight.
- **Evidence.** `logs/studies/pixel_ground_path/RESULTS.md`; `EXP-PROJ-AMP`.
- **State.** `ACCEPTED` — keep, with an explicit non-claim: because flatness holds by
  construction, no result in this thesis bounds behaviour on a real floor.

### A03 — Camera intrinsics, optics, resolution and timing are fixed within an arm

- **Need.** Makes the estimator, not the hardware, the independent variable.
- **Plausibility.** Trivially satisfiable in simulation; in deployment it means "do not swap
  or reconfigure cameras mid-study".
- **Sensitivity / justification.** This is a control, not a hypothesis. Enforce by pinning the
  world SDF, camera model and runtime-contract keys. Note that the repository *does* contain
  640×360 @ 5 Hz (`external_camera_*_laptop`) and 640×360 @ 3 Hz (`*_laptop_3hz`) variants of
  all four cameras. They exist to lower simulator load uniformly, are not a designed
  heterogeneity axis, and changing resolution changes the detector's input scale — so using
  them per-camera would confound A08 as well as A03.
- **Consequence if violated.** Any source or estimator ranking could have been produced by a
  hardware difference.
- **Evidence.** Nominal configuration `src/sim/models/external_camera/model.sdf`: 1280×720,
  90° HFOV, 5 Hz render; effective camera→belief update ≈ 2.2 Hz, bound upstream of the
  detector by Gazebo rendering.
- **State.** `ACCEPTED` — keep.

### A04 — Geometry provenance is declared and distinct

- **Need.** "Depth" currently spans an evaluation-only CAD oracle and a genuinely sensed
  height map that differ materially on the same evaluation (exact CAD AUROC 0.890 versus real
  depth frame 0.968 on `stack_capture2`). Collapsing them makes RQ03 unanswerable.
- **Plausibility.** Definitional. The risk is silent violation, not implausibility.
- **Sensitivity / justification.** The ladder in §4 is the control: every arm declares exactly
  one rung, and each rung carries a class, a staleness clock and a commissioning cost.
- **Consequence if violated.** An oracle upper bound is read as a deployable sensor and the
  benchmark's headline result is unusable.
- **Evidence.** `docs/reliability_prior_sensing_survey.md`;
  `scripts/geometry_visibility/{depth_realism,depth_source_comparison,sensed_height_prior,mono_depth_occlusion_prior}.py`.
- **State.** `ACCEPTED` — keep, and treat as **contract-B-blocking**: B cannot be frozen until
  the primary rung is chosen (§8 U1).

### A05 — Unknown or missing depth invokes an explicit conservative fallback

- **Need.** Real height maps have missing cells by default, not by exception: the
  sensor-realistic model cuts off near 10 m, while the maximum camera-to-reachable-cell range
  in the four-camera world is roughly 26 m.
- **Plausibility.** High.
- **Sensitivity / justification.** Missing-cell ablation at 0 / 10 / 30 percent, plus the
  range-limited rung D2 which produces its own realistic hole pattern.
- **Consequence if violated.** A raycast through unknown space returns *visible*, which is the
  unsafe direction: the planner routes into a shadow it believes is observed.
- **Evidence.** `scripts/geometry_visibility/depth_realism.py` degradation model; the
  falsified footprint-only prior (AUROC 0.669) and segmentation prior (0.780, below the 0.782
  camera-only baseline) recorded in `docs/reliability_prior_sensing_survey.md` §1, §3.4–3.5.
- **State.** `ACCEPTED` — keep.

### A06 — Static and dynamic occluders are reported as separate regimes

- **Need.** Shelves are static and mappable; people, pallets and forklifts are neither. A
  geometric source is structurally blind to the dynamic regime, whereas a learned source
  averages over it — an aggregate score hides exactly that difference.
- **Plausibility.** High as a regime split; the *scope* choice (static only) is the
  restrictive part.
- **Sensitivity / justification.** Static benchmark first; an injected dynamic occluder is a
  later arm, not part of B's frozen matrix.
- **Consequence if violated.** Aggregate results conceal each family's true failure mode
  (RQ14).
- **Evidence.** None, and this is the point: neither world contains a dynamic occluder. Both
  contain static racks, walls and static AWS props only.
- **State.** `DEFERRED` — keep. Must appear as a limitation wherever occlusion is discussed.

### A07 — Lighting and real-image domain transfer are outside current claims

- **Need.** The detector is trained on Gazebo renders and both worlds use a fixed synthetic
  lighting rig (one directional sun plus point lights).
- **Plausibility.** As a scope statement, certain. As a claim about the world, it would be
  false — real deployments have variable lighting.
- **Sensitivity / justification.** Not tested. State as a limitation; make no hardware claim.
- **Consequence if violated (i.e. if someone reads a transfer claim in).** The learned
  estimator and the detector both fail under image-domain shift, and every `p_use` number
  moves.
- **Evidence.** Locked decision "evidence is Gazebo-only" (`09_decisions_and_risks.md`).
- **State.** `DEFERRED` — keep.

### A08 — Detector weights and confidence threshold are frozen within comparisons

- **Need.** Quality sources must see the same perception process, or the ranking is a
  detector artefact.
- **Plausibility.** A control; trivially satisfiable.
- **Sensitivity / justification.** Weights are hash-pinned (`ART-DETECTOR-4CAM-V3`,
  `cb1f4249…`). The threshold is **not** consistently frozen: the offline usable-observation
  gate uses confidence ≥ 0.25 (node default) while the active four-camera mission config and
  the single-camera runtime contract both use 0.05. Because `p_qual` is *defined* by that
  threshold, the offline label set and the runtime are not currently the same gate (§8 U6).
- **Consequence if violated.** Source ranking can be caused by a threshold change rather than
  by the source.
- **Evidence, and an important refinement.** On the *measurement* path the frozen detector
  contributes essentially nothing: predicted box edges sit within ±0.34 px of the detector's
  own labels (sd 0.46–0.97 px) at a 99.7 % detection rate, and replacing the detector with
  label boxes moves radial sd from 42.4 mm to 42.6 mm — 0 mm in quadrature
  (`logs/studies/pixel_ground_path/RESULTS.md`, e2/e3). The freeze therefore protects the
  *availability* path (`p_det`), not the covariance path.
- **State.** `ACCEPTED` — keep; resolve U6 before B's labels are generated.

### A09 — Robot dimensions and appearance are fixed and documented

- **Need.** Both the visibility label and the pixel→ground map are target-specific.
- **Plausibility.** Certain within a study; zero transfer to another robot.
- **Sensitivity / justification.** TurtleBot3 Burger: body box 0.140 × 0.140 × 0.143 m centred
  at x = −0.032 m relative to the pose origin, `base_link` 0.010 m above `base_footprint`,
  LDS at 0.172 m above `base_link` (physical top ≈ 0.20 m), planner collision radius 0.125 m.
  Optional pose markers are off by default.
- **Consequence if violated.** Measured, not hypothetical: the −0.032 m body offset rotating
  with an unobserved heading is the *dominant* error term in the deployed projection. Its
  yaw-marginal irreducible spread is 30.3 mm radial / 22.2 mm lateral and is roughly constant
  in metres across 0–16 m, unlike the pixel term. Change the robot and every covariance
  constant in the pipeline is wrong by a known mechanism.
- **Evidence.** `src/sim/robot_description/urdf/turtlebot3_burger.urdf.xacro`;
  `logs/studies/pixel_ground_path/RESULTS.md` (e1, e3, e4, e5).
- **State.** `ACCEPTED` — keep. See A17 and §8 U11 for the separate question of the *field*
  target height, which is 0.35 m and does not match this silhouette.

### A10 — One robot removes multi-target association ambiguity

- **Need.** Matches every capture and campaign to date.
- **Plausibility.** True of the current setup; false of most real warehouses.
- **Sensitivity / justification.** Explicit scope limit; the opportunity schema already
  reserves `INVALID_ASSOCIATION`, and the track check is disabled because no runtime tracker
  exists.
- **Consequence if violated.** `p_use` must gain an association term, and the residual floors
  measured here fold in mis-association error that is not currently present.
- **Evidence.** `docs/usable_observation/data_contract.md` §4 (checks 7–8).
- **State.** `ACCEPTED` — keep.

### A11 — Heading is odometry-backed and camera updates correct 2-D position

- **Need.** Matches the locked estimator: `heading_update_mode: camera_xy_only`,
  `no_keypoint_or_visual_heading: true`. A YOLO-pose keypoint heading was built (8.6° median),
  never wired in, and removed on 2026-07-08.
- **Plausibility.** Medium-high for a differential-drive robot with encoders and IMU over a
  bounded run. The operational odometry channel carries correlated slip and additive noise
  (α ≈ 0.8) rather than being exact.
- **Sensitivity / justification.** Now quantified end to end. Solving for (x, y) with a
  yaw-aware mesh model: 50.4 mm yaw-blind; 17.9 mm at 0° heading error; 19.3 mm at 10°;
  29.2 mm at 30°; 39.7 mm at 45°; 66.2 mm at 90°. **Break-even is ≈ 45° of heading error**, so
  the assumption is comfortably safe *and* leaves a 2.8× accuracy headroom unclaimed.
- **Consequence if violated.** A yaw-aware inversion helps while heading error remains below
  roughly 45° and becomes harmful as the error grows beyond that break-even point. With no
  usable heading, `p_use` and `R_cond` must be marginalised over yaw. This is what makes U2
  consequential.
- **Evidence.** `logs/studies/pixel_ground_path/e5_yaw_aware_headroom/summary.json`.
- **State.** `ACCEPTED` — keep.

### A12 — Timestamp synchronization and latency are measured operational inputs

- **Need.** A residual is geometry error plus stale-state error; the two cannot be separated
  without timing.
- **Plausibility.** High in simulation; in deployment it requires explicit clock discipline.
- **Sensitivity / justification.** Measured, not assumed: detection↔odometry association rate
  0.9993 at a 0.15 s tolerance across three commissioning captures; per-camera inter-sample
  gaps with median 0–10 ms, p99 ≤ 28.8 ms and max 80 ms; odometry at 50 Hz; effective
  camera→belief update ≈ 2.2 Hz. The join between detections and odometry is sound — no
  re-capture is required.
- **Consequence if violated.** Residual floors mix geometry with stale-state error, and the
  per-step process noise becomes rate-dependent.
- **Evidence.** `logs/studies/operational_residual_rcond/exp1_timing_and_coverage/timing_and_coverage.json`.
- **State.** `TESTED` — keep.

### A13 — Camera errors may be temporally persistent and mutually correlated

- **Need.** This is the current paper's mechanism, not a nuisance term. Repeated views of one
  robot through a shared target model do not create independent evidence.
- **Plausibility.** High: a fixed camera's calibration error is a constant over a run, and all
  four cameras share the same robot geometry and the same detector.
- **Sensitivity / justification.** NEES and coverage against the stated ellipse,
  leave-one-camera-out folds, and a correlation-floor ablation.
- **Consequence if violated (i.e. if independence is assumed anyway).** Measured on 1424
  steps: trust-everything gives median NEES 4.22, mean 6.68, 95 % coverage 0.58 and an
  unearned-confidence fraction of 0.42; factorized fusion is *worse* (median NEES 5.11, 50 %
  coverage 0.076). The driver is camera C's real +78 mm lateral bias.
- **Evidence.** `logs/studies/bayesian_filter_showcase/exp1_graceful_vs_trusting/summary.json`
  and `exp2_does_it_generalize/summary.json`; `EXP-RCOND`, whose null shows the floor is
  bias-bound rather than data-bound (per-camera conditional covariance ties or loses to
  pooled).
- **State.** `TESTED` — keep.

### A14 — Evidence is simulation-only and supports no hardware-deployment claim

- **Need.** Enables controlled faults and exact evaluation truth.
- **Plausibility.** Certain as a scope statement.
- **Sensitivity / justification.** No sim-to-real study is planned in either contract.
- **Consequence if violated.** External validity is unproven; every quantitative constant is
  world- and renderer-specific.
- **Evidence.** Locked decision in `09_decisions_and_risks.md`.
- **State.** `ACCEPTED` — keep.

### A15 — Evaluation truth is unavailable to all deployed interfaces

- **Need.** Prevents oracle leakage into anything a policy can read at runtime.
- **Plausibility.** Definitional, and mechanically enforced rather than promised.
- **Sensitivity / justification.** `OperationalReliabilitySample` recursively rejects keys
  containing GT/oracle/collision/clearance/breach/localization-error/outcome tokens;
  `state_source == GT` raises `LeakageError` at gate entry; the registry records
  operational-interface and evaluation-only inputs per experiment.
- **Consequence if violated.** Results become oracle-assisted and invalid.
- **Evidence.** `docs/reliability_contracts/schema.md`; `docs/usable_observation/data_contract.md`
  §3; `EXP-NET-COMMISSION` recovers the actionable camera decision without truth.
- **Caveat to record.** *Truth-free is not unbiased.* The anchored leave-one-out reference
  understates camera A by about 4.2×, which is why leave-one-camera-out is mandatory rather
  than optional; and sizing covariance from cross-camera disagreement alone fails by ~4×.
  Truth-free commissioning bounds a *decision*, not a covariance.
- **State.** `ACCEPTED` — keep, with the caveat carried into the commissioning narrative.

### A16 — Four cameras provide geometric and residual-structure diversity but not optical archetype diversity

- **Need.** The four cameras are the same model at the same height, pitch, FOV, resolution and
  rate. Their position, occlusion exposure, handover role and measured residual structure
  differ; E6 shows that the residual structure cannot yet be attributed uniquely to camera
  calibration rather than route, region, yaw or robot silhouette.
- **Plausibility.** Certain — it is a statement about the configuration, not a hypothesis.
- **Sensitivity / justification.** Historical per-camera fits report residual floors of
  7.1 / 12.3 / 76.8 / 32.8 mm for A / B / C / D, plus a best-precision floor share of
  31.0 / 28.0 / 14.8 / 26.2 % against an essentially uniform 25 % best-coverage share.
  These establish heterogeneous installed-view residuals, not four independent camera
  biases. RQ15 and the WS05 identifiability gate control any stronger attribution.
- **Consequence if violated (i.e. if optical archetypes are claimed).** Any statement about
  wide-angle versus narrow, high versus low resolution, or vendor differences would be
  unsupported by construction. See `06_world_camera_design.md` §6 for the exact
  supported/unsupported list.
- **Evidence.** `logs/studies/achievable_precision_map/exp1_precision_vs_coverage/summary.json`;
  `EXP-BIAS`; `EXP-COMMISSION`.
- **State.** `ACCEPTED` — keep.

---

## 3. Accepted additions

### A17 — The candidate-pose grid and target height are frozen and shared across arms

- **Need.** Two sources evaluated on different grids or at different target heights are not
  comparable, and the target height silently defines what "visible" means.
- **Plausibility.** A control.
- **Sensitivity / justification.** Current fields for `warehouse_full_4cam` use a 240 × 184
  grid over x ∈ [−11.7, 11.7], y ∈ [−9.0, 9.0] (≈ 0.098 m cells) at `target_height` 0.35 m,
  while the planner-side profile declares 470 × 360 over the same extent. Both must be pinned
  per arm. Note the height mismatch flagged in §8 U11: 0.35 m sits ~0.15 m above the Burger's
  physical top, and for a camera at H = 6.10 m the predicted shadow behind an occluder of
  height h shortens by roughly d·Δz/(H − h) — about 0.36 m at 10 m behind a 1.90 m rack. The
  error is in the anti-conservative direction.
- **Consequence if violated.** Differences between sources are partly grid-resolution and
  target-height artefacts, and every occlusion prediction is biased toward "visible".
- **Evidence.** `logs/visibility_comparison/spawn_grid_20260727/gp/camera_*/manifest.json`;
  `src/experiments/config/world_profiles.yaml`;
  `src/sim/robot_description/urdf/turtlebot3_burger.urdf.xacro`.
- **State.** `ACCEPTED`. The exact grid and target height remain blocked by U11; acceptance
  freezes equality across arms, not the unresolved numeric value.

### A18 — Sample budgets count unique sites, not frames

- **Need.** The commissioning-cost claim (RQ12) is meaningless if a budget can be inflated by
  repetition.
- **Plausibility.** A control, but currently violated in spirit by the collection design:
  `repeats_per_route: 5` × 3 lateral offsets × 2 speeds multiplies frames ~30× without adding
  sites.
- **Sensitivity / justification.** Report performance against unique sites per camera; the
  candidate ladder is in `06_world_camera_design.md` §9.
- **Consequence if violated.** The commissioning curve is a repetition curve, and a GP looks
  cheaper than it is.
- **Evidence.** `experiments/multicamera_commissioning_bigwarehouse/config/study.yaml`;
  the four-camera GP manifests aggregate 2202 events into 1101 points per camera.
- **State.** `ACCEPTED`.

### A19 — Splits use preregistered grouping appropriate to the estimand

- **Need.** Random cell-level splits leak: the GP length scale is 1.2 m and events are
  aggregated at 0.3 m, so neighbouring held-out cells are not independent.
- **Plausibility.** A control, and currently unmet. All four deployed four-camera GP manifests
  record `heldout_event_count: 0` and an empty `route_disjoint_validation_csv`; the 6-fold
  validation that does exist shows prior-only beating the fitted GP on Brier for every camera
  (A 0.128 vs 0.157, B 0.119 vs 0.164, C 0.128 vs 0.145, D 0.131 vs 0.167).
- **Sensitivity / justification.** Use route groups for route transfer, spatial blocks at
  least twice the kernel length scale (≥ 2.4 m) for interpolation/extrapolation, and yaw
  groups when the estimand depends on robot appearance or heading. These are complementary
  tests, not a requirement that every split simultaneously group on all three dimensions.
  Two of the three "independent" bias captures are straight lines at fixed yaws and the third
  repeats the first route (median nearest-neighbour 0.006 m), so the calibration-identifiability
  study must control both route/region and yaw.
- **Consequence if violated.** A learned source scores its own interpolation and the GP null
  is falsely overturned.
- **Evidence.** `logs/visibility_comparison/spawn_grid_20260727/gp/camera_*/manifest.json` and
  `validation/camera_*/validation_summary.csv`; `logs/studies/pixel_ground_path/RESULTS.md` §1.
- **State.** `ACCEPTED`.

### A20 — The pixel statistic and the inversion plane are one coupled design choice

- **Need.** The selected pixel and the plane it is inverted onto are not independently
  swappable: box-bottom belongs with a contact/floor plane, box-centre with the
  grid-search-optimised z\* = 0.085 m plane and a mesh model.
- **Plausibility.** A control, derived from measurement.
- **Sensitivity / justification.** Mixing them is not a small error: bottom-at-0.05 m gives
  110.2 mm, bottom-at-floor 66.6 mm, centre-at-0.085 m 50.4 mm on the same 1844 scored
  detections (1849 index rows).
- **Consequence if violated.** A calibration arm labelled as testing one mechanism silently
  tests two, and the covariance constants no longer correspond to the estimator in use.
- **Evidence.** `logs/studies/pixel_ground_path/RESULTS.md`;
  `logs/studies/pixel_ground_path/e3_mesh_model_and_covariance/summary.json`.
- **State.** `ACCEPTED`, and now **moot for the deployed path**: the frozen pair is box
  bottom-centre + floor plane with no correction of any kind, so there is no longer a plane
  constant or a per-camera term that could be mismatched to the statistic (§8 U5, resolved
  2026-08-07; evidence `logs/studies/pixel_ground_path/e7_ipm_zero_parameter/RESULTS.md`).
  The coupling remains true of the *historical* artifacts and is why v2 scores 68.2 mm with
  its along-bearing term but 110.2 mm with the plane alone.

---

## 4. Geometry and depth provenance ladder

This is the precise meaning of A04/A05. **Every arm that consumes geometry must declare
exactly one rung and one staleness state.** "Depth" alone is not an admissible label.

| Rung | Name | What it is | Class | In-repo instantiation | Staleness clock | Deployment cost |
|---|---|---|---|---|---|---|
| D0 | Complete map upper bound | Every occluder parsed from the world file with exact heights, including walls and visual overlays | **Evaluation-only oracle**; admissible only as a clearly labelled upper bound | `unav_common.occlusion_geometry.parse_occlusion_scene_from_world`; the COMPLETE-CAD reference in `depth_realism.py` | None by definition | Not obtainable in deployment |
| D1 | Commissioning scan, idealised | One perfect depth frame per camera at commissioning, same pose and intrinsics as the RGB sensor | Operational **only if** an idealised depth sensor is declared | `depth_camera` sensor in the camera models (`always_on 0`); `depth_occlusion_prior.py` | Frozen at scan time | One capture per camera |
| D2 | Commissioning scan, sensor-realistic | D1 degraded by a ~10 m usable range, range-growing noise and dropout | Operational | `depth_realism.py` | Frozen at scan time | One capture plus a declared sensor model |
| D3 | Maintained static map | D1 or D2 kept current by a stated re-scan policy | Operational | Not implemented | Bounded by the policy interval | Recurring capture |
| D4 | Stale map | A D2 map captured before a layout change and not re-scanned | Operational, adversarial | Not implemented; `whatif_layout_change.py` *predicts* the effect but never measures it | Unbounded | Zero, which is why it is the realistic failure |
| D5 | Rescanned map | D4 followed by one re-scan of the changed region only | Operational | Not implemented | Reset for the rescanned region | One partial capture |
| D6 | Live sensed depth | Depth read per frame at planning time rather than from a map | Operational | Sensor exists, disabled at runtime | None | Continuous bandwidth and compute |
| D7 | Footprint-only, no depth | 2-D drivable-map holes with an assumed height | Operational, **already falsified** | The dropped freespace prior | n/a | Zero |
| D8 | Monocular-depth estimate | DL depth on the fixed camera's RGB, affine-corrected using floor pixels whose true depth is analytically known from calibration | Operational, **unvalidated** | `mono_depth_occlusion_prior.py`, `mono_depth_ablation.py` | Frozen at inference | Zero hardware, one inference |
| D9 | Sampled-CAD "sensed" cloud | A point cloud sampled from CAD and reconstructed into a height map | **Circular** — validates plumbing only | `sensed_height_prior.py` | n/a | n/a |

Notes that must travel with the ladder:

- **The known separation.** On the same evaluation (`stack_capture2`, uniform teleport grid):
  camera-only geometry AUROC 0.782, complete CAD 0.890, real depth frame 0.968. The
  viewpoint-matched sensed frame beats the exact CAD oracle because occlusion for a camera is
  determined by the surfaces *that camera sees*. D0 is therefore not even a strict upper bound
  on occlusion quality — only on geometric completeness.
- **Falsified rungs stay in the register.** D7 (0.669) and semantic segmentation without depth
  ordering (0.780, below the camera-only baseline) are settled negatives and may appear only
  as reference nulls.
- **D9 may never be cited as evidence.** Its ~0.97 score is circular by construction.
- **D6 changes the claim.** With live depth, `p_use` at the *current* pose becomes partly
  observed rather than predicted; it remains a legal *future*-pose predictor only through the
  static structure it reveals, never through the current detection outcome.
- **Missing cells.** Every rung except D0 must declare its missing-cell fallback: `unavailable`
  (excluded from the field), `FOV fallback` (geometry-only prediction), or `conservative prior`
  (assume occluded). Silent interpolation is prohibited.

---

## 5. Frozen-control contract A — correlated-error closed-loop paper (`EXP-CL-CAL`)

**Independent variable:** the projection/calibration arm, and nothing else. Configuration
generation must prove that the arms differ only in the declared keys.

| Control | Frozen value | Source of record |
|---|---|---|
| World | `warehouse_full_4cam.world.sdf`, pinned by world hash | `src/sim/gazebo_worlds/worlds/` |
| Camera set | A/B/C/D at (∓6.00, ∓10.00, 6.10), pitch 0.92 rad, yaw ±1.5708 | `docs/warehouse_full_4cam_layout.md` |
| Overview camera | (0, 0, 26.0), media only; excluded from GP, fusion and planner interfaces | layout doc |
| Optics / rate | 90° HFOV, 1280×720, 5 Hz render, ≈2.2 Hz effective update | camera model SDF |
| Detector | `ART-DETECTOR-4CAM-V3` `cb1f4249…`, trained imgsz 960, inference 640, IoU 0.45, masks off | `registry.yaml`, missions config |
| Confidence threshold | 0.05 at runtime — **inconsistent with the 0.25 offline gate** | §8 U6 |
| Pixel statistic | `bbox_bottom` (deployed). WS04's `bbox_center` path is opt-in and must not be wired | A20, §8 U5 |
| Projection plane and calibration artifact | **PENDING WS05** (v2 / v3 / v4 arm conflict) | WS05 handoff |
| Reliability fields | `spawn_grid_20260727` fused four-camera GP plus the per-camera manager templates; frozen, never refit per arm | missions config |
| Belief and correction | `state_correction_mode: per_camera`; runtime NIS gate 9.21; `pixel_max_correction_jump_m` 0.5; `pixel_timeout_s` 1.25; `skip_stale_pixel_correction: true`; `max_predict_speed_mps` 0.6 | missions config; §8 U7 |
| Planner | hierarchical; global 75 × 0.4 s; local horizon 6 at 5 Hz; `v_max` 0.6; `nogo_mode: keep_in`, `warning_band` 0.05, weight 2000; belief-nogo κ = 1.0; `r_visible_uv` 2.5 | missions config |
| Miss endpoint | `r_miss_uv` 40 offline vs 120 runtime default — quoting either is blocked by `MissEndpointPolicy.require_reconciled()` | §8 U8 |
| Actuation / encoder noise | `use_command_noise` and `use_encoder_noise` on, with the listed slip/additive constants and α = 0.85 / 0.80; identical across arms; paired seeds | missions config |
| Tasks | `mc_central_ns`, `mc_south_we`, `mc_north_we` | missions config |
| Seeds | documented 0–4; generated `_clv2.yaml` / `_clv3.yaml` currently contain seed 0 only | §8 U9 |
| Independent unit | the matched `(task, seed)` run; frames and detections are within-run samples | WS05 |
| Primary endpoint | **PENDING WS05** (clean goal / belief calibration / p95 localization error) | WS05 |
| Nominal noise | none added — no synthetic pixel jitter, latency, dropout or injected drift in the primary comparison | §9 of `06_world_camera_design.md` |
| Evaluation firewall | `gt_*` / `eval_*` evaluation-only; `use_ground_truth_metrics: true`, `allow_odom_metric_fallback: false` | A15, runtime contract |
| World/field compatibility | **BLOCKED** — the current world postdates the July GP fields, whose manifests bind no world hash | §8 U4 |

Contract A deliberately does **not** freeze the reliability-source question: it compares
calibration arms under one fixed field set, and therefore cannot establish that the
correlation-flooring / leave-one-out package as a whole improves navigation.

---

## 6. Frozen-control contract B — reliability-source benchmark (`EXP-USABLE`)

**Independent variable:** the source of `p_use,c(s)`. Everything below is identical across
arms; a violation of any row invalidates the comparison rather than degrading it.

| Control | Frozen requirement |
|---|---|
| Target | One quantity, `p_use,c(s)` = usable-observation probability for camera *c* at candidate pose *s*. Whether *s* includes heading is §8 U2 and must be settled before any fitting |
| Label definition | Schema `obs_opportunity_v1` with the frozen gate `usable_observation_gate_v1`; one record per camera per synchronized opportunity, **including misses** |
| Detector and threshold | A08, one value, applied identically to labels and to every arm |
| Candidate-pose grid and target height | A17, one grid and one height for all arms |
| Splits | A19 preregistered route, spatial-block and/or yaw grouping chosen for the question, identical across arms |
| Budget | A18 unique sites per camera, reported as a curve |
| Probability calibration | One allowance (e.g. a single held-out isotonic/Platt stage) granted identically to all arms, or to none |
| Seeds and sites | Paired; the same seeds and the same sites for every arm |
| Downstream representation | Shared frozen `R_cond`, shared persistent bias / correlation floor, shared freshness rules, and the shared posterior `E[P⁺] = p_use·P_hit + (1 − p_use)·P⁻`. No arm refits the observation representation |
| Planner | Frozen; the same weights and route library for every arm at the offline route-discrimination gate |
| Geometry provenance | A04 §4: one declared rung and one staleness state per arm, with a declared missing-cell fallback |
| Nominal noise | None. Synthetic noise appears only in the OFAT sensitivity arms of `06_world_camera_design.md` §9 |
| Design | One-factor-at-a-time. The Cartesian product of sensitivities is prohibited |
| Gate order | Offline prediction → failure audit → commissioning curve → transfer/staleness → offline route discrimination → closed loop. A method that fails an earlier gate consumes no Gazebo time |

### Feature legality at candidate poses

A predictor for a *future* pose may use only information that will be available before the
robot is there.

| Feature | Legal for `p_use` prediction? | Reason |
|---|---|---|
| Camera identity | Yes | Constant per camera |
| Candidate position (x, y) | Yes | The estimand's own argument |
| Candidate heading | **Only if U2 resolves to include it**, and then only as the *planned* heading | Otherwise it is a current-state leak |
| Calibrated camera pose and intrinsics | Yes | Commissioning output |
| Range, bearing, incidence, projection Jacobian at the candidate pose | Yes | Derived from calibration and the candidate pose |
| 2-D drivable map | Yes | Declared deployment input |
| Geometry rungs D1–D5, D8 | Yes, with declared provenance and fallback | §4 |
| Complete-map rung D0 | **No** as a deployed input; yes only as a labelled upper bound | §4 |
| Fitted GP field plus its epistemic support | Yes | Commissioned artifact |
| Instantaneous detector confidence | **No** | Not available at a pose the robot has not reached; it is a camera-management signal |
| Current detection validity, recent detection history | **No** | Same reason |
| Belief covariance at the candidate pose | **No** as a feature; it may weight *training* observations (expected-kernel) | It is a property of the estimator, not of the pose |
| Rendered imagery at the candidate pose | **No** | Future rendering is not an operational input |
| Any `gt_*`, `eval_*`, oracle or outcome field | **No**, rejected in code | A15 |

---

## 7. Supported and unsupported generalization

| Axis | Supported | Not supported | Why |
|---|---|---|---|
| Camera hardware | Position, occlusion exposure, handover role, measured installed-view residual structure | Optical archetypes, resolution or frame-rate diversity, vendor or hardware transfer; camera-specific attribution before RQ15 closes | A16; four identical models. See `06_world_camera_design.md` §6 |
| Detector | A frozen-detector contract; on the measurement path specifically, results are unusually detector-robust (0 mm in quadrature) | Any claim across detector families | A08; RQ08 is a standing LIMITATION |
| Image domain | Nothing | Lighting, texture or real-image transfer | A07, A14 |
| Robot | One TurtleBot3 Burger | Any other platform | A09; the dominant projection error is this robot's own 32 mm body offset, so transfer is not merely untested but wrong by a known mechanism |
| Targets | Single target, no association | Multi-robot, cluttered-target scenes | A10 |
| Occlusion | Static, mapped structure | Dynamic occluders (people, forklifts, moved pallets) | A06; no dynamic occluder exists in either world |
| Layout stability | Static layouts | Changed, stale or rescanned layouts | A04/§4 D3–D5 are unimplemented; `whatif_layout_change.py` predicts but does not measure |
| Calibration drift | Detection precedes harm under a controlled injected ladder | Real drift processes, drift isolation, or multi-camera simultaneous drift | A01 |
| Worlds | Two worlds with measured properties, under the two-world rule | "Warehouses in general"; and no clean world-to-world transfer claim, because the two worlds differ in mount height *and* detector | RQ10; see `06_world_camera_design.md` §1–§3 |
| Truth-free commissioning | Recovering an actionable per-camera *decision* without truth | Sizing a covariance without truth | A15 caveat; disagreement-based sizing fails by ~4× |

---

## 8. Unresolved decisions — returned, not invented

Each item blocks something concrete. None can be inferred from an existing locked decision.

| # | Question | Why it cannot be inferred | Blocks | Owner |
|---|---|---|---|---|
| U1 | Which geometry rung (§4) is contract B's **primary operational depth** arm? | The survey recommends D8 on cost grounds but validates nothing; D1/D2 are implemented but never frozen; D3–D5 do not exist | B design freeze, RQ03, the whole depth arm | WS07 + supervisor |
| U2 | Does `p_use,c(s)` take `s = (x, y)` or `(x, y, heading)`? | A11 keeps heading odometry-backed, but the pixel-path evidence shows heading conditioning is worth 2.8× on *accuracy* — a different quantity from *availability*, and the two need not share an argument | Grid size, commissioning budget, A17, every arm's feature vector | WS01/WS07 + supervisor |
| U3 | What counts as a **local** versus **global** layout change, and where may a changed-layout variant live? | Only one concrete instance exists (a 2.6 m pallet in `warehouse_aws` aisle A2, prediction only). The two-world rule reserves `warehouse_full_4cam` for frozen-method evaluation, so a changed variant of it is either a rule exception or a third world | B4 transfer gate, D4/D5, the third world split in `06` §3 | WS02 → supervisor |
| U4 | Is the current world compatible with the July GP fields? | The world hash postdates the fields, whose manifests bind no world hash. Shared exposure may preserve a within-study contrast but not external interpretation | Contract A readiness (fail-closed) | WS06 |
| ~~U5~~ | **RESOLVED 2026-08-07: `bbox_bottom` intersected with the floor plane (inverse perspective mapping), NO calibration artifact at all.** The `bbox_center` @ 0.085 m candidate is rejected as method and retained as the comparison that justifies the choice; `contact_z_m = 0.05` is superseded. Rationale: IPM is the textbook path with zero statistic-level tuned scalars and 66.6 mm mean error, versus 50.4 mm for a candidate carrying two grid-search-chosen scalars, versus 110.2 mm for the previous default. This went further than first decided: e7 then measured v4's two surviving cross-bearing constants against the same 1844 detections and they made it **worse** (70.1 mm vs 66.6 mm raw), inverting the per-camera lateral bias they existed to remove (camera C +18.8 mm -> -58.7 mm). All correction degrees of freedom were therefore **deleted from the runtime**, not merely unselected. This also closes the registry's E6 caveat by removing the terms it warned about rather than identifying them. | — | — | closed |
| U6 | Is the frozen detector confidence threshold 0.25 or 0.05? | The offline gate contract and the runtime configs disagree, and `p_qual` is *defined* by the threshold, so the offline labels and the runtime are not currently the same gate | B's label generation, A08, any `p_qual` number | WS07 + integration |
| U7 | Which NIS policy is frozen — runtime 9.21 or offline 5.991? | They are different policies applied to the same mechanism; neither is documented as superseding the other | Contract A acceptance statistics, cross-study comparability | WS05 |
| U8 | What is the reconciled miss endpoint (`r_miss_uv` 40 vs 120)? | `MissEndpointPolicy.require_reconciled()` blocks quoting either until the residual-tail measurement is made on real data | Any published EFE/covariance figure | WS04/WS05 |
| U9 | Contract A's seed matrix: seed 0, or seeds 0–4? | The documentation says 0–4; the generated configs contain seed 0 only. Fifteen matched pairs support a mechanism or null report, not a safety-superiority claim | Contract A sample size and claim strength | WS05 |
| U10 | What legal DL input is scientifically distinct from a spatial MLP or GP? | Without one, the DL challenger is a reparameterised GP and its admission gate is vacuous | B's DL arm | WS07 + supervisor |
| U11 | Is the 0.35 m field target height deliberate or inherited? | The Burger's physical top is ≈ 0.20 m. A taller assumed target shortens every predicted shadow — the anti-conservative direction — by roughly d·Δz/(H − h) | A17, every occlusion-based source, all existing fields | WS02 → supervisor |
| U12 | What are the minimum meaningful effects for Brier, terminal belief and navigation outcome? | Preregistration requires them; nothing in the locked evidence implies a threshold | B1 and B5 gates, contract A's null interpretation | supervisor |
| U13 | Is there a genuinely **unavoidable** camera-poor region in `warehouse_full_4cam`? | The negative control requires one, and the candidate (the west wall-backed one-sided shelf lane) has not been certified against the measured per-camera fields | Route archetype R4 in `06` §7 | WS02 measurement, then WS07 |
