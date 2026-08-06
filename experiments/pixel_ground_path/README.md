# pixel_ground_path — the locked pixel→ground path for infrastructure-camera localization

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-PIXEL-GROUND
status: READY
claim_ids:
- C1
- C2
assumption_ids:
- A01
- A02
- A03
- A08
- A09
- A14
reviewer_question_ids:
- RQ08
- RQ14
figure_ids:
- F01
- F07
dependencies:
- ASSET-RUNTIME
operational_inputs:
- pixel_statistic
- camera_calibration
- robot_geometry
evaluation_only_inputs:
- ground_truth_pose
- rendered_silhouette
primary_metric: held-out calibration of propagated ground covariance
promotion_gate: Promote only the minimal calibrated path that improves held-out covariance
  honesty.
evidence_paths:
- experiments/pixel_ground_path/README.md
- logs/studies/pixel_ground_path/e4_covariance_calibration/summary.json
- logs/studies/pixel_ground_path/e5_yaw_aware_headroom/summary.json
archive_rule: Preserve summaries and the chosen minimal model; archive superseded
  fitted corrections.
next_action: Decide whether this is required infrastructure for EXP-USABLE or supporting
  analysis only.
```

<!-- RESEARCH-METADATA:END -->


**Question.** What is the ONE defensible way to turn a detection in a fixed overhead camera
into a metric ground position with an honest covariance — logical, standard, and
commissionable by an integrator?

**Chapters served.** `research_story/04_factorised_observation_model` (the observation model)
and `research_story/09_multicamera_handover_fusion` (per-camera covariance and fusion).
Supersedes the *correction* line in `experiments/external_camera_bias_model/`: that study's
audit findings stand, its fitted artifacts (v2, v4) do not.

**Results:** [`logs/studies/pixel_ground_path/RESULTS.md`](../../logs/studies/pixel_ground_path/RESULTS.md).
Read that before this file — it contains the numbers, and it records three claims from the
first draft of this design that the experiments **falsified**.

---

## 1. The path

Estimand: the vertical projection of the robot's `base_footprint` origin onto the floor.

| step | choice | why |
|---|---|---|
| 1 | **Object model = the URDF visual meshes** (body box, two tyres, LiDAR puck), assembled in the `base_footprint` frame | The semantic mask renders the visual meshes, not the collision primitives, and they differ by 6.8 mm at the top. A bounding cylinder is wrong by −4.3 px in width. |
| 2 | **Pixel statistic = the box centre**: `u = u_centre`, `v = v_bottom + 0.5·(v_top − v_bottom)` | Chosen at design time by minimising yaw-marginal error over a CAD grid. The α curve is a plateau over [0.3, 0.6]; a pre-registered tie-break prefers the named statistic. |
| 3 | **Yaw-aware inversion**: Gauss-Newton on `(x, y)` so the modelled box statistic matches the observed one, at the filter's current heading | The dominant error is the body's 32 mm x offset seen through the heading. Conditioning on heading takes 50.4 mm → **17.9 mm**, and still wins at 45° of heading error. |
| 3b | **Fallback when heading is unavailable or unreliable**: back-project the box centre onto the plane `z* = 0.085 m` | Closed form, no iteration, no heading. 50.4 mm. Both arms share the same object model and statistic, so the fallback is a degradation, not a different method. |
| 4 | **Validity gate**: predict the box size at the solved position and gate on its Mahalanobis distance | Catches border clipping and shelf occlusion. Size is a gate, never an estimator — 1 px of box height is 0.89 m of range. |

`z* = 0.085 m` is *derived* (it minimises the CAD yaw-marginal error), not chosen. Contrast
the deployed `contact_z_m = 0.05`, a free operator constant worth 94 mm of radial bias.

## 2. Calibration procedure

| # | step | method | parameters |
|---|---|---|---|
| C1 | intrinsics | planar-target calibration (Zhang) | exact in sim |
| C2 | extrinsics / plane | ≥4 surveyed floor points → DLT homography | exact in sim |
| C3 | object model | the robot's URDF/CAD | **0 fitted** |
| C4 | statistic (α, z\*) | design-time minimisation over a CAD grid | **0 fitted to data** |
| C5 | detector edge noise | run the accepted detector over its labelled val split; compare predicted box to label box | GT-free; measured **±0.34 px bias**, so the per-camera gate leaves Δ = 0 |
| C6 | `Σ_yaw` | spread of the yaw-marginal error over the CAD grid | **0 fitted**: 30.3 mm radial / 22.2 mm lateral |
| C7 | `Σ_uv` total | **needs a commissioning run with robot poses** | see the weakness below |

Nothing is fitted per camera. C5's per-camera numbers came out inside the gate, so the
shipped configuration has **no per-camera parameter at all** — against twelve before.

## 3. Uncertainty per detection

```
R = J(u, v; z*) · Σ_uv · J(u, v; z*)ᵀ  +  Σ_yaw
```

Two terms, because they scale differently: the pixel term grows with range (radial
1.35 → 4.23 cm/px over 5–16 m, as `(H² + d²)/(f·H)`), while `Σ_yaw` is roughly constant in
metres. Collapsing them into one pixel constant gave NEES running 26 at short range to 7 at
long range; separating them gives **2.83, uniform 2.54–3.18 across camera, range and yaw**.

Drop `Σ_yaw` when the yaw-aware arm is active — it exists only to cover the heading you
chose not to use — and replace it with `J·(∂/∂θ)·σ_θ²·(∂/∂θ)ᵀ` from the filter's heading
variance. e5's degradation curve is the empirical version of that term.

## 4. Assumptions

1. **Intrinsics and extrinsics are exact.** True in simulation by construction. In a real
   warehouse this becomes C1/C2 and is the first thing to re-audit; the whole method inherits
   any homography error directly.
2. **The floor is planar and at z = 0** over the operating area.
3. **The robot's CAD is trusted to ~1 mm**, and the *rendered/observed* geometry matches it.
   In sim this is checkable; on real hardware the visual appearance (cables, payload, lids)
   will not match CAD as well.
4. **One robot, one class, one known target.** The method is a known-object-model inversion;
   it does not extend to unknown or variable-size targets without re-deriving (α, z\*).
5. **The detector's box tracks the true silhouette bbox**, which holds because it was trained
   on exactly that label definition. A detector trained on differently-defined boxes would
   need C5 re-run and could need a non-zero Δ.
6. **Heading is available to ~30° or better** for the yaw-aware arm. Break-even is 45°.
7. **Detections are unoccluded and fully in frame.** Every sample in the evidence base is
   `occlusion_state == clear`.

## 5. Weaknesses

1. **`Σ_uv` cannot be sized without robot poses.** Inter-camera disagreement — the obvious
   GT-free route, 575 pairs available — returned 4× too small a variance, because the
   dominant error is a shared cause that partly cancels between views. A CAD-computed
   correction for that shortfall was also wrong (11.13 predicted vs 4.11 needed). So
   commissioning needs a run with truth, or a deliberate conservative inflation: the
   design-time-only covariance gives NEES 3.52, i.e. over-confident by 1.76× in variance.
2. **The residual covariance is still 1.4× over-confident** (NEES 2.83 vs 2.0) even with
   poses. It is uniform, so a single scalar fixes it, but that scalar is fitted.
3. **Four discrete yaws.** `Σ_yaw` and every yaw-marginal number average a periodic function
   at 0/90/180/270°. The 45° diagonal, the worst case for a rectangular footprint, is not in
   the data.
4. **Occlusion is entirely untested.** The size gate in step 4 is designed but not validated,
   and occlusion is where the bottom-edge path fails worst — so the *comparison* against the
   deployed path is also untested in its worst regime.
5. **Open-loop only.** No closed-loop run, no navigation claim. The 9.21 chi-square gate and
   `pixel_max_correction_jump_m` interact with a 50 mm→18 mm change in a way nobody has
   measured.
6. **The yaw-aware arm is iterative.** Gauss-Newton converges in 2–3 steps from the
   closed-form seed, but it is no longer a single matrix multiply, and its runtime cost in
   the ~3 Hz perception loop is unmeasured.
7. **Single-robot evidence.** All 1849 samples are one TurtleBot3 Burger in one world.

## 6. Failure modes

| failure | trigger | symptom | mitigation | tested? |
|---|---|---|---|---|
| Clipped box at the image border | robot at FOV edge | box centre wrong by half the clip; error grows without bound | size-consistency gate (step 4) | **no** |
| Bottom occluded by a rack | robot behind foreground shelf | 1.47 cm per clipped pixel through the centre path, 2.91 through the bottom path | size gate + `visible_height_fraction` | **no** |
| Heading estimate diverges | filter loses heading lock | yaw-aware arm degrades past 45° error and becomes **worse** than yaw-blind | monitor `σ_θ`; fall back to arm 3b above a threshold | curve measured, switch not built |
| Wrong object model | different robot, added payload, changed LiDAR | radial bias returns; ~7 mm of height error ≈ 13 mm of position error | compare predicted vs observed box height as a monitor | partially (mesh vs primitives) |
| Extrinsic drift | camera knocked or sags | direct position bias; this method has **no** self-check for it | the existing calib-drift health monitor is the right owner | not in this study |
| Detector regresses to a different box convention | retrain, different label policy | Δ becomes non-zero and per-camera | C5 re-run gates it | mechanism in place |
| Two robots / false positive | another mover in view | highest-confidence box may not be the robot | out of scope — needs association | **no** |
| Very short range (< 3 m) | robot near nadir | untested territory; the design grid stops at 2 m | — | thinly (32 positions at 0–5 m) |

## 7. Where this sits in the literature

- **Ground-plane homography / IPM** — Mallot et al., *Biol. Cybern.* 1991; Hartley &
  Zisserman for homography estimation and first-order covariance propagation.
- **Foot-point projection** is the baseline convention in fixed-camera multi-view tracking
  (Khan & Shah, *TPAMI* 2009; WILDTRACK, Chavdarova et al., *CVPR* 2018). Step 2 is a
  documented departure, with the measurement that motivates it.
- **Known-size object model rather than one foot pixel** — POM (Fleuret, Berclaz, Lengagne &
  Fua, *TPAMI* 2008) models people as fixed-size shapes and explains the observed silhouette;
  Berclaz et al. (KSP), *TPAMI* 2011. Steps 1–3 are the same idea with a robot whose
  dimensions are known exactly, which is the easy case they did not have.
- **Projecting at a non-zero height plane** — Khan & Shah's multiple scene planes; the feature
  perspective transformation of MVDet (Hou, Zheng & Gould, *ECCV* 2020). Step 3b is that
  construction with the height derived rather than assumed.
- **Model-to-box fitting for a calibrated fixed camera** is standard traffic-surveillance
  practice (Dubská, Herout & Sochor, automatic roadside camera calibration).
- **Marginalising a nuisance parameter into the covariance**, then conditioning on it when an
  estimate exists, is ordinary nonlinear filtering — `Σ_yaw` and step 3 are the two halves.

## 8. Next

1. **Occlusion**: build the size-consistency gate and test it on deliberately occluded
   captures. This is the largest untested area and it is where the deployed path is worst.
2. **A yaw-diverse capture**: rotate in place on a grid so yaw is decoupled from position and
   the 45° diagonal is covered. Cheap, and it removes weakness 3.
3. **Wire the yaw-aware arm** into `reliability/projection.py` behind a flag, with the
   `σ_θ`-triggered fallback, and re-measure NEES in a closed loop.
4. **Re-audit the 1424-detection residual set by yaw, not by capture** — and note that
   `fusion_handover_20260721` is the same route as `smoke1_20260716`, so it is two folds.

## Reproduce

```bash
python3 experiments/pixel_ground_path/e0_pixel_statistic_geometry.py        # geometry (see correction)
python3 experiments/pixel_ground_path/e1_object_model_vs_real_silhouettes.py
python3 experiments/pixel_ground_path/e1b_cad_derived_statistic.py
python3 experiments/pixel_ground_path/e2_detector_edge_characterisation.py  # needs GPU once
python3 experiments/pixel_ground_path/e3_mesh_model_and_covariance.py
python3 experiments/pixel_ground_path/e4_covariance_calibration.py
python3 experiments/pixel_ground_path/e5_yaw_aware_headroom.py
```

e1 onward read the dataset payload **read-only from cold storage**
(`logs/perception_datasets/COLD_STORAGE.md`); nothing is copied back into the workspace.
`robot_silhouette_model.py` is the shared object model — import it, do not restate URDF
numbers.
