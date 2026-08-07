# pixel_ground_path — candidate pixel→ground path for infrastructure-camera localization

> ## OUTCOME 2026-08-07: the candidate was NOT adopted
>
> The deployed measurement is **inverse perspective mapping** — box bottom-centre
> intersected with the floor plane (`contact_z_m = 0.0`, artifact
> `projection_calibration_v4`). This study's box-centre @ `z* = 0.085 m` candidate is
> **rejected as method** and retained as **the evidence that justifies the choice**. All six
> scripts stay runnable and in place; nothing here was moved or deleted.
>
> | path | mean | radial bias | tuned scalars |
> |---|---|---|---|
> | bottom @ floor — **DEPLOYED**, textbook IPM | 66.6 mm | −27.3 mm | 0 |
> | centre @ 0.085 m — this study's candidate, rejected | 50.4 mm | +4.0 mm | 2 (`alpha`, `z*`) |
> | bottom @ 0.05 m — *previously* deployed, now superseded | 110.2 mm | −94.1 mm | 1 (free constant) |
>
> The 16 mm the candidate wins is not worth two search-chosen scalars and a paragraph of
> defence. The 44 mm the *old* default lost is what actually mattered, and that is fixed.
>
> **Correction to the wording below:** §1 calls `z* = 0.085 m` *derived*. That overstates it.
> It is the argmin of a grid search (81 values of z on [0, 0.20] × 11 values of alpha) over a
> CAD silhouette grid. The true optimum was `alpha = 0.4, z = 0.06` (37.4 mm RMS); a
> pre-registered 2 mm plateau tie-break snapped it to the named box centre (38.3 mm). Zero
> robot measurements entered the objective — but it is a numerical optimum, not geometry.

<!-- RESEARCH-METADATA:START (generated; edit research/registry.yaml) -->

```yaml
experiment_id: EXP-PIXEL-GROUND
status: LOCKED
claim_ids:
- C1
assumption_ids:
- A01
- A02
- A03
- A08
- A09
- A14
- A20
reviewer_question_ids:
- RQ08
- RQ14
- RQ15
figure_ids:
- F01
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
- paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/RESULTS.md
- paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/summary.json
- paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/provenance.json
archive_rule: Preserve summaries candidate model provenance and the external E6 challenge
  permanently.
next_action: Preserve as supporting analysis only; keep candidate code experiment-local
  until held-out covariance occlusion yaw and sequential-correlation gates pass.
```

<!-- RESEARCH-METADATA:END -->


**Question.** What is the ONE defensible way to turn a detection in a fixed overhead camera
into a metric ground position with an honest covariance — logical, standard, and
commissionable by an integrator?

**Claim served.** C1 (quality representation), as registered in `research/registry.yaml`.
This study challenges the interpretation of the fitted v2/v4 correction terms; it does not
silently supersede their artifacts. The calibration components must first be made
identifiable in a yaw-diverse, route-disjoint study.

**Results:** [`logs/studies/pixel_ground_path/RESULTS.md`](../../logs/studies/pixel_ground_path/RESULTS.md).
Read that before this file — it contains the numbers, and it records three claims from the
first draft of this design that the experiments **falsified**.

**Audit decision, 2026-08-06: supporting analysis only.** The mean and covariance equations
are coherent after fail-closed input checks, and the experiment-local implementation is now
*proved* to be the path the locked numbers were measured on rather than a lookalike — but
this is not required shared
infrastructure for `EXP-USABLE` and is not ready for a hidden runtime substitution. The later
source benchmark must freeze one observation definition across every source arm; changing from
the historical bottom-centre/v4 path to this one would change `p_qual`, its covariance, and
its gating semantics rather than merely supply an implementation detail. Today the evidence is
one simulated robot, clear boxes, four discrete cardinal yaws and open-loop projection, while
the proposed size-consistency gate is unvalidated. Carry this work as the candidate
measurement arm and supporting error analysis until the integration gates below are met.

The second audit pass (same day) confirmed the decision and added three reasons it is the
right one, none of which were visible in the first:

1. **`R` is not licensed for sequential fusion.** `Σ_yaw` is temporally correlated, so the
   per-detection NEES 2.83 says nothing about what a filter integrating it will believe. This
   is a blocker in its own right and is the single thing most likely to be assumed away by a
   downstream reader (§3).
2. **There is deliberately no runtime opt-in.** The candidate statistic and covariance stay
   under this experiment until the observation contract, correlated-error treatment and
   validity gate can be promoted together (§1).
3. **The evidence was not re-runnable** — all six scripts pointed at a path the cold-archive
   move deleted. Fixed here; the evidence base is 1849 index rows and 1844 scored detections,
   both now confirmed against the recorded summaries.

---

## 1. The path

Estimand: the vertical projection of the robot's `base_footprint` origin onto the floor.

| step | choice | why |
|---|---|---|
| 1 | **Object model = the URDF visual meshes** (body box, two tyres, LiDAR puck), assembled in the `base_footprint` frame | The semantic mask renders the visual meshes, not the collision primitives, and they differ by 6.8 mm at the top. A bounding cylinder is wrong by −4.3 px in width. |
| 2 | **Pixel statistic = the box centre**: `u = u_centre`, `v = v_bottom + 0.5·(v_top − v_bottom)` | Chosen at design time by minimising yaw-marginal error over a CAD grid. The α curve is a plateau over [0.3, 0.6]; a pre-registered tie-break prefers the named statistic. |
| 3 | **Inversion: back-project the box centre onto the plane `z* = 0.085 m`** | Closed form, one matrix multiply, no heading input, no iteration. **50.4 mm, unbiased.** |
| 4 | **Proposed validity gate, not implemented or validated**: predict the box size at the solved position and gate on its Mahalanobis distance | Intended to catch border clipping and shelf occlusion. Size may be a gate, never an estimator — 1 px of box height is 0.89 m of range. |

`z* = 0.085 m` is *derived* (it minimises the CAD heading-marginal error), not chosen.
Contrast the deployed `contact_z_m = 0.05`, a free operator constant worth 94 mm of radial
bias.

**Model choice within this supporting arm:** the heading-blind path above is the one retained.
e5 showed that
conditioning the inversion on the filter's heading estimate reaches 17.9 mm instead of
50.4 mm, and that it still beats heading-blind up to 45° of heading error. It is deliberately
**not** adopted: it needs an iterative solve, a heading-quality monitor with a fallback, and a
defence of the heading estimate itself. 5 cm unbiased is enough, and this version is defensible
in four sentences. The heading-aware arm stays recorded as measured headroom, not as method.

Experiment surface only: `box_projection.py` contains the point and covariance equations and
their frozen candidate constants. Production `reliability.projection`, detector selection,
observation contracts and launch/config interfaces remain unchanged. Adoption requires one
explicit end-to-end promotion task; there is no half-wired `bbox_center` option.
The statistic and its plane are **one coupled choice** — the box centre with the old contact
plane is worse than either pairing.

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
candidate has **no per-camera correction parameter**. That does not make the full covariance
data-free: C7 is a truth-backed global commissioning quantity for this robot/detector/mount
combination.

## 3. Uncertainty per detection

```
R = J(u, v; z*) · Σ_uv · J(u, v; z*)ᵀ  +  Σ_yaw
```

Two terms, because they scale differently: the pixel term grows with range — the projection
Jacobian's radial sensitivity measures **1.6 cm/px at 5 m to 4.8 cm/px at 16 m** on these
mounts — while `Σ_yaw` is roughly constant in metres. Dropping `Σ_yaw` gives mean NEES
**15.21** with 59 % of samples above the 9.21 chi-square gate; keeping both gives **2.83,
uniform 2.54–3.18 across camera, range and yaw**, with 1.4 % above the gate.

Two corrections to earlier drafts of this section, both found by
[`verify_candidate_matches_evidence.py`](verify_candidate_matches_evidence.py):

- the per-range figures "NEES 26 at short range to 7 at long range" were **never recorded**
  by e4, which only stratifies the full model. The recorded contrast is the aggregate above.
  The direction is still right — omitting a range-independent term hurts most where the pixel
  term is smallest — but the numbers are retired rather than softened;
- the closed form `(H² + d²)/(f·H)` is a small-angle approximation and over-predicts the
  measured Jacobian by ~24 % at 12 m and ~56 % at 16 m, because angular resolution per pixel
  falls off-axis. The code differentiates numerically for that reason, and the propagated
  covariance is insensitive to the step (worst relative trace change 1.2e-5 over 0.005–2 px).

`Σ_yaw` is the price of being heading-blind, and since we chose to be heading-blind it is
always on. `project_box_to_world_with_covariance(..., sigma_yaw_m=None)` drops it, and exists
only so the heading-aware arm can be tried later without touching this code.

Implementation frame convention: `J` maps `(u,v)` perturbations directly into map `(x,y)`.
`Σ_yaw = diag(σ_radial², σ_left-lateral²)` is separately rotated from the camera-to-estimate
bearing frame into map axes, then added. A zero horizontal bearing has no defined radial
frame and fails closed. Non-finite or non-positive-area boxes, alpha outside `[0,1]`,
non-finite/negative noise, a plane below the floor or at/above the camera, invalid ray-plane
intersections and degenerate numerical Jacobians also fail closed. These numerical checks are
not an occlusion or apparent-size gate.

e4 builds the same matrix in the *bearing* frame and scores NEES there; the candidate module builds it
in map axes. They are one matrix in two frames, verified to 6.5e-19 m² over 1600 boxes on the
four real cameras, and NEES is invariant under the rotation. One deliberate difference: e4
rotates `Σ_yaw` using the bearing to the **true** pose, while the candidate uses the bearing to
the **estimated** point, because that is the only one a deployed caller has. Measured over all
1844 detections that is worth at most 1.3e-5 m², i.e. **1.4 % of σ_radial²**.

> **`Σ_yaw` is not per-frame white noise, and this is the largest single obstacle to using
> `R` in a filter.** It is the marginal spread of a *deterministic* function of the robot's
> heading, so successive detections carry very nearly the same offset rather than independent
> draws. NEES 2.83 is a per-detection calibration over 1849 independent poses; it is not a
> licence to integrate this `R` sequentially as if it were measurement noise. A filter that
> does will shrink its covariance as 1/n against a floor that does not shrink — precisely the
> mechanism `bayesian_filter_showcase` measured (stated 1.9 cm against 5.3 cm actual, 41.9 %
> of truth outside the stated 95 % ellipse). Sequential use needs the heading term carried as
> a correlated component — a per-camera floor, a bias state, or the heading conditioning of e5
> — not as `R`.

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
6. **Detections are unoccluded and fully in frame.** Every sample in the evidence base is
   `occlusion_state == clear`.

The mean requires **no heading input**. The covariance is not assumption-free: it treats the
CAD-grid yaw marginal as representative of deployment. The 30 mm `Σ_yaw` term expresses that
chosen marginal rather than guaranteeing coverage for every possible heading distribution.

## 5. Weaknesses

1. **`Σ_uv` cannot be sized without robot poses.** Inter-camera disagreement — the obvious
   GT-free route, 575 pairs available — returned 4× too small a variance, because the
   dominant error is a shared cause that partly cancels between views. A CAD-computed
   correction for that shortfall was also wrong (11.13 predicted vs 4.11 needed). So
   commissioning needs a run with truth, or a deliberate conservative inflation: the
   design-time-only covariance gives NEES 3.52, i.e. over-confident by 1.76× in variance.
2. **The reported NEES is not held-out commissioning validation.** e4 sizes the silhouette
   component of `Σ_uv` and evaluates NEES on the same 1844 truth-backed rows. The detector
   component uses its labelled validation split, but the full covariance still needs a
   separate commissioning/evaluation split before promotion.
3. **The residual covariance is still 1.4× over-confident** (NEES 2.83 vs 2.0) even with
   poses. It is uniform, so a single scalar fixes it, but that scalar is fitted.
4. **Four discrete empirical yaws.** The CAD design grids sample yaw more densely (e4 uses
   16 headings), but every real-box/NEES row is at 0/90/180/270°. The 45° diagonal, the worst
   case for a rectangular footprint, is therefore modelled but not empirically validated.
5. **Occlusion is entirely untested.** The size gate in step 4 is designed but not validated,
   and occlusion is where the bottom-edge path fails worst — so the *comparison* against the
   deployed path is also untested in its worst regime.
6. **Open-loop only.** No closed-loop run, no navigation claim. The 9.21 chi-square gate and
   `pixel_max_correction_jump_m` interact with a 110 mm→50 mm change in a way nobody has
   measured.
7. **We are knowingly leaving 30 mm on the table.** `Σ_yaw` is the largest term and it exists
   only because the estimator ignores a heading estimate the filter already has. That is a
   deliberate simplicity trade, not an oversight — but a reviewer may well ask, and the answer
   has to be "measured, 17.9 mm, chose not to".
8. **Single-robot evidence.** All 1849 samples are one TurtleBot3 Burger in one world.
9. **`Σ_yaw` is temporally correlated**, so the per-detection NEES does not transfer to
   sequential filtering. See the box in §3; this is an integration blocker, not a caveat.
10. **The shipped constants are rounded.** `Σ_uv` ships as `(1.15, 0.77)` px against e4's
    `(1.1516, 0.7680)`. That is ~0.3 % of the pixel-term variance — visible only in the
    no-`Σ_yaw` variant (15.17 recomputed against 15.21 recorded), invisible in the full model.
11. **The mount is load-bearing and was only prose.** `α`, `z*` and `Σ_yaw` are functions of
    the 6.10 m / 0.92 rad viewing geometry, which all four evidence cameras share exactly.
    `box_projection.BOX_STATISTIC_REFERENCE_MOUNT` and `box_statistic_mount_deviation`
    now make that checkable; the deviation is deliberately *reported*, not gated, because the
    tolerance is an integration decision and not a property of the projection.

## 5a. Integration gates and proposal (no runtime change in this workstream)

Treat adoption as a new, explicit measurement arm rather than a calibration-v4 replacement:

1. Freeze a new measurement ID tying together `bbox_center`, `z*=0.085 m`, the robot visual
   model, detector/label convention, camera mount family and covariance provenance. Never
   combine the centre pixel with the historical contact plane or v4 corrections.
2. Extend the observation contract and diagnostics to carry `bbox_center` (or consume the
   already-carried full `bbox_xyxy`) and admit `selected_pixel_source='bbox_center'`; add an
   explicit opt-in parameter to both detector paths. Keep `bbox_bottom` as the default and
   run bit-for-bit/default regression tests.
3. Commission `Σ_uv` with robot poses on a split distinct from the NEES evaluation split.
   Re-derive alpha/plane/`Σ_yaw` for any different camera height/pitch or robot appearance;
   do not reuse these constants merely because intrinsics match.
4. Implement and validate the size-consistency gate on deliberately clipped and occluded
   detections, including a fail-closed fallback to no update. The existing minimum-area check
   is not evidence for this model-based gate.
5. Run a yaw-diverse validation including 45° diagonals, then a separately named closed-loop
   A/B arm against the frozen historical path. Only after calibration, gate, NEES-stratum and
   closed-loop acceptance may the source benchmark choose this as its one shared measurement
   definition; all source arms must then use it unchanged.

## 6. Failure modes

| failure | trigger | symptom | mitigation | tested? |
|---|---|---|---|---|
| Clipped box at the image border | robot at FOV edge | box centre wrong by half the clip; error grows without bound | size-consistency gate (step 4) | **no** |
| Bottom occluded by a rack | robot behind foreground shelf | 1.47 cm per clipped pixel through the centre path, 2.91 through the bottom path | size gate + `visible_height_fraction` | **no** |
| — (heading is not an input) | — | the heading-blind path cannot fail this way; that is what `Σ_yaw` buys | n/a | n/a |
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
3. **Identify calibration components** on a grid that varies camera, region and yaw
   independently. The current C/D correction signal is explainable by object geometry.
4. **Only after those gates, consider a runtime arm** with an explicit observation-contract
   change and a correlated-error treatment for `Σ_yaw`.

## Reproduce

```bash
# verification: no dataset re-run, writes nothing into logs/, exits non-zero on any drift
python3 experiments/pixel_ground_path/verify_candidate_matches_evidence.py
python3 -m pytest tests/experiments/test_pixel_ground_box_projection.py -q

python3 experiments/pixel_ground_path/e0_pixel_statistic_geometry.py        # geometry (see correction)
python3 experiments/pixel_ground_path/e1_object_model_vs_real_silhouettes.py
python3 experiments/pixel_ground_path/e1b_cad_derived_statistic.py
python3 experiments/pixel_ground_path/e2_detector_edge_characterisation.py  # needs GPU once
python3 experiments/pixel_ground_path/e3_mesh_model_and_covariance.py
python3 experiments/pixel_ground_path/e4_covariance_calibration.py
python3 experiments/pixel_ground_path/e5_yaw_aware_headroom.py
python3 experiments/pixel_ground_path/e6_external_log_validation.py       # no GPU, no dataset
```

`verify_candidate_matches_evidence.py` is the guard against the failure this study is most
exposed to: e3/e4/e5 each carry a **private copy** of the pixel→ground math, and the candidate
module was written afterwards, which is exactly how `operational_residual_rcond/exp3`
found the detector node projecting with one fewer degree of freedom than the camera manager.
It re-derives e4's four NEES variants *through the candidate module* on the real 1844 detections and
reproduces every recorded number (45.39 / 3.52 / 15.17 / 2.83, gate fractions
0.902 / 0.028 / 0.589 / 0.014), plus the mount, the frozen constants and the frame change.

**Dataset location.** e1–e5 read the payload (images and label polygons) from cold storage;
the 2026-08-05 archive pass moved it and the hard-coded workspace path in all six scripts died
with it. `dataset_paths.py` now resolves it, preferring a root that actually carries `labels/`
and failing with every location it tried rather than silently analysing nothing. The
in-workspace `logs/perception_datasets/.../merged` copy is metadata only — enough to read the
calibration index, not to re-derive the silhouette term.

e6 is the only one that does not touch the purpose-built capture: it scores the object model
against the deployed driving logs in
`logs/studies/external_camera_bias_model/exp1_residual_characterization/residuals.csv`, and
so is the external check on e1–e5. It also records the provenance fact that reframes that
older dataset — `smoke1` is one heading (+90.0° on 2475/2475 rows) and `smoke2` is another
(0.0° on 1140/1140), which is why no correction fitted there ever generalised.
The permanent, claim-bounded E6 summary and provenance live under
`paper_artifacts/correlated_error_icra/results/F01/pixel_ground_e6/`; the per-sample table
remains ignored generated output.

e1 onward read the dataset payload **read-only from cold storage**
(`logs/perception_datasets/COLD_STORAGE.md`); nothing is copied back into the workspace.
`robot_silhouette_model.py` is the shared object model — import it, do not restate URDF
numbers.
