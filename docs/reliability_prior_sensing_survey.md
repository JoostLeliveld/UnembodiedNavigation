# From camera-geometry prior to a good reliability GP: a survey of occlusion-sensing extensions

*Research memo, 2026-07-13. No code — decision document. Evidence tags follow the module convention:
**[REAL]** = actual captured data / real pipeline; **[SYNTHETIC]** = sampled-CAD or emulated; **[HYPOTHETICAL]** = invented input.*

## 0. Fixed frame of reference (not up for debate here)

- **Deployment inputs at t=0:** camera calibration (intrinsics `world_profiles.yaml` + per-world mounted pose) and the 2D drivable-region map (13 lane prisms, `driveable_geometry_json`, footprints only, no heights). Nothing else.
- **Initial GP** = camera geometry over drivable cells: FOV membership × projection-Jacobian range/obliquity, projected at marker height 0.35 m. It **cannot** model occlusion and will wrongly call rack-shadow cells "visible". By design.
- **Update** = drive, log per-cell detect/miss, Beta-Bernoulli grid + RBF GP in logit space (`fit_visibility_gps.py`, plus the belief-aware variants). Working code, [REAL]-validated.
- GT positions and CAD prisms (18 occluders, max height 1.90 m) are **evaluation-only**.
- "Better R_plan → better navigation" is proven in the paper; not re-derived here.

## 1. How big is the problem the initial prior can't see?

Grounded numbers, all from this repo:

| Fact | Number | Evidence |
|---|---|---|
| Detection rate over the **uniform** GP-fitting samples | mean 0.60; **39 % of positions unreliable (<0.5)** | [REAL] `usability_gp_uniform.py`, 139 positions / 557 samples |
| Camera-only prior on uniform teleport grid | AUROC **0.782** | [REAL] stack_capture2, 108 samples, 35 % occluded |
| + occlusion knowledge (real depth frame) | AUROC **0.968** | [REAL] `depth_occlusion_prior.py` |
| + occlusion knowledge (exact CAD) | AUROC **0.890** | eval-only reference |
| Data-only GP, leave-one-route-out | AUROC **0.77** (vs 0.99 for geometry prior) | [REAL] `gp_usability_validation.py` — the driven GP interpolates its routes, generalizes poorly to unvisited regions |
| Driving update on **visited** cells | RMSE 0.207 → **0.064** (3×) | [REAL] `online_update_demo.py` on honest_campaign_v1 |

So: occlusion is a large, real effect (not marginal — the earlier "92 % detected, no problem" was a route-biased sampling confound), the driving update fixes it **where visited**, and nothing currently generalizes it to **unvisited** cells. That is precisely the gap jobs (a) and (b) split across.

Two facts that shape everything below:

1. **Viewpoint-matching matters.** Occlusion *for this camera* is fully determined by the surfaces *this camera sees* (the blind volume is the shadow of visible front faces). That is why the real depth frame (same viewpoint as the RGB detector) beat exact CAD (0.968 vs 0.890): it is automatically consistent with what the detector experiences.
2. **The initial prior errs in one direction.** Camera geometry can only *over*-predict reliability (occlusion subtracts, never adds). It is an upper bound. This asymmetry is free information for the GP initialization regardless of any sensing.

## 2. Job (a): learning true reliability by driving — extensions

This job is largely solved in the repo; the realistic extensions are refinements, not new sensors.

**(a1) Encode the upper-bound asymmetry in the prior strength.** Initialize Beta cells as α₀ = n₀·p_geom, β₀ = n₀·(1−p_geom) with **small n₀ everywhere in-FOV** (a handful of virtual observations). Consequence: a genuinely blind cell flips to "unreliable" after a few misses instead of fighting a confident wrong prior; a genuinely visible cell loses almost nothing. Combined with the existing conservative planning field `sigmoid(μ − β·σ)`, this makes the wrong-in-shadows prior *safe* rather than *correct* — which is all the initial phase needs. Cost: zero. The `prior_pseudocount` machinery in `geometry_visibility.py` already implements the concept (built for the occlusion-aware variant, but the camera-only version is a strict simplification).

**(a2) Active (epistemic) survey routing.** "Just driving" is slow only if the robot drives task routes. Your active-inference planner already carries an epistemic term; the reliability GP's posterior variance is exactly an expected-information-gain field. A commissioning "survey mode" that routes through high-σ cells is the principled accelerator for (a) and is thesis-native (EFE exploring its own observation model — a genuinely nice story). This is standard informative-path-planning territory (GP field estimation with IPP, e.g. Hitz et al., JFR 2017), so it is citable, and it competes directly with structure sensing: both attack the "unvisited cells" gap, one with wheels, one with pixels. Safety during the survey comes from (a1)'s conservatism plus the onboard 2D scan (its actual job).

**(a3) Position-uncertainty handling: keep it naive.** Established [REAL] finding: naive ≈ covariance-spread ≈ oracle here (Brier 0.0136 vs 0.0137), because belief error (p95 0.127 m) ≪ GP lengthscale (0.9–1.35 m). Spread-by-covariance is the right mechanism *if* localization ever degrades to lengthscale-order, and hard gating is strictly worse (discards the informative uncertain samples). Do not spend more effort here.

**(a4) The sharp-boundary problem — the real coupling to job (b).** Occlusion shadows have step edges; a stationary RBF at ls ≈ 1 m smears them, so learned reliability bleeds ~1 m into and out of shadows, and convergence is slowest exactly at boundaries the planner cares about. Fixes in ascending cost: shorter lengthscale (needs more data everywhere), non-stationary kernels (complexity, weak thesis payoff), or **put the sharpness in the prior mean from sensed structure and let the GP learn smooth residuals** — which is the strongest single argument *for* job (b) beyond coverage: sensing doesn't just pre-fill unvisited cells, it supplies the discontinuity structure the kernel can't express.

## 3. Job (b): sensing 3D structure to predict occlusion early

All options below **create** structure by sensing (allowed); none assumes a pre-existing height map (forbidden). All plug into the GP the same way, via already-built machinery: point cloud → `height_map_from_points` → `raycast_min_clearance` from the camera pose → occlusion factor on the prior mean + a pseudo-count field (low n₀ near shadow boundaries / unknown cells, per §2(a1)). Domain stays drivable cells; the driving update is unchanged.

### 3.1 Monocular depth DL on the fixed camera's RGB

- **Senses:** dense per-pixel depth from the single existing RGB view (Depth Anything V2, Metric3D v2, UniDepth class); back-project through the known K, R, t.
- **Occlusion quality (expected):** the *ceiling* is the [REAL] 0.968 result — identical pipeline, identical (perfectly viewpoint-matched) viewpoint, with depth error as the only degradation. Because shadows are cast by rack *front faces and top edges*, which the camera sees at close-to-moderate range, the needed depths are the well-estimated ones. Height error → shadow-length error scales like h·d/(H−h) (H = 4.8 m, h ≤ 1.9 m), i.e. moderate sensitivity; a 10 % height error shifts a shadow boundary decimeters, not meters — and (a1)'s low n₀ at boundaries absorbs that.
- **The scale problem, and why this camera partially escapes it:** zero-shot monocular *metric* depth carries scene-level scale/shift bias (order 0.4 m MAE in recent benchmarks — [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Metric3D v2](https://arxiv.org/html/2404.15506v4), [wildlife benchmark](https://arxiv.org/abs/2510.04723)). But here the drivable map projected into the image yields floor pixels whose true depth is known *analytically* from calibration (z = 0 plane), spanning meters of depth range across the oblique view — a strong per-scene affine (scale, shift) correction using only permitted inputs. This is the single biggest de-risking trick available and needs no segmenter.
- **Hardware/cost:** zero. One offline inference on a median frame at commissioning (structure is static).
- **Constraint compliance:** cleanest of all options.
- **Failure modes:** (i) **OOD viewpoint** — depth nets are trained on ego-level imagery; a 4.8 m overhead-oblique warehouse view is underrepresented: the genuinely open risk. (ii) Reflective/textureless floor (floor depth is discarded anyway — heights are what matter). (iii) Sim-specific caveat: Gazebo RGB is itself OOD for these nets, so a sim validation is *indicative, not conclusive*; a sanity check on a real overhead warehouse photo should accompany it. (iv) Errors can point the unsafe way (under-estimated height → shadow predicted shorter) — mitigated by boundary-n₀, not eliminated.
- **Validation cost in this repo: nearly free and uniquely so.** The captured RGB frame and the saved real depth (`depth.npy`, 720×1280) already exist; run a monocular model on the RGB, compare against real depth, re-run the exact `depth_occlusion_prior.py` AUROC evaluation on stack_capture2. One afternoon, no new capture, and it directly replaces the circular 0.97 with an honest number.

### 3.2 Stereo / RGB-D co-located at the camera

- **Senses:** real metric depth from the camera's viewpoint — same viewpoint-matched advantage as 3.1, without the DL scale risk.
- **Critical split:** **consumer RGB-D is under-ranged for this geometry.** Optimum ranges: RealSense D435i ~6 m, D455 ~3–6 m, Kinect-class ToF ~0.5–5 m ([sensor comparison](https://www.researchgate.net/publication/383004609_Comparative_Evaluation_of_Intel_RealSense_D415_D435i_D455_and_Microsoft_Azure_Kinect_DK_Sensors_for_3D_Vision_Applications), [ToF overview](https://arxiv.org/pdf/2012.06772)); far drivable cells sit 8–12 m from the mount. Bolting a RealSense next to the camera covers the near field only. **Wide-baseline stereo is the viable variant:** infrastructure mounting allows B ≈ 0.5–1 m (impossible on a robot); σ_z ≈ z²·δd/(f·B) gives ~0.1 m at 10 m with B = 1 m — ample for 1.9 m racks.
- **Repo status — honesty flag:** the existing 0.97 (`sensed_height_prior.py`) is **[SYNTHETIC], circular** (cloud sampled *from* CAD, prior recovered *is* CAD + noise). It validates plumbing only. The honest next test is real stereo *matching* (SGM) on two rendered Gazebo views — a real-pipeline test on sim imagery.
- **Hardware/cost:** one extra camera + mount + cross-calibration (~€100s). Key cost-collapser: capture can be **one-time and temporary** — bring a rig at commissioning, capture, remove. A *permanent* second camera is a different proposition (see 3.6).
- **Failure modes:** textureless rack faces defeat matching (boxes/racks usually textured enough; floor irrelevant); extrinsic calibration error between rig and camera skews shadow geometry; one-time capture misses later layout changes — but that is exactly what the driving update catches (blind cells resume failing), so the phases cover each other.
- **Constraint compliance:** clean (senses structure).

### 3.3 Robot 3D LiDAR

- **Senses:** cm-accurate 3D structure along driven routes. Unique property among robot-mounted options: **extends prediction beyond visited cells** — driving aisle A1, the LiDAR ranges the racks and predicts the camera-shadow in A2 before ever entering it. A height map is viewpoint-agnostic, so non-viewpoint-matching is harmless here (it sees *more* than the camera, never less of what matters... with one exception below).
- **Occlusion quality:** high where the sensor sees rack tops. **Vertical-FOV censoring is the systematic failure:** a ±15° unit at ~0.2 m height sees the top of a 1.9 m rack only within ~6 m, and a 2.6 m stack (the `high` occluder variant) even closer; unseen tops mean height is a *lower bound* → shadow under-predicted → error in the **unsafe** direction. Must be handled as censored data ("≥ h observed"), which complicates the otherwise-clean pipeline.
- **Hardware/cost:** €3–10 k, plus compute, mass, and integration on a TurtleBot-class platform.
- **Strategic mismatch:** the thesis premise is an *unembodied* robot — currently a 2D scan (0.12–3.5 m) and an IMU, with all localization through the external camera. A robot carrying a 3D LiDAR could largely localize itself, which undercuts the problem statement the reliability GP exists to serve. Technically sound; narratively self-defeating.
- **Constraint compliance:** clean, but see above.

### 3.4 Robot 2D laser scan

- **Senses:** occupancy in one plane at ~0.2 m. The robot already has it (360 samples, 3.5 m range, `/scan`).
- **Occlusion quality: ~none, with direct repo evidence.** A planar scan yields footprints — which the drivable map *already encodes* — and **no heights**. It cannot distinguish a 0.4 m tote (no shadow from a 4.8 m camera) from a 2.6 m stack (long shadow). Turning footprints into shadows requires *assuming* a height, i.e. the footprint-only Level-2 prior that was already built, measured (weak lift, 0.669), and **deliberately dropped** (`freespace_prior.py`, deleted). A 2D laser re-derives the dropped rung with extra hardware steps.
- **Residual roles (real but not job (b)):** change detection — a scan hit where the drivable map says free flags layout change, a principled trigger to *reset pseudo-counts* in that region so the GP relearns; and its existing short-range safety job during (a2) survey drives.
- **Verdict:** not a candidate for occlusion prediction. Zero cost, zero (b)-value.

### 3.5 Semantic segmentation on the fixed camera

- **Settled failure as a standalone, [REAL]:** `segmented_occlusion_prior.py` — a cell is flagged occluded if its projected pixel lands on the occluder silhouette; without depth ordering this wrongly condemns floor *in front of* racks and cannot cast shadows *behind* them; AUROC **0.780**, below the camera-only baseline it was meant to improve. Not relitigated.
- **Residual component roles:** masking dynamic objects out of the median frame before 3.1's depth inference; semantic class → height *priors* ("rack ⇒ tall") fused with monocular depth as regularization. Note the floor-pixel-identification role is already covered without a segmenter by projecting the drivable map (3.1).
- **Verdict:** component, never a standalone occlusion source.

### 3.6 Multi-camera triangulation

- Two distinct values, worth keeping separate: (i) **structure sensing** — two calibrated overlapping views = wide-baseline stereo, i.e. 3.2; (ii) **coverage** — a cell blind to cam-1 may be visible to cam-2, which *dissolves* occlusion rather than predicting it. The repo's 39 %→78 % union-coverage figure is **[HYPOTHETICAL]** (invented cam-2) — directionally obvious, evidentially empty.
- **Industry context:** the RAIL deployment at Volvo ([Brorsson et al., arXiv 2512.15215](https://arxiv.org/abs/2512.15215)) runs 15 top-down PoE cameras at ~8 m (~60 m² each), robots with **no onboard LiDAR or cameras**, and handles occlusion purely by camera density and top-down mounting — no reliability map at all. Two readings: (1) at scale, industry buys cameras, not priors; (2) your single-oblique-camera setting is the deliberately hard, sparse-infrastructure end of the spectrum where a reliability model earns its keep. Both readings help the thesis if stated plainly.
- **The strongest framing for this thesis:** the reliability GP is exactly the objective function for *camera placement* — "where does the next camera buy the most reliability" falls straight out of the field. That turns multi-camera from a competing method into a downstream application, without building any of it.
- **GP integration:** per-camera reliability fields fused at the planner (e.g. complement-product of miss probabilities); prior and update machinery extend per-camera unchanged.
- **Cost:** cameras are cheap; mounting, cabling, and cross-calibration dominate. Out of scope to build; in scope to discuss.

### Comparison table

| Option | Occlusion-prediction quality | Hardware cost | Constraint-clean? | Killer weakness | Repo evidence status |
|---|---|---|---|---|---|
| 3.1 Monocular depth on fixed cam | High expected (viewpoint-matched; ceiling = 0.968 [REAL]) | **None** | Yes | OOD overhead viewpoint for depth nets | Pipeline [REAL]-validated with true depth; mono itself untested — near-free test available |
| 3.2 Wide-baseline stereo at mount | Highest (metric, viewpoint-matched) | Low (can be temporary rig) | Yes | 0.97 currently circular; consumer RGB-D under-ranged | [SYNTHETIC] only; honest SGM-in-sim test possible |
| 3.3 Robot 3D LiDAR | High near, censored heights far/tall | High (€3–10 k) | Yes | Cost + contradicts unembodied premise; unsafe-direction censoring | None in repo |
| 3.4 Robot 2D laser | ~None (footprints only) | Already on robot | Yes | No heights = the dropped Level-2 prior | [REAL-adjacent]: footprint prior built & dropped (0.669) |
| 3.5 Semantic segmentation | Negative standalone (0.780 < baseline) | None | Yes | No depth ordering | [REAL] failure, settled |
| 3.6 Multi-camera | Dissolves rather than predicts | Medium–high, per camera | Yes | Changes the problem; coverage figure hypothetical | [HYPOTHETICAL] |

## 4. Is structure sensing needed at all, versus just driving?

**The honest answer is: not *needed* for the thesis claim; cheaply *worth it* for deployment speed and first-visit safety — and only one option is cheap enough to clear that bar.**

Case for "just drive": the update provably learns occlusion where visited (3× RMSE, [REAL]); with (a1)'s upper-bound-aware initialization the wrong prior is safe rather than dangerous; (a2)'s epistemic survey routing attacks the coverage gap with the planner you already have; shadows announce themselves densely (blind cells fail every pass, so per-cell convergence is a handful of traversals); and the two-phase story — deliberately poor geometric prior, honestly calibrated by experience — is complete, defensible, and doesn't hinge on any sensing working.

Case for sensing: 39 % of uniform positions are unreliable, and the driven GP generalizes poorly off-route (0.77) — so until coverage is achieved, the planner is flying on an anti-conservative prior exactly in the cells where the thesis's own failure mode (confident routing into a blind shadow) lives; and per §2(a4), sensing contributes the sharp boundary structure the RBF kernel cannot learn efficiently. The cost of "just driving" is not wrongness at convergence — it converges — but transient risk and commissioning time, which is precisely what a deployment argument cares about.

The decision therefore reduces to price. Robot 3D LiDAR (3.3) and permanent multi-camera (3.6) buy accuracy that a few survey laps also buy, at hardware and narrative cost. The 2D laser (3.4) and segmentation (3.5) are already-falsified rungs. Stereo (3.2) is cheap-ish but currently unvalidated (circular 0.97). Monocular depth (3.1) is the only option whose marginal cost — zero hardware, one inference, an afternoon's validation against data that already exists — is low enough that "is it needed?" stops being the right question.

## 5. Recommendation

1. **Do regardless of any sensing (job a):** encode the upper-bound asymmetry — low pseudo-count Beta initialization from the camera-geometry prior + the existing conservative planning field. This is the cheapest correctness improvement available and makes the deliberately-poor prior *safe*.
2. **Primary (b) extension: monocular depth on the fixed camera's RGB**, deployed as an *optional bolt-on prior refiner*: median frame → monocular depth → floor-plane affine correction via the projected drivable map → height map → raycast → refined prior mean + boundary-aware pseudo-counts. The update phase is untouched either way — clean modularity, clean thesis story (zero-hardware Level-3). **First step is the validation, not the feature:** run a monocular model on the already-captured frame, score against the saved real depth, and re-run the existing AUROC evaluation. If the OOD-viewpoint risk kills it, you've spent an afternoon and the survey's negative result is itself reportable.
3. **Secondary, if any hardware is on the table:** a *temporary* wide-baseline stereo capture at commissioning — but only after replacing the circular 0.97 with a real stereo-matching test (two rendered views + SGM in sim). A *permanent* second camera should be framed as coverage/placement (the GP as camera-placement objective), not as an occlusion sensor.
4. **Reject for job (b):** robot 2D laser (footprints ≡ the dropped Level-2 prior; keep it for change-detection-triggered pseudo-count resets), semantic segmentation standalone (settled [REAL] failure; component roles only), robot 3D LiDAR (sound but expensive and premise-contradicting).
5. **Frame "just driving" as the baseline, not the fallback:** with (1) + epistemic survey routing, driving alone is a complete, safe, slow solution; sensing options are accelerators ranked by cost, and only 3.1 is cheap enough to be unconditionally worth attempting.

## 6. Consolidation note and current scale-out status (2026-07-15)

The exploratory runs and artifacts from retired testbeds were archived during the
2026-07-15 world consolidation. Their figures, route outcomes, and depth-prior
scores are no longer active evidence: the corresponding source worlds and
tooling are intentionally absent from this repository.

The active scale testbed is `warehouse_full_4cam`: a 24.5 × 20.5 m warehouse
with four wall-mounted external cameras at `(-6, -10)`, `(-6, 10)`, `(6, -10)`,
and `(6, 10)` m. The inward camera columns deliberately enlarge the adjacent
camera overlap used for calibration, source selection, and handover testing.
The day-zero calibration-only artifact reports 99.2% union coverage and 42.2%
multi-camera overlap. It contains no detector training records, no measured
per-camera reliability field, and no validated fused estimate.

The next evidence chain is therefore explicit: collect D0/D1 records for each
camera; fit the four frozen-pipeline GPs independently; collect synchronized D2
overlap records; then evaluate selection, handover, and conservative fusion.
The live A–D views and top-down overview in
`logs/studies/multicamera_commissioning_bigwarehouse/four_camera_showcase/`
are layout/runtime evidence only. They demonstrate that the four-stream system
is live; they do not support a reliability or planner-performance claim.

### Sources
- [Brorsson et al., Infrastructure-based AMRs for Internal Logistics (RAIL), arXiv 2512.15215](https://arxiv.org/abs/2512.15215)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) · [Metric3D v2 (TPAMI)](https://arxiv.org/html/2404.15506v4) · [monocular metric depth benchmark, arXiv 2510.04723](https://arxiv.org/abs/2510.04723) · [MDPI survey on monocular metric depth](https://www.mdpi.com/2073-431X/14/11/502)
- [RealSense/Kinect sensor comparison](https://www.researchgate.net/publication/383004609_Comparative_Evaluation_of_Intel_RealSense_D415_D435i_D455_and_Microsoft_Azure_Kinect_DK_Sensors_for_3D_Vision_Applications) · [ToF depth camera overview, arXiv 2012.06772](https://arxiv.org/pdf/2012.06772)
- Hitz et al., "Adaptive continuous-space informative path planning for online environmental monitoring," JFR 2017 (GP-field IPP, for §2(a2))
