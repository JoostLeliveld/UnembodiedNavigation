# Thesis master plan — two-world strategy, storyline, contributions (2026-07-15)

> Provenance: plan delivered by Joost on 2026-07-15 (from an external planning conversation).
> Saved verbatim as the storyline source of record. The repo mapping of this plan lives in
> [README.md](README.md); statuses in [registry.yaml](registry.yaml).
> "Original warehouse" = `warehouse_aws.world.sdf` (1 external camera). "Large warehouse,
> four cameras" = the big multicamera world (chapter 08).

# 1. Use the two warehouses for different scientific purposes

The cleanest division is:

* **Original warehouse, one camera:** controlled research testbed. Use it to establish causality, compare models fairly, debug interfaces, and demonstrate the main contribution.
* **Large warehouse, four cameras:** scale and deployment testbed. Use it only after the single-camera model and metrics are frozen. It should demonstrate coverage, camera heterogeneity, handover, fusion and commissioning burden.

Do not discover the uncertain-input GP, tune its target, redesign R_plan, and debug four-camera fusion simultaneously in the large warehouse. That would make failures impossible to attribute.

The original warehouse is particularly suitable because it already has a locked one-camera surface, four tasks, five seeds and constant-versus-reliability-aware planning results. Preserve that as the reference campaign rather than modifying it in place.

## Recommended thesis structure

### Contribution 1 — Primary

> **Learning spatial external-camera trust from uncertain robot beliefs during ordinary warehouse driving.**

Use the original warehouse.

Include:

* uncertain-input GP;
* trust-target comparison;
* simple confidence calibration;
* weak FOV/range prior as an ablation;
* frozen trust-to-R_plan interface;
* downstream one-camera navigation evaluation.

### Contribution 2 — Choose one

Either:

> **Factorised camera observation quality:** detection availability and conditional measurement noise are modelled separately.

Primarily original warehouse, then cross-camera validation in the large warehouse.

Or:

> **Reliability-aware multi-camera selection, handover and conservative fusion.**

Large four-camera warehouse.

### Separate future contribution

> **Active commissioning routes that reduce trust-map uncertainty efficiently.**

Large warehouse.

Geometry/occlusion modelling should only become a contribution if it uses realistically available geometry and substantially exceeds range/FOV baselines. Otherwise, keep it as a prior ablation. The current project rules already require geometry predictions to remain `MODEL ONLY` until checked against detector evidence.

---

# 2. Storyline from beginning to end

The research folders and presentation should follow one causal chain:

```text
00 Warehouse problem and existing evidence
        ↓
01 Operational sensing and belief interface
        ↓
02 What camera trust should mean
        ↓
03 Learning trust at uncertain robot locations
        ↓
04 Converting trust into R_plan
        ↓
05 One-camera navigation evidence
        ↓
06 Cold-start priors and data efficiency
        ↓
07 Scaling to the four-camera warehouse
        ↓
08 Camera selection, handover and fusion
        ↓
09 Active commissioning
        ↓
10 Final thesis campaign and evidence package
```

This matches the intended project story:

```text
camera evidence
→ spatial camera reliability
→ effective planning covariance R_plan(s)
→ reliability-aware route behaviour
```

and avoids treating the GP, covariance and planner as one black box.

---

# 3. Recommended repository structure

Keep reusable software separate from research claims. The numbered folders should hold experiments and evidence, not duplicate implementations.

```text
thesis/
├── README.md
├── experiment_registry.yaml
├── current_runtime_contract.yaml
│
├── src/
│   ├── perception/
│   ├── projection/
│   ├── localization/
│   ├── reliability/
│   │   ├── point_gp.py
│   │   ├── uncertain_input_gp.py
│   │   ├── availability_model.py
│   │   ├── conditional_noise_model.py
│   │   └── priors.py
│   ├── observation_model/
│   │   └── trust_to_rplan.py
│   ├── multicamera/
│   │   ├── selector.py
│   │   ├── handover.py
│   │   └── fusion.py
│   ├── commissioning/
│   └── evaluation/
│
├── data/
│   ├── original_warehouse/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── manifests/
│   └── large_warehouse_4cam/
│       ├── raw/
│       ├── processed/
│       └── manifests/
│
└── research_story/
    ├── 00_problem_and_existing_baseline/
    ├── 01_operational_belief_and_logging/
    ├── 02_trust_target_and_calibration/
    ├── 03_uncertain_input_gp/
    ├── 04_factorised_observation_model/
    ├── 05_trust_to_rplan/
    ├── 06_original_warehouse_navigation/
    ├── 07_weak_priors_and_geometry/
    ├── 08_large_warehouse_scaling/
    ├── 09_multicamera_handover_fusion/
    ├── 10_active_commissioning/
    └── 11_final_thesis_campaign/
```

Every research folder should use the same evidence structure:

```text
03_uncertain_input_gp/
├── README.md
├── claim.md
├── assumptions.md
├── experiment_matrix.yaml
├── configs/
├── manifests/
├── scripts/
├── artifacts/
│   ├── figures/
│   ├── tables/
│   ├── videos/
│   ├── metrics/
│   └── logs/
└── evidence.yaml
```

The README should always contain:

1. claim;
2. realistic assumptions;
3. forbidden assumptions;
4. inputs and outputs;
5. baselines;
6. validation gate;
7. exact commands;
8. frozen configurations;
9. figures and tables;
10. caveats.

That is consistent with the project's evidence-first module template.

Use manifests to reference shared datasets rather than copying the same logs into several folders.

---

# 4. World allocation by research direction

| Research direction            | Original warehouse                  | Large four-camera warehouse    | Status                           |
| ----------------------------- | ----------------------------------- | ------------------------------ | -------------------------------- |
| Uncertain-input GP            | **Primary proof**                   | Frozen-method generalisation   | Main contribution                |
| Trust-target comparison       | **Main ablation**                   | Cross-camera consistency check | Supporting                       |
| Confidence calibration        | **Develop and validate**            | Test transfer per camera       | Supporting or minor contribution |
| Factorised availability/noise | **Controlled proof**                | Per-camera validation          | Possible second contribution     |
| Trust-to-R_plan               | **Mechanism demonstration**         | Integration at scale           | Interface                        |
| Larger warehouse              | No method development               | **Scale demonstration**        | Evaluation setting               |
| Onboard sensors               | State/safety interface              | State/safety interface         | Assumption/infrastructure        |
| Weak FOV/range prior          | **Controlled data-efficiency test** | Commissioning-scale test       | Ablation                         |
| Geometry/occlusion            | Controlled model ablation           | Only with realistic geometry   | Optional separate contribution   |
| Multi-camera fusion/handover  | Not necessary                       | **Primary environment**        | Separate contribution            |
| Active commissioning          | Tiny sanity test only               | **Primary environment**        | Separate contribution            |

---

# 5. Folder-by-folder research programme

## `00_problem_and_existing_baseline`

### Purpose

Establish that spatially constant camera covariance is inadequate and that spatial reliability can affect navigation.

### Warehouse

Original one-camera warehouse only.

### Existing evidence to preserve

The current locked campaign has:

* four routes;
* five seeds;
* C1 constant camera covariance;
* C2 GP-scaled camera covariance;
* 15/20 versus 20/20 clean goals;
* 4/20 versus 0/20 GT geometry breaches;
* zero physics contacts.

These are the present active results and must remain separate from older submitted-paper numbers.

### Presentation figures

**Figure 00A — Warehouse problem**

Top-down warehouse map with:

* camera position;
* approximate FOV;
* camera-good region;
* camera-poor region;
* two possible paths to the same goal.

**Figure 00B — Existing paired route**

C1 and C2 trajectories on the same map.

**Figure 00C — Current campaign table**

| Condition                | Clean goals | GT breaches | Contacts |
| ------------------------ | ----------: | ----------: | -------: |
| Constant R               |       15/20 |        4/20 |        0 |
| Reliability-aware R_plan |       20/20 |        0/20 |        0 |

### Video

**V00 — Existing motivation video**

Split screen:

* left: constant covariance;
* right: learned covariance;
* overlay: belief ellipse, current R_plan, detector status.

### Conclusion

> Spatially varying camera trust can matter downstream, but the current evidence does not establish how that trust map can be learned realistically from uncertain robot poses.

This creates the opening for the new thesis.

---

## `01_operational_belief_and_logging`

### Purpose

Show how ordinary driving produces realistic training records.

### Warehouse

Original warehouse.

### Required data record

For every camera frame:

(t_k, μ_k^-, P_k^-, opportunity_k, m_k, c_k, u_k, v_k, Δt_k)

The camera outcome must be paired with the **prior** belief before that frame updates the robot pose.

### Initial plots

**Figure 01A — Commissioning route with covariance ellipses**

Top-down route showing:

* accepted camera updates;
* misses;
* camera-unobserved sections;
* belief ellipses growing during odometry-only movement.

**Figure 01B — Covariance through time**

Three aligned traces:

* time since last accepted camera update;
* tr(P_xy);
* evaluation-only actual position error.

**Figure 01C — Covariance calibration**

Expected versus empirical 50%, 90% and 95% ellipse coverage.

**Figure 01D — Causal timing diagram**

```text
odometry prior
→ record μ−, P−
→ detector outcome
→ save trust observation
→ optional camera update
```

### Video

**V01 — Data collection**

Top view plus camera image:

* belief ellipse expands outside observation;
* detector hit/miss shown;
* ellipse contracts after an accepted camera update;
* collected GP samples appear live.

### Gate

Do not proceed to uncertain-input fitting until the covariance is at least directionally related to actual error. The project workflow explicitly treats poor covariance calibration as a stop condition.

---

## `02_trust_target_and_calibration`

This folder combines **trust-target comparison** and ordinary **confidence calibration** because both answer:

> What operational camera signal should the spatial GP learn?

They are not yet separate contributions.

### Warehouse

Original warehouse first. Use the four-camera warehouse later only to check whether conclusions transfer between cameras.

### Candidate targets

* raw YOLO confidence;
* calibrated confidence;
* hit/miss or usable-detection indicator;
* track continuity;
* innovation magnitude;
* NIS acceptance;
* combined scalar trust.

Keep evaluation-only targets separate:

* projected bottom-centre BEV error;
* camera localisation error;
* state-estimation improvement.

The current detector is specifically required to be judged through its projected bottom-centre localisation point, not just mAP or confidence.

### Initial plots

**Figure 02A — Confidence versus localisation error**

Scatter or hexbin: YOLO confidence versus GT projected BEV error.

Colour by:

* distance;
* image edge distance;
* occlusion category;
* pose uncertainty.

This immediately tells the supervisor whether raw confidence is a defensible proxy.

**Figure 02B — Confidence reliability diagram**

Predicted confidence bins versus empirical usable-detection frequency.

**Figure 02C — Regional miss-rate map**

Spatial map of P(usable detection | s).

**Figure 02D — Target comparison**

| Target | Held-out NLL | Brier | Correlation with BEV error | Predicts NIS rejection |
| ------ | -----------: | ----: | -------------------------: | ---------------------: |

**Figure 02E — Failure examples**

A panel of:

* high confidence, large projection error;
* low confidence, accurate projection;
* miss in nominal FOV;
* ambiguous opportunity.

### Video

**V02 — Detector reliability failure reel**

Short camera clips with overlays of:

* confidence;
* projected point;
* evaluation-only true robot position;
* resulting BEV error.

### Possible conclusions

1. Raw confidence predicts availability but not localisation accuracy.
2. Calibration improves detection probabilities but does not solve geometric bias.
3. Hit/miss and conditional error should be modelled separately.

The third conclusion justifies the factorised model.

---

## `03_uncertain_input_gp`

This is the primary methodological folder.

### Warehouses

* Synthetic diagnostic first.
* Original warehouse for the scientific proof.
* Large warehouse only as frozen-method generalisation.

### Baselines

| ID | Method                                          |
| -- | ----------------------------------------------- |
| U0 | Global constant reliability                     |
| U1 | Point-input GP                                  |
| U2 | Point-input GP with larger learned length scale |
| U3 | Gaussian spatial smoothing                      |
| U4 | Covariance-weighted point-input GP              |
| U5 | Uncertain-input expected-kernel GP              |
| U6 | GT-position GP, evaluation-only upper reference |

### Initial investigations

**Figure 03A — One-dimensional explanation**

Same measurements, with:

* point-input GP;
* uncertain-input GP;
* increasing input covariance.

This is the clearest explanation slide.

**Figure 03B — Two-dimensional anisotropic synthetic case**

Show:

* true field;
* uncertain training points with ellipses;
* point GP;
* Gaussian blur;
* uncertain-input GP;
* posterior uncertainty.

The key visual question is whether U5 differs meaningfully from ordinary smoothing.

**Figure 03C — Original warehouse training observations**

Route with:

* green detections;
* red misses;
* covariance ellipses;
* confidence colour;
* nominal camera support.

**Figure 03D — Original warehouse trust maps**

Side-by-side:

* U1 point GP mean;
* U3 smoothing;
* U4 weighted GP;
* U5 uncertain-input GP;
* each method's posterior uncertainty.

**Figure 03E — Uncertainty-scaling ablation**

P_k^(α) = α P_k

Plot held-out NLL or Brier score for α ∈ {0, 0.5, 1, 2, 4, 8}.

A convincing result shows that U5 degrades more gracefully as meaningful pose uncertainty grows.

**Figure 03F — Performance by time since observation**

Held-out performance versus:

* seconds since last accepted external-camera update;
* covariance trace;
* distance travelled odometry-only.

### Video

**V03 — Same data, different GP assumptions**

Animate the construction of:

* point-input trust map;
* weighted map;
* uncertain-input map.

Display covariance ellipses so the mechanism is visible.

### Primary conclusion

> Modelling the uncertain spatial locations of camera-reliability observations improves held-out trust calibration when odometry uncertainty is significant.

### Negative but useful conclusion

> Under the uncertainty regime of this warehouse, a simple smoother performs equivalently, so the full expected-kernel treatment is unnecessary.

That is a legitimate research result.

---

## `04_factorised_observation_model`

This is the clean location for the possible second contribution.

### Research claim

> Detection availability and localisation accuracy given detection are different spatial phenomena and should not be collapsed into one confidence-derived trust value.

### Model

p_det(s) = P(usable detection | s)

and

R_cond(s) = Cov(e_camera | usable detection, s).

Then:

R_plan(s) = g(p_det(s), R_cond(s)).

### Warehouses

Original warehouse for method development. Four-camera warehouse for checking whether different cameras have different availability/noise structures.

### Initial plots

**Figure 04A — Two maps, not one**

Side-by-side:

* probability of receiving a usable observation;
* conditional localisation-error magnitude or covariance.

This may be the strongest figure in the second contribution.

**Figure 04B — Four-region taxonomy**

Examples of:

1. frequent and accurate;
2. frequent but inaccurate;
3. rare but accurate when detected;
4. rare and inaccurate.

**Figure 04C — Innovation calibration**

NIS histogram and expected chi-square reference for:

* constant R;
* confidence mapping;
* spatial conditional noise;
* factorised model.

**Figure 04D — Ablation**

| Model                  | Availability NLL | Innovation NLL | NIS coverage | Navigation |
| ---------------------- | ---------------: | -------------: | -----------: | ---------: |
| Confidence scalar      |                  |                |              |            |
| Availability only      |                  |                |              |            |
| Conditional noise only |                  |                |              |            |
| Factorised             |                  |                |              |            |

### Video

**V04 — Same confidence, different measurement consequence**

Show two robot detections with similar confidence but different projected error and filter correction.

### Contribution decision

Call this a separate contribution only when:

* both components are learned;
* both are independently validated;
* the factorisation beats scalar trust;
* it changes R_plan or navigation in a measurable way.

Otherwise, it remains a target ablation inside Contribution 1.

---

## `05_trust_to_rplan`

This is an interface, not normally a contribution.

### Warehouse

Original warehouse first; reuse unchanged in the large warehouse.

### Required plots

**Figure 05A — Semantic separation**

```text
detector reliability
≠ conditional measurement noise
≠ GP uncertainty
≠ R_plan
```

**Figure 05B — Mapping curve**

τ ↦ σ_plan².

Show:

* lower bound;
* upper bound;
* monotonicity;
* treatment of GP-unsupported regions.

**Figure 05C — Map triptych**

Same floor plan with:

1. learned trust;
2. GP epistemic uncertainty;
3. effective R_plan.

**Figure 05D — Parameter sensitivity**

Route choice or predicted belief uncertainty versus the mapping slope and endpoints.

### Video

Usually no standalone video is needed. Display R_plan live in the navigation video.

### Conclusion

> The learned reliability field affects planning only through a frozen, bounded observation-covariance interface.

The project explicitly requires the GP to remain distinct from R_plan, and the matrix must retain clear shape and units.

---

## `06_original_warehouse_navigation`

This is where the complete primary storyline is showcased.

### Conditions

| Condition | Meaning                          |
| --------- | -------------------------------- |
| N0        | No external-camera enhancement   |
| N1        | Constant covariance              |
| N2        | Existing/reference trust map     |
| N3        | Point-input operational GP       |
| N4        | Uncertain-input operational GP   |
| N5        | Factorised model, when available |

N2 can be an optimistic or historical reference. N4 does not need to beat an oracle-like reference; it should beat realistic operational baselines.

### Routes

Use the existing four-route set, plus one specially designed uncertainty route:

* long odometry-only segment;
* camera reacquisition;
* route choice between short camera-poor and longer camera-good corridors;
* one control route where all methods should behave similarly.

### Final figures

**Figure 06A — Paired mechanism case**

For one route:

* map and planned path;
* trust along path;
* R_plan along path;
* belief covariance along path;
* detector hit/miss timeline.

**Figure 06B — All traces**

All seeds and conditions, with failures explicitly marked.

**Figure 06C — Outcome table**

* clean goal rate;
* GT geometry breach;
* physical contacts;
* fallback;
* travel time;
* path length;
* planning latency.

**Figure 06D — Estimation calibration**

* position RMSE, GT evaluation only;
* NIS;
* covariance coverage;
* rejection rate.

### Video

**V06 — Primary thesis video**

Three-way comparison:

* constant covariance;
* point-input GP;
* uncertain-input GP.

Overlays:

* active detector;
* belief ellipse;
* R_plan;
* planned route;
* current trust value.

This should be the main thesis-defense video.

---

## `07_weak_priors_and_geometry`

Keep two distinct subfolders.

```text
07_weak_priors_and_geometry/
├── 07a_fov_range_prior/
└── 07b_geometry_occlusion_model/
```

## `07a_fov_range_prior`

### Status

Supporting ablation.

### Warehouses

Original warehouse for controlled analysis. Large warehouse for commissioning-scale evidence.

### Conditions

* constant prior;
* distance-only;
* distance plus image-edge/FOV;
* empirical GP without prior;
* weak prior plus empirical updates.

### Key plots

**Figure 07A — Prior maps**

Distance, image-edge and combined weak prior.

**Figure 07B — Learning curve**

Held-out NLL versus:

* number of frames;
* driven metres;
* commissioning minutes.

**Figure 07C — Incorrect-prior recovery**

Show how detector evidence corrects a wrong range/FOV prediction.

### Possible contribution claim

Not normally separate:

> A weak calibration-derived prior reduces early commissioning burden.

---

## `07b_geometry_occlusion_model`

### Status

Separate only with realistic geometry.

### Inputs that may be acceptable

* sensed LiDAR/RGB-D/stereo geometry;
* maintained CAD;
* approximate camera calibration.

### Forbidden shortcut

Do not use perfect Gazebo shelf heights or a complete visibility map as an operational assumption.

### Key plots

**Figure 07D — Geometry prediction versus detector evidence**

* predicted observable/occluded region;
* empirical miss rate;
* false-visible and false-occluded regions.

**Figure 07E — Baseline comparison**

| Model                   | Held-out NLL | Recall of misses | Calibration |
| ----------------------- | -----------: | ---------------: | ----------: |
| Range only              |              |                  |             |
| Range + FOV             |              |                  |             |
| Geometry                |              |                  |             |
| Geometry + empirical GP |              |                  |             |

### Go/no-go rule

If geometry does not clearly beat range/FOV, report that range and obliquity dominate in the tested warehouse. That honest result is already encouraged by the evidence rules.

---

## `08_large_warehouse_scaling`

The large warehouse is an **evaluation environment**, not a contribution.

### Purpose

Show that the frozen single-camera pipeline scales to:

* larger drivable area;
* more route diversity;
* sparse coverage;
* multiple camera footprints;
* greater commissioning burden.

### First experiment: treat each camera independently

Before fusion, fit one model per camera: τ_1(s), τ_2(s), τ_3(s), τ_4(s).

Use the same:

* target;
* GP kernel;
* calibration method;
* train/validation split rules;
* R_plan endpoints

chosen in the original warehouse.

### Required figures

**Figure 08A — Large warehouse overview**

Top-down map with:

* four cameras;
* nominal supports;
* overlap regions;
* camera gaps;
* commissioning routes.

**Figure 08B — Per-camera data coverage**

Four small multiples showing detector observations and covariance ellipses.

**Figure 08C — Per-camera trust fields**

Four matched maps using the same colour scale.

**Figure 08D — Camera heterogeneity**

Per-camera:

* detection rate;
* calibration;
* conditional error;
* trust-map NLL;
* range and image-edge distribution.

**Figure 08E — Scaling**

Runtime and memory versus:

* number of observations;
* warehouse area;
* number of cameras.

### Video

**V08 — Four-camera warehouse overview**

Aerial/top-down animation:

* current robot position;
* cameras observing it;
* each camera's current predicted trust;
* coverage gaps.

### Conclusion

> The frozen single-camera learning pipeline produces distinct, camera-specific trust fields at larger warehouse scale.

Do not yet call this fusion.

---

## `09_multicamera_handover_fusion`

This is a genuinely separate contribution.

### Research question

> How should multiple external cameras with spatially varying and potentially correlated quality be selected or fused without producing an overconfident robot belief?

### Conditions

| ID | Method                                        |
| -- | --------------------------------------------- |
| M0 | Best fixed single camera                      |
| M1 | Nearest camera                                |
| M2 | Highest detector confidence                   |
| M3 | Highest predicted trust                       |
| M4 | Naive independent fusion                      |
| M5 | Measurement-calibrated fusion                 |
| M6 | Conservative fusion under unknown correlation |
| M7 | Oracle best camera, evaluation only           |

### Four-camera warehouse only

The original warehouse does not add much here unless a second camera can be installed cheaply for a unit test.

### Required plots

**Figure 09A — Selection regions**

Map showing which camera M3 would choose at every point.

**Figure 09B — Handover timeline**

Against time:

* selected camera;
* each camera's trust;
* detection availability;
* handover events;
* belief covariance.

**Figure 09C — Overlap-region disagreement**

‖z_i − z_j‖ as a spatial heatmap.

**Figure 09D — Fusion calibration**

* localisation RMSE;
* innovation NLL;
* NIS coverage;
* covariance overconfidence rate.

**Figure 09E — Trajectory smoothness**

Trajectory overlays and displacement/jitter metrics.

The uploaded measurement-calibrated fusion paper is useful here because it shows that calibrated fusion may improve trajectory variance and smoothness even when absolute accuracy gains are modest.

**Figure 09F — Camera dropout**

Performance when:

* one camera is disabled;
* a frame stream becomes stale;
* two overlapping cameras disagree.

### Videos

**V09A — Handover**

Robot passes through four camera regions; active camera colour changes smoothly.

**V09B — Naive versus conservative fusion**

Split screen with belief ellipse and trajectory jitter.

**V09C — Camera failure**

One camera drops out; the system transitions to another source or conservative fallback.

### Contribution conclusion

> Camera-specific trust maps support more stable camera selection and prevent overconfident fusion in overlap regions.

---

## `10_active_commissioning`

This should only start after the GP model and four-camera maps work.

### Research question

> Can the robot learn the four camera trust fields with less driving by selecting informative but safe commissioning routes?

### Large warehouse is the natural setting

A one-camera original-warehouse test can validate implementation, but the scientific result should use the large warehouse because there is a meaningful coverage-allocation problem.

### Conditions

| ID | Route policy                                                     |
| -- | ---------------------------------------------------------------- |
| A0 | Ordinary task routes                                             |
| A1 | Uniform raster or waypoint coverage                              |
| A2 | Random safe routes                                               |
| A3 | Maximum GP posterior variance                                    |
| A4 | Information gain without pose uncertainty                        |
| A5 | Information gain accounting for predicted robot pose uncertainty |

### Required plots

**Figure 10A — Equal-budget routes**

Routes after the same:

* distance;
* time;
* number of camera frames.

**Figure 10B — Learning curves**

Held-out NLL or Brier score versus:

* driven metres;
* commissioning minutes;
* number of collected samples.

**Figure 10C — Per-camera uncertainty reduction**

Integrated GP posterior variance for cameras 1–4 over time.

**Figure 10D — Sample redundancy**

Fraction of samples collected in already well-known regions.

**Figure 10E — Safety and cost**

* minimum clearance;
* route duration;
* planner compute time;
* fallback events.

### Video

**V10 — Active versus ordinary commissioning**

Side-by-side accelerated video:

* both robots receive the same driving budget;
* live trust uncertainty maps update;
* final map quality shown at the end.

### Contribution conclusion

> Pose-uncertainty-aware commissioning reaches a specified reliability-map quality with less driving than ordinary or uniform coverage.

This is a different claim from using the trust map for navigation, so it should remain a separate contribution.

---

# 6. What to show in the next supervisor presentation

Do not present eleven completed research directions. Present a **decision deck** showing what can be measured now and what each option would prove.

## Slide 1 — Warehouse problem

One camera is useful but spatially uneven. Show existing C1/C2 route evidence.

## Slide 2 — Existing pipeline and realism gap

```text
camera observations
→ trust field
→ R_plan
→ planner
```

Highlight:

> The current missing question is how to learn the trust field from ordinary driving without exact robot positions.

## Slide 3 — Two-world experimental strategy

| Original warehouse     | Large four-camera warehouse |
| ---------------------- | --------------------------- |
| isolate mechanisms     | demonstrate scale           |
| one camera             | four cameras                |
| controlled routes      | coverage and overlap        |
| fit and compare models | freeze and transfer model   |
| causal evidence        | systems evidence            |

## Slide 4 — Initial data-collection investigation

Show one original-warehouse route with:

* covariance ellipses;
* camera hits and misses;
* camera-unobserved sections.

This confirms that the proposed uncertain-input problem actually occurs.

## Slide 5 — Is the belief covariance meaningful?

Show:

* covariance trace versus actual error;
* 90% ellipse coverage;
* error versus time since camera update.

This is the first go/no-go gate.

## Slide 6 — Is confidence a useful target?

Show:

* confidence versus projected localisation error;
* confidence calibration curve;
* miss-rate map.

This tells the supervisor whether the thesis should remain scalar or become factorised.

## Slide 7 — Uncertain-input GP option

Show the synthetic point-GP versus uncertain-input-GP figure and the planned U0–U6 baselines.

## Slide 8 — Three possible contribution packages

### Package A — Lowest risk

Uncertain-input GP + target ablation + R_plan + original warehouse navigation.

### Package B — Stronger statistical thesis

Package A + factorised availability and conditional noise.

### Package C — Stronger systems thesis

Package A + four-camera selection/handover/fusion.

## Slide 9 — Large warehouse role

Show the four-camera layout and planned outputs:

* per-camera trust maps;
* handover map;
* overlap disagreement;
* camera dropout.

Label any unrun plots as `PLANNED` or `HYPOTHETICAL`, not current evidence. The current contribution guidance specifically warns against presenting hypothetical multi-camera plots as experimental evidence.

## Slide 10 — Scope recommendation

Recommend:

> Complete Contribution 1 on the original warehouse first. Then select either factorisation or multi-camera fusion as the second contribution. Keep active commissioning as future work unless the first two complete early.

---

# 7. Highest-value initial plots to prepare now

These plots will allow a meaningful supervisor decision without implementing every extension.

## Immediate from existing original-warehouse logs

1. **Route + belief covariance ellipses + detector hits/misses.**
2. **Covariance trace and actual position error versus time.**
3. **YOLO confidence versus projected bottom-centre BEV error.**
4. **Confidence reliability diagram.**
5. **Spatial hit-rate and miss-rate map.**
6. **Current trust map versus simple distance/FOV baseline.**
7. **Existing C1/C2 paired route and outcome table.**

## Small offline prototypes

8. **One-dimensional uncertain-input GP demonstration.**
9. **Synthetic anisotropic point-GP versus smoothing versus uncertain-input GP.**
10. **Preliminary original-warehouse point-GP versus uncertain-input-GP maps.**
11. **Trust, GP uncertainty and R_plan triptych.**
12. **Preliminary availability map versus conditional-error map.**

## Large-warehouse planning visuals

13. **Four-camera nominal coverage map.**
14. **Planned per-camera data collection routes.**
15. **Overlap and handover regions.**
16. **Estimated dataset size and runtime scaling table.**

The first twelve support an actual method decision. The last four support scope and feasibility planning.

---

# 8. Video package for the eventual thesis

| Video                                              | Warehouse          | Purpose                      |
| -------------------------------------------------- | ------------------ | ---------------------------- |
| V00 Existing constant vs learned covariance        | Original           | Motivation                   |
| V01 Passive data collection with covariance growth | Original           | Realism of uncertain inputs  |
| V03 Point GP vs uncertain-input GP animation       | Original/synthetic | Explain method               |
| V06 Three-condition navigation comparison          | Original           | Main practical evidence      |
| V08 Four-camera coverage overview                  | Large              | Scale                        |
| V09 Reliability-aware camera handover              | Large              | Multi-camera contribution    |
| V09C Camera dropout and fallback                   | Large              | Robustness                   |
| V10 Active commissioning comparison                | Large              | Optional future contribution |

For every video, show the data source explicitly:

* `BELIEF`;
* `PIXEL`;
* `MODEL`;
* `GT—evaluation only`.

The ground-truth firewall requires GT to remain evaluation-only and visibly labelled.

---

# 9. Recommended final contribution split

## Primary thesis contribution

```text
02 trust target investigation
        +
03 uncertain-input GP
        +
05 frozen R_plan interface
        +
06 original-warehouse navigation
```

Contribution statement:

> **A passive commissioning method for learning spatial external-camera trust from ordinary robot driving while accounting for uncertainty in the robot locations assigned to camera observations.**

The trust-target comparison and ordinary confidence calibration support this claim but are not separate contributions.

## Optional second contribution A

```text
04 factorised availability and conditional noise
```

Statement:

> **A factorised external-camera observation model separating measurement availability from conditional localisation noise.**

This is the more coherent statistical continuation of the primary contribution.

## Optional second contribution B

```text
08 per-camera scaling
        +
09 camera selection, handover and fusion
```

Statement:

> **Reliability-aware selection and conservative fusion across a network of fixed warehouse cameras.**

This is the stronger large-warehouse systems contribution.

## Separate future paper

```text
10 active commissioning
```

Statement:

> **Safe informative route selection for efficient commissioning of camera-reliability fields.**

## Supporting only

* onboard sensors;
* larger warehouse;
* simple confidence calibration;
* weak FOV/range prior;
* standard sparse-GP runtime work;
* standard trust-to-R_plan mapping.

## Separate only under strong evidence

* geometry/occlusion modelling;
* novel confidence calibration;
* novel R_plan optimisation.

---

# 10. My recommended execution order

1. Freeze the current original-warehouse baseline.
2. Build the operational prior-belief logger.
3. Produce the covariance and confidence diagnostic plots.
4. Implement point, smoothing, weighting and uncertain-input GP baselines.
5. Complete route-disjoint trust prediction on the original warehouse.
6. Freeze the trust target and trust-to-R_plan mapping.
7. Run the final original-warehouse navigation campaign.
8. Transfer the frozen pipeline to each of the four cameras independently.
9. Choose **factorised modelling or multi-camera fusion** as the second contribution.
10. Attempt active commissioning only after the earlier evidence chain is complete.

The main thesis should therefore be proven in the original warehouse and **showcased at scale** in the four-camera warehouse. Multi-camera handover and active commissioning belong specifically to the large warehouse; uncertain-input GP fitting, target selection and R_plan validation should be settled before reaching it.
