# Evidence ledger: completed paper versus extension work

This ledger deliberately separates results that can motivate the paper from
results that can be used as a paper claim.  It resolves several tempting but
unsafe shortcuts: exact SDF/CAD geometry, ground-truth pose, and CAD-sampled
depth may validate an implementation but are not deployment inputs.

## Status keys

| Tag | Meaning |
| --- | --- |
| **Established** | Existing, frozen or reproducible evidence supports a narrow claim. |
| **Measured in simulation** | Derived from logged simulator/detector data, but not yet a cross-layout or closed-loop paper result. |
| **Model / plumbing** | First-principles, CAD-assisted, synthetic, or interface validation.  Useful to choose experiments; not an empirical deployment claim. |
| **Open** | Needs a new prospective experiment or evidence chain. |

## Claims and their honest status

| Extension | What the local artifact currently says | Status to present | What it may support now | What it must not support yet |
| --- | --- | --- | --- | --- |
| Offline reliability-aware planning | The completed paper/current campaign establishes that `R_plan` shaped by a learned reliability field changes route behaviour and robustness. | **Established** | Motivation and downstream anchor. | A claim about day-zero commissioning. |
| Geometry explains a learned field | The original warehouse geometry prior ranks the empirical GP at Spearman 0.730 over 22,917 driveable in-FOV cells. | **Model compared with measured simulation data** | Geometry is a plausible prior feature source. | CAD is a deployable input; geometry alone transfers across layouts. |
| Depth/occlusion comparison | The existing capture benchmark reports camera-only 0.782 AUROC and viewpoint-matched depth 0.968 AUROC; source and sensing realism need to remain explicit. | **Measured simulated-sensor benchmark** | A direct justification for testing viewpoint-matched sensed structure. | Real warehouse depth performance or monocular-depth deployment. |
| CAD-sampled sensed-height prior | The `sensed_height_prior.py` path samples/corrupts CAD to exercise height-map plumbing. | **Model / plumbing** | Interface and unknown-cell handling. | “Stereo has been validated” or a 0.97 sensing result. |
| Fresh geometry-prior capture | Stage-3 visual/summary shows a 0.68 association with a fresh simulated capture. | **Measured in simulation, exploratory** | Candidate changed-world validation protocol. | Cross-layout generalisation or day-zero navigation success. |
| Online update from driving | Existing logged trajectories, detections, and belief covariance drive the update demos.  The documented visited-cell map error drops 0.207 → 0.064. | **Measured in simulation** | Driving has information; update-policy ablations are feasible. | A live planner already consumes online updates; prospective unseen-cell sample-efficiency. |
| Position-uncertainty weighting | Current five-fold event CV shows naive, covariance-spread, and uncertainty-weighted variants close; hard gating is not favoured by the logged result. | **Measured in simulation** | Hard gating should be a negative control, not the assumed method. | A universal claim that covariance spreading is superior. |
| Big warehouse / zero-shot figures | Scripts and captured data exercise a larger-world setup and produce visual zero-shot/close-loop analyses. | **Exploratory simulation** | A scalable testbed and media asset. | A frozen benchmark conclusion without a registry, configuration, seeds, and outcome audit. |
| Two/four-camera maps and fusion | Layouts, per-camera contracts, priors, overlap checks, and handover inflation are implemented; camera-B detector evidence is explicitly pending. | **Model / plumbing + open** | A well-scoped future ablation and a compelling system layout. | “Multi-camera fusion improves localization” or a closed-loop handover result. |

## Quantities safe to use in a pitch — with their scope

- **0.730 Spearman:** geometry prior versus the empirical single-camera GP on
  the existing warehouse grid.  It is an explanatory/model comparison, not a
  cross-layout deployment result.  Source:
  [`VALIDATION.md`](../../logs/studies/geometry_visibility_prior/warehouse_aws_v0/VALIDATION.md).
- **0.207 → 0.064:** visited-cell residual map RMSE in the logged-driving online
  update demonstration.  It motivates a prospective learning-curve experiment;
  it is not live online navigation.  Source:
  [`online_update_demo.py`](../../scripts/geometry_visibility/online_update_demo.py).
- **0.782 versus 0.968 AUROC:** calibration-only versus viewpoint-matched-depth
  prior on the documented uniform capture benchmark.  Describe this as a
  simulated-sensor test and re-run it under a frozen cross-layout protocol.
  Source: [`reliability_prior_sensing_survey.md`](../reliability_prior_sensing_survey.md).
- **39% → 78% coverage:** modelled geometric union for adding a second camera
  in the current world.  It is coverage potential, not detector or navigation
  improvement.  Source:
  [`geometry_visibility_deployment.md`](../geometry_visibility_deployment.md).

## Evidence required before submitting the next paper

1. Freeze worlds, cameras, detector checkpoints, driveable maps, prior inputs,
   task set, seeds, outcome taxonomy, and all data splits in a new runtime
   contract/registry.
2. Capture detector hit/miss labels prospectively in at least one unseen layout
   and one targeted changed-shadow layout.  Keep GT and exact SDF geometry on
   the evaluation side of the firewall.
3. Produce a true sensed-structure prior from the allowed sensor pipeline
   (not a CAD-sampled cloud) and log its unknown regions.
4. Evaluate day-zero prior calibration on spatially disjoint, unvisited cells
   before showing any updates.
5. Run a fixed-budget, prospective update campaign with held-out cells/layouts.
   Record all hits and misses, belief covariance, update timestamps, and map
   versions.
6. Integrate the selected online map path into the planner only after the
   offline evidence is frozen; then run matched closed-loop conditions.
7. If multi-camera remains in scope, collect camera-B detector data, prove
   overlap calibration without GT, and benchmark handovers before adding it to
   a navigation claim.

## Presenting uncertainty well

Use a three-label legend in every meeting slide: **measured simulation**,
**model/plumbing**, and **open experiment**.  This makes the research case
stronger: the new paper is exactly the work needed to turn a promising chain
into a validated deployment method.
