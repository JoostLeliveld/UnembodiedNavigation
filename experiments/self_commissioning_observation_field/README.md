# Self-commissioning Bayesian observation field

This offline successor keeps three different questions separate:

\[
p_{use}(x,c), \qquad b(x,c,m), \qquad R_{cond}(x,c,m).
\]

The frozen A3 artifact supplies the monocular-depth geometry prior plus learned
availability-GP residual. Operational detections then commission conditional
bias and covariance. Ground truth is used only to build and audit the artifact;
the planner reads only the frozen fields, camera calibration and its belief.

The current four-camera evidence is `PG-IPM-CURRENT`, so its response is the
YOLO bounding-box bottom centre. It contains no front/rear keypoint visibility
modes. The present `clear` and `marginal` labels are therefore **A3 probability
strata**, not physical occlusion labels. A four-camera, multi-session keypoint
campaign remains required before making a keypoint-mode claim.

## What is implemented

- per-camera/per-probability-stratum Normal-inverse-Wishart bias and `R`;
- a finite-rank RBF GP approximation for spatial bias and log variance;
- nested spatial fitting/calibration/testing, with the constant model retained
  if the spatial candidate fails;
- exact enumeration of all 16 hit/miss subsets at every four-camera belief
  update; and
- an offline route ablation between `p_use` only and the complete observation
  field over one frozen solved-route library.

Run:

```bash
python3 experiments/self_commissioning_observation_field/freeze_inputs.py
python3 experiments/self_commissioning_observation_field/commission_field.py
python3 experiments/self_commissioning_observation_field/planner_ablation.py
python3 experiments/self_commissioning_observation_field/make_figures.py
```

Outputs are under `logs/studies/self_commissioning_observation_field`. Meeting
copies `15_self_commissioned_per_camera_p_and_R.png` and
`16_self_commissioned_planning_ablation.png` are placed in
`logs/studies/availability_paper/figures`.

This is an offline, frozen-input mechanism audit—not a closed-loop navigation
claim and not yet confirmatory evidence for a paper.
