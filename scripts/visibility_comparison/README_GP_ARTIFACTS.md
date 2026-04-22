# GP Visibility Artifacts

These artifacts (`.npz` files) contain the fitted Gaussian Process visibility fields used by the EFE planner.

## Generation
Artifacts are produced by fitting empirical observations of target visibility using:
```bash
python3 scripts/visibility_comparison/fit_visibility_gps.py --gp-targets data/gp_targets.csv
```

## Schema
Current artifacts (v2) must contain:
- `xs`, `ys`: Grid coordinates (1D arrays).
- `P_mean_map`: Predicted mean visibility probability.
- `P_conservative_plan_map`: Lower-bound probability (mean - β * std) for risk-averse planning.
- `F_std_map`: Latent GP standard deviation (logit space).
- `camera_pos`: Position of the observer during sampling.
- `target_height`: Z-height of the targets.
- `geometry_json`: Environment prism geometry for plotting.

## Reproducibility
The fitting process is now tied to `sklearn.gaussian_process.GaussianProcessRegressor` with an RBF kernel. 
Ensure `random_state` is set if re-fitting for exact binary equivalence.
