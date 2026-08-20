# Static probability in planning — four-camera experiment

This folder contains the first executable planning experiment for a static usable-observation
probability. It contains no slides or PDFs.

The comparison changes only how planning treats measurement availability:

1. `availability_blind_shortest` ignores the field when choosing a route;
2. `r_over_p_shortcut` performs one Gaussian update with
   `R_plan = R_cond / p_use`;
3. `explicit_hit_miss` propagates
   `E[P+] = p_use P_hit + (1-p_use) P_miss`.

All conditions use the same four frozen camera fields, route candidates, dynamics,
conditional measurement covariance and runtime hit/miss filter. A single camera is selected
per grid cell using a frozen expected-information rule; simultaneous fusion is deliberately
outside this first experiment.

## Evidence status

This is an **offline mechanism experiment with model-based stochastic replay**, not a
closed-loop Gazebo campaign. The planning fields come from the existing 8,808 detector-event
four-camera spawn grid. The conservative GP field is available to the planner; the full-fit
posterior mean supplies replay probabilities. Because both come from the same commissioning
dataset, the replay is not independent probability-calibration evidence.

The conditional covariance is derived from `PG-IPM-CURRENT`: camera-measurement residual
component SDs versus commanded ground truth, current zero-parameter floor IPM, balanced
four-camera set-pose dataset, 1,844 detections. Ground truth and residual statistics are used
offline only.

## Run

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
MPLCONFIGDIR=/tmp/mpl_static_puse \
PYTHONPATH=src/reliability:src/planning:src/unav_common \
python3 experiments/usable_observation/supervisor_comparison/11_static_probability_planning/run_experiment.py
```

If numerical trials completed but rendering was interrupted, add `--reuse-runs` to reuse the
validated run table. The default always recomputes the experiment.

Generated files are written to `results/` and `figures/`. See `RESULTS.md` for the numerical
outcome and limitations.

The separate `closed_loop_gazebo/` folder contains the actual four-camera Gazebo feasibility
pilot, its exact-run metrics and PNG figures. Its result is currently inconclusive because all
three planners selected an infeasible near-stationary global solution; see
`closed_loop_gazebo/PILOT_RESULTS.md` before using it in a comparison.
