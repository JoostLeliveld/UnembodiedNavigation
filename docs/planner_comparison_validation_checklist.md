# Planner Comparison Validation Checklist

## Core comparison hygiene

- [ ] Every comparison run uses the same world, task, planner mechanics, and stopping logic.
- [ ] Only the visibility GP artifact and intended live perception backend differ between methods.
- [ ] `run_summary.json` reports completed runs only, and incomplete runs are excluded from comparison figures.
- [ ] Time-series plots are aligned to `first_cmd_stamp`, not node startup.
- [ ] GP and ambiguity backgrounds correspond to the same method artifact used for that run.

## State and uncertainty propagation plots

- [ ] `experiment.csv` contains actual state and inferred state.
- [ ] `experiment.csv` contains state covariance or explicitly logs NaN covariance.
- [ ] Actual path and inferred path are visually distinguishable.
- [ ] Covariance ellipses are shown at regular time intervals.
- [ ] Time axis is aligned to first command, not startup.
- [ ] Ambiguity-region contours are generated for each method.
- [ ] Path-over-ambiguity-region plots exist for each method.
- [ ] Path-profile plots sample `p_vis`, ambiguity, risk, and uncertainty along the realized path.
- [ ] Summary CSV reports mean/max state error and mean/max state uncertainty.
- [ ] Report includes the uncertainty-propagation sheets.

## Honest interpretation

- [ ] If covariance is unavailable, plots describe it as an uncertainty proxy rather than state certainty.
- [ ] Oracle visibility is treated as a reference field, not a runtime perception method.
- [ ] YOLO confidence is described as an uncalibrated detector score.
- [ ] Any path difference claim is checked against the corresponding GP and ambiguity maps first.
