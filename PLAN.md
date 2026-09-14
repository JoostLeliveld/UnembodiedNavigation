# Thesis execution plan

The thesis evaluates whether commissioning a fixed external-camera network improves predicted
belief, route choice, and navigation reliability.

## Locked chain

```text
known-pose commissioning survey
  -> selected raw-box correction and its matched R_i(p, psi)
  -> position-only availability q_i(p)
  -> expected belief along candidate routes
  -> existing IWAI route objective
  -> matched navigation comparison
```

The detector, deterministic gate, raw box-bottom projection, estimator architecture, planner,
controller, tasks, and seeds are shared. A visual hull is not part of the current observation
or any correction candidate. Process covariance `Q_k` is fixed by the experiment configuration
and is not a model-selection problem.

## Evidence stages

1. Warehouse, robot, camera map, dataset, and YOLO11n-960 detector: locked.
2. Raw detector-box admission outcomes: rebuild under the current deterministic gate.
3. Correction and covariance: compare raw-box candidates on complete development drives;
   fit a fresh `R` to each candidate's out-of-fold residuals; freeze the selected pair.
4. Availability: fit `q_i(p)` from every expected camera opportunity, pooling headings at
   each position; assess probability calibration on held-out complete drives.
5. Runtime fusion: use the selected correction and matched covariance in the one robot filter.
6. Planning: freeze one route per task and condition using the same candidates and feasibility
   rules; vary only constant/commissioned future `q` and `R` forecasts.
7. Navigation: execute the four-arm, four-task, five-seed matched campaign and retain every run.
8. Reporting: admit only manifest-bound results under the localization-metrics contract.

## Only open method decision

The correction family and its matched covariance remain open until the declared
development-drive comparison is complete. This includes whether the 16x16 visibility residual
earns selection and whether residual dependence justifies an augmented bias state.

Everything else above is fixed. Superseded studies are recoverable from Git and must not be
used to change the method.

