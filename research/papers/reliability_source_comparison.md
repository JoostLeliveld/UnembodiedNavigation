# Reliability-source comparison thesis chapter

## Purpose

Compare operational sources for predicting future usable observations and expose their
deployment trade-offs. This starts only after the correlated-error paper package is closed.

## Arms

1. constant null model;
2. calibrated FOV/range;
3. sensed-depth or raycast geometry with explicit provenance;
4. Gaussian process;
5. geometric prior plus operational updates;
6. a DL challenger only after offline admission gates.

All arms predict the same `p_use` target and use only information available at future
candidate poses. Evaluation truth and current detector outcome are forbidden operational
features.

## Subquestions

- Which source predicts held-out usable observations most accurately and honestly?
- Which failures are explained by occlusion, unsupported space, stale geometry, or layout
  shift?
- Which sources change expected belief and discriminate meaningful route alternatives?
- What commissioning, runtime, transfer, and update costs purchase those gains?
- After fields are frozen, which camera-selection policy best uses them without becoming
  overconfident?

## Gate sequence

Feature legality → held-out calibration → failure audit → offline route discrimination →
closed-loop navigation → deployment decision matrix. No method skips a gate.
