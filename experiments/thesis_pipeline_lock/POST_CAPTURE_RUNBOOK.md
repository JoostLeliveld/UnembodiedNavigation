# THESIS-FINAL-PIPELINE-V1: post-capture runbook

This runbook continues the canonical reference-position campaign locked by
`reference_position_campaign_lock.json`. The master capture has 2,569 physical positions,
four declared headings and five expected cameras per heading. Its four position-disjoint
roles are not interchangeable:

```text
D_mu (1,301 positions) -------> shared visibility-informed correction
D_R (689 positions) ----------> R0/R1/R2 from frozen out-of-sample corrected residuals
D_dev (429 positions) --------> frozen hyperparameter/model selection only
final_audit (150 positions) --> one full-stack evaluation; fits or selects nothing
```

## Stage 04 — finish, audit and label the dataset

1. Require `status=complete`, 10,276 pose batches, 51,380 rows, exactly five cameras per pose, no
   failed rows, all referenced RGB/masks present, and a maximum batch span of 50 ms.
2. Run the slow decoded-pixel audit once.  This is integrity-only access to `final_audit`, not
   model evaluation:

   ```bash
   python3 experiments/thesis_pipeline_lock/audit_master_capture.py \
     --capture logs/thesis_final_pipeline_v1/master_capture \
     --protocol experiments/thesis_pipeline_lock/label_dataset_protocol.json \
     --verify-pixels \
     --output logs/thesis_final_pipeline_v1/stage04_dataset/capture_audit.json
   ```
3. Treat semantic masks as offline label/integrity support only. They never enter the runtime
   gate, correction, covariance query, fusion, estimator or planner.
   support rule.  A positive must satisfy all area, width, height, projected-hull-support,
   ground-contact and border rules in `label_dataset_protocol.json`.  Zero-mask images are
   negatives.  Visible but insufficiently supported views are retained in the ledger but
   excluded from gradients and detector-validation metrics.
4. Produce per-camera/per-role counts and reproducible contact sheets of positives,
   negatives and every ambiguous-reason class.  If the frozen rule makes the dataset
   infeasible, stop and version the protocol before training.  Never repair it using
   `final_audit`.
5. Freeze the capture/index/protocol hashes and the generated label ledger as Stage 04.

## Frozen detector

The detector is the YOLO11n checkpoint and SHA-256 named in
`reference_position_campaign_lock.json`. Capture stores raw data and does not run the detector.
All downstream inference uses that exact checkpoint. Replacing or retraining it requires a
dated lock amendment before any final fitting; final-audit data may never choose it.

## Stage 06 — freeze the detector-to-measurement gate

Run the frozen detector once over `D_mu`, `D_R` and `D_dev`. A YOLO return is not automatically
a localization measurement. Freeze a fixed belief-independent gate using only declared
detector and image/box validity fields. Semantic masks, projected hulls, commanded pose, ground
truth, innovations and NIS may score diagnostics offline but may never enter the runtime gate.
Retain every expected camera opportunity, including misses and refusals.

## Stages 07 and 08 — correction, covariance and planning precision

Use the frozen-detector opportunity ledger and the position-disjoint roles in the campaign lock:

- correction: deploy the shared structured-plus-16x16-visibility model; raw and structured-only
  models are diagnostic ablations;
- conditional covariance: compare exactly global-full R0, per-camera-full R1 and spatial
  per-camera-full R2, fitted to out-of-sample residuals from the frozen correction;
- planning precision: query the matched runtime covariance at each camera and planner-grid
  position, rotate it into the world frame and export its direct inverse. Do not fit a second
  field or modify it using opportunities, misses, refusals, availability or NIS outcomes.
  Weak residual support already lowers M2 precision because its covariance approaches the
  broad prior.

For every fit and fold, keep all cameras and headings of a position together. Give each physical
position equal total weight regardless of retained opportunity count. Repetitions at a position
count as one spatial support location. Freeze model files, scalers, features, thresholds, hashes
and exact reproduction checks before opening `final_audit`.

## Final audit and downstream campaign

Only after YOLO, gate, correction, R0/R1/R2 and matched planning-precision exports are immutable may `final_audit` be opened for
one full-stack evaluation.  Report detector precision/recall and false positives, gate coverage,
conditional camera-reading error, covariance calibration and planning-precision diagnostics,
stratified by camera, range, visible size and occlusion class.  No tuning follows this audit.

Then freeze Stage 07/08, run the registered navigation campaign, evaluate each estimate at its
own timestamp using the localization-metrics contract, retain every attempt, and finally build
the thesis figures/tables only from manifests referenced by `pipeline_lock.json`.
