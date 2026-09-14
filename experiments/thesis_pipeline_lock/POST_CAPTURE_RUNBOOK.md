# THESIS-FINAL-PIPELINE-V1: post-capture runbook

This runbook continues the locked warehouse, blue robot and 400-position capture map.  The
master capture is one dataset with four spatially disjoint roles.  They are not interchangeable:

```text
detector_fit (80 positions) ----------> YOLO gradients
detector_validation (40 positions) ---> early stopping + checkpoint/resolution choice
commissioning_fit (240 positions) ----> frozen YOLO outputs -> gate -> correction + R_hit + q
final_audit (40 positions) -----------> opened once after the complete stack is frozen
```

## Stage 04 — finish, audit and label the dataset

1. Require `status=complete`, 3,200 poses, 16,000 rows, exactly five cameras per pose, no
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
3. Generate labels only for `detector_fit` and `detector_validation` using the frozen semantic
   support rule.  A positive must satisfy all area, width, height, projected-hull-support,
   ground-contact and border rules in `label_dataset_protocol.json`.  Zero-mask images are
   negatives.  Visible but insufficiently supported views are retained in the ledger but
   excluded from gradients and detector-validation metrics.
4. Produce per-camera/per-role counts and deterministic contact sheets of positives,
   negatives and every ambiguous-reason class.  If the frozen rule makes the dataset
   infeasible, stop and version the protocol before training.  Never repair it using
   `final_audit`.
5. Freeze the capture/index/protocol hashes and the generated label ledger as Stage 04.

## Stage 05 — train and freeze a sensible detector

1. Initialize from the local COCO-pretrained `local_artifacts/base_models/yolo11n.pt`; never
   initialize randomly.
2. Train a declared seed schedule in two phases: detection head/backbone-frozen warm-up, then
   upper-layer fine-tuning at a smaller learning rate.  Use position-held-out
   `detector_validation` only for early stopping and checkpoint selection.
3. Training-only augmentation may change appearance, crop and translation.  Context-randomized
   copy-paste examples must remain a declared minority and use only `detector_fit` sources and
   backgrounds.  Validation remains 100% real.  Mosaic is moderate and disabled near the end;
   it must not manufacture implausibly tiny targets.
4. Compare 640, 960 and 1280 input resolutions on the same declared validation positions.
   Select using precision/recall, false positives and recall stratified by visible robot width,
   not aggregate mAP alone.
5. Preserve all trials.  Freeze one checkpoint, its SHA-256, Ultralytics/Torch versions,
   seed, augmentation configuration, selected epoch and the full validation table.

## Stage 06 — freeze the detector-to-measurement gate

Run the frozen detector once over `commissioning_fit`.  A YOLO return is not automatically a
localization measurement.  Freeze a runtime-observable gate using confidence, box pixel size,
border contact and agreement with the projected hull.  Semantic masks and commanded pose may
score candidate gates offline but may never enter the runtime gate.  Report coverage beside
conditional localization accuracy.  Include a small robot-absent warehouse false-positive
challenge before freezing the gate.

## Stages 07 and 08 — one commissioning set, three fitted products

Use only the gated, frozen-detector outputs on the same `commissioning_fit` spatial positions:

- mean/location correction: start with no correction and simple per-camera corrections before
  accepting a neural model; choose by spatially held-out commissioning folds;
- conditional `R_hit`: fit the R0--R4 ladder to out-of-fold residuals and stop at the simplest
  rung that improves proper score, containment and sharpness;
- availability `q`: model the probability that a camera produces a gate-usable measurement,
  including misses rather than conditioning them away.

For all three, keep every heading of a position in the same fold.  Predeclare nested spatial
subsets (for example 25, 50, 100, 150, 200 and 240 positions) to produce data-efficiency
curves without changing the final model-selection population.  Freeze model files, scalers,
features, thresholds, hashes and exact reproduction checks.

## Final audit and downstream campaign

Only after YOLO, gate, correction, `R_hit` and `q` are immutable may `final_audit` be opened for
one full-stack evaluation.  Report detector precision/recall and false positives, gate coverage,
conditional camera-reading error, covariance calibration and availability proper scores,
stratified by camera, range, visible size and occlusion class.  No tuning follows this audit.

Then freeze Stage 07/08, run the registered navigation campaign, evaluate each estimate at its
own timestamp using the localization-metrics contract, retain every attempt, and finally build
the thesis figures/tables only from manifests referenced by `pipeline_lock.json`.
