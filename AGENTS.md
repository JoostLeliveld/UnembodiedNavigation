# Repository instructions: how to talk about a number

Before answering any question involving camera accuracy, localization error, belief error,
calibration, coverage, RMSE, bias, or a comparison of runs, read [`PLAN.md`](PLAN.md),
[`docs/localization_metrics.md`](docs/localization_metrics.md), and
[`docs/localization_metrics_registry.json`](docs/localization_metrics_registry.json), then load the run through
[`experiments/fusion_on_fixed_routes/aligned.py`](experiments/fusion_on_fixed_routes/aligned.py).

[`docs/open_questions.md`](docs/open_questions.md) says what is unresolved and which
implementation limits are known. Check it before presenting anything as settled.

## The clean sheet

Every study dated before **2026-08-25** is superseded. Do not reuse any of its numbers, and
do not quote a figure from a `RESULTS.md` without re-deriving it from the drives on disk.
A stale artifact that still parses is the most expensive kind of wrong here.

## Hard rules

1. **Name what the number is.** Never say "camera error" or "localization error" without
   saying which quantity it is (a camera's reading, the fused correction, or the filter's
   belief), which statistic (median, mean, p95, signed bias), against which reference, over
   which runs, and how many samples.
2. **Do not compare across those three layers.** A camera's reading error, the fused
   correction's error, and the belief's error are different quantities. So are median error,
   p95, signed bias, and calibration. None of them are interchangeable.
3. **Score each quantity at the instant it describes.** A per-camera reading is scored at its
   capture stamp, the fused answer at its own stamp, the belief at the belief's stamp. Pairing
   an estimate with a later truth charges the robot's own travel to the sensor: at 0.22 m/s a
   0.1 s offset is 2.2 cm, which is larger than the effect being measured.
4. **Count each detection once.** A held message is re-logged every tick. `aligned.py`
   deduplicates; anything not going through it does not.
5. **Never pool run directories by glob or "latest".** Use a frozen manifest with exact run
   IDs, conditions and seeds, and fail on a missing or extra run.
6. **One drive is not a result.** Time samples within a drive are correlated and cannot
   substitute for replication. Aggregate within a run first, then across seeds.
7. **`gt_*` is the reference.** Legacy `truth_x/y` is wheel odometry and is diagnostic only.
   Ground truth scores results; it is never an online input, and it never terminates a run.
8. **A camera never knows ground truth**, its own error, or its own calibration. State which
   inputs the method actually had online and which fields the offline evaluator used.

## Say the thing, not the label

Report in centimetres, percentages and seconds — not pixels, ratios or log-likelihoods.
Lead with the verdict, then the mechanism, then the caveat. If a result is a null, an
artifact, or a correction to something said earlier, say so in the first sentence.

Write the sentence the number would appear in, in the paper. If you cannot write that
sentence, do not report the number.
