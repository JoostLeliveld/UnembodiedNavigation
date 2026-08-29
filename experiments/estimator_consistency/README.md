# Where does the overconfidence come from?

**Question.** The fused camera correction claims a covariance. Scored against the truth at
its own stamp, its NEES is 7-35 where a consistent planar estimator would give 2. Something
is wrong by a factor of 4-17. Is it **bias** — a systematic offset the covariance was never
meant to describe — or is the **covariance itself understated**, because the cameras are
fused as though their errors were independent when they are not?

The two have completely different consequences. A bias is a calibration job: measure it once,
subtract it, move on. An understated covariance because of shared error is a modelling
failure that gets *worse* the more cameras you add, and no amount of commissioning fixes it.

**Method.** Entirely offline, on the schema-4 drives already on disk. No simulator.

1. Score each layer at the instant it describes: a camera reading at its capture stamp, the
   fused answer at its own stamp. `../fusion_on_fixed_routes/aligned.py` does the alignment
   and the deduplication; nothing here re-implements either.
2. NEES as logged, per camera and fused.
3. **Remove the bias and recompute.** Subtract the mean residual — globally, per camera, and
   resolved along and across each camera's own viewing ray, since that is the direction the
   projection makes errors in. Whatever NEES survives is not bias.
4. If it survives, ask whether independence is the reason: measure the correlation between
   two cameras' residuals at the same detector batch, and compare the fused covariance the
   manager published against the empirical covariance of the fused residuals.
5. Report the inflation factor the stated covariance would need, and the shared-error
   standard deviation that would produce it.

**Ground truth is used to form residuals and for nothing else.** No result here is available
to the runtime.

**Outputs.** `logs/studies/estimator_consistency/`.
