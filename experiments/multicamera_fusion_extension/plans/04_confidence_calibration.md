# Plan 04 — Detector-confidence calibration (§8.1–8.2)

## Purpose
Map raw YOLO score `c` to `p^C = P(usable | detection, c)`. Raw confidence is
never used as a probability or covariance. Note the repo precedent: raw
confidence was already abandoned once for detection RATE (calibration
invariance, ch.02). This module is the disciplined way to bring per-frame
confidence back — and if calibrated confidence adds nothing over the spatial
field, that is a reportable finding, not a failure (confidence gate, §21).

## New code
`src/reliability/reliability/confidence_calibration.py` + tests. Pure Python
(package style: no sklearn dependency).

```python
class IsotonicCalibrator:        # PAV algorithm, monotone step function
    fit(scores, labels)          # labels = usable G from plan 02
    predict(score) -> p
class LogisticCalibrator:        # logit(p) = a0 + a1·logit(c); Newton/IRLS fit
class MultivariateLogisticCalibrator:
    # + log bbox_area, û, v̂, range r  (§8.2) — tests whether confidence stays
    # informative after geometry is known
def reliability_curve(scores, labels, n_bins) -> bins  # feeds shared metrics/figures
```

Per-camera calibrators (detector OOD differs per viewpoint); a pooled variant
as ablation. Serialize as JSON artifacts with fit provenance + hashes.

## Data & splits
Labels from plan 02 (`usable` given detection). Fit on the **calibration
split** (runs disjoint from GP training and from test; §16 grouped by run).
Isotonic first choice; logistic fallback when calibration data is thin
(isotonic overfits small samples — report both).

## Gates (§21 confidence gate)
- Calibrated beats raw on held-out Brier/NLL + ECE (via `scripts/shared/metrics.py`).
- Transfers to held-out routes (reliability diagram within tolerance).
- Still informative after spatial reliability is known: multivariate model's
  confidence coefficient significant under grouped bootstrap — else record
  "confidence adds nothing beyond geometry" disclosure and drop it from the stack.
- Unit tests: PAV correctness vs brute force on small cases, monotonicity,
  degenerate inputs (all-one-class), round-trip serialization.
