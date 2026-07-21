# Module acceptance gates + stop rules

The programme-wide discipline for when a module may be used downstream, and when
to stop and reconsider. From the two-paper roadmap Part I §3–§4, reconciled with
the existing per-study README + gate practice. See
[`../PROGRAMME_ROADMAP_2026-07-21.md`](../PROGRAMME_ROADMAP_2026-07-21.md).

## Module acceptance rule (the 10-point README)

Before any module is used downstream, its README (or plan file) must state:

1. **Claim** — the one-sentence assertion the module supports.
2. **Realistic assumptions** — what the module assumes is available operationally.
3. **Non-assumptions** — what it explicitly does NOT assume (esp. no GT online).
4. **Literature anchor** — the paper/method it extends or reproduces.
5. **Interface** — inputs/outputs, message shapes.
6. **Units and frames** — px vs m, image vs world, which frame.
7. **Validation gate** — the pass/fail criterion (below).
8. **Baselines** — what it is compared against, on identical data.
9. **Caveats** — known limitations, honest nulls, ambiguities.
10. **Exact commands and artifacts** — how to reproduce; artifact hashes.

A module that cannot fill all ten is not ready to be depended on.

## Per-module validation gates (Paper 1)

Condensed from roadmap §21 / Part II. Full text in each WP plan.

- **Perception (G1):** bottom-centre localization visually checked; per-region
  miss map exists; projected p95 error known; failure frames documented.
- **Service model / GP (G3 = C1):** test routes spatially separated; learned beats
  constant AND beats-or-justifies-complexity-over geometry-only; calibration curves
  acceptable; GP σ high in unsupported areas; result does NOT depend on GT labels.
- **Covariance (G4 = C2):** positive definite; units/frames explicit; monotonic
  mapping; endpoints validated; NOT inflated everywhere (coverage AND sharpness
  both reported); consistency improves or stays acceptable.
- **Health monitor (G5 = C3):** fault-detection delay acceptable; healthy-camera
  false-rejection low; controlled + physical faults identified; uses NO GT signal.
- **Fusion (P2-G4 = C5):** exact Toro baseline reproduced; identical detections
  across methods; NIS consistency reported; gains are not merely smoothing from a
  differently-tuned Q.
- **Planning (G6 = C4):** obstacle/no-go geometry identical across conditions;
  direct visibility reward stays zero in the principal comparison; route changes
  correspond to predicted belief behaviour; current confidence is NOT used as
  future information.

## Stop rules

Stop and document before continuing when any of these is true:

1. A module needs ground truth **operationally** (not just for scoring).
2. A simpler baseline performs similarly (report it; do not hide it).
3. The system appears safe only because covariance is inflated everywhere.
4. Route behaviour changes for an **unrelated** cost (leaked term).
5. A visual contradicts a numerical metric.
6. The health monitor rejects healthy cameras more often than it isolates faulty
   ones (roadmap E6 critical failure criterion → do not use it downstream).
7. Multi-camera fusion is overconfident because camera errors are correlated.
8. The current single-camera system is not reproducible (fails G0).

A stop is not a failure — it is the point at which the honest finding (a null, a
tie, a leak) becomes the result to report.
