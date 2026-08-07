# PROPOSED — 2-D footprint commissioning capture

**Status: draft for approval. Nothing here has been run, and no registry entry cites it.**

One capture that closes two independent blockers. It is not a navigation run and it is not
a campaign arm; it is a commissioning dataset whose only job is to vary the things the
existing logs hold together.

## Why one capture serves both blockers

| blocker | what it needs varied independently |
|---|---|
| **E6** — the per-camera "calibration" constants are confounded with robot silhouette and fixed route yaw | camera × region × **yaw** |
| **EXP-PROJ-AMP** — range, image row and projection conditioning are collinear at ρ ≥ 0.97 | position across **both** footprint axes |

Both reduce to: sample the footprint in two dimensions, and put several headings at every
location. These were being treated as separate campaigns. They are one dataset.

## What the existing data cannot do

Measured on `residuals.csv` (1,424 detections, three captures), not asserted:

| camera | n | x span | y span | image column span | populated 0.5 m cells |
|---|---|---:|---:|---:|---:|
| A | 125 | 3.1 m | 6.9 m | 96 px of ~1280 | **5** |
| B | 295 | 3.1 m | 7.3 m | 98 px | **7** |
| C | 474 | 2.0 m | 9.4 m | 279 px | 19 |
| D | 530 | 2.0 m | 8.4 m | 297 px | 22 |

The visible floor per camera is roughly 22 m × 17 m. Detections occupy a thin cross through
it. Only two headings exist in the whole dataset (0° and 90°), and one of the three captures
has no recorded yaw at all. Collinearity along the sampled ribbon:

```text
rho(sigma_max(J), range) = 0.976 / 0.969 / 0.999 / 0.996     (A/B/C/D)
rho(image row v, range)  = -0.947 / -0.776 / -0.996 / -0.984
```

Any model comparison on this data is a comparison of one variable wearing four names.

## Design

**Grid.** Per camera, over the drivable *and* visible floor: spacing **1.0 m**, target
**30–50 locations**. Cover both footprint axes, not a route through them. A lawnmower or
zig-zag path is acceptable if stopping at each node is impractical, provided the pass
direction alternates so image column is not confounded with traverse order.

**Headings.** At each location, **ψ ∈ {0°, 90°, 180°, 270°}**, plus the **45° diagonals** at a
declared subset of locations — E6 showed the diagonals are the missing stratum, and 45° is
also where the yaw-aware/yaw-blind crossing is currently bracketed but unmeasured.

**Repeats.** **10–30 usable frames per pose**, so a per-cell covariance is estimable rather
than a single realisation. Frames at one pose are within-unit observations, not replicates
(A18): the independent unit is the commanded site.

**Runs.** At least **five independent passes per camera**, so splits are by run, never by
shuffled frame. Adjacent frames are 0.2 s apart and near-identical; random frame splitting
leaks the answer across the split.

Order of magnitude: 30 sites × 4 headings × 20 frames ≈ **2,400 detections per camera** —
comparable to today's total across all four, but structured instead of collinear.

## What gets recorded per detection

| field | source | role |
|---|---|---|
| camera id, timestamp, run/pass id | configuration, runtime | grouping |
| detector hit/miss, `(u,v)`, confidence, bbox | detector | operational |
| projected `(x̂, ŷ)` | deployed projection | operational |
| `J(u,v)` | camera model | derived |
| commanded site id, commanded ψ | experiment metadata | design |
| reference `(x, y, ψ)` | Gazebo GT | **evaluation only** |

Ground truth measures the residual and supplies the controlled yaw. It never enters a
projection, a Jacobian, a covariance, or the planner. That is the standing firewall (A15).

## Analysis this unlocks, in order

1. **Bias then variance.** Per cell, `ē_k` separately from `Σ̂_k = Cov(eᵢ)`. Today's data
   already shows the bias is 20–85 mm per cell and dominates; a covariance model must be
   scored against `Σ̂_k`, never against total error.
2. **Does yaw-aware mean correction remove the bias?** Directly answerable once several
   headings exist at one location — currently impossible.
3. **Does the remaining spread vary with position?**
4. **Model ladder at equal parameter count:** constant → range-only → `JJᵀ` → learned.
   **The decisive comparison is range-only vs `JJᵀ`**, and it is the one today's data cannot
   make.
5. Report held-out NLL for auditability *and* stated σ / measured RMS / coverage in
   millimetres for reading.

## Gates before this is worth running

- [ ] target height resolved (**U11** — 0.35 m assumed vs ≈0.20 m physical; every predicted
      shadow depends on it)
- [ ] detector confidence threshold reconciled (**U6** — 0.25 offline vs 0.05 runtime)
- [ ] grid certified against each camera's measured field, so sites are actually observable
- [ ] cell size and minimum count frozen in config before collection, not after
- [ ] estimated wall-clock, given the detector runs at ~3 Hz and the P2000 is inference-bound

## What this capture does not do

It is open-loop commissioning evidence. It authorises no navigation claim, no safety claim,
and no closed-loop result. `EXP-CL-CAL` stays blocked on its own protocol; this only removes
the identifiability blocker standing in front of it.
