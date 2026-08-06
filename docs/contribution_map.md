# Runtime contribution map

The active implementation chain is:

```text
camera frame
→ frozen detector and pixel statistic
→ calibrated ground projection
→ operational residual/health contract
→ usable probability + conditional covariance + correlation floor
→ belief correction
→ route planning
→ evaluation-only metrics
```

| Runtime responsibility | Implementation |
|---|---|
| Simulation and camera worlds | `src/sim/` |
| Detection and operational image evidence | `src/perception/` |
| Projection and state conventions | `src/state/`, `src/reliability/reliability/projection.py` |
| Reliability fields, health, selection, and conservative fusion | `src/reliability/` |
| Belief correction and planning | `src/planning/` |
| Campaign contracts and logging | `src/experiments/`, `scripts/visibility_comparison/` |

The scientific interpretation of this chain is controlled by
[`../research/`](../research/README.md). In particular, the GP and historical precision
blend are baselines rather than the current paper contribution.
