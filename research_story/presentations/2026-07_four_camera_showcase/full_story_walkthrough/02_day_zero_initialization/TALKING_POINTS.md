# 02 — Day-zero initialization

## Show

1. `figures/dayzero_reliability_atlas.png`
2. `figures/best_camera_and_reliability.png`
3. `figures/overlap_handover_corridor.png`

## Say

“Before the first drive, calibration provides four cautious spatial hypotheses:
where each camera projects to the floor, how oblique or distant that projection
is, and which regions geometrically overlap. The union covers 99.2% of the
planning grid, with 42.2% seen by two or more calibrated cameras.”

## Evidence boundary

- The maps are calibration-only day-zero priors.
- They contain no detector training records or ground-truth labels.
- The availability union is not a fused coordinate measurement and is not a
  learned GP result.

## Transition

“Geometry cannot see detector failure or shelf shadows. The next step is to
challenge each prior with operational detections *and misses* during driving.”
