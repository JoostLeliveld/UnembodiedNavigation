# Four-camera depth and visibility test

This folder contains the machine-readable Camera A–D evaluation. The supervisor-facing plots
are one level up in `../figures/04_*`, `05_*`, and `06_*`.

- `results.json`: depth, commissioning-fit, scale-gate, per-camera visibility, fused
  visibility, and dynamic-shadow metrics.
- `maps/`: compressed inferred and oracle visibility arrays for every tested model.
- `predictions/`: RGB-only monocular predictions and metadata for the eight evaluated frames.
- `online_anchor_ablation.json`: legacy vs enhanced anchors vs enhanced + temporal
  Bayesian anchoring, including per-frame provenance and method-only timing.

Protocol: fit each model/camera affine on the clear frame at 0.4 s using fixed calibration,
the floor plane, and the planner's 2-D traversable regions; reuse that fit unchanged at 1.2 s
after the pallet appears. Gazebo depth and geometry-raycast visibility are evaluation-only.

See [`../README.md`](../README.md) for interpretation, limitations, and the compact result
table. Recompute metrics and plots from saved predictions with:

```bash
python3 experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/four_camera_study.py --plots-only
```

Recompute the online-anchor ablation from the same saved predictions with:

```bash
python3 experiments/usable_observation/supervisor_comparison/10_monocular_depth_results/online_anchor_ablation.py
```

That ablation has only two timestamps per camera. It verifies the update mechanism and
guards against an immediate regression. The follow-up complete 21-update replay is in
[`../temporal_anchor_sequence/`](../temporal_anchor_sequence/); it supersedes this
two-timestamp artifact for temporal interpretation.
