# warehouse_layout_sketches — what a realistic camera network would look like

**Status:** exploratory. **Not** a proposal to change the evaluation world — see the
recommendation in the results.

## Question

The current 4-camera world is four-fold symmetric, cameras at (±6, ±10), 6.1 m, 52.7°.
Would a realistic, asymmetric warehouse network look meaningfully different, and does it
matter?

## Answer (exp1)

Yes, but not mainly because of symmetry. Three realistic layouts (8 / 12 / 16 cameras,
racking occlusion, dock doors, misaligned north–south aisles) leave **25–43 %** of the
drivable floor seen by no camera. **The current world leaves ~1 %.**

Two takeaways:

1. **Placement dominates budget.** The first 16-camera layout scored *worse* on
   redundancy than the 12-camera one until its cameras were re-aimed up the storage
   aisles rather than along the cross-aisle (28.0 % → 37.5 %, same count).
2. **The current world is unrealistically easy**, which quietly weakens any claim about
   planning to stay observable — there is almost nowhere to be blind.

## Run

```bash
python3 experiments/warehouse_layout_sketches/exp1_layout_candidates.py
```

Seconds, offline, geometry only. Outputs →
[`logs/studies/warehouse_layout_sketches/exp1_layout_candidates/`](../../logs/studies/warehouse_layout_sketches/exp1_layout_candidates/).

## Reuse map

| need | reused from |
|---|---|
| camera projection + FOV | `unav_common.camera_model.ObliqueCameraModel` |
| pose → look_at construction | mirrors `reliability.projection.camera_model_from_world` |
| repo paths | `scripts/shared/paths.repo_root` |

Occlusion is a line-of-sight test written here (sampled ray vs rack height); the
`unav_common.occlusion_geometry` prism machinery is the heavier alternative if these
sketches ever become a world.
