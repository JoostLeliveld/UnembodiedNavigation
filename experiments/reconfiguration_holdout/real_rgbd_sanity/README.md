# Physical RGB-D component sanity check

This folder adds a small, reproducible physical-image check for the
reconfiguration study's monocular depth component.  It uses the official
[TorWIC-SLAM](https://github.com/Viky397/TorWICDataset) real warehouse release,
which is distributed under CC BY-NC 4.0 plus the release's
[additional dataset terms](https://github.com/Viky397/TorWICDataset/tree/main/License).

Read `PREREGISTRATION.md` first.  The protocol was frozen before any TorWIC
image was downloaded or viewed.

## Reproduce

Only 48 preregistered ZIP members (20.5 MB compressed) are fetched; the 3.75 GB
archive is never downloaded.

```bash
python3 experiments/reconfiguration_holdout/real_rgbd_sanity/fetch_torwic_subset.py
python3 experiments/reconfiguration_holdout/real_rgbd_sanity/evaluate.py --device cuda
```

Use `--device cpu` when CUDA is unavailable.  Model/device metadata is stored
with every cached prediction.  Outputs are written under
`logs/studies/reconfiguration_holdout/real_rgbd_sanity/`.

## What is and is not tested

The two Azure Kinect views are mobile and low-mounted, not fixed overhead
cameras.  The release supplies calibrated, RGB-aligned metric depth and
model-generated semantic masks, but it does not supply the sensor-rack height
above the floor.  Frame `000000` therefore provides a one-time, sensor-assisted
floor-plane proxy for each camera.  A UniDepth affine is fitted on that frame
and applied unchanged to seven later frames.  Later sensor depth is used only
to score warehouse-structure pixels; floor/anchor pixels are excluded.

This evaluates physical RGB-to-depth transfer of a once-commissioned affine.
It does not evaluate detection availability, route selection, localization, a
changed layout, or full-system physical deployment.

## Frozen result

The preregistered interpretation rule passed all five gates:

- both planes and both robust affine fits passed;
- all 14 held-out camera-frames had at least 1,000 scored structure pixels;
- all 14 paired frames improved;
- median MAE improved in both cameras;
- the across-frame median fell from **1.979 m raw to 0.592 m anchored**
  (**70.1% lower**), versus a non-deployable per-frame oracle-affine median of
  0.201 m.

The camera-specific median MAE changes were 1.998 to 0.478 m (left) and 1.716
to 0.885 m (right).  The frozen plane remained consistent with held-out floor
depth to 0.029 m median MAE across frames.  The result is not uniformly
high-accuracy: the right-camera median delta-1 after anchoring was only 0.025,
and its frame `000805` retained 6.612 m MAE.  This is evidence that the anchor
usually corrects a large physical-domain scale bias, not evidence that the
component is ready for unconstrained real deployment.

Machine-readable results:

- `logs/studies/reconfiguration_holdout/real_rgbd_sanity/results/results.json`
- `logs/studies/reconfiguration_holdout/real_rgbd_sanity/results/per_frame.csv`
- `logs/studies/reconfiguration_holdout/real_rgbd_sanity/torwic_subset/manifest.json`

No pixel-level significance test is reported because pixels within a frame are
not independent, and the 14 frames come from only one route and two cameras.

