# Frozen protocol: TorWIC physical RGB-D sanity check

Frozen on 2026-08-19 before downloading or viewing any TorWIC RGB, depth, or
segmentation image and before running monocular-depth inference.  The archive
directory was inspected only to establish filenames, compressed sizes, and
random-access feasibility.

## Scope and claim boundary

This is a deliberately narrow out-of-domain sanity check of the monocular
depth and floor-affine component.  It is **not** a physical replication of the
fixed overhead-camera reconfiguration experiment, a visibility/detection
evaluation, or a navigation experiment.  The TorWIC cameras are mobile and the
provided semantic masks are model-generated.  A positive result supports only
the statement that a once-commissioned floor affine can improve held-out
metric depth on physical warehouse imagery.

## Public source and frozen sample

- Dataset: Toronto Warehouse Incremental Change (TorWIC)-SLAM release.
- Official repository: <https://github.com/Viky397/TorWICDataset>
- Official terms: CC BY-NC 4.0 plus the dataset-specific terms under the
  repository's `License/ReadMe.md`.
- Collection/archive: `Oct. 12, 2022/Aisle_CCW.zip`, Google Drive file ID
  `1hplx0_5tDKz4zF6iRgTfwuQNIund0URn`, advertised size 3,753,328,323 bytes.
- Calibration: official `calibrations.txt`, Google Drive file ID
  `1NVnNEi-9QDoeyrnkxtlv8dHZl4Sc79zw`.
- Cameras: `left` and `right` Azure Kinect RGB-D cameras.
- Commissioning frame for each camera: `000000`.
- Held-out test frames for each camera: `000115`, `000230`, `000345`,
  `000460`, `000575`, `000690`, and `000805`.
- Modalities: `image_{side}`, `depth_{side}`, and
  `segmentation_greyscale_{side}` only.  Exact ZIP members are extracted with
  HTTP byte ranges; the multi-gigabyte archive is never downloaded.

The indices are a fixed, evenly spaced traversal sample visible in the ZIP
directory.  No image content or model outcome informed their selection.

## Inputs and preprocessing

The official calibration supplies each camera's `fx`, `fy`, `cx`, `cy`, image
size, and OpenCV distortion coefficients.  RGB, aligned depth, and semantic
mask are remapped once to the same pinhole grid using that calibration and the
original intrinsic matrix as the output matrix.  RGB uses bilinear resampling;
depth and labels use nearest-neighbour resampling.  TorWIC `uint16` depth is
converted to optical-axis metres with the documented factor `0.001`; zero and
non-finite depths are invalid.

The frozen monocular model is `unidepth_v2_vits14`, the same default used by
the availability and reconfiguration studies.  It receives only RGB and the
official pinhole intrinsics.  No TorWIC image is used for model selection or
fine-tuning.

## One-frame floor-plane proxy and frozen affine

TorWIC publishes camera-to-LiDAR extrinsics but not the sensor-rack height
above the floor.  Therefore this dataset cannot reproduce the study's fully
analytic floor intersection from calibration alone.  The following one-time
proxy is frozen in advance and must be disclosed wherever the result is used:

1. In commissioning frame `000000`, take semantic class 1 (`Driveable Ground`),
   erode it with a 7-by-7 square, retain valid depths in `[0.4, 20]` m, and
   stride the pixel grid by four in each direction.
2. Back-project those sensor-depth pixels through the official intrinsics.
3. Fit a plane with deterministic RANSAC (seed 20260819, 2,000 trials,
   0.04 m point-to-plane threshold), then refit by SVD on its inliers.  Require
   at least 1,000 inliers and an inlier fraction of at least 0.70.
4. Intersect commissioning-frame floor rays with this plane to obtain analytic
   floor depths.  Fit the repository's robust affine ground anchor from the
   UniDepth prediction to those depths, using its standard validity gates.
5. Freeze both the plane and affine separately for each camera.  Do not use
   test-frame sensor depth, test-frame floor depths, or test-frame affine
   refitting in the evaluated method.

This one sensor-assisted commissioning frame substitutes only for the missing
rig-to-floor transform.  It prevents a claim that the TorWIC result itself is
calibration-free.

## Held-out evaluation

For every held-out frame, apply that camera's frozen affine to its raw
UniDepth prediction.  Evaluate against the aligned sensor depth only on valid
pixels from the warehouse-structure labels
`{4,5,6,7,8,9,10,12,13,14,15}` (wall/fence/pillar, static feature,
rack/shelf, goods, fixed machinery, cart/pallet jack, pylons, non-static
feature, person, forklift/truck, and dynamic feature).  Class 1 floor pixels,
including all anchor pixels, are excluded from the primary score.

Per camera-frame, report pixel MAE, RMSE, AbsRel, delta-1, bias, valid pixel
count, raw-versus-anchored difference, and the non-deployable oracle-affine
ceiling fitted on that frame's scored truth.  Aggregate with medians across the
14 camera-frames and also report each camera separately.  Pixel counts are not
treated as independent trials and no pixel-level significance test is allowed.

As an evaluation-only assumption diagnostic, compare the frozen plane's
predicted depth with sensor depth on eroded class-1 floor pixels in each held-out
frame.  This diagnostic may explain a failure but may not update the plane or
affine.

## Frozen interpretation rule

Call the sanity check supportive only if all of the following hold:

1. both commissioning plane and affine fits pass their frozen validity gates;
2. at least 12 of 14 held-out camera-frames contain 1,000 or more scored
   structure pixels;
3. median anchored structure MAE is lower than median raw MAE for each camera;
4. the overall median anchored structure MAE is at least 10% lower than raw,
   and at least 10 of 14 paired frames improve.

Otherwise report it as mixed or negative.  Regardless of outcome, describe it
as a one-route component sanity check and retain the simulation-only limitation
for the full system.

