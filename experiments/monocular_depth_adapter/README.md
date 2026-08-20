# monocular_depth_adapter — one wrapper over several RGB-to-depth networks

## What this is

A camera on the warehouse wall sees the floor and the racks, but the pipeline
only ever asks it "where is the robot's bottom edge". A monocular depth network
turns the same picture into a distance for every pixel, which is what an
occlusion or clearance prior would need. Before any of that can be trusted,
something has to run those networks the same way every time and write down
exactly what came out.

That is all this study builds: **RGB image plus camera intrinsics in, depth plus
metadata out, saved to disk.** It does not decide what the depth is good for.

Three model families are wired up: **Depth Anything V2**, **Metric3D v2** and
**UniDepthV2**. They do not return the same kind of number, and the adapter's
main job is to stop that difference from being lost.

## The thing that goes wrong if you skip this

Four different quantities all get called "depth":

| convention | what one number means | can you threshold it in metres? |
|---|---|---|
| `metric_z` | metres along the optical axis | yes |
| `euclidean_range` | metres along the ray from the lens | yes, but it is a *different* number for the same point |
| `relative_depth` | unitless, bigger = further, unknown scale | no |
| `inverse_depth` | unitless, bigger = **nearer**, unknown scale | no |

The first two differ by the secant of the angle off the optical axis. For these
90-degree cameras that is a factor of about **1.5 in the image corners** — not a
rounding error. The last two cannot be turned into metres at all without an
anchor from somewhere else in the scene.

So every prediction carries its convention, and the only conversion the adapter
will perform is between the two metric ones, which needs nothing but the
intrinsics. Ask it to turn relative depth into metres and it raises.

## What it must not do, and how that is enforced

The adapter must not see obstacle ground truth, read simulator depth, anchor
anything to the floor, decide what is visible, or alter a camera calibration.

That is structural, not a promise in a comment. A backend's only entry point is
`infer_batch(images, intrinsics)`; there is no argument through which truth or
scene geometry could arrive. The `monodepth/` package imports nothing from this
repository — no `unav_common`, no ROS, nothing under `logs/` — and a test parses
its imports on every suite run to keep it that way.

Predictions are written to disk. No ROS node, no topic, on purpose: this stage is
for measuring the models, not for deploying one.

## Layout

```text
monodepth/                 the adapter — self-contained, no repo imports
  types.py                 DepthConvention, CameraIntrinsics, DepthPrediction, valid-mask rules
  conventions.py           the two metric conversions; refuses the rest
  adapter.py               MonocularDepthAdapter: batching, timing, memory, uncertainty
  uncertainty.py           native confidence / flip consistency / temporal spread
  determinism.py           seeding and the deterministic-kernel switches
  storage.py               npz + json sidecar, and the loader that refuses unlabelled depth
  backends/                depth_anything_v2.py, metric3d_v2.py, unidepth_v2.py
frozen_set.py              the repo-side boundary: pins frames by SHA-256, derives K
build_frozen_set.py        build / --verify the manifest
run_inference.py           run models over the frozen set, save predictions + costs
benchmark_report.py        multi-axis comparison; explicitly declares no winner
make_figures.py            the two figures that go with a run
capture_depth_truth.py     matched RGB + real Gazebo depth (EVALUATION ONLY)
evaluate_against_truth.py  score the models against that depth, floor-anchored
frozen_sets/               the tracked manifests (small; the images stay in logs/)
```

Outputs go to `logs/studies/monocular_depth_adapter/<run>/<model>/`. Budget about
**6 MB per frame per model** — a 24-frame run over all seven checkpoints is
roughly 1 GB, in the ignored `logs/` tree.

## Using it

```python
from monodepth import MonocularDepthAdapter, DepthRequest, CameraIntrinsics

K = CameraIntrinsics(fx=640.0, fy=640.0, cx=640.0, cy=360.0, width=1280, height=720)
with MonocularDepthAdapter("unidepth_v2_vits14", batch_size=4,
                           uncertainty="native+flip") as adapter:
    preds = adapter.predict([DepthRequest("frame0", rgb_uint8, K)])

p = preds[0]
p.depth            # (720, 1280) float32, in p.convention's units, unmodified
p.convention       # DepthConvention.METRIC_Z
p.valid            # (720, 1280) bool — NaNs, non-positives, out-of-range marked
p.uncertainty      # (720, 1280) float32, kind named in p.uncertainty_kind
p.timing.forward_s, p.memory.gpu_peak_allocated_mib
p.model.checkpoint, p.model.parameter_count
```

Command line:

```bash
cd experiments/monocular_depth_adapter
python3 build_frozen_set.py --verify           # re-hash the frozen frames
python3 run_inference.py --batch-size 1 --uncertainty native+flip --skip-failing
python3 benchmark_report.py --run bs1_native_flip
python3 make_figures.py --run bs1_native_flip
```

`run_inference.py` merges into an existing run rather than overwriting it, so a
model that needs the whole card to itself can be re-run alone into the same
directory:

```bash
python3 run_inference.py --models dav2_metric_indoor_large --run-name bs1_native_flip
```

## The frozen image set

`frozen_sets/monodepth_frozen_v1.json` pins 24 existing real-Gazebo frames by
SHA-256. Nothing is rendered or synthesised here. Frames carry one of two roles,
and they are not interchangeable:

- **`method_development`** — 12 frames, camera A, `warehouse_aws`. Every
  accuracy, stability, and model-comparison statement comes from these.
- **`batch_plumbing_only`** — 12 frames, cameras A/B/C/D, `warehouse_full_4cam`,
  three robot poses seen from all four mounts. Used **only** to show that batch
  inference runs across four calibrations and to record what that costs.

The split exists because of the repo's two-world rule: method development
happens in the original warehouse, and the four-camera world is reserved for
evaluating frozen methods. The acceptance criterion "batch inference works for
all four cameras" is a plumbing question, so it is answered with plumbing-only
frames and no model ranking is drawn from them. Both the manifest and the
benchmark report carry that restriction in writing.

## Uncertainty

Three signals, all optional, chosen with the `uncertainty=` argument:

- **native** — the model's own confidence head. Metric3D v2 and UniDepthV2 have
  one; Depth Anything V2 does not, and in that case the adapter returns `None`
  and says why rather than inventing something. Each model's scale is its own;
  the raw values are passed through unnormalised.
- **flip** — a second forward pass on the left-right mirrored image (with the
  principal point mirrored to match), unflipped and differenced. Costs one extra
  pass. For non-metric models the mirrored prediction is affine-aligned first,
  because a relative-depth network is free to return a different scale and
  differencing that free parameter would report gauge as uncertainty.
- **temporal** — per-pixel spread across several frames from one fixed camera,
  computed offline by `run_inference.py`. The scene did not move, so whatever
  moved is either the robot or the model being unstable.

## Benchmarking without declaring a winner

`benchmark_report.py` reports six axes side by side and refuses to collapse
them: what each model returns, what it costs, how stable it is, how well the
models agree (in metres where both are metric, by rank agreement always), whether
a model's own confidence tracks where the models disagree, and what focal length
a self-estimating model thinks it is looking through.

None of these is accuracy — the frozen set has no depth labels, and the adapter
has no route to any. A single depth RMSE would produce a ranking that would not
survive contact with what the depth is actually for, which is why it is not
computed here.

## Was any of it right? (ground truth)

The adapter cannot answer that — it has no route to truth, by design. A separate
evaluation does, using the camera's own co-located Gazebo depth sensor:

```bash
# with the simulator up and the depth topic bridged (see the script's docstring)
python3 capture_depth_truth.py              # 12 matched RGB + depth pairs
python3 evaluate_against_truth.py           # raw / floor-anchored / oracle arms
```

The model side is allowed exactly one outside assumption, the one a real
warehouse can also make: **the floor is a plane at z = 0 and the camera
calibration is known.** That makes the true depth of any open-floor pixel
computable analytically, so a scale and a shift can be fitted with no sensor and
no CAD, and then scored on the pixels that are *not* floor.

Full numbers in [`logs/studies/monocular_depth_adapter/RESULTS.md`](../../logs/studies/monocular_depth_adapter/RESULTS.md).
The short version: raw metric depth is off by 1.6–5.2 m, floor-anchoring brings
the best model to **0.247 m MAE / 3.7% relative**, and that is within 1.2 cm of
an affine fitted against truth itself.

## Environment

Verified on this machine 2026-08-11: Python 3.10.12, torch 2.5.1+cu118,
transformers 5.13.1, **Quadro P2000 with 4 GB of VRAM**, which is the binding
constraint on model size.

Dependencies added for this study: `timm`, `einops`, `mmengine` (Metric3D config
loading), and `unidepth` installed with `--no-deps` from the authors' GitHub.
Two import shims live in the backends and are documented there: a three-symbol
`mmcv.utils` stub backed by mmengine, and a `wandb` stub, both avoiding heavy
dependencies that only the models' training paths need.

`mmengine` pulls numpy 2.x by default; numpy was pinned back to **1.26.4** after
installing it, which is what the rest of the repo expects.

## Tests

```bash
python3 -m pytest tests/perception/test_monocular_depth_adapter.py -q      # ~1 s, in the default suite
MONODEPTH_GPU_TESTS=1 python3 -m pytest tests/perception/test_monocular_depth_adapter.py -q   # + the real networks
```

The fast tests drive the full adapter path through a deterministic stub backend,
so batching, valid masks, timing, memory, uncertainty and storage are all covered
without a GPU. The real networks are opt-in because they take minutes.

## Status and what is not done

This study has no `research/registry.yaml` entry yet, so `hygiene_check.py`
reports it as an unregistered experiment directory. It serves the operational
depth arm of the reliability-source benchmark (WS07), which owns the registry
and the decision about whether monocular depth stays an arm at all — that entry
is theirs to add, not this study's to assume.

Not built, and deliberately: any metric anchoring of the non-metric models, any
back-projection to a height map, any comparison against a depth sensor or CAD,
any ROS integration, and any statement about which model is best.
