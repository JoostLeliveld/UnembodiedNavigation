# REUSE_MAP — monocular_depth_adapter

What this study reuses instead of reimplementing, and the one thing it
deliberately does not touch.

| need | reused from | why not local |
|---|---|---|
| camera intrinsics `K` for a capture | `unav_common.camera_model.ObliqueCameraModel` (`.K`) | `f = (W/2)/tan(fov/2)` written here would be a second definition of the cameras' calibration that can drift from the runtime one. Used in `frozen_set.py` only. |
| mount pose (x,y,z,r,p,y) -> look-at point | `experiments.core.world_profiles.compute_look_at_from_pose` | same function the capture scripts used; reproduces the stored `look_at` for camera A exactly (checked 2026-08-11). |
| Spearman rank correlation | `scripts/shared/metrics.py:spearman` | repo hard rule — an audit found three divergent hand-rolled Spearman formulas. Used in `benchmark_report.py`. |
| real Gazebo imagery, camera A, `warehouse_aws` | `logs/visibility_comparison/warehouse_visibility_capture_v1` | existing real capture; no new rendering, and no synthetic imagery anywhere in this study. |
| real Gazebo imagery, cameras A-D, `warehouse_full_4cam` | `logs/visibility_comparison/commissioning_grid_20260807` | the only existing four-camera capture. Used for batch plumbing only — see the two-world note in the README. |
| Metric3D v2 preprocessing recipe | reproduced from the hub checkout's `hubconf.py` `__main__` block | the authors only expose it inside a script guard, so it cannot be imported. Copied verbatim with the source named in the module docstring. |

## Not reused on purpose

`scripts/geometry_visibility/mono_depth_occlusion_prior.py` runs Depth Anything V2
output through a floor-plane affine correction and into an occlusion prior. It
consumes a pre-computed `monodepth_stack_large.npy` whose producer was never
committed and whose input frames are no longer in the tree
(`logs/geometry_visibility_prior/frames/` is gone), so it cannot currently run.

This study supplies the missing half — the inference that produces such an array,
with the metadata that one lacked — but does not adopt its floor-plane
correction. That correction is a scene anchor, and anchoring is outside this
adapter's remit by design. Wiring the two together is a downstream decision.

## What this study owns

`monodepth/` is a self-contained package with no repo imports at all, enforced by
`tests/perception/test_monocular_depth_adapter.py`. If it is later promoted out
of `experiments/` into shared code or a ROS node, it moves as-is; nothing has to
be untangled first.
