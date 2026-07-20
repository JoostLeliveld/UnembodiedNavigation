# C2 belief / camera / timing diagnosis & fix

Investigation of the truth ↔ plan ↔ belief ↔ camera discrepancy in C2 runs.
Diagnostics-first: every module was checked against ground truth in live runs.

## TL;DR

The discrepancy had **two causes**, and a single perception-node change fixed both:

1. **Detector staleness** (the dominant cause). In-run YOLO inference inflated to
   ~290 ms (idle floor on the Quadro P2000 is ~100 ms, *not* the 17 ms in old
   notes), giving ~1.3–2 Hz updates and **0.6–0.9 s belief staleness**. Root
   cause = GIL/launch contention between the detector's daemon inference thread
   and `rclpy.spin` in one process.
2. **Heading drift**. `heading_update_mode=camera_xy_only` never corrects yaw, so
   the belief dead-reckoned noisy odom to **30–70°** — even in fully-visible
   regions — which steered the robot into obstacles. This was the actual
   collision cause.

**Fix (perception node only, machine-independent, bit-identical detections):**
run YOLO inference *synchronously in the image callback* (single Python thread,
no GIL contention) + a startup warmup. Queue depth 1 preserves latest-frame-only.

**Result:** staleness 0.6–0.9 s → **0.14 s**; truth-belief position error 0.13–0.22 m
→ 0.08–0.19 m; truth-belief heading error **70° → 6–11°** (the EKF couples
position↔heading, so fresher position updates indirectly tamed heading — no
explicit heading change was needed); C2 outcomes flipped from collision to
**4/5 goal-reached, 0 collisions**.

## What was checked (trust no module)

| Stage | Method | Verdict |
|---|---|---|
| YOLO bounding box | camera-frame overlay vs truth-projected model box (`diag_bbox_overlay`) | **SANE** — box frames the robot at all frames |
| Pixel→world projection | capture-time (latency-removed) error vs truth | **SANE** — 0.054 m median |
| Affine vs planner-offset calibration | reconstructed planner path vs `/state/bev` affine | **immaterial** — 0.051 vs 0.055 m; no fix needed |
| EKF update | innovation / NIS / accept-reject ledger | **SANE** — NIS small, 89–100% accepted |
| Belief drift | truth vs belief over time, occlusion-shaded | TIMING staleness + indirect heading drift |
| Controller | tracking-yaw source trace | uses belief heading; fine once belief heading is fixed |

## What was NOT the problem (ruled out with data)

- Camera/projection geometry (capture-time error 0.05 m).
- The affine-vs-offset calibration inconsistency (0.004 m difference).
- TorchScript export — **abandoned**: not faster (109 vs 100 ms) and not
  bit-identical on this seg model (loads as `task=detect`).
- An explicit visual/displacement heading correction — **not needed**; the
  timing fix kept heading at 6–11° across all tasks including the occluded one.

## Code changes

- `src/perception/perception/nodes/yolo_robot_detector_node.py`:
  `inference_in_callback` (default true) runs inference in the callback;
  `warmup_iters` (default 3); `use_torchscript` (default false, parameter-gated).
- Param plumbing in `visibility_launch_common.py`,
  `warehouse_primary_comparison.launch.py`, `run_visibility_campaign.py`.

## Diagnostic tooling (scripts/visibility_comparison/diag/)

- `analyze_run.py` — timing/projection/update/drift summary, phased before/after first cmd.
- `bench_detector.py` — idle inference benchmark + TorchScript bit-identity gate.
- `diag_bbox_overlay.py` — camera-feed GIF + bbox-vs-truth montage.
- `diag_belief_drift.py` — belief-vs-truth over the reliability map, error decomposed, occlusion-shaded.
- `diag_update_projection.py` — projection-path compare + EKF update timeline.
- `diag_route_animation.py` — per-task paired C1/C2 route animation (error grows occluded / recovers visible).

## Open item

- `route_west_to_a1_upper` ends **stuck** (no crash): the far-west service lane is
  seen at a grazing camera angle (periphery projection error ~0.5 m) and is
  low-ρ; the belief degrades and the robot stops safely. This is consistent with
  the intended C2 "stop safely in unreliable region" behavior, but is the one
  task where the camera cannot help enough — i.e. failure occurs in a
  poorly-observed region, as intended.

(Numbers above from the `_diag_baseline`, `_timingfix`, and `_validate_c2` runs;
paired C1/C2 results appended after the C1 batch.)
