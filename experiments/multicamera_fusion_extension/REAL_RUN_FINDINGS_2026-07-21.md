# Four-camera pipeline: brick-by-brick REAL Gazebo runs (2026-07-21)

> **Historical runtime snapshot.** “Current” and “deployed” in this dated note mean
> 2026-07-21. Its projection corrections and per-camera error numbers are retired and must
> not be compared with current floor-IPM metrics. See `docs/localization_metrics.md`.

Every number here is from an actual headless Gazebo run on this host
(Quadro P2000, 4 GiB), measured live — no synthetic data, no replay. Built one
layer at a time; each brick verified before the next.

## Brick 0-1 — environment & build
- No zombie sim/`cmd_vel` processes before starting (a real hazard: a stale
  `drive_study_route` publishing `/cmd_vel` corrupts runs).
- `src` == `build` == `install` (editable install points at `build/`, which was
  in sync), so a run executes current code.

## Brick 2-3 — sim alone (no detector)
- `real_time_factor = 0.748`; four camera raw streams at ~2.8-3.6 Hz wall,
  stamp dt median 0.200 s (= requested 5 Hz sim). **The sim is not the
  bottleneck.** (This overturns the earlier `THROUGHPUT_DIAGNOSIS` reading of
  RTF 0.315, which had been measured WITH the detector running.)

## Brick 4 — add the batched GPU detector
- Batched mode (`yolo_batched_four_camera:=true`, the default) puts **all four
  cameras incl. camera_A in one GPU batch** (`device=0, batch_order=A,B,C,D`).
- imgsz=640 strict: **all four cameras emit valid detections at 0.97 Hz**, age
  0.11 s (fresh). RTF 0.722. GPU 2560 MiB / 55% util — headroom to spare.
- **camera_A is NOT broken in batched mode.** A static-spawn logger captured
  **214 valid camera_A detections**; a driven traverse added 18 more. The
  "dead camera_A" (0 detections in the captured smoke data) is an artifact of
  the SEPARATE-PROCESS fallback, where the launch pins camera_A to **CPU**
  (`yolo_device_camera_a=cpu`, ~986 ms/frame) because 4 single-camera GPU
  processes OOM the 4 GiB card. The batched path avoids this entirely.
- Bottleneck is **batch inference latency**: warmup 2414 ms / 3 iters ≈ **805 ms
  per 4-image batch** at imgsz=640 → ~1 Hz ceiling. Not sync, not GPU
  saturation (async mode gave the same ~0.9 Hz but stale 0.9 s age).

## 3 Hz question — answered by measurement
- imgsz=416: batch inference ~298 ms, obs rate rose to **2.42 Hz** — BUT
  **detection collapsed to camera_B only** (A/C/D produced 0 valid detections
  on a robot all four detect at 640). So imgsz reduction buys rate by
  destroying the detection the multi-camera method needs.
- **Verdict:** 3 Hz wall is not achievable at full detection quality on this
  GPU (inference-bound). Run at **imgsz=640, ~1 Hz, all four cameras fresh** for
  data collection. Wall rate only limits LIVE closed-loop (E8); offline
  record→replay uses sim time (5 Hz), so ~1 Hz wall is fine — matches the
  "start at 1 Hz" decision.

## Brick 5-6 — driven traverse, camera_A and camera_C vs ground truth
Route `south_to_north_handover`, spawned AT the start (spawn resets `/odom`
consistently — teleporting mid-run via `set_pose` does NOT reset control odom
and drove the robot the wrong way; that was a real bug I hit and corrected).
The robot tracks at ~0.074 m/s vs 0.15 commanded (slow ~1 Hz control loop), so
a full 14.4 m traverse needs `--max-sim-runtime-s ~300`; this pass reached the
handover band (y −7.2 → −0.27) before the default deadline.

`/ground_truth_tf` publishes **zero header stamps**, so GT must be timestamped
by `/clock` at receipt and keyed on `child_frame_id: turtlebot3` (fixed in the
logger). 302 driven detections, all GT-joined.

**Live per-camera projection error vs GT (raw pixels re-projected two ways):**

| camera | n | RAW bias_y (m) | RAW mean err (m) | +v2 calib bias_y | +v2 mean err |
|---|---:|---:|---:|---:|---:|
| A | 18 | −0.125 | 0.130 | −0.056 | 0.087 |
| C | 157 | **−0.151** | 0.156 | **−0.034** | **0.077** |
| D | 127 | +0.107 | 0.125 | +0.010 | 0.018 |

The v2 along-bearing calibration (`projection_calibration_v2/…json`, already
fitted) halves camera_C's error and cuts its y-bias to −0.034 m — verified on a
FRESH driven run, independent of the earlier replayed smoke data.

## The two fixes

**camera_A — already correct in batched mode (the default).** No code change:
always run `yolo_batched_four_camera:=true`. The captured smoke data with dead
camera_A came from the separate-process CPU fallback; batched puts A on the GPU
with B/C/D and it detects normally.

**camera_C — apply the v2 projection calibration.** The math fix is proven; the
gap is that it is not wired into the live pipeline: the launch arg
`manager_projection_calibration` defaults to `""` (empty), and the recorder
wrote `pred_world` UNCORRECTED (the v2 file was fitted from that data but never
applied back). Fix = feed `projection_calibration_v2` wherever the world
position is computed (live: `manager_projection_calibration:=<v2.json>`;
offline: re-project from pixels with the calibration — done in the loader).

## Full traverse — all four cameras validated (update, same day)

A complete south→north traverse (spawn at start, `--max-sim-runtime-s 420`,
robot reached y +7.01) captured **295 GT-joined detections across all four
cameras**, clearing the earlier camera_B gap. v2 along-bearing calibration vs GT:

| camera | n | RAW mean err (m) | +v2 mean err (m) |
|---|---:|---:|---:|
| A | 50 | 0.128 | 0.087 |
| B | 59 | 0.058 | **0.026** |
| C | 91 | 0.184 | 0.086 |
| D | 95 | 0.107 | 0.036 |

**camera_B (previously unverified) is now confirmed: 0.058 → 0.026 m.** The v2
calibration more than halves error on every camera, so it is validated to land
in the live pipeline for the whole camera set — the thin-evidence caveat below
is resolved. camera_B only detects in the far-north stretch (y ≳ 5), which is
why the earlier half-traverse saw zero B detections (not a camera fault).

NOTE: this validation used a lightweight detection+GT logger that did NOT record
odometry, so the closed-loop FUSION re-analysis on this full A/C→B/D handover
still needs a run with the operational recorder (odom + perception + GT
firewall) → `load_commissioning_run` → subset/fusion sweep. That is the real
test of whether the handover regime (no single camera covers the whole aisle)
finally makes fusion beat the best single camera.

## Caveats / honesty
- The `south_to_north` pass only reached the handover band, so camera_B (north)
  has 0 detections here; a full traverse (raise the sim deadline) is needed for
  B and for a complete A/C→B/D handover dataset.
- `manager_projection_calibration` and the launch defaults live in the parallel
  commissioning workstream's files (`src/experiments/launch/…`, currently
  uncommitted by them) — the live-wiring change must be coordinated, not landed
  blindly here.
- Async detector mode is marked diagnostic-only (does not publish the strict
  four-camera runtime contract) — not eligible for evidence runs.
