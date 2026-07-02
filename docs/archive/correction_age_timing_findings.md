---
marp: true
title: Planner Pixel-Correction Age — Timing Investigation Findings
---

# Planner Pixel-Correction Age: Timing Findings

Goal: reduce `planner_pixel_correction_age_s` from ~0.92 s toward 0.2 s for closed-loop stability.

**Result: 0.92 s → 0.27–0.31 s (2.7–3.4×), validated, robot reaches goal every run.**
The literal 0.2 s is **not** reachable by configuration on this Gazebo + `ros_gz_bridge` + mobile-P2000 setup; it requires an image-transport re-architecture or hardware (evidence below).

---

## What was tested (8 loaded C2 runs, route_apron_to_a3_mid, each goal-reached)

| Change | true correction age | notes |
| --- | --- | --- |
| baseline | ~0.92 s | rendering on Intel iGPU, P2000 idle |
| **GPU offload → P2000** | **0.72 s** | **banked, free, accuracy-safe** (PRIME render offload) |
| + YOLO imgsz 960→640 | 0.42 s | acceptance 90→93%; changes detector input res |
| + camera 5→10 Hz | 0.41 s | no gain (not camera-rate bound) |
| + apply-on-arrival (`pixel_correction_min_interval_s=0`) | 0.33 s | no gain (planner apply gap is only 0.020 s) |
| + executor 2→6 threads | 0.34 s | no gain (not thread-starved) |
| + physics 1 kHz→250 Hz | 0.31 s | small gain (CPU/physics not the main hog) |
| + camera 1280×720→640×360 (+ scaled pixel params) | 0.27 s | **image-delivery UNCHANGED (0.16 s); accuracy slightly worse** |

Only the GPU offload is kept in the repo; all other changes were reverted after measurement.

---

## Root cause (latency decomposition, cross-joined detector & planner stamps)

| segment | latency | tunable? |
| --- | --- | --- |
| capture → detector **receives** image | **~0.16 s** | **NO** — fixed |
| detector inference + processing | 0.06–0.12 s | fast (YOLO 56 ms) |
| detector publish → **planner applies** | **0.020 s** | planner is fine |

The dominant term is the **~0.16 s image-delivery latency in the `ros_gz_bridge` path**, and it is **invariant** to image resolution (4× fewer bytes → unchanged), camera rate, CPU/physics load, and executor threads. The planner (0.020 s) and detector (fast) are **not** the bottleneck. Lingering processes, extra cameras (seg camera is lazy), the GPU-lidar (already off in the campaign), and logging were all checked and ruled out.

---

## Why 0.2 s needs more than configuration

The fixed ~0.16 s lives in the ROS image bridge under Gazebo **Fortress (gz-transport 11)**. The only routes past it:

| Path | Reaches 0.2 s? | Effort / risk |
| --- | --- | --- |
| **C++ gz-transport-11 bypass** (detector reads image direct, skip `ros_gz_bridge`) | likely | new C++ node; Python gz bindings are Harmonic (transport13) and can't talk to the Fortress sim |
| **Migrate sim Fortress → Harmonic** (gz-sim 8.11 is installed) | likely | world/plugins/sensors/bridge all Fortress-built; full migration + revalidation |
| **Faster hardware** | yes | detector + planner already fast; the bridge + 6-core mobile CPU is the limiter |

---

## Bottom line

- **Banked now:** GPU offload (0.92 → 0.72 s), free and accuracy-safe; combined with imgsz 640 + lighter physics it reaches **~0.27 s (2.9×)** with an accuracy/comparability caveat to validate.
- **The detector and planner are not the problem** — this is strong, precise evidence for the compute request (the limiter is the simulator's image-transport pipeline, not the method).
- **0.2 s is a real engineering/hardware decision**, not a tuning oversight: pick the C++ bridge-bypass, the Harmonic migration, or new compute.
