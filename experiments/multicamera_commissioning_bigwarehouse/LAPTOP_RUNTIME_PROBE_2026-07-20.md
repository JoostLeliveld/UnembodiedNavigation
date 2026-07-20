# Laptop runtime probe — 2026-07-20

This is diagnostic provenance for the isolated
`warehouse_full_4cam_laptop_640x360` world and
`detector_4cam_v3_laptop_640x360.yaml`. It is not detector, calibration, or
paper evidence.

## Configuration

- Host GPU: Quadro P2000 Mobile, 4 GiB VRAM.
- Four source cameras: 640×360 RGB, 5 Hz requested simulated cadence.
- Detector: one native `warehouse_yolo_detector_v1/model.pt`, four-image
  batch, `imgsz=640`, GPU device 0.
- Integrity policy: strictly new four-camera batches, no image reuse,
  0.10 s maximum stamp skew, 0.50 s pending-frame expiry.

## Result

The profile started successfully and its camera/ROS topic identities matched
the paper runtime. It did **not** pass its throughput gate:

| Stream | Observed wall-clock rate |
| --- | ---: |
| camera A raw RGB | ~2.57 Hz |
| camera B raw RGB | ~2.52 Hz |
| camera C raw RGB | ~2.66 Hz |
| camera D raw RGB | ~2.82 Hz |
| accepted camera-A `CameraObservation` | ~0.93–1.10 Hz |

The batched detector logged repeated `stamp_skew` rejections. The same
one-second accepted-batch cycle was observed at 1280×720 and 640×360, so a
fourfold source-pixel reduction does not solve the timing bottleneck on this
host. No tolerance was widened: doing so would violate the stated 100 ms
temporal-integrity contract.

A non-evidence ablation disabled the ROS bridge for all 40 world contact
sensors (each configured at 60 Hz). Accepted output rose only to about
1.15 Hz, so contact-topic forwarding is not the primary cause. It may remain
disabled only for timing diagnosis: collision-valid runs keep it enabled.

## Decision

`detector_4cam_v3_laptop_640x360.yaml` remains
`commissioning_candidate_blocked_from_evidence`. The final paper simulations
should use a stronger host with the unchanged 1280×720/v2 runtime, unless a
separate future synchronization redesign is specified and validated.
