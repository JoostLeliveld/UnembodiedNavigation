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

An explicit NVIDIA PRIME/EGL run (`use_nvidia_prime_offload:=true`) removed the
Gazebo EGL/Dri warnings and raised raw RGB throughput to approximately
3.35–3.82 Hz. Accepted strict batches nevertheless remained about 1.10 Hz:
the remaining limitation is timestamp alignment, not rendering throughput or
YOLO inference.

## Detector-off and asynchronous follow-up

The detector-off baseline, with the same 640×360 world, PRIME/EGL selection,
and contact bridge disabled, sustained raw wall-clock rates of **3.578 Hz**
(A), **3.577 Hz** (B), **3.441 Hz** (C), and **3.441 Hz** (D). Rendering and
image bridging therefore clear the 3 Hz raw-stream gate on this machine.

v4 then tested a diagnostic-only shared-model asynchronous microbatch path. It
preserves a latest unseen image per camera but publishes no strict runtime
contract and cannot record evidence. With its 20 ms coalescing window, live
raw streams fell to **2.322–2.460 Hz** and all four `CameraObservation`
streams measured **0.742 Hz**. A 120 ms window was no better: raw streams were
**2.303–2.453 Hz** and observations **0.709–0.712 Hz**. A representative
diagnostic reported about **200.3 ms** inference wall time but **1050.9 ms**
end-to-end callback time.

This rules out both the strict 100 ms stamp gate and a simple microbatch-window
tuning as the primary explanation. Running YOLO concurrently materially slows
the integrated Gazebo/ROS schedule; the laptop cannot meet the 3 Hz
end-to-end requirement for this four-camera simulator/detector stack.

## Decision

`detector_4cam_v3_laptop_640x360.yaml` remains
`commissioning_candidate_blocked_from_evidence`. The final paper simulations
should use a stronger host with the unchanged 1280×720/v2 runtime, unless a
separate future synchronization redesign is specified and validated.

`detector_4cam_v4_laptop_async_microbatch.yaml` is retained as a versioned
negative diagnostic. It is `diagnostic_candidate_blocked_from_evidence`; it
does not authorize campaign recording, calibration, training, or paper claims.

## Corrected executor and input-size follow-up

The early raw-topic rate probes were themselves intrusive: each `ros2 topic
hz` subscriber deserializes a full RGB image, adding four extra image consumers
to an already constrained run. Mixed raw/output measurements that produced an
output rate above a raw rate are therefore invalid and are not used for any
decision. Later probes measured only the four small `CameraObservation` topics.

Separating asynchronous image reception from inference with a two-thread ROS
executor improved the 640×360, 416-input diagnostic to **2.384–2.410 Hz**.
This is a real improvement over the previous single-executor result, but does
not meet the 3 Hz gate. A 384 input was worse (**1.909–1.938 Hz**), a 3 Hz
camera-cadence successor (v5) was worse (**1.169–1.194 Hz**), and a 250 ms
coalescing window was worse (**1.090–1.236 Hz**). None is promotable.

An isolated GPU benchmark measured a native Ultralytics 416 four-image batch
at **93.2 ms median** (95.0 ms p90), so the remaining deficit is not network
inference throughput alone. The existing TorchScript export is fixed to a
960-square anchor layout and fails for the intended input shape; it is not an
acceleration option without a newly exported, validation-gated artifact.

The laptop remains worth pursuing, but the next ablation must reduce
render/bridge image-transport cost (for example, an isolated lower-source-
resolution successor) while retaining a fresh detector and calibration
validation plan. No candidate crosses into evidence until it sustains 3 Hz.
