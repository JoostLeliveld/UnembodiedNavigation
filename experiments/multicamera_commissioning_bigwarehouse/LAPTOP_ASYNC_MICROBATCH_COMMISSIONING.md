# Laptop asynchronous microbatch diagnostic

`detector_4cam_v4_laptop_async_microbatch.yaml` is the diagnostic successor to
the 640x360 laptop profile. It keeps one shared native YOLO model but releases
each camera's latest unseen image after a 20 ms wall-clock coalescing window.
It does not match camera timestamps, reuse frames, or fuse observations.

## Scope and exclusion

This mode is intended to determine whether the P2000 laptop can sustain at
least 3 Hz output for each camera once strict four-camera stamp admission is
removed. Each published observation retains its source image stamp. A consumer
must apply an explicit temporal-association rule before combining observations
across cameras.

No strict four-camera runtime contract is published in this mode. Existing
recorders consequently fail closed, so this profile is barred from D0--D4,
training, calibration fitting, mapping, GP fitting, campaign ledgers, and paper
claims. It is not a workaround for the 100 ms strict-synchronization gate.

## Pilot invocation

```bash
source install/setup.bash
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=<unused-domain-id> IGN_PARTITION=laptop_async_probe

ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  world:=warehouse_full_4cam_laptop_640x360.world.sdf \
  world_name:=warehouse_full_4cam_laptop_640x360 headless:=true \
  reset_world:=false shadow_manager:=false bridge_contacts:=false \
  use_nvidia_prime_offload:=true \
  yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
  yolo_imgsz:=640 yolo_batched_four_camera:=true yolo_batched_device:=0 \
  yolo_synchronization_mode:=asynchronous yolo_async_coalesce_wall_s:=0.02 \
  yolo_batched_cpu_num_threads:=2 yolo_cpu_num_interop_threads:=1 \
  yolo_opencv_num_threads:=1 yolo_max_pending_wall_s:=0.50 yolo_use_masks:=false
```

Measure raw RGB and `CameraObservation` wall-clock rates independently for A--D
over the same steady-state window. Record source-stamp age, duplicates, GPU
memory, renderer selection, and the exact invocation. Do not use the output
rate alone as a cross-camera fusion result.

## Promotion boundary

A new evidence-eligible successor requires fresh 640x360 calibration and
detector validation, an immutable runtime identity, and a specified,
independently validated temporal association/fusion contract. That successor
must state how it treats readings whose camera stamps differ; it cannot inherit
the strict-runtime evidence status from v2 or v3.

The separate v5 asset (`detector_4cam_v5_laptop_3hz_img416.yaml`) paces the
same 640×360 source cameras at 3 Hz and uses a 416-square YOLO input for the
next diagnostic. It remains blocked from evidence for the same reasons and
must pass the complete rate, freshness, calibration, detector, and temporal
association gates before promotion.
