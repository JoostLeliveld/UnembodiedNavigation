# Laptop 640×360 commissioning successor

`detector_4cam_v3_laptop_640x360.yaml` and
`warehouse_full_4cam_laptop_640x360.world.sdf` are a separate, four-camera
commissioning profile for the Quadro P2000 laptop. The paper world and its
locked v2 configuration remain unchanged at 1280×720.

The successor keeps the camera poses, 90-degree horizontal FOV, 5 Hz simulated
sensor cadence, model checkpoint, topics, and strict fresh-frame policy. Each
RGB/depth/semantic camera image is instead 640×360, reducing rendering pixels
by four while preserving 16:9 geometry. Its model instances retain the normal
`external_camera` through `external_camera_d` names, so the existing bridge and
detector topic contract is unchanged.

## Status

This profile is **commissioning-only**. It is blocked from D0–D4 evidence,
training, GP fitting, or paper claims. Do not point the campaign ledger or a
paper recorder at it. In particular, the v2 projection calibration is invalid
for this source resolution.

## Diagnostic launch

```bash
source install/setup.bash
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=<unused-domain-id> IGN_PARTITION=laptop_640x360_probe

ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  world:=warehouse_full_4cam_laptop_640x360.world.sdf \
  world_name:=warehouse_full_4cam_laptop_640x360 \
  headless:=true reset_world:=false shadow_manager:=false \
  yolo_model:=logs/perception_models/warehouse_yolo_detector_v1/model.pt \
  yolo_imgsz:=640 yolo_batched_four_camera:=true yolo_batched_device:=0 \
  yolo_batched_cpu_num_threads:=2 yolo_cpu_num_interop_threads:=1 \
  yolo_opencv_num_threads:=1 yolo_max_batch_stamp_skew_s:=0.10 \
  yolo_max_pending_wall_s:=0.50 yolo_use_masks:=false
```

The detector's `yolo_imgsz:=640` is unchanged: it is the network's square
letterbox input size, while 640×360 is the rendered source geometry.

## Promotion gates

Create a new evidence-eligible successor only after all of these are preserved
as immutable artifacts for this exact world and model:

1. Every RGB stream and every `CameraObservation` stream sustains at least
   3.0 wall-clock Hz over the declared window.
2. Accepted four-camera batches remain at or below 0.10 s stamp skew, with no
   reused images, and pass the existing freshness/age gate.
3. A new per-camera projection calibration is fit from laptop-world outputs;
   the 1280×720 calibration must not be rescaled by hand.
4. A source-matched detector data/validation set passes both range detection
   and bottom-point localization audits, including the 12–16 m range.
5. The runtime readiness report, source/config/model hashes, and all gate
   outputs are recorded before any campaign plan is created.

If any gate fails, retain the diagnostic result and use a stronger host for the
unchanged 1280×720 paper configuration. Do not relax the 0.10 s synchronization
bound to manufacture throughput.

`bridge_contacts:=false` is available only for a non-evidence timing ablation.
It suppresses the ROS forwarding of the 40 configured contact streams; it does
not make a collision-valid run and the first ablation did not materially solve
the throughput failure.
