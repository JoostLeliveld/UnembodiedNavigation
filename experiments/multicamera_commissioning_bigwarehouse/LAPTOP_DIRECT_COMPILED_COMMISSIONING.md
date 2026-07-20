# Direct-Gazebo compiled-runtime laptop candidate

`detector_4cam_v6_laptop_direct_gz_torchscript416.yaml` retains the 640×360
source images and routes the four Gazebo RGB payloads directly to the detector.
It bypasses three `ros_gz_bridge` RGB conversions and runs a fixed-shape,
batch-four TorchScript export at 416-square network input. The normal ROS
detection outputs remain unchanged.

## Rate result

On 2026-07-20, the isolated laptop rate probe measured `CameraObservation`
wall-clock output rates of 3.386 Hz (A), 3.387 Hz (B), 3.391 Hz (C), and 3.391
Hz (D). The source resolution was unchanged at 640×360. Only small output
topics were observed; raw image monitors are prohibited because they add image
deserialization load and perturb the measurement.

## Status

This is **not evidence-ready**. It is a commissioning candidate blocked from
recording, campaign ledgers, calibration fitting, training, mapping, and paper
claims. It deliberately publishes no strict runtime contract, and its direct
input plus compiled model must be independently validated.

Before any evidence-eligible successor, freeze and validate all of the
following for the exact source-checkpoint and compiled-artifact hashes in v6:

1. Native-versus-TorchScript detection equivalence, including masks, selected
   bottom points, scores, misses, and class filtering across the independent
   test surface.
2. Fresh 640×360 projection calibration and detector range/localization audits
   at the 416 runtime input.
3. A longer (at least 10-minute) 3 Hz rate, frame-age, duplicate, and GPU
   memory soak, without raw-image monitoring subscribers.
4. An explicit cross-camera temporal-association/fusion contract; direct input
   does not restore the strict 100 ms global batch guarantee.
5. Immutable source, export-command, environment, and artifact-hash records.

The reproducible export starts from the `.pt` checkpoint and uses
`format=torchscript`, `imgsz=416`, `batch=4`, `device=0`, `nms=False`. Never
overwrite the legacy fixed-960 export; write the v6 artifact to the configured
path and verify its SHA-256 before launch.
