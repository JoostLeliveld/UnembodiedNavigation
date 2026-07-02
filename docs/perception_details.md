# Perception Model and YOLOv11 Detector Details

This document provides a comprehensive log of the YOLOv11 instance segmentation detector configuration, training datasets, optimization parameters, and inference settings.

---

## 1. Detector Model Details
- **Architecture:** YOLOv11 instance segmentation network (nano variant).
- **Fine-Tuning Checkpoint:** `logs/perception_models/warehouse_yolo_detector_v1/model.pt` (local, not tracked in git).
- **Public Metadata:** `paper_artifacts/perception/warehouse_yolo_detector_v1/`
- **Base Model:** YOLOv11n-seg.
- **Target Class:** `robot` (mapped to Class ID `0` at inference; mapped from Gazebo semantic segmentation index `23`).
- **Input Image Size:** 
  - **Training:** $960 \times 960$ pixels (clean occlusion-gated retrain, `warehouse_yolo_detector_v1`)
  - **Inference:** `imgsz = 640` (the campaign runs the low-latency arm; the 960-trained model is evaluated at 640 for faster detector callbacks)

---

## 2. Training Dataset Details
- **Source World:** `warehouse_aws.world.sdf`
- **Image Collection Protocol:**
  - Collected by teleporting the robot through a 2D spatial grid of size $24 \times 20$ with 4 uniform yaw headings per position ($0.0$, $\pi/2$, $\pi$, $3\pi/2$ rad), yielding $960$ candidate poses.
  - Simulator-native semantic segmentation was used offline to automatically generate ground-truth labels and masks.
  - Frame pairs were accepted only if the timestamp difference between the RGB frame and semantic label frame was $\le 60$ ms.
  - A settling period of 3 simulator frames was enforced after each teleport before capture.
- **Dataset Size:** 852 total images (683 training, 169 validation).
- **Split Strategy:** Deterministic group-level `spatial_yaw_bucket` split to eliminate adjacent-pose and orientation data leakage.
- **Augmentation Hyperparameters:**
  - Horizontal Flip (`fliplr`): 0.5 probability
  - Translation (`translate`): 0.1 fraction
  - Scaling (`scale`): 0.5 fraction
  - Mosaic (`mosaic`): 1.0 probability
  - Random Erasing (`erasing`): 0.4 probability

---

## 3. Training & Optimization Configuration
- **Epochs:** 30
- **Batch Size:** 8
- **Optimizer:** `auto` (ultralytics standard AdamW/SGD)
- **Learning Rate Schedule:**
  - Initial Learning Rate ($\text{lr0}$): 0.01
  - Final Learning Rate fraction ($\text{lrf}$): 0.01
  - Momentum: 0.937
  - Weight Decay: 0.0005
  - Warmup: 3.0 epochs (warmup momentum = 0.8, warmup bias lr = 0.1)
- **Seed:** 0 (deterministic)

---

## 4. Inference & Runtime Integration
- **Confidence Threshold:** $\tau_{\mathrm{conf}} = 0.05$
- **IoU Threshold:** 0.45
- **Centroid Selection:** Bounding-box bottom-centre projected to the planar ground coordinates $(x, y)$ using a calibrated projection model.
- **Missed Detection Handling:** Missing detections (scores $< \tau_{\mathrm{conf}}$ or no selected detection) are recorded as a zero-score ($c_i = 0$), and no camera-based EKF correction is applied.
- **Runtime Signals:** Figure availability plots use the raw YOLO detection flag and score. The Normalized Innovation Squared (NIS) gate is **active** with threshold $9.21$ ($\chi^2$ at 2 DOF, $0.99$): camera measurements whose innovation NIS $> 9.21$ are rejected so an outlier cannot collapse the belief covariance. The self-heal recovery is **disabled** (`pixel_correction_nis_reject_cov_scale = 1.0`); only the standard gate remains.

---

## 5. Training Performance Plots

### Training Metrics over 30 Epochs
![YOLO Training Results](../paper_artifacts/perception/warehouse_yolo_detector_v1/results.png)
