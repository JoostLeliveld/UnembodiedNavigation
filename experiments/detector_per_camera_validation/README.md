# detector_per_camera_validation

**Question:** the frozen detector reports one pooled validation score. Is it equally good at
all five camera viewpoints, or would a per-camera localization difference partly be a
detector artifact?

`validate_per_camera.py` partitions the detector's own validation split by camera prefix and
scores each viewpoint separately. It retrains nothing, adds no data, and refuses to run if
the detector weights no longer match the hash in `docs/reproducibility_inputs.md`.

Results and interpretation: `logs/studies/detector_per_camera_validation_20260907/`.

**Headline:** camera D is the weak viewpoint (mAP50-95 0.7763, recall 0.8725) against
0.9321-0.9713 for the other four. Camera C, which returns the fewest field detections, is
near the top — so C's low yield is occlusion, not recognition failure.
