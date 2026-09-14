# Reproducibility inputs

Current input identities are recorded only in the active pipeline manifests:

- `experiments/warehouse_v2_sketches/world_freeze_manifest.json`;
- `experiments/thesis_pipeline_lock/robot_target_manifest.json`;
- `experiments/thesis_pipeline_lock/camera_capture_map_manifest.json`;
- `experiments/thesis_pipeline_lock/stage04_dataset_manifest.json`;
- `experiments/thesis_pipeline_lock/stage05_detector_manifest.json`;
- the future Stage 06–11 manifests after they are frozen.

Do not duplicate hashes in prose or copy them from superseded protocols. The active manifest
is the authority. A changed byte requires a new manifest identity and an explicit lock update.

