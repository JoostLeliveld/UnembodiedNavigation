# `scripts`

The active perception script surface is now intentionally small:

- [`perception/capture_external_camera_images.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/capture_external_camera_images.py)
- [`perception/drive_robot_pattern.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/drive_robot_pattern.py)
- [`perception/test_yolo_out_of_box.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/test_yolo_out_of_box.py)
- [`perception/make_preview_grid.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_preview_grid.py)
- [`perception/train_yolo_seg.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/train_yolo_seg.py)
- [`perception/make_redmask_pseudolabels.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/make_redmask_pseudolabels.py)
- [`perception/capture_simseg_dataset.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/capture_simseg_dataset.py)
- [`perception/capture_projected_bbox_dataset.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/capture_projected_bbox_dataset.py)
- [`perception/analyze_dataset_robustness.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/perception/analyze_dataset_robustness.py)

The active visibility-comparison backbone now lives under:

- [`visibility_comparison/capture_visibility_samples.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/capture_visibility_samples.py)
- [`visibility_comparison/extract_perception_targets.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/extract_perception_targets.py)
- [`visibility_comparison/build_gp_targets.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/build_gp_targets.py)
- [`visibility_comparison/fit_visibility_gps.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/fit_visibility_gps.py)
- [`visibility_comparison/plot_gp_and_ambiguity_maps.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/plot_gp_and_ambiguity_maps.py)
- [`visibility_comparison/plot_gp_signal_figure.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/plot_gp_signal_figure.py)
- [`visibility_comparison/plot_planned_paths.py`](/home/joostleliveld/Thesis/UnembodiedNavigation/scripts/visibility_comparison/plot_planned_paths.py)

Everything else in this folder should be treated as secondary or legacy unless a specific experiment still depends on it.
