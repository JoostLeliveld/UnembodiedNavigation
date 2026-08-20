# Keypoint measurement: replacing the box-bottom pixel with a marked point on the robot

## The question

Today the camera's reading is the bottom-centre pixel of a YOLO box, pushed through plain
IPM onto the floor. That pixel is not a physical point on the robot: where the bottom edge of
the silhouette sits depends on which way the robot is facing and how far away it is. The
measured consequence is a **lean of about 8.6 cm** in the single-camera warehouse — the same
error frame after frame, which no per-frame noise model can represent and which repeated
looks cannot average away.

The alternative tested here: put two small coloured disks on the robot at known positions in
`base_link`, detect **those** instead, and make the observation model the plain projection of
a point whose location is defined rather than inferred. If that removes the lean, the camera
term becomes something a filter can honestly consume: mean structure in `h`, remaining spread
in `R`.

Two keypoints rather than one also means the reading carries **heading**, which the box bottom
never did.

## What is here

| file | what it does |
|---|---|
| `evaluate_keypoint_model.py` | scores a keypoint model as a *measurement*: pixel residual, floor error in cm, heading error, each split by range / heading / apparent size |

The dataset it reads comes from `scripts/perception/capture_projected_keypoint_dataset.py`
(driven by `scripts/perception/capture_keypoint_dataset.sh`), which teleports the robot over a
grid and writes, per pose, the rendered image plus the analytically projected marker pixels.

## The camera pose is the dataset

The labels are the projection of the markers through *one specific camera*. Move the camera
and every label in the dataset is wrong, so a model trained on it is measuring against a
camera that no longer exists.

That is exactly what had happened to the previously trained model `yolo_pose_aws_v3`
(captured 2026-06-05): it was built with the single camera at **(0, −4.90, 4.50)**, and commit
`991742cf` moved it to **(0, −5.50, 4.80)** while also reshaping the warehouse around it
(floor 11×10 → 12×10 m, south wall from y = −5.0 to −5.6). Projected onto the marker plane,
that camera move shifts the scene by up to **87 px** near the camera and shrinks the
front-to-rear marker separation by 6–12%, which is the quantity heading precision is inversely
proportional to. So v3 is evaluated here as an out-of-distribution baseline, not as the
candidate.

Every capture writes `capture_manifest.json` with the camera pose it used. Check it before
believing any number computed from that dataset.

## Occluded poses are not silently labelled as visible

A projected keypoint always lands on some pixel, including when a rack is in the way. The
capture therefore also asks whether the marker was actually *rendered* at that pixel: a
marker-coloured pixel that also differs from a robot-free background frame. Colour alone is
not enough — the blue rack rails anti-alias against the yellow racks into pixels that pass any
colour test loose enough to survive a 10 m view — and the background test removes them because
racks do not move. Poses where neither marker renders are rejected with reason
`markers_not_rendered` rather than written as confident labels of a hidden robot.

## Going to the four-camera world

The four cameras there sit at (±6, ±10, 6.10) — nothing like the single camera — so that world
needs its own capture and its own trained model, exactly as the box detector already has one
(`warehouse_yolo_detector_4cam_v3_960`). Under the two-world rule this is the *evaluation* step
for a method frozen in the single-camera world, not more method development.

The capture reads **every camera at each teleported pose**, so four cameras cost the same
teleports as one:

```bash
CAMERAS=external_camera,external_camera_b,external_camera_c,external_camera_d \
WORLD=warehouse_full_4cam.world.sdf \
DATASET=logs/perception_datasets/projected_keypoint_dataset_4cam_v1 \
bash scripts/perception/capture_keypoint_dataset.sh
```

Two things this has to get right, both covered by
`tests/perception/test_keypoint_multicamera_capture.py`:

- **The train/val split belongs to the pose, not the image.** All four views of one pose go to
  the same side of it, or the same pose sits in train and val and "held out" means nothing.
- **Each reading is back-projected through its own camera.** Cameras b and d face south while a
  and c face north, so using the wrong one produces a plausible position on the far side of the
  warehouse rather than an obvious error.

A camera that fails to deliver a *fresh* frame after a teleport is recorded as
`no_fresh_frame` and dropped, rather than reusing its last frame — which would belong to the
previous pose.

## Related

- `experiments/filter_notebook/` — where the 8.6 cm lean and the pixel-space decomposition of
  the box-bottom reading were measured.
- `docs/METHOD.md` — the pipeline this reading would slot into.
