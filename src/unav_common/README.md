# `unav_common`

This package contains shared geometry and manifest utilities used by multiple active packages.

## Why This Folder Exists

Several packages need the same camera and world-geometry logic. This folder prevents those small but important utilities from being reimplemented in multiple places.

## Main Files

| File | Role |
| --- | --- |
| [`unav_common/camera_model.py`](unav_common/camera_model.py) | camera projection and inverse projection helpers |
| [`unav_common/occlusion_geometry.py`](unav_common/occlusion_geometry.py) | world-geometry parsing and occlusion support |
| [`unav_common/manifest.py`](unav_common/manifest.py) | manifest helpers used by experiment logging |
| [`unav_common/geometry.py`](unav_common/geometry.py) | general geometry helpers |

## Who Uses This Package

- `perception`
- `state`
- `planning`
- `experiments`

## What To Read First

1. `unav_common/camera_model.py`
2. `unav_common/occlusion_geometry.py`
3. `unav_common/manifest.py`

## Caveat

This is support code. It is important for understanding how camera geometry and world geometry are handled, but it is not where the thesis comparison itself is defined.
