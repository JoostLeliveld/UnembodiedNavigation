# `state`

This package converts pixel-space observations into a planner-facing planar state estimate.

## Why This Folder Exists

The planner does not consume raw image coordinates. This package performs the homography-based conversion from image-space observation to BEV state.

## Inputs And Outputs

- **Inputs**
  - `/perception/pixel_pose`
  - `/perception/detection_diagnostics`
  - `/odom`
- **Outputs**
  - `/state/bev`

## Central Files

| File | Role |
| --- | --- |
| [`state/nodes/pixel_to_bev_state_node.py`](state/nodes/pixel_to_bev_state_node.py) | runtime state-estimation node |
| [`state/core/pixel_to_bev.py`](state/core/pixel_to_bev.py) | geometric conversion helper |
| [`state/core/noise.py`](state/core/noise.py) | covariance construction helper |

## What To Read First

1. `state/nodes/pixel_to_bev_state_node.py`
2. `state/core/pixel_to_bev.py`
3. `state/core/noise.py`

## Implemented Now

- camera-based `x,y` projection into BEV
- odometry heading fallback
- optional motion-based heading fallback
- covariance publication with the state estimate

## Important Caveat

The current main estimator is hybrid:

- `x,y` from the external camera
- `theta` from odometry fallback in the primary image path

That limitation should be stated explicitly in any presentation or write-up.
