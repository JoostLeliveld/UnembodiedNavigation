# `state`

This package converts pixel-space observations into a BEV state estimate.

The planner-facing observability model uses planar position:

\[
\hat s_t = [\hat x_t,\hat y_t]^\top .
\]

## Inputs And Outputs

- Inputs: `/perception/pixel_pose`, `/perception/detection_diagnostics`, `/odom`
- Outputs: `/state/bev`, `/state/heading_diagnostics`

## Central Files

| File | Role |
| --- | --- |
| [`state/nodes/pixel_to_bev_state_node.py`](state/nodes/pixel_to_bev_state_node.py) | runtime pixel-to-BEV conversion and heading fallback publication |
| [`state/core/pixel_to_bev.py`](state/core/pixel_to_bev.py) | camera geometry conversion helper |
| [`state/core/noise.py`](state/core/noise.py) | covariance construction helper |

## Heading Caveat

The stable method claim is camera-derived `x,y`. Heading is a runtime setting and must be reported from the manifest:

- segmentation/detection model with odometry fallback: `theta` from odometry fallback
- diagnostic displacement-heading mode: `theta` from consecutive camera-derived position updates in the planner correction path
- pose model with `keypoint_marker_world_z > 0`: `theta` from front/rear keypoints back-projected to BEV, with odometry fallback

The current paper-facing campaign configs and Task A figure manifests use odometry-backed heading.
