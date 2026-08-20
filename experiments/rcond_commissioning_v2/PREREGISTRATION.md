# Frozen pilot/full-campaign contract

- Development world: `warehouse_aws.world.sdf`.
- Reading: `yolo_pose_aws_v4/model.pt`, two-marker keypoint position.
- Conditional set: detections with both rendered markers visible.
- Bias features fixed before the session holdout: intercept, range, sin(yaw),
  cos(yaw), normalized image horizontal coordinate.
- Spatial split: whole `(x_idx,y_idx)` anchors, including every heading/repeat.
- Session split: full Gazebo restart; final session is evaluation only.
- Independent component: covariance about each `(session,anchor,heading)` mean.
- Persistent component: covariance of those means after subtracting the fitted
  bias and finite-repeat contribution.
- Naive repeated-measurement covariance: `R_iid/n`.
- Proposed covariance: `B_persistent + R_iid/n + bias posterior covariance`.
- Primary held-out endpoints: Gaussian NLL and 50/80/95% ground-position ellipse
  coverage. Report sharpness and sample counts with every coverage number.
- A range-conditioned covariance is accepted only if it improves held-out NLL,
  does not worsen absolute 95%-coverage error, and has non-degenerate variation.
- No filter or planner run is authorized from pilot evidence.

