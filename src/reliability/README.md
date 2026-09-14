# Reliability and commissioned sensing

This package contains the paper-facing camera-network measurement path.

The active contract is:

1. a frozen YOLO box is admitted by a deterministic, belief-independent sensor gate;
2. the raw observation is the box bottom centre projected to the ground plane;
3. the selected correction is either the raw-box baseline, a box-feature MLP residual, or
   the same MLP with a gated image residual;
4. each correction uses its own covariance model fitted from whole-drive out-of-fold
   residuals;
5. availability is the position-only field `q_i(p)`, with headings pooled at each position;
6. a single robot filter consumes each camera frame once.

Visual-hull observations, hull-derived correction inputs, belief-dependent admission,
heading-conditioned availability, and process-noise selection are outside the thesis
method. See `docs/COMMISSIONED_SENSOR_MODEL_CONTRACT.md` for the normative definitions.

Current runtime entry points are `reliability/nodes/camera_manager_node.py`,
`reliability/commissioned_availability.py`, and `reliability/commissioned_visibility.py`.
