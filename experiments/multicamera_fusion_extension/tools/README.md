# Operational data builders

These builders create the training surface for the factorised reliability
models. They only accept `operational/` data. Ground truth, oracle labels, and
paths under `evaluation_only/` are rejected before a row is written.

`build_opportunity_dataset.py` combines exported operational samples with an
operational prediction sidecar. Each sidecar row is keyed by `sample_id` and
contains the camera ID, predicted image position/covariance, predicted robot
height, stream-liveness flag, and belief-to-image association delta. It emits a
row only when the expected robot support lies in the validated image region,
the predicted scale is sufficient, the stream is live, and association timing
is within the frozen bound. An out-of-support pose is omitted, not relabelled as
a detector miss.

`build_loo_labels.py` then joins an operational leave-one-camera-out reference.
The reference must name the labelled camera in `excluded_camera_id`; otherwise
the command fails. It writes the image-space residual and usability label, but
leaves a miss unlabelled for conditional usability rather than inventing a
residual.

The actual sidecar/reference producers will be enabled only after the locked
four-camera detector and projection commissioning gate passes. The schema and
firewall are implemented now so that collection has a stable target.
