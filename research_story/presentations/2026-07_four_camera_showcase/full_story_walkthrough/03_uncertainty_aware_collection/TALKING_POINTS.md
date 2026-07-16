# 03 — Uncertainty-aware collection

## Show

`figures/collection_record_protocol.png`

## Say

“We do not turn a route into a map by simply dropping detections onto fixed
coordinates. Every record retains its camera identity, timestamp, detector
quality, estimated robot pose, and pose covariance. A miss is evidence too.
Ground truth is kept outside this operational stream for later scoring.”

## Executed pilot evidence

- Two routes were executed in the new world: the south-to-north handover and
  a dedicated A/C overlap pass.
- The actual route, detector, miss, and noisy-odometry figure is now in
  [`../07_real_commissioning_execution/figures/01_real_routes_and_observations.png`](../07_real_commissioning_execution/figures/01_real_routes_and_observations.png).
- The present pilot uses a declared 0.10 m odometry covariance floor because
  the current encoder-noise stream carries positions but zero covariance
  entries. This is explicit in the run manifest.

## Transition

“Those uncertainty-stamped records give each camera enough evidence to update
only its own initial hypothesis.”
