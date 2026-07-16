# 04 — Per-camera GP learning

## Show

`figures/per_camera_gp_learning_protocol.png`

## Say

“The result is not one blended warehouse map. It is four independently fitted
GPs. Each starts from its own calibration prior, is updated only by its own
uncertainty-stamped observations, and reports both a posterior mean and a
posterior uncertainty. A local failure of camera A is not silently copied to B,
C, or D.”

## Executed pilot evidence

- Four canonical expected-kernel GPs were fitted from the 60–62 aligned
  camera-specific operational events gathered in the two pilot routes.
- The prior/posterior/uncertainty small multiples are in
  [`../07_real_commissioning_execution/figures/02_actual_per_camera_gp_updates.png`](../07_real_commissioning_execution/figures/02_actual_per_camera_gp_updates.png).
- These are pilot fits, without route-disjoint held-out scoring. They show that
  the update machinery works; they do not establish generalisation yet.

## Transition

“Only after the individual maps are earned can overlap become a controlled test
surface for source agreement and handover.”
