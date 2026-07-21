# 01 — Facility and live streams

## Show

1. `figures/facility_layout.png`
2. `figures/live_four_camera_montage.png`
3. `figures/overview_live.png`

## Say

“This is one large operating field observed by four independent, wall-mounted
cameras. They do not see the same thing: racks, props, viewing angle, and
occlusion differ by source. The overhead view is included only to explain the
world to the audience; it never enters localization, GP learning, selection, or
fusion.”

## Evidence boundary

- These are live Gazebo/runtime and layout assets.
- They establish that four camera topics and the presentation overview work.
- They do **not** establish detector reliability or localization improvement.

## Transition

“Because the streams are distinct, the system begins with four distinct initial
expectations rather than pretending one camera's reliability applies everywhere.”
