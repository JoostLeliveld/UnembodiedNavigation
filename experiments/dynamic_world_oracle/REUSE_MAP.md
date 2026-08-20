# Reuse map — dynamic_world_oracle

What this study borrows, and the one thing it adds.

## Borrowed, not reimplemented

| need | comes from | why not local |
|---|---|---|
| camera geometry from a world SDF | `reliability.projection.camera_model_from_world` | the runtime projection path derives `look_at` from the SDF pose this exact way; a second derivation would let the oracle and the runtime disagree about where a camera points |
| CAD prisms from a world SDF | `unav_common.occlusion_geometry.parse_collision_scene_from_world` | the collision boxes *are* the occluders; parsing the SDF again risks picking up visuals or floor paint |
| obstacle parts from a model SDF | `unav_common.occlusion_geometry.parse_occlusion_scene_from_world` (collision tags) | the same parser, pointed at a spawnable model, so the oracle's obstacle geometry and the simulator's are one object — bounding obstacles by hand cost 29% false occlusions before this |
| single-ray occlusion | `unav_common.occlusion_geometry.segment_occluded` | THE occlusion predicate in this repo |
| pixel projection | `unav_common.camera_model.ObliqueCameraModel` | same model the detector's projection uses |
| repo root | `scripts/shared.paths.repo_root` | preferred over `parents[N]` for new code |
| the warehouse itself | `scripts/geometry_visibility/make_warehouse_full.py` | extended with a `--variant dynamic` flag rather than forked; the static default still writes the frozen flagship world byte-for-byte |

## Added here

`oracle.segments_hit_any_prism` — a vectorised slab test over all grid cells and
all prisms at once. It is a **speed twin of `segment_occluded`, not a fork**: a
92×69 grid against 32 prisms is 380k ray-box tests per camera per frame, which is
minutes of scalar Python and ~25 ms vectorised.

Because it duplicates a predicate that already exists, both the acceptance run
(`verify_acceptance.py`, 2400 rays) and the test suite
(`tests/experiments/test_dynamic_world_oracle.py`, 250 rays) cross-check it
against `segment_occluded` on random rays. If the two ever disagree, that is a
failure, not a tolerance.

## Deliberately not reused

`scripts/visibility_comparison/capture_visibility_samples.py` teleports a robot
and captures frames through the ROS bridge. This study does not use it: the
bridge on this machine is pinned to a different Gazebo ABI than the Python
gz-transport bindings, and a scenario needs a *stepped* server rather than a
free-running one. The relevant knowledge — set-pose service, image topics,
`ObliqueCameraModel` construction — was reused; the transport was not.
