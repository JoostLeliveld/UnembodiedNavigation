# WS04 — pixel-to-ground measurement path

## Objective

Review and finish the isolated pixel-to-ground work without letting it silently alter the
current campaign. Decide whether the box-centre projection is (a) required shared
infrastructure for the future source benchmark, (b) supporting analysis only, or (c) not yet
promotable. Preserve the existing work and make its provenance/assumptions internally
consistent.

## Existing uncommitted work

These paths are one coherent, currently dirty change set and belong exclusively to WS04:

- `src/perception/perception/core/yolo_selection.py`
- `src/reliability/reliability/projection.py`
- `tests/experiments/test_pixel_ground_box_projection.py`

The changes add an opt-in `bbox_center` pixel, a derived plane at `z=0.085 m`, propagated
pixel covariance and a yaw-marginal covariance. Historical `bbox_bottom` remains default,
so the active campaign should not change unless another config explicitly wires it.

## Ownership

Writable:

- the three dirty paths above
- `experiments/pixel_ground_path/`

Read-only:

- `logs/studies/pixel_ground_path/`
- calibration/world/detector artifacts and cold-storage manifests
- current closed-loop campaign and all launch/config files
- `research/registry.yaml`, status and numbered research documents

Do not wire this path into runtime launch/config, touch the current paper campaign, refit
calibration, edit the registry, or claim closed-loop benefit.

## Requirements and assumptions

- Estimand: floor projection of `base_footprint`, one known robot, one class.
- The pixel statistic and inversion plane are one coupled design choice.
- Exact camera calibration, locally planar floor, fixed rendering/box convention and known
  robot visual geometry must be explicit.
- Separate CAD-derived yaw marginalization from detector/commissioning-derived pixel noise.
- Do not say a term is “not fitted to data” if its scale uses truth-backed commissioning
  poses. Name each source and its operational cost.
- Constants derived for the current 6.10 m/0.92 rad camera mounts cannot be presented as
  camera-independent.
- Current evidence is clear/unoccluded, one robot, four discrete yaws, simulation only and
  open loop.
- A size-consistency validity gate is designed but unvalidated; do not imply it exists in
  runtime.
- Runtime default must remain bit-compatible with the historical bottom-centre path.

## Known questions to resolve

1. Does the projection covariance implementation match the evidence equation and frame
   convention for all camera bearings?
2. Are invalid boxes, nonfinite/negative noise, degenerate camera bearings and invalid plane
   intersections fail-closed?
3. Does `select_best_detection` expose enough box data for `project_box_to_world`, or is the
   current API only an offline helper?
4. Are `BOX_STATISTIC_SIGMA_UV_PX` and `BOX_STATISTIC_SIGMA_YAW_M` correctly described as
   design-time, detector-validation or truth-backed commissioning quantities?
5. Does adopting this path change the meaning of calibration v4 and therefore require a
   separate closed-loop arm rather than a hidden implementation swap?

## Deliverables

1. A short decision: infrastructure, supporting analysis, or blocked—with reasons.
2. Consistent provenance and assumption wording in code and experiment README.
3. Focused numerical/edge-case tests for statistic, projection and covariance.
4. Reproduction check against e3/e4/e5 summaries without altering raw evidence.
5. Exact downstream integration proposal, but no runtime wiring.
6. Handback listing tests and proposed registry status/next action.

## Acceptance criteria

- Existing bottom-centre callers behave identically.
- Focused tests and relevant perception/reliability tests pass.
- No unsupported claim of GT-free covariance commissioning remains.
- Constants, coordinate frames, mounting dependence and failure modes are documented.
- No launch, planner, detector node or campaign config changes occur.
- The downstream owner can tell precisely what must change to opt in.

## Paste-ready prompt

```text
Work only on the isolated pixel-to-ground workstream in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Preserve the existing uncommitted changes in:
- src/perception/perception/core/yolo_selection.py
- src/reliability/reliability/projection.py
- tests/experiments/test_pixel_ground_box_projection.py

You may edit only those paths and experiments/pixel_ground_path/. Read the corresponding
logs/studies/pixel_ground_path evidence read-only. Do not edit registry/status, launch or
campaign configs, planner/camera-manager behavior, calibration artifacts, or current paper
files. Do not run Gazebo or wire this method into runtime.

Audit whether the opt-in bbox-center + z=0.085 m path and covariance are scientifically and
numerically coherent. Preserve bbox_bottom as the exact default. Check invalid/nonfinite
inputs, box geometry, Jacobian/frame rotation, camera-bearing degeneracy, plane validity and
mount dependence. Reconcile provenance wording: the README admits Sigma_uv sizing needs
truth-backed commissioning poses, so code must not imply every constant is data-free.
Current evidence is one robot, simulation, clear detections, four discrete yaws and open
loop; the size gate is not validated.

Return a decision whether this is required infrastructure for the later source benchmark,
supporting analysis only, or blocked. Add focused tests and documentation corrections as
needed, run only relevant unit/reproduction checks, and provide an integration proposal
without changing runtime behavior. Do not commit; the integration chat owns commits and
registry changes.
```
