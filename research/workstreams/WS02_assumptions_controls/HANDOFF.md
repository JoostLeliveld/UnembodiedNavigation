# WS02 — assumptions, controls, worlds and noise

## Objective

Make every assumption testable or explicitly scoped, and freeze the controls for the current
paper separately from the later reliability-source benchmark. Define world, camera, route,
geometry provenance and sensitivity levels without implementing experiments.

## Ownership

Writable:

- `research/03_assumptions.md`
- `research/06_world_camera_design.md`

Read-only:

- `research/registry.yaml`, `research/09_decisions_and_risks.md`
- `docs/warehouse_full_4cam_layout.md`
- `src/experiments/config/world_profiles.yaml`
- `src/experiments/config/tasks.yaml`
- `scripts/visibility_comparison/warehouse_full_4cam_missions.yaml`
- `experiments/multicamera_commissioning_bigwarehouse/config/study.yaml`
- calibration, detector, GP and world manifests

Do not edit runtime code, configs, artifacts, experiments, registry or status.

## Requirements

- Every assumption must state why needed, plausibility, sensitivity/justification,
  consequence if violated, evidence and `ACCEPTED`/`TESTED`/`DEFERRED` proposal.
- Paper A and benchmark B receive separate frozen-control tables.
- A nominal comparison changes one factor. Sensitivities are one-factor-at-a-time, not a
  Cartesian product.
- Detector weights/threshold, camera configuration, target, planner/controller, splits and
  evaluation firewall are fixed within comparisons.
- “Depth” must distinguish complete SDF/CAD upper bound, commissioning scan, maintained
  static map, stale map, rescanned map and live sensed depth.
- World descriptions use measurable properties: occlusion density, aisle openness, overlap,
  asymmetry, range, stability, camera-poor regions and route alternatives.
- Camera descriptions use height/pitch, FOV/focal length, resolution, rate, overlap,
  occlusion exposure, calibration bias and residual correlation—not unsupported archetypes.

## Required route and split design

- Short poor-observation route versus modest visible detour.
- Equal-length routes differing in occlusion.
- Overlap/handover route.
- Unavoidable camera-poor negative control.
- Changed-layout route through the changed region.
- Uniformly good-quality route where no method should alter behaviour.
- Grouped leave-one-route-out, spatial block, changed-layout and optional second-world
  holdouts. Camera-mount holdout supports mount-role transfer only.

## Candidate sensitivity ladder

Keep nominal evidence free of extra synthetic noise. Record the following as candidate OFAT
levels to be reviewed and frozen after a non-degenerate pilot:

| Factor | Candidate levels |
|---|---|
| Pixel output jitter | 0, 0.5, 1.0, 2.0 px SD |
| Calibration yaw | 0, 0.1, 0.25, 0.5 degrees |
| Calibration translation | 0, 0.025, 0.05, 0.10 m |
| Latency | 0, 50, 100, 200 ms |
| IID dropout | 0, 5, 15, 30 percent |
| Burst dropout | 0.5, 1.0, 2.0 s |
| Missing depth | 0, 10, 30 percent cells |
| Layout | nominal, local change, global change |
| Commissioning budget | 0, 50, 100, 250, 500, 1000 unique sites/camera |

Do not silently declare these final: flag levels needing a pilot or supervisor decision.

## Deliverables

1. Completed assumption register covering A01-A16 and any proposed additions.
2. Exact frozen-control table for the current closed-loop study.
3. Exact frozen-control and feature-legality table for the source benchmark.
4. World/camera property schema and route archetype catalogue.
5. Primary-versus-sensitivity noise contract.
6. Supported/unsupported generalization table and unresolved decisions.

## Stop conditions

Hand back rather than choose silently if the primary depth provenance, heading dependence of
`p_use`, layout-change definition, second-world requirement, or DL legal representation
cannot be inferred from existing decisions.

## Paste-ready prompt

```text
Audit and refine assumptions and experimental controls in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Read research/03_assumptions.md, research/06_world_camera_design.md,
research/registry.yaml, research/09_decisions_and_risks.md,
docs/warehouse_full_4cam_layout.md, src/experiments/config/world_profiles.yaml,
src/experiments/config/tasks.yaml,
scripts/visibility_comparison/warehouse_full_4cam_missions.yaml, and
experiments/multicamera_commissioning_bigwarehouse/config/study.yaml.

You may edit only:
- research/03_assumptions.md
- research/06_world_camera_design.md

Do not edit the registry/status, code, experiment configs or artifacts. Do not launch
experiments.

Create separate frozen-control contracts for (A) the current correlated-error closed-loop
paper and (B) the later constant/distance/FOV/depth/GP/hybrid/DL benchmark. For each
assumption record need, plausibility, sensitivity/justification, consequence, evidence and
state. Define depth provenance precisely; distinguish complete-map upper bound, scan,
maintained/stale/rescanned map and live sensed depth. Define measurable world/camera
properties, grouped splits, route archetypes and OFAT sensitivity levels. State exactly what
can and cannot be claimed from four optically identical cameras. Return unresolved decisions
instead of inventing consequential definitions.
```
