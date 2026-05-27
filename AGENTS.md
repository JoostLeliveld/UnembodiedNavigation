# AGENTS.md

Shared guidance for coding agents in this repository. This mirrors
`CLAUDE.md` so different assistants do not drift into different claims.

## Research Boundary

- The compact `warehouse_occ_light.world.sdf` benchmark is the paper-facing
  evidence core.
- `warehouse_aws.world.sdf` is exploratory unless the final world geometry,
  AWS detector, AWS GP capture, smoke runs, seeded campaign logs, and figures
  are all complete and registered.
- Do not edit TeX files unless explicitly requested.
- Do not implement mission waypoints or task scripts that force the desired
  route. Route choice must come from the planner objective.
- Keep sparse planning as future work unless the user asks for a separate
  algorithm design. A fair sparse planner would score candidate routes; it must
  not inject mission waypoints into the local controller.

## Scientific Vocabulary

- Use `known driveable / forbidden-zone layer` for 2D planner constraints.
- Use `learned observation reliability` for the GP-derived detector reliability.
- Use `3D occlusion affecting camera observations` for visual effects from
  shelves, boxes, distance, perspective, or calibration.
- Use `planner-facing covariance` for the covariance used by EFE.
- The GP affects camera `(x, y)` observation covariance only; heading comes from
  odometry in the current paper-facing runs.

## Result Claims

Every claimed result needs the complete chain: world, detector, GP, config, logs,
metrics, and figures. If any link is missing, mark the result as diagnostic,
exploratory, or future work.
