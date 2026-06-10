# Decision Log

Short, dated decisions that prevent the project from re-litigating the same
scientific choices.

## 2026-06-10

- Lock the external camera at **z=4.8 (+0.3), y=−5.5 (+0.6 back)** to fix corner/FOV
  coverage and open A1; the south wall (+0.4 height) + side walls + dock bumpers
  moved south with the camera so the raised wall stays behind it (no self-occlusion).
  Full GP recapture on the locked geometry → **`aws_gp_v7`** (912 frames, 647/912
  detected = 71%, driveable-only sample filter, tuned **length_scale 0.90, noise_var
  0.05, beta 0.5**). camera_pos in the artifact = (0,−5.5,4.8); consistency OK.
- **A1 made observable by hyperparameters, not de-occlusion.** The A1-mid "blind band"
  was a GP over-smoothing artifact: the R2-occluded A1-**east** lane (raw YOLO ≈0.002)
  was bleeding into the genuinely-visible A1-**west** lane (raw YOLO 0.74–0.83) via the
  1.20 m length scale. Shortening to 0.90 m respects the occlusion boundary and lifts
  A1-west ρ_plan 0.24→0.55 while keeping the genuinely-occluded A4 rack-shadow low
  (≈0.005) — faithful to the data, not inflated. (ls=0.70 over-pulled C2 to hug the
  west boundary, seg_clear=−9 mm; 0.90 is the clean choice.)
- **F87 offline rollout (aws_gp_v7, ls=0.90): Gate PASS.** C1→NW-blind reaches
  (d=0.19); **C2→south-visible drives through A1** (d=0.35, seg_clear=+0.002 clear),
  preferring the visible route over NW-blind (J 5299 < 5310). C2 went from stuck at
  d≈4.25 (old camera) to reaching via A1 — the route-split now holds with C2 actually
  traversing A1. Figures regenerated from v7: `problem_setup_camera.pdf`,
  `gp_pipeline_aws_v7.pdf`, `driveable_region_alignment.{png,pdf}`,
  `F87_offline_rollout_v7.png`.

- Paper camera-view cleanup of `warehouse_aws.world.sdf`: replace the hand-authored
  `known_driveable_green_boundary` (62 green segments, not matched to the planner)
  with a generated `known_driveable_boundary` model — **blue** outer driveable
  boundary (prism-union bbox x[-5.65,5.15] y[-3.45,4.8]) + **green** no-go region.
  The green no-go is the *complement of the driveable corridors*, not the tight
  obstacle outline: each inter-aisle column spans corridor-to-corridor (so green
  touches the driveable aisles), running the full rack band top-to-bottom but
  **split at the physical rack mid-gap** so the open R2-R5 connectors stay
  driveable (R1, whose mid-gap is filled, is one solid column). Emitted from
  `driveable_geometry_json` (column x-extents) + the world `<collision>` rack
  geometry (mid-gap split) by
  `scripts/paper_figures/generate_driveable_overlay_sdf.py` (no hand-tuned coords).
  Removed visual-only clutter: `mission_floor_markers` (disks),
  `low_floor_stock_and_apron_context` (red apron spots, black label stripes, label
  panels, charging pad, cone, low crates), `aws_staging_mesh_visuals`, the 5
  `box_spot_R*` red disks, and the `aws_pallet_jack_receiving` include. Collision
  link set is **byte-identical** (23 links) → v6 GP `geometry_json` (collision-based)
  stays valid; only the render changes, so the pending route-split GP re-recapture
  uses this clean world as its base. Alignment verified by
  `scripts/paper_figures/make_driveable_region_alignment.py` →
  `logs/paper_figures/driveable_region_alignment.{png,pdf}`. Colour semantics
  locked: blue = outer driveable boundary, green = internal no-go.
- Make the figure match the PLANNER ground truth on the rack mid-gaps: added 4
  mid-gap connector prisms (`connector_A1_A2_R2gap`, `..._A2_A3_R3gap`,
  `..._A3_A4_R4gap`, `..._A4_east_R5gap`; x = inter-aisle column, y[1.25,2.2]) to
  `driveable_geometry_json` in `aws_f86a_camera_xy_config.yaml`. R1 has NO connector
  (its mid-gap is physically filled), so A0→A1 still requires going around R1. The
  planner now allows the R2-R5 mid-gaps (driveable), matching the physical world
  and the figure. Gate re-checked (`make_f86_heading_compare.py`): route choice
  UNCHANGED vs pre-connector v6 — C1→NW (d=0.13, reaches), C2 both seeds safe-stop
  at d=4.34 (identical to before; the pre-existing camera-poor-goal finding, not a
  regression — connectors are east of the A0→A1 corridor). The SDF green no-go
  (split columns, open mid-gaps) already equals the planner no-go (complement of
  aisles+connectors), so no camera re-render was needed.

- Replace the no-go `log_barrier` penalty with a hinged-log `warning_band`
  penalty (`nogo_cost.py`). Root cause of the "nogo_weight>200 collapses the C2
  route split" failure: `log_barrier` penalizes every valid interior state, so it
  always prefers the wider aisle; raising the weight amplified this width bias
  until both C2 seeds fell into the south basin. The warning-band penalty is
  exactly zero for valid interior states (clearance ≥ b=0.05), a soft log warning
  inside the band, and a strong quadratic violation term — so `nogo_weight` (now
  2000) crushes real violations without biasing valid-route choice. Offline gate
  PASS at weight=2000 with the nogo term negligible; route choice driven by
  risk/ambiguity. The earlier `project_to_driveable` post-solve waypoint clip is
  reverted/removed. See `docs/F86_method_and_runtime_contract.md` §5.
- F86a v4 world geometry: replace the ugly 2.6 m `rack_R4_highstack_occluder`
  monolith with a sensible tapered occluding crate stack (~1.9 m) on the
  box_spot_R4 pad (still occludes the A4 lane by height); fill the R1 left-shelf
  mid gap (`rack_R1_mid` + rails + shelf boxes + DAE mesh) so the left shelf is
  one continuous barrier — a physical reason the robot cannot cut west-service →
  A1 mid-shelf. Both edits stay in non-driveable bands (no driveable prism
  touched; `driveable_geometry_json` unchanged and re-verified). World geometry
  edit invalidates aws_gp_v5 → GP recaptured as aws_gp_v6.

## 2026-05-20

- `warehouse_aws.world.sdf` is now the paper benchmark. `warehouse_occ_light` was the original candidate but superseded before seeded Gazebo validation. It is the simplest
  validated setting for showing state-dependent observation uncertainty.
- Keep `warehouse_aws.world.sdf` exploratory. It requires final geometry,
  detector retraining/validation, visibility capture, GP fitting, smoke tests,
  seeded logs, and figures before it can support a claim.
- Remove mission waypoint support. Route choice must emerge from the planner
  objective, not from a mission script that changes the goal sequence.
- Reject the AWS visible-goal route-choice probe as paper evidence. The baseline
  already took the detour-like route, while the learned condition stalled when
  ambiguity was weighted aggressively.
- Reject the AWS dark-final-goal route-choice probe as paper evidence. It mixed
  route visibility with a camera-poor final goal, making the result hard to
  interpret.
- Treat sparse planning as future work. A scientifically fair version may score
  coarse route candidates with the same objective terms, but it should not
  inject route-forcing waypoints into the local controller.

## 2026-05-27

- Move AI/research authority to `/home/joostleliveld/Thesis/CLAUDE.md`. The
  `UnembodiedNavigation` and `thesis-report` guidance files are supplements only.
- Keep broad Claude permissions for speed, but encode stronger behavioral rules:
  no destructive cleanup without explicit delete lists, no YOLO/GP recapture
  before accepted geometry, no Gazebo campaigns before offline sanity checks, and
  no paper claim without the full artifact chain.
- Retire repo-local agent prompts in favor of root agents:
  `experiment-designer`, `rollout-runner`, `planner-diagnostician`,
  `figure-analyst`, and `paper-rigor-writer`.
- Preserve multistart. It is allowed as condition-neutral optimizer basin
  handling and must be reported. It is not a mission waypoint mechanism.
- Treat long-horizon/multistart timing results as a useful diagnostic:
  they can show that the visibility-aware solution exists in the objective, but
  current solve times are a scalability limitation.
- Prefer general planner mechanisms such as goal-prior scheduling/annealing and
  normalized costs over simply increasing ambiguity weight.

## 2026-05-28

- Lock the runtime method contract in `docs/runtime_method_contract.md`.
- Define C1 as constant-observability EFE with both risk and ambiguity active.
  C1 differs from C2 by not querying the GP and by using spatially constant
  observation covariance, not by removing ambiguity.
- Define C2 as learned-observability EFE with both risk and ambiguity active.
  The GP affects planner-facing camera `(x, y)` covariance only.
- Use condition-neutral multistart as optimizer basin handling. Candidate
  generation may use the known 2D driveable floor and local maneuvers, but not
  learned visibility or the condition label.
- Use a shared 2-sigma belief-tube driveable-region log barrier for AWS
  diagnostics. Non-driveable floor is a forbidden-zone/traversability layer, not
  an observation-reliability tradeoff.
- Lock AWS Gazebo diagnostics to a robotics-faithful hierarchical runtime:
  longer global EFE route solve, short local tracker, command/encoder noise on,
  and crash/contact as terminal tracked failures.
- Permit modest lane-graph optimizer seeds generated from the known 2D
  traversability layer. This addresses supervisor feedback about local optima
  without scripting the desired visibility-aware route. The seeds must be shared
  by C1/C2 and must not use GP visibility.
- Treat any older diagnostic in which `constant_R_efe` has `ambiguity_cost=0`
  as stale for C1/C2 interpretation.

## Stable Wording Decisions

- Use `known driveable / forbidden-zone layer` for 2D planner constraints.
- Use `learned observation reliability` for the GP-derived reliability map.
- Use `3D occlusion affecting camera observations` for shelves, boxes, distance,
  perspective, and calibration effects.
- State that the GP affects camera `(x, y)` observation covariance only; heading
  is odometry-backed in the paper-facing runs.
