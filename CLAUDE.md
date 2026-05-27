# CLAUDE.md

This repository supports a thesis/paper on visibility-aware planning for an
external-camera navigation setup. Treat this file as the shared operating
contract for AI assistants working in this repo.

## Non-Negotiables

- Do not edit TeX files unless the user explicitly asks for TeX edits.
- The compact `warehouse_occ_light.world.sdf` benchmark is the current paper
  core.
- `warehouse_aws.world.sdf` is an exploratory Experiment B world until it has a
  complete detector, GP, smoke-run, seeded-log, and figure chain.
- Do not add or use mission waypoints to force route choice. A route difference
  must emerge from the planner objective.
- Every claimed result needs a named world, detector artifact, GP artifact,
  config, run logs, and generated figures.
- Keep 2D traversability separate from 3D visibility. Known driveable /
  forbidden-zone layers are planner constraints. Learned observation reliability
  affects camera `(x, y)` covariance only, not heading directly.
- Sparse planning is future work only in the current cleanup pass. It may be
  discussed as fair route-candidate scoring, not as mission waypoint scripting.

## Maintained Context

Read these files before changing experiment logic or paper-facing docs:

- `docs/active_research_state.md`
- `docs/decision_log.md`
- `docs/experiment_registry.md`
- `docs/paper_alignment.md`
- `docs/runtime_dataflow.md`
- `docs/paper_runtime_contract.yaml`
- `docs/publication_checklist.md`

## Evidence Standard

Paper evidence must pass the full artifact chain:

1. world geometry and camera pose fixed;
2. detector trained or selected for that world;
3. visibility samples captured in that same world;
4. GP fit from those samples with `P_conservative_plan_map`;
5. campaign config records the exact detector and GP artifacts;
6. seeded logs complete without hidden fallbacks;
7. figures/metrics are generated from those logs.

AWS Experiment B should be described as a planned or exploratory extension until
that chain exists.

---

## Thesis claim (short form — long form in memory)

**Stability and safety, not goal-reaching speed.**

- On easy tasks (a visible alternative to the goal exists), C1 and C2 both reach the
  goal; C2 may take a slightly safer trajectory but no real penalty.
- On hard tasks (no visible path to the goal), C2 **stops safely** at the shadow
  boundary while C1 risks a blind traversal and may crash. C2 not reaching the goal in
  the hardest case is a **success**, not a failure.

Long form: `~/.claude/projects/-home-joostleliveld-Thesis/memory/project_thesis_stability_claim.md`.

## Ideal demonstration behaviours

- **Lateral preference**: in a uniform aisle whose one side is more visible to the
  external camera, C2 hugs the visible side; C1 drifts to centerline and may lose track.
- **Late commit**: when the goal is in shadow, C2 stays in the visible region as long
  as possible and only enters the dark zone for the last 1–2 m of approach.
- **Safe stop**: when no visible path to the goal exists at all, C2 refuses to enter
  shadow and stops safely. No collision is the success criterion.

## Where context lives

**Auto-memory** (`~/.claude/projects/-home-joostleliveld-Thesis/memory/`):
- `MEMORY.md` — index.
- `project_thesis_stability_claim.md` — the reframed stability claim (above).
- `project_world_design_intent.md` — shelf-end pads, blockage from height not mid-aisle.
- `project_shadow_behaviour_design.md` — what shadow exposure is acceptable.
- `project_iwai_experiment.md` — broader campaign protocol.
- `project_supervisor_feedback_may2026.md` — last supervisor meeting action items.

**Repo reference docs** (`docs/`):
- `docs/PLANNER_HYPERPARAMETERS.md` — every knob, what it does, recipes for eliciting
  specific behaviours.
- `docs/CONSISTENCY_CHECKLIST.md` — the World ↔ YOLO ↔ GP ↔ Costmap chain. Run it before
  any new campaign.
- `docs/AGENT_BOUNDARIES.md` — what each subagent may read / propose / never touch.

## Subagents (in `.claude/agents/`)

Five specialised agents form the iteration loop. Three are loop agents
(propose → run → diagnose); two are guardrails (consistency + writeup):

| agent | role | use when |
|---|---|---|
| `scenario-designer` | proposes the next experiment — new task OR hyperparameter change | "what should we try next?" / "design a task that shows X" |
| `rollout-runner` | executes the proposed change (offline-first; Gazebo only on user-approval) and produces canonical artifacts | after scenario-designer; before plot-analyst |
| `plot-analyst` | plots the rollout result, identifies what is or isn't emerging, recommends the next intervention | after rollout-runner; or when looking at any existing run |
| `consistency-guardian` | runs the World↔GP↔YOLO↔Costmap checklist | before any rollout; after any SDF / world_profiles / tasks.yaml / GP edit |
| `rigor-writer` | writes up converged results in thesis-quality prose; audits claims | when an iteration is paper-ready, or before sharing a writeup |

Typical loop: `scenario-designer → consistency-guardian → rollout-runner → plot-analyst →
(loop back to scenario-designer with the recommendation) → rigor-writer`.

Invoke via the Agent tool with `subagent_type=<name>`.

## Hard rules (applies to humans + agents + main session)

- **Clean EFE invariant**: no new cost terms in
  `src/planning/planning/core/casadi_efe.py`. The objective stays
  `risk + ambiguity + control + nogo + terminal_progress_penalty`. Tune existing weights;
  do not add accumulators.
- **GP / YOLO are not auto-touched**. World geometry edits invalidate the GP. Decisions
  to recapture or retrain go through the user; the `experimentalist` agent (Stage B) may
  invoke the capture pipeline only on user approval.
- **Memory files are append-mostly**, owned by the main session. Agents propose updates;
  the main session writes them.

## Recurring-pain reminders

The user has re-explained each of these multiple times. Read the linked memory first
before responding to:

1. "C2 should stop safely, not chase the goal" → `project_thesis_stability_claim.md`.
2. "Boxes at shelf-end pads, blockage from height" → `project_world_design_intent.md`.
3. "Don't add cost terms — tune existing ones" → `docs/PLANNER_HYPERPARAMETERS.md`.
4. "Did the GP get recaptured after the world change?" → `docs/CONSISTENCY_CHECKLIST.md`.

## Typical agent composition (worked example)

User: *"design a task in A1 that shows lateral preference"*

1. `scenario-designer` proposes the task with sampled-GP evidence and a paste-ready
   `tasks.yaml` entry; predicts what C1 vs C2 should do differently.
2. `consistency-guardian` validates the proposal (start/goal in green region, GP support
   present, no stale GP / SDF drift).
3. `rollout-runner` executes the task offline against the current planner; emits
   canonical artifacts (trajectory CSVs + plots) into a rollout output directory.
4. `plot-analyst` inspects the artifacts, identifies whether lateral preference is
   actually emerging, hypothesises the cause if not, and recommends the next intervention
   (back to `scenario-designer` with a specific tuning, OR forward to `rigor-writer` if
   the result is already publication-ready).
5. (loop) `scenario-designer` applies the recommendation; back to step 2.
6. When converged, `rigor-writer` writes the thesis-quality summary into the
   dedicated rollout writeup or a maintained paper-results directory registered
   in `docs/experiment_registry.md`.
