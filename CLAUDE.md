# UnembodiedNavigation Supplement

The root file `/home/joostleliveld/Thesis/CLAUDE.md` is the authoritative AI
operating contract for this thesis. This file only adds repository-specific
context.

## Repository Role

This folder contains the simulation, planner, detector/GP artifacts, diagnostics,
and experiment logs for the visibility-aware external-camera navigation project.

The current paper benchmark is `warehouse_aws.world.sdf` with the B1 route-choice
task (`F31_b1_apron_a3_mid`). The `warehouse_occ_light` world was the original
benchmark candidate but was superseded before seeded Gazebo validation.
New multistart probes, long-horizon studies, and goal-prior annealing experiments
remain exploratory until the full artifact chain is complete and registered.

## Maintained Repo Docs

Read these before changing experiment logic or paper-facing docs:

- `docs/active_research_state.md`
- `docs/decision_log.md`
- `docs/experiment_registry.md`
- `docs/paper_alignment.md`
- `docs/runtime_dataflow.md`
- `docs/paper_runtime_contract.yaml`
- `docs/publication_checklist.md`
- `docs/CONSISTENCY_CHECKLIST.md`
- `docs/PLANNER_HYPERPARAMETERS.md`

## Repo-Specific Guardrails

- World geometry edits invalidate downstream visibility data and GP artifacts.
- Do not recapture YOLO/GP artifacts until world geometry and camera view are
  accepted by the user.
- Do not run Gazebo campaigns before offline rollout/objective sanity checks.
- Keep multistart if used, but report it as neutral optimizer basin handling.
- Do not reintroduce mission waypoints or route-forcing task scripts.
- Always give the path to generated figures and logs.

## Maintained Agents

Use the root agents in `/home/joostleliveld/Thesis/.claude/agents/`. The old
repo-local agent prompts were retired to avoid conflicting instructions.
