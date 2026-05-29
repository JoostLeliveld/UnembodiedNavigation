# Documentation Index

This folder is the maintained bridge between the codebase and the paper. Root
`/home/joostleliveld/Thesis/CLAUDE.md` is the authoritative AI operating
contract; these docs are repo-specific references.

| File | Role |
| --- | --- |
| `active_research_state.md` | current truth, active hypothesis, valid/exploratory lines, and next decision |
| `decision_log.md` | short dated decisions, including rejected AWS probes |
| `experiment_registry.md` | valid, exploratory, and invalid run families plus the required artifact chain |
| `paper_alignment.md` | code-to-paper contract, assumptions, and paper wording boundaries |
| `paper_runtime_contract.yaml` | machine-readable checklist for paper-facing runs |
| `runtime_method_contract.md` | locked C1/C2, multistart, driveability, and pre-Gazebo method contract |
| `runtime_dataflow.md` | offline artifact flow and online ROS topic flow |
| `publication_checklist.md` | release-readiness checklist and remaining decisions |
| `CONSISTENCY_CHECKLIST.md` | world, detector, GP, costmap, and task consistency gates |
| `PLANNER_HYPERPARAMETERS.md` | planner knobs, intended effects, and tuning cautions |

For the current state, start with `active_research_state.md`. For code entry
points, start with the repository `README.md`.

For AI/agent workflow, start with the root files:

- `/home/joostleliveld/Thesis/CLAUDE.md`
- `/home/joostleliveld/Thesis/.claude/README.md`
