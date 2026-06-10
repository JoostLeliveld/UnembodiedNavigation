# Documentation Index

This folder is the maintained bridge between the codebase and the paper. Root
`/home/joostleliveld/Thesis/CLAUDE.md` is the authoritative AI operating
contract; these docs are repo-specific references.

| File | Role |
| --- | --- |
| `active_research_state.md` | current truth, active hypothesis, OPEN blockers, and next decision |
| `decision_log.md` | short dated decisions, including rejected AWS probes |
| `experiment_registry.md` | current artifacts, evidence status, superseded/archived, and invalid lines |
| `paper_runtime_contract.yaml` | **the single runtime contract** (machine-readable) for paper-facing runs |
| `paper_alignment.md` | code-to-paper contract, assumptions, and paper wording boundaries |
| `runtime_dataflow.md` | offline artifact flow and online ROS topic flow |
| `CONSISTENCY_CHECKLIST.md` | world, detector, GP, costmap, and task consistency gates |
| `PLANNER_HYPERPARAMETERS.md` | planner knobs, intended effects, and tuning cautions |
| `perception_details.md` | YOLO detector architecture, dataset, and inference settings |

(The former three runtime contracts were consolidated into
`paper_runtime_contract.yaml`; stale v5/compact-benchmark docs were removed in the
2026-06-10 cleanup.)

For the current state, start with `active_research_state.md`. For code entry
points, start with the repository `README.md`.

For AI/agent workflow, start with the root files:

- `/home/joostleliveld/Thesis/CLAUDE.md`
- `/home/joostleliveld/Thesis/.claude/README.md`
