# Agent Boundaries

Per-agent read/edit boundaries for the five specialised subagents in `.claude/agents/`.
This is the single source of truth — each agent's system prompt references it.

## The five agents and their roles

| agent | role |
|---|---|
| `scenario-designer` | proposes the next experiment (new task or hyperparameter tuning) |
| `rollout-runner` | executes the proposed change (offline-first; Gazebo on approval) |
| `plot-analyst` | plots the rollout result, diagnoses what's working / not, recommends next step |
| `consistency-guardian` | World↔GP↔YOLO↔Costmap chain validator; runs before any rollout |
| `rigor-writer` | thesis-quality writeup + claim audit on converged results |

## Stage A (current, this session) — minimal edit footprint

Three agents (`scenario-designer`, `consistency-guardian`) are read/propose-only. Two
(`rollout-runner`, `plot-analyst`, `rigor-writer`) write to scoped output directories
only — they do NOT touch source files.

| File class | scenario-designer | rollout-runner | plot-analyst | consistency-guardian | rigor-writer |
|---|---|---|---|---|---|
| `src/planning/` (planner code) | read | read | read | read | read |
| `src/planning/planning/core/casadi_efe.py` (EFE math) | read | read | read | read | read |
| `src/sim/gazebo_worlds/worlds/*.sdf` | read | read | read | read | read |
| `src/experiments/config/world_profiles.yaml` | **propose** | read | read | **propose** | read |
| `src/experiments/config/tasks.yaml` | **propose** | read | read | **propose** | read |
| `scripts/visibility_comparison/aws_*.yaml` (campaign configs) | **propose** | read | read | **propose** | read |
| `scripts/visibility_comparison/plot_*.py` | read | read | read | read | read |
| `scripts/visibility_comparison/_rollout_driver_*.py` (temporary) | n/a | **write** | read | n/a | read |
| `scripts/visibility_comparison/_plot_*.py` (temporary) | read | read | **write** | n/a | read |
| `logs/visibility_comparison/rollouts/<new-dir>/` | n/a | **write** | **write** | n/a | **write** |
| `logs/visibility_comparison/rollouts/<new-dir>/*.md` | read | read | read | read | **edit** |
| `docs/experiment_registry.md` | read | read | read | read | **edit** |
| `~/.claude/projects/.../memory/*.md` | read | read | read | read | **propose** |
| `docs/PLANNER_HYPERPARAMETERS.md` | read | read | read | read | **propose** |
| `docs/CONSISTENCY_CHECKLIST.md` | read | read | read | **propose** | read |
| `docs/AGENT_BOUNDARIES.md` (this file) | read | read | read | read | read |
| GP `.npz` artifacts | read | read | read | **flag-only** | read |
| YOLO model directory | read | read | n/a | **flag-only** | read |

**Legend**
- `read` — may read; never write to.
- `propose` — emits a Markdown diff block; the main session applies it.
- `edit` — may use `Edit`/`Write` directly within the scoped path.
- `write` — may create a new file at the scoped path.
- `flag-only` — may report inconsistencies but cannot edit the binary artifact.

## Tool budgets (Stage A)

| agent | tools |
|---|---|
| `scenario-designer` | `Read, Grep, Glob, Bash` (Bash read-only) |
| `rollout-runner` | `Read, Grep, Glob, Bash, Write` (Bash includes `efe_offline_lab.py`; Gazebo only on user-approved invocation) |
| `plot-analyst` | `Read, Grep, Glob, Bash, Write` (Bash for plotting scripts; Write for figures + analysis.md) |
| `consistency-guardian` | `Read, Grep, Glob, Bash` (Bash read-only verification) |
| `rigor-writer` | `Read, Grep, Glob, Write, WebSearch, WebFetch` |

## Stage B (target, future session) — broader edit capability

Once the Stage A loop has been used in real iterations, the following promotions:

| agent | gains in Stage B |
|---|---|
| `scenario-designer` | `Edit` on `tasks.yaml`, `aws_*.yaml` so it can apply its own proposals after a user "go" |
| `rollout-runner` | broader `Bash` (including `run_visibility_campaign.py`) gated on per-invocation user approval; can recapture GP via the pipeline ONLY on explicit user approval |
| `plot-analyst` | nothing extra |
| `consistency-guardian` | `Edit` on `world_profiles.yaml`, `tasks.yaml`, `docs/CONSISTENCY_CHECKLIST.md` (still cannot touch SDF or GP) |
| `rigor-writer` | direct `Edit` on memory files (currently propose-only) |

A `git diff --stat` after every Stage-B agent finishes will surface any edit outside the
agent's allowed paths, and the main session reverts those changes.

## Hard rules (both stages)

These apply to every agent, every action, regardless of stage:

1. **Clean-EFE invariant.** No agent — ever — adds a new cost term to
   `src/planning/planning/core/casadi_efe.py`. Tune existing weights; do not add
   accumulators. The objective stays
   `risk + ambiguity + control + nogo`.
2. **World SDF and YOLO are the user's domain.** No agent edits the SDF or the YOLO
   model. Changes to either require user decision and downstream re-derivation
   (GP recapture, possibly YOLO retrain).
3. **GP `.npz` files are write-once.** Agents may read them; only the capture/fit
   pipeline writes them; the pipeline is invoked by the user (Stage A) or by
   `rollout-runner` with explicit user approval (Stage B).
4. **Memory files are append-mostly, owned by main session.** Agents propose memory
   updates; the main session writes them (Stage A). `rigor-writer` is the only candidate
   for direct memory editing in Stage B, and even then only after the propose-then-apply
   pattern is well-tested.
5. **No route-seeding in the live planner.** The planner should not add
   route-specific cold-start trajectories, rotated initialisations, or mission
   waypoints to elicit a result. Any proposal that changes optimizer
   initialisation or route-candidate handling must be explicitly flagged as
   future work or a diagnostic tool.

## Conflict resolution

If this doc and an agent's system prompt disagree on a boundary, this doc wins; update
the system prompt to match. If an agent attempts an out-of-scope action in Stage B, the
main session reverts that specific change and adds a note for the user.

For anything ambiguous (e.g., "is `scripts/visibility_comparison/plot_*.py` editable by
plot-analyst?"), default to **no — write a new `_plot_*.py` in the scoped temporary path
instead**, and escalate to the user if that doesn't work.
