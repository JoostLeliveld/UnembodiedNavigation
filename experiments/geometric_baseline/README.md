# geometric_baseline — C0 conventional-navigation baseline

Investigation study adding a **third experimental condition C0** to the primary
warehouse route-choice comparison. C0 is a *geometry-only shortest-path planner*
("conventional navigation"): the reviewer-requested naive baseline that ignores
external-camera reliability entirely.

## What C0 is

| | C0 (geometric_shortest_path) | C1 (constant_R_efe) | C2 (visibility_aware_efe) |
|---|---|---|---|
| Global route | **shortest** valid lane-graph route | one-shot global EFE solve | one-shot global EFE solve |
| Camera-reliability model | none | none (constant R) | GP-scaled R_plan |
| EFE / ambiguity / obs-risk reasoning | **none** (global solve skipped) | yes (constant R) | yes (GP R_plan) |
| Driveable region + no-go geometry | same as C1/C2 | same | same |
| Local tracker | same `turn_then_go` | same | same |

C0 chooses, over the **same** driveable region and no-go geometry as C1/C2, the
shortest of the condition-neutral lane-graph route seeds
(`generate_route_seeds`), measured by total polyline length, and hands that route
to the **same** geometric local tracker. It does no camera-reliability or
expected-free-energy reasoning: the one-shot global EFE optimisation is skipped
outright (`global_planner_mode='geometric_shortest_path'`). If the lane graph
yields zero valid route seeds it falls back to a straight start→goal route (with
a logged warning).

The point of the baseline: C0 is what a conventional planner does — take the
shortest feasible path and drive it. Comparing C0/C1/C2 isolates the value of
(a) EFE route reasoning at all (C0 vs C1) and (b) spatially-varying camera trust
(C1 vs C2), against a genuinely naive reference.

## Which research_story chapter it serves

**Chapter 06 — `06_original_warehouse_navigation`** (Contribution 1's closing
closed-loop evidence, world `warehouse_aws`). C0 is the reviewer-requested
conventional-navigation baseline alongside that chapter's constant-R (N1) and
reliability-aware (N2/N4) conditions; it also reinforces the chapter 00
"does spatially-varying camera trust matter?" framing by anchoring the low end
of the comparison. Chapter 00 (`honest_campaign_v1`) is LOCKED and is **not**
touched — C0 is evaluated in a **new** campaign
(`warehouse_visibility_campaign_honest_v2.yaml`), log-root
`logs/visibility_comparison/honest_campaign_v2` (append-only; passed via
`--log-root`).

## How to run (the human drives the sim — GPU/zombie management)

The campaign runner (`run_visibility_campaign.py`) has no per-run filter flags;
it runs the whole matrix built from the config. Two ways to exercise ONE C0 run:

Preview the exact per-run launch commands without starting Gazebo:

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/warehouse_visibility_campaign_honest_v2.yaml \
  --log-root logs/visibility_comparison/honest_campaign_v2 --dry-run
```

Smoke-test ONE C0 run (route_west_to_a1_upper, seed 0) via the launch file
directly — this is exactly the command the campaign runner would issue for that
run (from `--dry-run`), just with a chosen `log_dir`. Note
`planner:=geometric_shortest_path`, `terminate_on_geom_collision:=False`, and
**no** `visibility_artifact_path` (C0 is camera-model-free). The launch file
derives `global_planner_mode:=geometric_shortest_path` from the planner name.

See the parent task report / `--dry-run` output for the full argument string.

Full 60-run v2 campaign (4 tasks × 3 conditions × 5 seeds):

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/warehouse_visibility_campaign_honest_v2.yaml \
  --log-root logs/visibility_comparison/honest_campaign_v2
```

The runner also supports `--resume` (skips already-completed runs).

## Reuse — no new runtime code

C0 required **no** new planner, cost model, or tracker. See `REUSE_MAP.md`. The
implementation is a small new branch in the existing agent node plus condition
plumbing; everything load-bearing is reused unchanged.

## Outputs

Study outputs (once run) belong in
`logs/studies/geometric_baseline/<expN_name>/` with a `RESULTS.md`; the raw
campaign runs live in `logs/visibility_comparison/honest_campaign_v2/`
(gitignored, append-only).
