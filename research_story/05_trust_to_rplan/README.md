# 05 — One frozen interface: trust → R_plan

**Question.** Can we freeze a single bounded, monotone mapping from learned trust to the
planner's observation covariance — so that every navigation result in ch.06/08/09 is
attributable to the *map*, never to interface drift?

**Status: PARTIAL — freeze currently BLOCKED.** World: original warehouse; reused unchanged
at scale. This is an interface, not a contribution; its "result" is discipline.

## What "done" looks like

```text
1 / R_plan = trust / R_visible + (1 - trust) / R_miss        (precision blend)
R_plan(x,y) = diag(σ_plan², σ_plan²)  px²                     (explicit shape + units)
```

- exactly ONE implementation, delegated to everywhere it's used;
- documented lower/upper bounds, monotonicity, and GP-unsupported-region behaviour;
- semantic separation kept explicit: detector reliability ≠ conditional noise ≠ GP epistemic
  uncertainty ≠ R_plan (Fig 05A).

## The results we're aiming for

- **Fig 05B** — the mapping curve τ ↦ σ²_plan with bounds and the unsupported-region rule.
- **Fig 05C** — triptych on one floor plan: learned trust / GP uncertainty / effective
  R_plan. The slide that proves the pipeline isn't a black box.
- **Fig 05D** — sensitivity: route choice and predicted belief growth vs mapping slope and
  endpoints. Aim: conclusions in ch.06 are robust across a reasonable parameter band.

## Implemented now

| Item | Tag | Note |
|---|---|---|
| Runtime precision blend in the planner (`unicycle_planner_node.py`) | established | `r_miss_uv = 120 px` |
| `reliability/single_camera_adapter.py` blend | established | the most tested copy |
| Offline copy in `geometry_visibility.py` | established | **divergent: `r_miss_uv = 40 px`** |
| exp2 operational mapping study (4 figs) | measured_in_sim | commissioning-data → map → held-out metrics |

## Gap → freeze checklist (the gate)

1. Reconcile `r_miss_uv` 40 px vs 120 px — **no R_plan numbers may be quoted before this**
   (CLAUDE.md known mismatch).
2. Make the offline copy delegate to the tested adapter (audit item D5).
3. Fix the dangling `world_profiles.yaml` `visibility_artifact` default — deferred while that
   file is multicam-branch WIP.
4. Write down bounds + unsupported-region treatment; then produce 05B–05D and stamp the
   interface FROZEN in this README and `registry.yaml`.

No standalone video — R_plan is displayed live in V06.
