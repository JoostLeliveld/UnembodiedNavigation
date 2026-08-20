# Availability-aware planning paper

**The question.** An external camera network delivers two different things, and the
system has been conflating them:

- **will an observation arrive here?** — usable-observation probability `p_use(x)`
- **how accurate is it, given one arrived?** — conditional covariance `R_cond(x)`

The old planning adapter folds the first into the second: low availability becomes an
inflated measurement covariance. The registered comparison also includes an explicit
hit/miss model. This study estimates availability from geometry or from the camera's
own depth and asks whether the deployed planner can act on the resulting field.

**Serves** `research/registry.yaml` experiments `EXP-AVAIL-CAL`, `EXP-AVAIL-VS-ACC`,
`EXP-AVAIL-ROUTE`, `EXP-AVAIL-CL`, and gate `GATE-ONLINE-EFE`.
**Outputs** go to `logs/studies/availability_paper/<eN>/`, never beside this code.
**Preregistration** is [`PREREGISTRATION.md`](PREREGISTRATION.md); the analysis for
each experiment is frozen in its own `experiment.yaml` before it runs.

## The ladder

| | What it asks | Claim | Needs Gazebo | State |
|---|---|---|---|---|
| **E1** [`e1_availability_calibration`](e1_availability_calibration/) | Which estimator predicts whether a detection arrives, on spatially held-out ground? | C1, C2 | no | **run** |
| **E2** [`e2_availability_vs_accuracy`](e2_availability_vs_accuracy/) | Are availability and conditional accuracy the same field? | C1 | no | **run** |
| **E3** [`e3_route_discrimination`](e3_route_discrimination/) | Does the field change the route, and does a deployable field choose as well as a surveyed one? | C3 | no | **run** |
| **E0** [`e0_online_efe_readiness`](e0_online_efe_readiness/) | Can the online planner reach a goal at all in this world? | — (gate) | yes | **passed 2026-08-18** |
| **E4** [`e4_closed_loop`](e4_closed_loop/) | Does availability-aware planning navigate better? | C4 | yes | **stopped at 12/45: identical global routes** |
| **E5** [`e3_route_discrimination/offline_efe_solve.py`](e3_route_discrimination/offline_efe_solve.py) | Why did the runtime objective ignore different fields? | diagnostic | no | **run** |
| **E6** [`e6_offline_map_plan_audit.py`](e6_offline_map_plan_audit.py) | Did maps differ, and did persisted planner solutions differ? | diagnostic | no | **run** |

E1–E3 are the completed offline source and route experiments. E0 passed. E4 began,
but was stopped after 12 of 45 runs once all persisted plans were found to contain
the same coordinates. E5 shows the availability term is present but too small beside
the frozen risk/obstacle terms to change route selection; E6 verifies that the fields
differ spatially while the saved global plans do not. Because the matched campaign is
incomplete, this package makes no comparative navigation-performance claim.

## Two things a reader should not misread

**The CAD raycast is not an upper bound.** It is used as the common reference in E3
because it is the most complete geometric field available, but it requires a surveyed
3-D model, and in E1 monocular depth matched it (Brier difference 0.006, sign test
p = 0.15 over 24 camera-folds). Calling it an "oracle" would be wrong in both
directions: it is not deployable, and it is not best.

**The cached GP fields leak.** `logs/visibility_comparison/spawn_grid_20260727/gp/`
holds GPs fitted on *every* spawn-grid event. Scoring them at held-out points scores
a model on its own training data — doing exactly that put the GP at Brier 0.021 and
AUROC 0.991, against 0.218 once refitted per fold. [`gp_refit.py`](gp_refit.py)
exists to prevent that, and it runs as a separate process because
`fit_belief_aware_gp` does `from common import ...` and this package also has a
`common`.

## Open governance question, for the supervisor

`research/06_world_camera_design.md` §2 states the two-world hard rule: method
development happens only in `warehouse_aws`; `warehouse_full_4cam` evaluates frozen
methods. E1 compares six availability estimators in `warehouse_full_4cam`.

The defensible reading is that E1 *selects among already-frozen estimators* rather
than developing one — every arm's apparatus (FOV/range, CAD raycast, monocular
depth, the GP fitter) existed and was built before this study, and the four-camera
world is where a four-camera availability field can be measured at all. The
alternative reading is that a source comparison is exactly the §3 method work the
rule reserves for `warehouse_aws`.

This is not mine to settle. It needs either an explicit registered exception or a
decision to re-run E1's geometric arms in the single-camera world. Flagging it here
rather than proceeding silently.

## Reproduce

```bash
# E1 — the GP refit first; it writes the leak-free held-out predictions.
python3 experiments/availability_paper/gp_refit.py \
    --out logs/studies/availability_paper/e1_availability_calibration/gp_fold_predictions.csv \
    --block-x-edges -11.7 -3.9 3.9 11.7 --block-y-edges -9.0 -0.25 9.0
python3 experiments/availability_paper/e1_availability_calibration/run_experiment.py

python3 experiments/availability_paper/e2_availability_vs_accuracy/run_experiment.py
python3 experiments/availability_paper/e3_route_discrimination/run_experiment.py
source install/setup.bash
python3 experiments/availability_paper/e3_route_discrimination/offline_efe_solve.py
python3 experiments/availability_paper/e6_offline_map_plan_audit.py
python3 experiments/availability_paper/make_figures.py

# E0 — check a finished run. LAUNCHES NOTHING.
# Before launching the run itself: pgrep -a "ros2 launch|ign gazebo|run_visibility_campaign"
python3 experiments/availability_paper/e0_online_efe_readiness/check_gate.py \
    --run 'logs/visibility_comparison/<campaign>/mc_blind_L/C1/seed0/experiment_*' \
    --goal <x> <y>
```
