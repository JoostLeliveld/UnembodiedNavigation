# offset_state_model — can a per-camera offset be commissioned, or only estimated?

**Question.** `bayesian_filter_showcase/demo_state_space_model.py` added a per-camera 2-D
offset to the filter state and beat `EXP-BELIEF`'s correlation floor (A4) on a strictly
proper score. This study is the gate that candidate has to pass before it can be a claim:
**is the estimated offset a property of the camera, or of the route it was fitted on?**

It serves **`EXP-BELIEF`** (whose `next_action` is to carry the frozen belief fields into a
matched closed-loop campaign) and speaks directly to **`RQ15`** — *are camera-specific
residuals identifiable from robot appearance, route and yaw?* — which is currently OPEN and
is the reason `EXP-CL-CAL` is BLOCKED.

## Verdict: it cannot be commissioned. Both experiments are negative.

| experiment | question | outcome |
|---|---|---|
| [`exp1`](../../logs/studies/offset_state_model/exp1_do_the_offsets_transfer/RESULTS.md) | does a frozen per-camera offset transfer to another capture? | **NO** — transfers on a repeat of its own route, actively harms across a heading change |
| [`exp2`](../../logs/studies/offset_state_model/exp2_offset_in_the_robot_frame/RESULTS.md) | does re-parameterising it in the robot's body frame rescue it? | **NO** — worse on every pair, and inconsistent even between two runs of the same route |
| [`exp3`](../../logs/studies/offset_state_model/exp3_model_the_geometry/RESULTS.md) | does modelling the geometry in the projection remove the need for a filter-side offset? | **PARTLY** — it halves RMSE (5.26 → 2.50 cm, zero fitted parameters) but the belief is still 41.5 % dishonest, so containment is still required |

`exp3` turns the two negatives into a method: **model the geometry in the projection, where it
is known and free; bound the residual in the filter, because what is left is not identifiable.**
Composed, that gives 2.4 cm RMSE at 0.9 % unearned confidence — the best pair anywhere in this
workstream, against 5.3 cm and 79.6 % for the deployed path with no containment. Geometry buys
accuracy and cannot buy honesty; the floor buys honesty and cannot buy accuracy.

Together they say the quantity is **not a constant in any frame**: not per-camera, not
per-robot. It is a function of viewing geometry — heading × camera bearing × elevation —
which is what `pixel_ground_path/e6`'s CAD object model computes with **zero fitted
parameters**. This is independent confirmation of E6 from the filter side rather than the
projection side.

**What survives** is narrower and still useful: the offset works well as an *online
adaptive state* (NEES 1.12 / 2.21 / 1.80 across the three captures, unearned confidence
0.0 / 7.6 / 0.1 %). It just may never be frozen. That is the same lesson `EXP-DRIFT` reached
from the other direction — a stale correction turns harmful — now measured across routes
instead of across time.

## Files

| file | what it does |
|---|---|
| `exp1_do_the_offsets_transfer.py` | fit offsets per capture, freeze, apply to another; four arms scored on each held-out capture |
| `exp2_offset_in_the_robot_frame.py` | one shared body-frame offset rotated by the odometry heading, same transfer test |
| `exp3_model_the_geometry_bound_the_residual.py` | re-project every detection by inverting the CAD silhouette model (heading from odometry), then re-run the arms and the transfer test on both measurement paths |

Both reuse rather than reimplement: the augmented filter and the proper scoring rule come
from `bayesian_filter_showcase.demo_state_space_model`, the A0–A4 arms from
`demo_how_the_filter_works.trace_arm`, and scoring from `exp1_graceful_vs_trusting.summarize`.
Outputs land in `logs/studies/offset_state_model/<exp>/`.

## Reproduce

```bash
python3 experiments/offset_state_model/exp1_do_the_offsets_transfer.py
python3 experiments/offset_state_model/exp2_offset_in_the_robot_frame.py
```

## Registry status

**No `research/registry.yaml` entry has been created for this study.** It is a negative gate
on a candidate, not a new contribution, and the recommended registry change is a
`next_action` edit on `EXP-BELIEF` plus evidence attached to `RQ15` — see the note at the end
of `exp1`'s RESULTS. That edit is left for explicit approval because statuses are the control
plane's single authority.
