# Fusion on one fixed route

**When more than one camera sees the robot at once, how should their readings be combined —
and what should the detector's box be taken to mean?**

Two results come out of one drive, so they share one page:

| | question | arms |
|---|---|---|
| **RQ2a** | Which multi-camera fusion rule? | F1 best single · F2 distance-and-angle heuristic · F3 independent · **F4 network** |
| **RQ2b** | What does a detector's box mean? | O1 raw box bottom-centre · O2 fixed offset · **O3 hull** |

This serves sentence 4 of [`PLAN.md`](../../PLAN.md): *calibrated external-camera observations
improve localization over onboard estimation alone without producing overconfident fusion.*
It is the step before the planner: **show the coverage is worth seeking before asking a
planner to seek it.**

---

## Two quantities, two names, and why they must not share a symbol

Every camera starts from the **same** detector noise — one commissioned number, in pixels:

```
R_pix = sigma_px^2 * I ,   sigma_px = 0.76 px      (calibration.json, frozen)
```

Push it through that camera's own imaging geometry and you get **that camera's estimate of
the robot's position, and its uncertainty**:

```
Sigma_c = J_c^-1 R_pix J_c^-T          in metres^2, per camera, per state
```

`J_c` is the Jacobian `observation.py` already computes for the hull prediction — how many
pixels the prediction moves per metre the robot moves. Written the other way round it is the
information form the fusion rules use: `Sigma_c^-1 ~ J_c^T R_pix^-1 J_c`.

**Consequence that decides the experiment:** distance and viewing angle are *already inside*
`Sigma_c`, because they are what changes `J_c`. A far or badly-angled camera gets a large
`Sigma_c` without anyone penalising it. So range and angle appear in this experiment **only
as F2, the heuristic baseline** — never as a hand-tuned term in the proposed method.

Do not call `Sigma_c` "R". The older code and docs use `R_xy = J R_uv J^T` for the same
object; in this study `R_pix` is the detector's noise and `Sigma_c` is camera *c*'s position
uncertainty, and no sentence uses one where it means the other.

---

## The route, before any arm is named

`fusion_network_traverse` in [`src/experiments/config/tasks.yaml`](../../src/experiments/config/tasks.yaml):
**the west dock door, 3.4 m from camera A, to the east cross aisle under camera D — 30.62 m.**
Frozen as a hash-bound artifact by
[`freeze_route.py`](freeze_route.py) into `route_fusion_network_traverse.json`; the campaign
refers to it by sha256, so every arm executes that polyline and not a route repaired at
launch. Figure: `logs/studies/deck_figures/fusion/01_the_route.png`.

Every arm drives this one route, so the only thing that differs between arms is how the
cameras' readings are combined.

| what the route meets | |
|---|---|
| cameras that contribute | **all five** — A 7.4 m, B 14.4 m, C 3.6 m, D 17.0 m, E 18.8 m |
| metres with 0 / 1 / 2 / 3 / 4 cameras at once | 2.4 / 9.6 / 9.2 / 5.2 / 4.4 |
| longest stretch expecting no camera | 2.0 m |
| corridor-distinct candidates the lane geometry offers | 6, from 30.7 m to 35.4 m |
| vertices the robot actually tracks | **10** — the corners, not the generator's 284 grid steps |
| measured clearance, tightest point | **0.354 m** at the aisle-mouth corner (10.45, −1.05) |

Four things about that table are choices, and they are stated rather than optimised:

- **The shortest candidate is driven.** Route choice is not the treatment here. It becomes
  the treatment in the planning experiment, where the menu must stay arm-neutral.
- **The start moved to the dock door** (from aisle A2|A3's south mouth) so camera A carries
  7.4 m instead of 4.8 m and every camera contributes. This is the reason the route is
  30.6 m rather than 19.8 m.
- **Camera C contributes 3.6 m, and that is a trade.** C is a corner camera looking
  south-east across the west aisles. The candidates that give it 9–21 m spend their length in
  the single-camera west aisles, which drops the two-or-more-camera share from 61 % to
  21–27 % — and the two-or-more share is the axis RQ2a is measured along. If camera C's own
  behaviour becomes the question, the 33.4 m candidate (C 8.8 m) is the route to drive, and it
  is a second run rather than a replacement.
- **The route is generated on a grid eroded by 0.50 m to clear 0.35 m.** A polyline joining
  0.10 m cell centres cuts diagonal corners and loses about 0.14 m of whatever erosion it was
  given: at 0.35 m erosion this route's tightest point measures 0.212 m, which the launch
  gate refuses and the planner's own 0.35 m keep-out would fight. `freeze_route.py` climbs the
  erosion ladder until the *measured* clearance passes, against the same geometry the launch
  gate uses.

- **The frozen polyline is 10 vertices, not 284.** The preselected-route contract executes
  every coordinate as a waypoint — deliberately, so the driven route is the selected one — and
  the generator walks a 0.10 m grid. Handed 284 waypoints, the tracker tried to face points a
  hand's width away, where a few centimetres of belief error swamp the bearing: measured, the
  robot span in place at 20.8 m of 30.6 and was flagged stuck. `freeze_route.py` now
  simplifies to the corners the route actually turns at, and re-measures clearance on the
  simplified polyline because that is the one driven.

**One consequence worth stating in the paper.** The robot's circumscribed radius is 0.486 m
and no candidate route to this goal clears that anywhere — so it cannot turn on the spot at
the tight corner. The arms therefore drive with a pursuit controller rather than
turn-then-go, and a collision would terminate the run and be reported.

Support numbers are the commissioned usable-sighting rate per camera (386 floor positions x
6 headings x 5 cameras), counted where that rate is 25 % or better. They are what the
commissioned model *expects* the route to meet — not a recorded drive.

---

## The six runs

All six use the same route, the same controller, the same frozen detector and the same
`sigma_px`. Runs 1–4 all use the hull observation model; runs 4–6 all use network fusion, so
run 4 is one run serving both questions.

| run | observation model | fusion rule | what it is there to answer |
|---|---|---|---|
| **1** | hull | **F1** single best camera, smallest `tr(Sigma_c)` | why fuse at all, if you can just pick the best camera? |
| **2** | hull | **F2** distance-and-angle weights | does covariance modelling beat knowing where the cameras are? |
| **3** | hull | **F3** independent Gaussian fusion | the standard principled baseline |
| **4** | **hull** | **F4** network fusion, exponents 1/N | **the proposed method** |
| **5** | raw box bottom-centre | F4 | what ignoring the observation model costs |
| **6** | fixed box-to-centre offset | F4 | whether a constant offset would have done |

### The four fusion rules, in one line each

**F1 — single best.** At each update take `c* = argmin_c tr(Sigma_c)` and use only that
camera. A strong, simple baseline: if one good camera is as good as five, the network claim
is dead and the paper should say so.

**F2 — distance and angle.** The intuitive engineering answer, frozen as written and
deliberately not tuned:

```
q_c = max(cos alpha_c, 0) / (d_c^2 + eps)      w_c = q_c / sum_j q_j      mu = sum_c w_c mu_c
```

Its covariance is the weighted combination, so it can be scored for honesty like the others.

**F3 — independent Gaussian fusion.** `Sigma^-1 = sum_c Sigma_c^-1`, mean
`Sigma * sum_c Sigma_c^-1 mu_c`. Already implemented and already the wired default:
`independent_measurement_fusion_2d` in `src/reliability/reliability/fusion.py`. It shrinks
its stated covariance like `1/N`, which is exactly the claim on trial.

**F4 — the network as one sensor.** Gaussian pooling with conservative exponents:

```
p_net(x) ~ prod_c N(x; mu_c, Sigma_c)^{w_c}      w_c = 1/N      Sigma_geo^-1 = (1/N) sum_c Sigma_c^-1
```

Two things do two separate jobs, and that separation is the point:

- **`Sigma_c` handles camera quality.** A precise camera already carries more information.
- **`w_c = 1/N` handles pooling.** It stops the network claiming N independent pieces of
  evidence when the cameras' errors may be correlated — same robot, same detector, same
  hull, same stock arrangement.

So the exponent must **not** also be set from `Sigma_c`. Quality is counted once.

### The three box interpretations

| | what the detector's box bottom-centre is taken to mean | fitted |
|---|---|---|
| **O1** | the robot's centre, directly | nothing |
| **O2** | the robot's centre plus one fixed offset | 2 numbers |
| **O3** | wherever the projected visual hull's boxed bottom-centre lands — the same operation the detector performs | nothing |

O3 is the frozen method (`experiments/measurement_commissioning/observation.py`,
`reliability/silhouette_observation.py`). O1 is worth about 30 cm and O2 cannot remove the
11 cm that swings with heading — those are commissioning measurements, so runs 5 and 6 are
there to show what that costs a *filter*, not to re-litigate the observation model.

---

## How the runs are produced: six live drives, one storyline each

**Six arms, six live closed-loop drives. One drive per arm for now, each with its own seed,
its own folder and its own storyline** — see [`OUTLINE.md`](OUTLINE.md). Nothing is paired and
nothing is averaged across arms yet: each arm is driven, scored and explained on its own, and
the cross-arm figures are built last, from folders that already exist.

Fusion feeds the belief, the belief predicts the box, and the predicted box decides which
detections are admitted. So the arms do *not* see the same detections: an arm whose fusion rule
is worse localizes worse, predicts the box worse, admits fewer sightings and drives slightly
differently. **That coupling is the method, not a confound to be engineered away.** An arm that
only wins when it is fed another arm's admitted detections has not won anything a robot could
use.

Three consequences, all reported rather than assumed away:

- **The claim is about deployable configurations** — "this way of combining cameras localizes
  better on this route" — not about a rule in isolation. That is the claim the paper wants.
- **The trajectory is a measured number.** Every arm gets the same waypoints and the same
  controller, but its own belief steers it. Each folder reports path length and the largest
  deviation from the commanded route. An arm that wanders off it has invalidated its own
  camera-count bins, and that is a result about the arm.
- **Bin by what actually happened**: the cameras that contributed to that update *in that run*,
  on that arm's own trajectory — never the commissioned expectation drawn in the route figure.

One drive per arm is not a variance claim, and no figure may imply one. Whether repeats and
paired seeds are needed is a decision to take after seeing one drive each.

At the planner's 0.22 m/s ceiling, 30.4 m is roughly two and a half minutes of driving per
traverse.

**Keep every log**: per-camera raw detections, admitted sightings, the belief and its stated
covariance, odometry, the commanded pose, and the reference pose kept aside for scoring only.
Not to build a replay comparison — to be able to explain a surprise afterwards.

## What already exists and what has to be built

| piece | where | state |
|---|---|---|
| the route and its candidate menu | `experiments/warehouse_v2_sketches/route_tasks.py`, task `dock_w__xaisle_e` | **done**, geometry only |
| the task entry | `src/experiments/config/tasks.yaml: fusion_network_traverse` | **done** |
| the route figure | `experiments/deck_figures/fusion/01_the_route.py` | **done** |
| `Sigma_c` from one `sigma_px` | `experiments/measurement_commissioning/{observation,uncertainty}.py` (`jacobian`, `ground_covariance`) | **done, frozen** |
| hull observation model, live | `reliability/silhouette_observation.py`, `camera_manager_node.py` (`silhouette_observation_correction`) | **done** |
| F3 independent fusion | `reliability/fusion.py: independent_measurement_fusion_2d` | **done, wired** |
| F1 best single by `tr(Sigma_c)` | `reliability/fusion.py: select_smallest_covariance` | **done, wired** |
| F2 distance-and-angle weights | `reliability/fusion.py: distance_angle_weighted_fusion_2d` | **done, wired**, frozen coefficients, no tuning |
| F4 network pooling, `w = 1/N` | `reliability/fusion.py: network_pooled_fusion_2d` | **done, wired** |
| a covariance profile that states `Sigma_c` | `camera_manager_node.py` (`covariance_profile: commissioned_sigma_px`) | **done, and the only supported profile** — the pre-clean-sheet metric-floor and 2.5/40 px profiles are gone |
| O1 raw box / O2 fixed offset | `camera_manager_node.py` (`observation_model`) | **done, wired** |
| the campaign driver: 6 arms x 4 routes x 5 seeds | `scripts/visibility_comparison/fusion_on_fixed_routes_campaign.yaml` | **done** |
| schema-4 logging: one assimilation row per detector batch | `experiments/nodes/experiment_logger.py`, `unicycle_planner_node.py` | **done, and required by `score.py`** |
| empty-warehouse frames (false positives) | — | **missing, and owed before this study**, per `PLAN.md` |
| a frozen run selection | `logs/studies/fusion_on_fixed_routes/frozen_runs.json` | **missing — this is the gate.** No number may be reported until it exists |

Two things to hold on to before reading any output:

1. **No fusion result is currently frozen.** Every drive so far is `logging_schema_version` 3
   and is diagnostic only; `score.py` refuses anything older than schema 4 and refuses a run
   whose corrections and assimilations do not correspond exactly. See
   `docs/localization_metrics_registry.json`.
2. **The operational heading error is still unmeasured**, and it enters the position error at
   about 0.23 cm per degree through the hull prediction (`heading_gate.py`). These drives are
   the first artefact that can measure it — log the estimator's heading and score it
   separately, because it is invisible in every commissioned number so far. The
   `camera_xy_only` versus `coupled` comparison has its own config,
   `scripts/visibility_comparison/heading_update_ablation_campaign.yaml`.

---

## How the runs are scored

**Headline: position error of the robot's belief, in centimetres**, median and 95th
percentile, over the fixed route. Never the single-sighting measurement error — that is a
different quantity (1.49 cm median, 3.50 cm RMSE) and conflating the two is the mistake
`PLAN.md` names explicitly.

**Beside it, always: is the stated uncertainty honest?** How often the reference position
falls inside the stated 95 % ellipse, and the stated 1-sigma next to the actual error. A rule
that widens its ellipse passes any calibration test and is useless.

**The plot to watch** — and the reason the route was chosen to span 0 to 4 cameras:

> error and **claimed** uncertainty, both against the number of cameras contributing at that
> moment.

The bins exist on this route: 9.4 m at one camera, 9.2 m at two, 5.0 m at three, 4.4 m at
four. If F3's claimed uncertainty keeps shrinking with camera count while its error does not,
and F4's claim stays honest, that is the evidence for treating the camera network as one
sensor. If both stay honest, the simpler rule wins and the paper says `1/N` was unnecessary —
that is a clean result, not a failed one.

**Supporting:** worst error, longest stretch with no admitted sighting (seconds), and the
admitted-sighting rate per camera, which is also the first operational check on the
availability map.

---

## Rules this study does not get to break

- **Ground truth scores; it never enters.** It forms residuals and metrics, and is not an
  input to the filter.
- **Availability does not weight a measurement that has already arrived.** It predicts the
  future; it does not judge the present. Availability belongs to the planner.
- **No usability definition may use a ground-truth error threshold.**
- **The visibility term is frozen method** and is not tuned to make a run look better.
- **Do not claim the network cancels bias.** Cameras fusing offsets produce a weighted
  combination of those offsets. A heading error is shared by all five, and fusing them removes
  1 % of it.
- **F2 stays untuned.** Its job is to be the honest simple answer, not a competitor that had
  a week of attention.
