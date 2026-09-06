# Outline: one folder, one storyline, per method

**Seven storylines, not one comparison table.** Each method is driven, scored and explained on
its own before any arm is set against another. The design and the six runs are in
[`README.md`](README.md); this file is the structure and the figure list.

Three decisions this outline implements:

1. **No paired seeds.** Each arm gets its own drive and its own seed, recorded. Nothing is
   paired, nothing is held fixed across arms except the route, the controller and the frozen
   detector. Repeats and pairing are a later decision, taken once we have seen one drive each.
2. **Each method owns a folder.** Its drive, its numbers, its figures and its story live
   together and can be read without any other arm.
3. **The predicted-bounding-box hull method gets its own storyline**, ahead of the arms. It is
   not a fusion arm — it is what every arm's measurement *means*, and it is the part nobody has
   seen explained end to end.

---

## Layout

```
experiments/fusion_on_fixed_routes/
  README.md          the design: the route, the six runs, the notation
  OUTLINE.md         this file
  arms.py            the seven definitions in one place: nothing else names an arm
  drive.py           run ONE arm live, end to end, into its own folder
  score.py           read one arm's drive, write its numbers
  story/
    00_hull.py       the predicted bounding box, explained
    01_best_single.py .. 06_fixed_offset.py    one script per arm
  compare.py         the cross-arm figures — runs only after all seven folders exist

logs/studies/fusion_on_fixed_routes/
  00_hull_observation/       figures + numbers.json   (no drive: commissioned data)
  01_best_single_camera/     drive/  figures/  numbers.json  story.md
  02_distance_angle/
  03_independent_fusion/
  04_joint_network_estimator/
  05_raw_box/
  06_fixed_offset/
  compare/                   only cross-arm figures live here
```

`story.md` in each folder is three paragraphs written after reading that arm's own figures:
what it did, what it claimed, where it broke. Written per arm, so a bad arm gets an
explanation rather than a row in a table.

---

## 00 — The predicted bounding box, explained

**The question:** a detector returns a box in an image. What in the world does that box mean?

This is the storyline that is still missing, and it is the one the whole measurement chain
rests on. Five figures, all from commissioned data already on disk — no drive needed.

| | figure | what it has to show |
|---|---|---|
| 1 | `01_the_box_is_not_the_robot` | three real frames, three cameras: the box's bottom-centre and the robot's true centre, **24–36 cm apart**, and the gap points a different way in each frame |
| 2 | `02_how_the_prediction_is_made` | **the new one.** One real frame, four panels: the robot's visual hull at the believed pose → projected into the image → boxed → bottom-centre taken. The same four steps the detector's box went through, on the prediction side |
| 3 | `03_why_not_convert_the_pixel` | the other direction is ill-posed: one pixel, three headings, three different floor positions. Converting needs a heading you do not have; predicting does not |
| 4 | `04_from_pixels_to_an_ellipse` | the Jacobian: how many centimetres a pixel is worth, near and far, and why the result is an ellipse **along the line of sight** rather than a circle |
| 5 | `05_what_is_left` | after the prediction is right, **0.29 cm** of lean remains against **2.2 cm** of per-sighting scatter — and the heading price: 0.23 cm per degree, break-even at 14° |

Figure 2 is the one to get right. It is the answer to "what is the hull method?" and it should
be readable with no equations: **we do not convert the detector's box into a position; we
predict what the box should look like and compare boxes.**

Say the two words apart, every time: the **observation model** is worth 30 cm and fits
nothing; the **offset** is worth half a centimetre and fits six numbers.

---

## 01–06 — One storyline per arm

Every arm's folder answers the same four beats, in the same order, so the storylines can be
read side by side later without being merged now.

| beat | figure | what it shows |
|---|---|---|
| **what it believes** | `01_the_rule` | this arm's rule, drawn: the same two-camera moment, and what this rule does with it. One picture, no equations |
| **what it did** | `02_the_drive` | the driven path over the warehouse, coloured by error, with the cameras that were contributing; the route figure's expectation drawn faintly beneath so promise and outcome are on one page |
| **is it honest** | `03_claim_vs_truth` | error and the stated 1σ against distance along the route, plus how often the truth fell inside the stated 95% ellipse. **Never one without the other** |
| **where it broke** | `04_worst_moment` | the single worst moment: the real camera frames at that instant, the boxes, the belief, the truth. If an arm has no failure worth showing, say so on the figure |

Per-arm numbers, written to `numbers.json` and quoted in `story.md`:

- position error of the belief: median, 95th, worst — **centimetres**
- how often the truth was inside the stated 95% ellipse, and the mean stated 1σ beside it
- longest stretch with no admitted sighting — **seconds and metres**
- admitted sightings per camera, and the rate at which each camera's detections were admitted
- path length and largest deviation from the commanded route
- completion: goal reached, timeout, or collision

The six arms, in the order their folders are numbered:

| folder | arm | the one sentence its storyline has to earn |
|---|---|---|
| `01_best_single_camera` | hull + single best camera by `tr(Σ_c)` | "one good camera is enough" — or it is not |
| `02_distance_angle` | hull + distance-and-angle weights | knowing where the cameras are is not the same as knowing how good they are |
| `03_independent_fusion` | hull + precisions add | the standard answer, and whether it grows overconfident with camera count |
| `04_joint_network_estimator` | hull + one robust batch estimate, then one Gaussian | the network as one sensor: disagreement becomes uncertainty |
| `05_raw_box` | box bottom-centre *is* the robot + joint network estimate | what ignoring the observation model costs a filter |
| `06_fixed_offset` | box bottom-centre + one fixed offset + joint network estimate | whether a constant would have done |

---

## compare/ — only after all seven exist

Three figures, and they are built last so that no arm's story gets written to fit them.

| figure | what it shows |
|---|---|
| `01_error_and_claim_vs_cameras` | **the plot the experiment exists for**: error and *claimed* uncertainty against the number of cameras contributing at that moment, one line per arm. The route was chosen to populate 0–4 cameras |
| `02_the_six_arms` | median and 95th error per arm with the honesty number beside each bar — sharpness and honesty never apart |
| `03_what_the_box_meant` | arms 4, 5, 6 on the same axes: the cost of the observation model, measured in a filter rather than in commissioning |

---

## The figure standard

The bar is the current deck (`experiments/deck_figures/`) — same house style, same palette,
same rules. Concretely, for every figure in every folder:

- **Built by exactly one script**, into its own folder, re-runnable from files on disk.
- **The title states the finding**, not the variables. Axis labels say which direction is good.
- **Real camera frames wherever the claim is visual.** A box-versus-centre argument drawn as a
  cartoon convinces nobody.
- **Honesty and sharpness always in the same figure.** A stated σ never appears without the
  actual error beside it.
- **The house palette**: `experiments/deck_figures/style.py`. Cameras keep their five colours
  everywhere, blue is the robot and its path, green a useful sighting, orange an outage.
- **Say what the data is** on the figure: how many sightings, which drive, which world, and
  whether it is a prediction or a recorded run.
- 170 dpi, wide canvas, type large enough to read from the back of a room.

---

## What has to be built before anything drives

Audited, not assumed — this is the critical path.

| # | blocker | where | why it blocks |
|---|---|---|---|
| 1 | ~~the launch path refuses any camera set that is not exactly A–D~~ | `visibility_launch_common.py` | **done.** The guard now validates against the perception layer's own `BATCHED_CAMERA_ORDER` (contract v2, five cameras), the manager's model-include map is derived from the world profile, and the camera bridges are derived from it too — camera E had no image bridge on this path |
| 2 | ~~no covariance profile states the commissioned `Σ_c`~~ | `camera_manager_node.py` | **done.** `covariance_profile=commissioned_sigma_px` reads σ_px from `calibration.json` and states `R_pix = σ_px² I` with no floor and no handover inflation. Confirmed live: *"R_pix = (0.7643 px)² I … pushed through each camera's geometry"* |
| 3 | ~~F1, F2, F4 do not exist~~ | `reliability/fusion.py`, `camera_manager_node.py` | **done.** `select_smallest_covariance`, `distance_angle_weighted_fusion_2d`, `joint_network_estimate_2d`, plus one `fusion_rule` parameter behind the shared disagreement gate. |
| 4 | ~~no fixed-offset observation mode~~ | `camera_manager_node.py` | **done.** `observation_model = hull \| raw_box \| fixed_offset`, the last pushing the reading a fixed distance away from its camera. The distance is one commissioned number: 30.9 cm, the mean box-bottom-centre-to-centre gap over 3351 admitted sightings |
| 5 | ~~no campaign driver for these arms~~ | `scripts/visibility_comparison/fusion_on_fixed_routes_campaign.yaml`, `freeze_route.py` | **done.** Six arms as campaign conditions on one hash-bound route; `--dry-run` exercises the same route and clearance gate as launch |
| 6 | empty-warehouse frames for the false-positive check | — | owed by `PLAN.md` before availability; **does not block these drives**, and is called out so it is not forgotten |

Order of work: **1 → 2 → 3, 4 → 5 → one smoke drive of arm 3 → the remaining six.** Arm 3 is
the smoke test because it is the one rule that already exists, so a failure there is the
runtime, not the new code.

Two things to check on that first drive, because they are invisible in everything commissioned
so far: the **operational heading error** (0.23 cm of position error per degree, through the
hull prediction) and whether **five cameras hold their cadence** in one batch on this machine.
