# Measurement commissioning

**Is the Camera+YOLO reading well enough behaved to be used as a global position
correction, and what does its error look like once commissioned?**

```bash
python3 experiments/measurement_commissioning/commission.py
```

Offline, a few minutes, reproducible from files already on disk. Writes to
`logs/studies/measurement_commissioning/`.

---

## One question per file

| file | the one question it answers | fits |
|---|---|---|
| `camera.py` | where are the cameras and how do they project? | nothing |
| `observation.py` | **the box is not the robot** — where should the box be for a given pose? | nothing |
| `admission.py` | is this sighting usable? | nothing |
| `detector.py` | what does the frozen detector actually do? | nothing |
| `capture.py` | what was collected, and which job may each part serve? | nothing |
| `offset.py` | what half-centimetre lean is left, and how is it removed? | **6 numbers** |
| `uncertainty.py` | how much should a sighting be trusted? | **1 number** |
| `confidence_covariance.py` | does confidence add held-out predictive value for post-geometry pixel covariance? | confidence bins, evaluated out of fold |
| `commission.py` | the driver | — |

**Information flows one way.** The first five are inputs to `offset` and `uncertainty`;
nothing measured in those two may change the others. Not pedantry — the detector was once
retrained to make a residual smaller, and a label-convention change then looked like a bias
result.

## Two things get called "bias" and only one is

**`observation.py` — the box-versus-centre problem. 30 cm. Not a bias, not fitted.**
The bottom edge of a detector's box is the robot's nearest point to the camera; its
horizontal middle is the midpoint of its widest part. Back-project the bottom-centre and it
lands **24–35 cm** from the truth, swinging **11 cm** as the robot turns. It is removed by
*predicting* the box the same way the detector measures it, so both sides are the same
physical quantity — not by converting a pixel into a position, which needs a heading and is
ill-posed. This is the largest error in the chain and the one that reaches the final RMSE.

**`offset.py` — the leftover lean. 0.5 cm. Six commissioned numbers.**
A calibration detail. Nothing downstream rests on it.

## What `uncertainty.py` actually does

The detector's error is a property of the **detector**, not of where the robot is standing.
So it is measured once, in pixels, and everything position-dependent is left to the camera
geometry. `fit_sigma_px` takes the spread of the pixel residuals — one number. That holds
here: the sideways spread is flat to within 10% across a fourfold change in distance.

`ground_covariance` then carries it into the world with the same Jacobian `observation.py`
already computes. One number produces a correctly shaped, correctly sized ellipse everywhere
— stretched along the viewing ray, 1.2 cm near a camera, 4.0 cm far. It tracks the observed
spread to within 15% while that spread changes sixfold, which is why it beat a fitted spatial
map with a thousand times the parameters.

**What it refuses to absorb:** heading error. That never travels through the pixel channel
and is *shared by every camera*. Folding it into a per-camera pixel noise would be the wrong
magnitude and would treat a shared error as independent, making the filter more overconfident
the more cameras are fused. `heading_term_cm` exists to *report* its size, never to add it.

## Two datasets, disjoint on purpose

| | offset | availability map |
|---|---|---|
| needs ground truth | **yes** — it is reading minus truth | **no** |
| how much | ~10–20 positions | 250+, still improving |
| given | 20 positions | 366 positions |

**How the offset spots are chosen matters more than how many.** Twenty spread evenly across
distance gave 133 sightings and 1.16 cm at long range; twenty chosen as the *most overlooked*
within each of four distance bands gave 192 and 0.89 cm. Same twenty parkings — one just puts
the robot where several cameras already look. The rule uses only camera mounts and building
geometry, so it works on a floor plan.

## The trap, paid for three times

**Never join files on coordinates.** The capture stores full floats, every written file
stores formatted ones, and the grid lands exactly on rounding ties — rounding,
lattice-snapping and decimal formatting each fail somewhere. It silently detached 10% of the
usable sightings from their attempt records (inflating "no camera can help here" from 15% to
24%) and twice miscounted which spots the offset was fitted on. Every position carries an
integer `position_id`, written into every file. Joins use it.

## Outputs

| file | what it is |
|---|---|
| `calibration.json` | everything frozen, with input hashes |
| `sightings.csv` | the 3 351 usable sightings and their pixel residuals |
| `availability.csv` | one row per (camera, pose) trial — 11 585, including the failures |
| `offset_positions.csv` | the 20 spots the offset was fitted on |

`availability_robustness.py` re-judges the same detections with a deliberately wrong pose, to
bound how far a truth-free availability map would drift from this truth-based one. At
10 cm / 2° the usable rate falls 0.70 → 0.56, and admission **fails safe** — wrongly-kept
stays near 1% while wrongly-dropped climbs to 15%.

## Known limits

1. **The pose is exact.** `h`, the Jacobian and admission are all evaluated at truth; at
   runtime all three use the robot's own estimate. The largest open risk.
2. **Admission selects for agreement**, discarding 30% of detections, so the measured
   residual is conditioned on the prediction already being close.
3. **Six headings, 60° apart** — enough to see heading structure, not to model it.
4. **False positives unmeasured** — no pictures of the empty warehouse.
5. **One stock state, one world, simulation.**
