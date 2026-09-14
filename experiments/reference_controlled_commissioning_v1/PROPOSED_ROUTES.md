# Proposed commissioning drives

Each partition is **one continuous drive**, not a set of separate straight runs. The robot
starts at the green marker, threads all five north-south aisles, crosses the east half and
returns along the south apron to where it began. Nothing is teleported and the drive is never
restarted, so the reference pose is tracked without a break and every camera opportunity is
logged in order along one unbroken trajectory.

Three sections are driven as a **sawtooth**, at an angle to the aisles, so the drive is not
axis-aligned everywhere. Every waypoint is checked against the real warehouse_v2 occluder
footprints: the fit drive never comes closer than 0.62 m to a
rack and the held-out drive never closer than 0.65 m.

## Why the drive is not all straight lines

An axis-aligned drive sees the robot at four headings only: north, south, east and west. The
covariance model takes heading as an input, so on such data a heading effect and a position
effect cannot be told apart -- every time the robot is at a given place it faces one of the
same two ways.

The sawtooth sections fix that. The fit drive spends 47 m of its
145 m at an angle, giving 9 occupied heading bins
instead of four:

| heading | distance driven |
|---|---:|
| 0 deg | 27.3 m |
| 60 deg | 4.6 m |
| 90 deg | 35.2 m |
| 135 deg | 14.3 m |
| 150 deg | 4.5 m |
| 210 deg | 6.8 m |
| 225 deg | 14.3 m |
| 270 deg | 36.0 m |
| 300 deg | 2.3 m |

They sit in the three places wide enough to hold them: the south apron, which is clear over
22 m by 2.2 m, the middle of the east hall, and the west end of the apron. Aisles are too
narrow to zigzag in, so those legs stay straight.

## The five aisles

The earlier four-line proposal missed one. The workspace has five north-south corridors:

| label | centre x | note |
|---|---:|---|
| A1 | -10.80 | far west, between the wall and the first rack -- **missed before** |
| A2 | -6.90 | between rack runs 1 and 2 |
| A3 | -3.05 | between rack runs 2 and 3 |
| A4 | 0.82 | between rack run 3 and the east half |
| A5 | 10.57 | east wall |

Both drives enter all five.

## Fit drive: 145 m, 33 turns

| # | waypoint | leg |
|---:|---|---|
| 0 | (-10.80, -6.80) |  |
| 1 | (-10.80, 8.60) | 15.4 m |
| 2 | (-6.90, 8.60) | 3.9 m |
| 3 | (-6.90, -9.00) | 17.6 m |
| 4 | (-6.90, -6.95) | 2.0 m |
| 5 | (-5.62, -5.05) | 2.3 m |
| 6 | (-4.33, -6.95) | 2.3 m |
| 7 | (-3.05, -5.05) | 2.3 m |
| 8 | (-3.05, -7.00) | 2.0 m |
| 9 | (-3.05, 8.60) | 15.6 m |
| 10 | (0.82, 8.60) | 3.9 m |
| 11 | (0.82, 6.50) | 2.1 m |
| 12 | (10.57, 6.50) | 9.8 m |
| 13 | (10.57, 2.95) | 3.5 m |
| 14 | (8.62, 1.80) | 2.3 m |
| 15 | (6.67, 2.95) | 2.3 m |
| 16 | (4.72, 1.80) | 2.3 m |
| 17 | (2.77, 2.95) | 2.3 m |
| 18 | (0.82, 1.80) | 2.3 m |
| 19 | (0.82, -1.00) | 2.8 m |
| 20 | (10.57, -1.00) | 9.8 m |
| 21 | (10.57, -5.05) | 4.0 m |
| 22 | (8.13, -6.95) | 3.1 m |
| 23 | (5.70, -5.05) | 3.1 m |
| 24 | (3.26, -6.95) | 3.1 m |
| 25 | (0.82, -5.05) | 3.1 m |
| 26 | (0.82, -9.00) | 4.0 m |
| 27 | (0.82, -6.95) | 2.0 m |
| 28 | (-1.12, -5.05) | 2.7 m |
| 29 | (-3.05, -6.95) | 2.7 m |
| 30 | (-4.99, -5.05) | 2.7 m |
| 31 | (-6.93, -6.95) | 2.7 m |
| 32 | (-8.86, -5.05) | 2.7 m |
| 33 | (-10.80, -6.95) | 2.7 m |
| 34 | (-10.80, -6.80) | 0.2 m |

## Held-out drive: 137 m, 33 turns

The same loop displaced 25 cm. The sawtooth amplitude is reduced by the same 25 cm so the
offset drive stays clear of the racks the fit sawtooth already runs close to.

| # | waypoint | leg |
|---:|---|---|
| 0 | (-10.55, -6.55) |  |
| 1 | (-10.55, 8.35) | 14.9 m |
| 2 | (-6.65, 8.35) | 3.9 m |
| 3 | (-6.65, -9.00) | 17.4 m |
| 4 | (-6.65, -6.70) | 2.3 m |
| 5 | (-5.37, -5.30) | 1.9 m |
| 6 | (-4.08, -6.70) | 1.9 m |
| 7 | (-2.80, -5.30) | 1.9 m |
| 8 | (-2.80, -6.75) | 1.5 m |
| 9 | (-2.80, 8.35) | 15.1 m |
| 10 | (1.07, 8.35) | 3.9 m |
| 11 | (1.07, 6.75) | 1.6 m |
| 12 | (10.32, 6.75) | 9.2 m |
| 13 | (10.32, 2.70) | 4.0 m |
| 14 | (8.47, 2.05) | 2.0 m |
| 15 | (6.62, 2.70) | 2.0 m |
| 16 | (4.77, 2.05) | 2.0 m |
| 17 | (2.92, 2.70) | 2.0 m |
| 18 | (1.07, 2.05) | 2.0 m |
| 19 | (1.07, -0.75) | 2.8 m |
| 20 | (10.32, -0.75) | 9.2 m |
| 21 | (10.32, -5.30) | 4.5 m |
| 22 | (8.01, -6.70) | 2.7 m |
| 23 | (5.70, -5.30) | 2.7 m |
| 24 | (3.38, -6.70) | 2.7 m |
| 25 | (1.07, -5.30) | 2.7 m |
| 26 | (1.07, -9.00) | 3.7 m |
| 27 | (1.07, -6.70) | 2.3 m |
| 28 | (-0.87, -5.30) | 2.4 m |
| 29 | (-2.80, -6.70) | 2.4 m |
| 30 | (-4.74, -5.30) | 2.4 m |
| 31 | (-6.68, -6.70) | 2.4 m |
| 32 | (-8.61, -5.30) | 2.4 m |
| 33 | (-10.55, -6.70) | 2.4 m |
| 34 | (-10.55, -6.55) | 0.2 m |

## Why hold anything out when the goal is to fit this warehouse

Fitting this warehouse is the goal, and a held-out drive does not conflict with it. Two things
still have to be caught, and both are live in the current results.

The covariance model chooses its own complexity. The learning curve showed the
range-and-bearing covariance is catastrophically wrong on unseen geometry below seven fit
drives, and that the failure is confident rather than noisy: at two drives its nominal 95
percent ellipse held 61 percent of residuals. On its own fit geometry it looks excellent.
Without a held-out drive the selection rule promotes a model that states a confident wrong
uncertainty, and the planner then trusts it.

The robot also never drives the commissioned line exactly. The controller has tracking error
and the planner picks its own path, so the question is whether the model still holds twenty
five centimetres to the side. That is what the offset loop measures.

One caveat on the offset. A 25 cm shift tests interpolation between fitted geometry, which is
the deployed perturbation, but it cannot detect a model that memorised the commissioned
corridors, because the offset drive stays inside the range and bearing envelope the fit drive
already covers. Keeping the sealed audit partition is worthwhile for that reason: it is opened
once, after the covariance model is frozen.

## What this buys over the current four straight lines

| | current | fit drive | held-out drive |
|---|---:|---:|---:|
| separate drives | 8 | 1 | 1 |
| aisles entered | 3 of 5 | 5 of 5 | 5 of 5 |
| turns | 0 | 33 | 33 |
| heading bins occupied | 4 | 9 | 9 |
| diagonal driving | 0 m | 47 m | 41 m |
| 1 m cells visited | 54 | 135 | 122 |
| camera_C bearing span | 1.40 rad | 2.82 rad | 2.71 rad |

Coverage rises from 54 to 135 one-metre cells. camera_C gains most, and it is the camera whose
0.12 rad of bearing support at two drives broke the geometry-conditioned covariance, so this
targets the measured weakness rather than adding repeats of what is already well covered.

## Practical notes

Drive the loop in both directions if time allows; reversing swaps the along-ray and
across-ray error split and the current data shows the two directions behave differently.

The robot must not stop and restart mid-loop. A break splits the drive into two shorter
sequences and removes exactly the continuity that makes temporal and cross-camera structure
measurable.
