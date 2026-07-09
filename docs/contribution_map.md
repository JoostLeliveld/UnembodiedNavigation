# Contribution Map

This project is easiest to present as a chain of small, testable contributions.
The planner result is not one monolithic planning trick; it comes from making
camera reliability explicit and carrying it through the stack.

![Contribution map](media/contribution_map.png)

## 1. External-Camera Perception

The warehouse has a fixed external camera, racks, oblique viewing angles, and
camera-poor regions. The perception contribution is a detector pipeline that
turns the RGB image into one localization point: the selected bounding-box
bottom centre. The raw detector score is logged as an empirical reliability
signal, but it is not treated as a calibrated probability.

Start here: [`../yolo/README.md`](../yolo/README.md).

## 2. Image-To-BEV Localization

The detector does not produce robot pose. The selected image point is projected
to the ground plane through the calibrated camera model, then refined by a
small affine correction fit on the capture grid. Heading is not directly
observed by the camera in the locked campaign; it is odometry-driven and can be
corrected only indirectly through belief covariance.

Start here: [`../estimation/README.md`](../estimation/README.md).

## 3. Stochastic Belief Story

The uncertainty terms have separate jobs:

| Term family | Role |
| --- | --- |
| Process/model noise | What the filter and planner assume while propagating belief. |
| Command/encoder noise | What motion execution and odometry inject into the run. |
| Measurement noise | Camera observation covariance used when image-space evidence is expected. |

This separation is important for the presentation: the system does not simply
"add noise". It models how uncertainty grows under motion and how camera
measurements can reduce that uncertainty when the camera is expected to be
trustworthy.

## 4. GP Reliability To `R_plan`

The GP part is a data-to-covariance story:

```text
sample pose -> detector score -> GP trust field -> sigma_plan^2 -> R_plan
```

The GP learns a spatial trust field from detector performance samples. It does
not learn `R` online. At planning time, the trust value scales the predictive
observation covariance through a precision blend:

```text
1 / R_plan = trust / R_visible + (1 - trust) / R_miss
```

In the locked setup, `R_plan` is a symmetric image-space covariance matrix with
zero off-diagonal terms and equal `u` and `v` variance:

```text
R_plan(x, y) =
[ sigma_plan^2(rho(x, y))      0                         ]
[ 0                            sigma_plan^2(rho(x, y))   ]  px^2
```

The ellipses in the README visuals are therefore circular glyphs whose size
changes across the map. They explain "camera measurement gets fuzzier here",
not "the robot footprint gets bigger here".

Start here: [`../gp/README.md`](../gp/README.md).

## 5. Planning With Reliability, Ambiguity, And No-Go Geometry

C1 and C2 use the same warehouse, seeds, driveable/no-go layer, local tracking,
and optimizer budget. The difference is planner-facing camera covariance:

```text
C1: constant camera covariance
C2: GP-scaled R_plan
```

Low camera trust increases future observation covariance. That changes expected
ambiguity and belief growth, which can make a short camera-poor route less
attractive than a longer route where localization remains useful. Obstacle and
no-go costs are separate from the GP field.

Start here: [`../planning/README.md`](../planning/README.md).

## 6. Campaign Evidence

The current honest campaign uses four routes, two conditions, and five seeds
per route/condition. The current packaged surface is C1 15/20 clean reaches
with four GT-geometry safety breaches, and C2 20/20 clean reaches.

Start here: [`../experiments/README.md`](../experiments/README.md).
