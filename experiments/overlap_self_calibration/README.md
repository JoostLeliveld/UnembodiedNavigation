# overlap_self_calibration — can the cameras find their own mounting error from where they overlap?

> **Not registered in `research/registry.yaml`.** Nothing here is a locked claim yet.

**Question.** When two cameras watch the same robot at the same instant and disagree, the
disagreement is information about the cameras, not about the robot. How much of a camera's
mounting error can be recovered that way — with no ground truth, no survey, and no target of
known position?

**Why now.** The question was unanswerable while the deployed reading was the bottom of a
detected box: its silhouette lean is 6–9 cm and swings by a factor of three with heading, which
is far bigger than any drift worth finding, so a per-camera offset absorbs the box and the camera
indistinguishably. The marked-point reading removed that — held-out mean error within 0.18 px on
every camera — and it gives **four numbers per sighting about three unknowns**, which is the
surplus the whole method lives on.

**Answer.** Aim yes, position never. Read
[`logs/studies/overlap_self_calibration/RESULTS.md`](../../logs/studies/overlap_self_calibration/RESULTS.md).

| | |
|---|---|
| **Aim up/down is measured, not assumed** | 0.0135° 1σ, unchanged across an 80× prior sweep — 18× finer than the drift that becomes harmful |
| **Mount height too** | 9.5 mm 1σ, prior-free |
| **A camera's own position never is** | posterior tracks the prior at 0.65 whatever the prior; sliding the whole network and the whole trajectory together changes no pixel |
| **Overlap does all the work** | poses one camera saw resolve 4 of 24 directions; poses two cameras saw resolve 20 — with 2.5× *fewer* sites |
| **Facing each other is worth 2.2–2.7×** | at 62 sites each, so it is the crossing angle and not the site count |

## What is measured versus what is assumed

Every number is reported with a **prior sensitivity sweep** (0.25× to 20×). A posterior that
holds still across it was measured; one that tracks the prior was assumed. This is the only
honest way to quote a self-calibration figure, and it is what separates the tilt claim
(prior-free) from the pan claim (a fixed 1/14 of whatever prior you bring).

## Faithfulness

The forward model reproduces the capture's own recorded marker pixels to **4.5 × 10⁻¹³ px** over
1000 held-out sightings, asserted before any analysis. `PinholeGroundCamera` agrees with the
deployed `ObliqueCameraModel` to 2.3 × 10⁻¹³ px, which is what licenses reusing the existing
drift parameterisation.

## Method in one paragraph

Every robot pose is treated as completely unknown — three free numbers per pose, 9528 across
3176 poses — and marginalised out by Schur complement, so what remains is what the data says
about the cameras rather than where the robot was. Six mounting numbers per camera (pan, tilt,
roll, and three position coordinates; no intrinsics, because a sagging bracket does not change a
lens) give a 24-dimensional information matrix. Its spectrum is reported in units of the smallest
drift the existing GT-free change detector fires on, and its per-parameter posterior is reported
against a stated prior.

## No fault is injected, and that is the point

The result is a property of the geometry and the noise, not of any particular fault, so it holds
for every drift at once. That is also why it runs offline in three minutes with no Gazebo.

## Files

| file | what |
|---|---|
| `calib_geometry.py` | forward model (pose → four pixels through a possibly-drifted camera) and its Jacobians |
| `observability.py` | information matrix with poses marginalised out; the arms and the overlap census |
| `sensitivity.py` | prior sweep, count-matched crossing angle, sites-needed ladder |
| `independence_check.py` | do two cameras err independently at a shared pose? (+0.05/+0.09 — yes, unlike the box bottom) |
| `fig_what_the_network_can_learn.py` | the summary figure |

## Reuse map

| need | reused from |
|---|---|
| camera model | `unav_common.camera_model.ObliqueCameraModel` |
| gazebo pose → look_at | `experiments.core.world_profiles.compute_look_at_from_pose` |
| mounting-drift parameterisation | `reliability.calibration_perturbation` (`PinholeGroundCamera`, `CalibrationPerturbation`, `perturb`) — the drift-lifecycle study's, imported not reimplemented |
| the reading, its geometry and Jacobian pattern | `experiments/localization_reading_story/keypoint_geometry.py` |
| pixel noise | `logs/studies/keypoint_measurement/v1_4cam_retrained/per_sample.csv` (held-out predictions of `yolo_pose_4cam_v1`) |
| capture geometry, visibility flags | `logs/perception_datasets/projected_keypoint_dataset_4cam_v1/{capture_manifest.json,capture_diagnostics.csv}` |
| detection threshold used as the unit | `experiments/calibration_drift_lifecycle` (0.1° / 0.025 m) |

## What this does not do

It bounds what is recoverable; it does not build the estimator. It also does not resolve the
**two-world rule** tension: overlap cannot be posed in the single-camera method-development
world, so this run deliberately only characterises the *existing frozen* four-camera geometry
and fits nothing. Building an estimator needs that decision made explicitly.
