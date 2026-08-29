# Unembodied Navigation

A small robot drives a simulated warehouse. Its wheel odometry drifts without bound, so it
leans on wall-mounted cameras that watch the floor. Cameras are not equally useful
everywhere: shelves block views, accuracy falls off with range and viewing angle, and a
camera can be quietly miscalibrated.

The question is not "can the robot see itself". It is **"how good is this particular
sighting, and does knowing that change how the robot should drive?"**

## What is being worked on right now

Three paper directions have been explored and switching between them is why this repository
looks like three projects. **The active one is fusion.**

| | direction | status |
|---|---|---|
| **B** | **Fusion on a fixed route** — how several cameras become one measurement, and what covariance that is entitled to claim | **ACTIVE** |
| A | Availability-aware planning — where the cameras can support localization, and routing on it | parked, blocked on data |
| C | Learned bias correction — a network that beats the analytic hull model | parked deliberately |

`PLAN.md` opens with what each one is, what exists for it, and why the parked two are parked.
Do not mix their artifacts: direction C's learned correction competes with the observation
model direction B holds constant.

## Start here

[`PLAN.md`](PLAN.md) is the plan of record — the sentences the paper has to earn, what is
done, what is still owed, and which dataset may serve which purpose. Read it before
proposing work.

[`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) is the start-to-finish walkthrough of the pipeline with
the glossary. Read it before reading any study.

[`docs/`](docs/) holds three contracts, not prose:
[`localization_metrics.md`](docs/localization_metrics.md) — which quantities may be compared
and what makes a drive scoreable; [`open_questions.md`](docs/open_questions.md) — what is
unresolved and which implementation limits are known;
[`reproducibility_inputs.md`](docs/reproducibility_inputs.md) — the hashed detector and
calibration bytes a machine needs to produce evidence at all.

**No fusion result is currently frozen.** Every drive so far predates the schema-4 logging
that identifies which detector batch produced which filter update, so all of them are
diagnostic. See [`docs/localization_metrics_registry.json`](docs/localization_metrics_registry.json).

## The pipeline

```text
wheel odometry (drifts)  ─┐
                          ├─► EKF: predict, then update through a gate chain ─► belief ─► planner
camera image                                                                   (mean +      (route +
  ─► YOLO box                                                                  covariance)   clearance)
  ─► bottom-centre pixel ─► ray ─► floor plane  = position   (plain IPM, zero fitted parameters)
  ─► pixel uncertainty pushed through J         = covariance (R_xy = J R_uv Jᵀ)
  ─► the cameras reconciled by camera_manager
```

The central object is the **belief**: a position *and* a stated uncertainty. Much of this
work is about whether that stated uncertainty is honest, not about whether the position is
accurate. They are separate claims, and a filter can pass one while badly failing the other.

## Layout

| Path | Role |
|---|---|
| [`src/`](src/) | The ROS 2 runtime: perception, reliability (the camera manager), planning, sim, experiments. Campaigns load from `install/`, so `colcon build` after editing. |
| [`experiments/`](experiments/) | One folder per investigation, each with a README saying what question it answers. |
| [`scripts/`](scripts/) | Campaign runner, perception dataset/model tooling, and the shared analysis library in [`scripts/shared/`](scripts/shared/). |
| [`tests/`](tests/) | `python3 -m pytest -q` from this directory. Fast; just run it. |
| [`config/`](config/), [`schemas/`](schemas/) | Shared configuration and data contracts. |
| `logs/` | Ignored. Run output, captured datasets, trained detectors. |

## Build and test

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
python3 -m pytest -q
```

## Running a campaign

**Check nothing is already running first.** A second Gazebo collides with the live one on
the same ROS topics and corrupts both.

```bash
pgrep -a "ros2 launch|ign gazebo|run_visibility_campaign"      # must be empty
source /opt/ros/humble/setup.bash && source install/setup.bash # or every run dies instantly

python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/fusion_on_fixed_routes_campaign.yaml \
  --dry-run
```

There are three campaign configs and each one names its arms in its own header:

| config | runs | what varies |
|---|---|---|
| `fusion_on_fixed_routes_campaign.yaml` | 120 | the fusion rule (F1–F4) and what a detector's box means (O1, O2), over four routes |
| `measurement_covariance_ablation_campaign.yaml` | 15 | how much the per-camera covariance knows (K0–K2) |
| `heading_update_ablation_campaign.yaml` | 10 | whether a position correction may move the heading (H0, H1) |

Every arm drives the same frozen, hash-bound route through the same controller, so the arm is
the only thing that differs.

Drop `--dry-run` to execute. Output lands in
`logs/visibility_comparison/<campaign>/<task>/<condition>/<seed>/experiment_*/`.
`run_summary.json` appears only when a run ends.

Reuse is fail-closed: a run is only reused when its manifest matches the current config,
the detector and calibration bytes, the campaign file, and the source tree it was produced
from. A changed checkout means a rerun, not a silent mixture.

## Reporting results

Every estimate is scored against the truth at the instant that estimate describes, through
the one loader in [`experiments/fusion_on_fixed_routes/aligned.py`](experiments/fusion_on_fixed_routes/aligned.py).
Two mistakes it exists to prevent — pairing a timestamped estimate with a later truth, and
counting one detection several times — were each made independently in more than one script
and were invisible in the output. Do not read the run CSVs without it.

Ground truth is used to score, and for nothing else. It is never an input the online
planner or filter can see.
