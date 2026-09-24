# Current thesis contract

## IMPORTANT: TeX files require explicit permission

Never create, edit, delete, rename, move, or restore a `.tex` file unless the user has
explicitly authorized that exact TeX change in the current conversation. Repository cleanup,
method alignment, prose review, and evidence cleanup do not imply permission to modify TeX.
Read and audit TeX when needed, but present proposed edits and wait for approval.

This repository exists to support the commissioned-camera-network thesis. Do not infer the
current method from old experiments, archived reports, superseded protocols, directory names,
or the newest-looking run.

Read these sources in order:

1. `docs/METHOD.md` — authoritative method, including its latest amendment.
2. `docs/STATE.md` — where the data and results are, current state, open items, history.
3. `pipeline/dataset_lock.json` and `logs/thesis/campaign/manifest.json` — the exact inputs
   by path and hash.

## Where things live

- `pipeline/` — the one pipeline: `dataset.py` (the only loader), `refit.sh`, final audit,
  `routes.sh`, `campaign.sh`, `score_collisions.py`, `analyze_campaign.py`, the campaign
  templates and `tasks.yaml`; `capture/`, `detector/`, `decisions/`, `ops/`.
- `world/` — the warehouse description (zones, cameras); the world itself is
  `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf`.
- `figures/` — one generator per manuscript figure.
- `src/` — the ROS runtime. `config/sensor_gate.yaml` — the sensor gate.
- `logs/thesis/` — captures, frozen detector, fits, evidence, routes, campaign.
  `logs/track_a_draft/` — only what today's draft figures still read; deleted once the
  figures are rebuilt from `logs/thesis/`.

If another document conflicts with these sources, it is wrong for the current thesis.

## Locked method

The runtime chain is:

```text
fixed YOLO11n detector
  -> deterministic belief-independent sensor gate
  -> raw bounding-box bottom-centre projected to the ground plane
  -> shared visibility-informed correction
  -> R0/R1/R2 covariance fitted to out-of-sample corrected residuals
  -> independent information-form camera fusion
  -> estimator NIS gate
  -> one robot-belief update
```

- The active method does not fit a separate `q_i(p)` or any other planner-information model.
  Planning uses the inverse of the matched runtime covariance at the same camera and position.
  Detector opportunities, misses, gate refusals and NIS outcomes do not enter the planning
  artifact. They affect runtime only when a camera frame is processed.
- Each planner arm is a direct export of the covariance used by runtime fusion in that arm.
  M0 uses the inverse of global R0, M1 uses the inverse of camera-specific R1, and M2 uses
  the inverse of spatial R2. The ray-frame matrix is rotated at the planning query position.
- The raw camera observation is always the projected bottom centre of the frozen YOLO box.
  No current correction candidate starts from, contains, or depends on a visual hull,
  hull-equivalent position, belief-projected box, CAD silhouette, or ground-truth geometry.
- The deployed correction predicts the offset from the raw projection to the known reference
  position. Its runtime inputs are the locked structured camera/box features, camera identity,
  and the declared 16x16 image-visibility matrix.
- The visibility-informed shared correction is the deployed correction. Raw and structured-
  only models are diagnostic ablations, not open deployment candidates.
- The active covariance comparison is exactly global-full R0, per-camera-full R1, and
  per-camera-spatial-full R2. All use the same broad inverse-Wishart covariance prior and
  out-of-sample residuals from the frozen deployed correction. The prior covariance is
  `100 I m2` with strength `2.5e-6`. Spatial R2 uses 16 neighbours and a 0.4 m Gaussian
  length scale, and approaches the 10 m standard-deviation prior as local support vanishes.
- `R_i^run` describes the corrected residual conditional on an admitted measurement. Spatial
  residual support enters R2 itself through the broad prior. There is no second support model
  in planning.
- Process-noise covariance `Q_k` is a fixed estimator/planner input frozen in the experiment
  configuration. Estimating, tuning, or rediscovering `Q` is not part of the thesis method.
- The planner is the existing IWAI EFE planner with a deterministic information-form belief
  approximation. The primary navigation matrix crosses R0/global, R1/per-camera and
  R2/spatial models with intact and camera-removal states. Each arm uses its corresponding
  runtime covariance and planner-information representation. All six arms share dynamics,
  process noise, objective, optimizer, route candidates, feasibility rules, controller,
  estimator structure, tasks, and seeds.
- The shared execution controller is `ff_fb`, the current-belief segment-feedback waypoint
  follower with route-tangent feedforward, bounded heading/cross-track feedback, corner
  preview and final-segment braking. Campaigns must set it explicitly and must not inherit
  the stale `turn_then_go` launch default.
- One robot filter consumes each physical camera frame once. Cascading a camera-filter
  posterior into another filter as a fresh independent measurement is prohibited.
- R2 uses 16 neighbouring positions and a 0.4 m length scale, fixed; they are never
  selected on data (the candidate grid is a sensitivity table only).
- Collision is the robot footprint at its true pose leaving the driveable region (site
  boundary, collision objects, zero margin), scored offline by `pipeline/score_collisions.py`
  from `ground_truth_pose.csv`. There is no contact sensing, and ground truth never stops a
  run.
- Campaign and check runs execute in lockstep (1 ms physics, 10 Hz control, 5 Hz cameras).
  Every route must pass the offline `ff_fb` replay inside the driveable region first.

## Evidence boundary

An artifact is current evidence only when an active thesis manifest names its exact path and
hash. All other results, plots, protocols, and run summaries are non-authoritative. Never use
an arbitrary `RESULTS.md`, a glob of run directories, or a value remembered from another chat.

For camera accuracy, localization error, belief error, RMSE, bias, NEES, coverage, or run
comparisons, also follow the reporting rules in `docs/METHOD.md` §11. State the layer,
statistic, reference, run set, and sample unit. Score readings, fused corrections, and beliefs
at their own timestamps. Use the declared independent unit: complete drive for older drive-
based development evidence, complete static position for the final reference survey, and
complete matched run/task/seed outcome for navigation comparisons.

## Working rules

- Ground truth is offline fitting/evaluation data only. It never enters the sensor gate,
  runtime correction, covariance query, fusion, estimator, planner, or stopping rule.
- The final reference campaign is partitioned by complete static reference position. All
  repetitions, headings, and cameras for one physical position remain in one partition.
- Older drive-based development artifacts may populate the complete draft for feedback, but
  final numerical claims are replaced rather than pooled with the final reference-position
  campaign. Follow the two-track boundary in the canonical lock.
- Preserve all expected opportunities and distinguish misses, refusals, admitted raw
  measurements, and corrected residuals.
- Search current source and active contracts before tests. Old names may survive in Git
  history but must not be revived.
- Before launching Gazebo or a campaign, run
  `pgrep -af "ros2 launch|ign gazebo|campaign_runner"`. Do not start a second run.

## One version of everything

There is ONE version of every artifact. Producing a new map, config, dataset, figure,
script or lock means moving or deleting the old one in the SAME step. Never leave two
live copies, and never leave a `_v2`, `_new`, `_proposal` or `_redraw` beside the thing
it replaces: parallel versions are how the wrong one gets used.

Everything written to disk is paper-facing. There are no scratch outputs that are allowed
to be stale or wrong. A proposal file is acceptable only while a decision is pending, and
is promoted into place or deleted once the decision is made.

A lock or manifest must never be rewritten in the same action as the thing it protects.
If it is, it certifies the change instead of detecting it.

## Showing plots

When the user asks to see a plot, figure, map or image, OPEN IT ON THEIR SCREEN. Do not
only write the path and do not rely on a markdown link: relative links resolve against the
editor's workspace root, which is not necessarily the working directory, so they often do
not click through.

    env -u LD_LIBRARY_PATH -u LD_PRELOAD -u GTK_PATH -u GIO_MODULE_DIR \
        DISPLAY=:0 setsid eog <file.png> </dev/null >/dev/null 2>&1 &

Clearing those four variables is required: snap paths leak into `LD_LIBRARY_PATH` and break
GTK apps with `undefined symbol: __libc_pthread_init`. `setsid` detaches the viewer so it
survives the tool call. Also send the file so it appears in the conversation, and quote the
absolute path. Never `pkill -f eog` to close a viewer - that pattern matches this agent's own
shell; list PIDs with `ps -eo pid,args | grep "[e]og"` and kill by PID.
