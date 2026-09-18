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

1. `docs/COMMISSIONED_SENSOR_MODEL_CONTRACT.md` — human-readable method contract.
2. `experiments/thesis_pipeline_lock/thesis_contribution_lock.json` — machine-readable thesis lock.
3. `experiments/thesis_pipeline_lock/pipeline_lock.json` — current evidence status.
4. `docs/localization_metrics.md` and `docs/localization_metrics_registry.json` — admissible results.

If another document conflicts with these sources, it is wrong for the current thesis.

## Locked method

The runtime chain is:

```text
fixed YOLO11n detector
  -> deterministic belief-independent sensor gate
  -> raw bounding-box bottom-centre projected to the ground plane
  -> selected correction
  -> covariance fitted to that correction's residuals
  -> correlation-aware camera fusion
  -> estimator NIS gate
  -> one robot-belief update
```

- `q_i(p)` is the probability that camera `i` returns a detector box that passes the fixed
  sensor gate at ground-plane position `p`. It depends on camera identity and two-dimensional
  position only. Commissioning outcomes are pooled over the headings sampled at each position.
  Heading is not an input to `q`.
- `q_i(p)` uses every expected camera opportunity, including detector misses and gate
  refusals. It excludes NIS and is never a runtime gate or route-feasibility gate.
- The raw camera observation is always the projected bottom centre of the frozen YOLO box.
  No current correction candidate starts from, contains, or depends on a visual hull,
  hull-equivalent position, belief-projected box, CAD silhouette, or ground-truth geometry.
- Learned correction candidates predict the offset from the raw projection to the known
  reference position. Runtime inputs may include camera identity, camera-relative geometry,
  raw-box geometry, detector confidence, and the declared 16x16 image-visibility matrix.
- The image candidate is a zero-initialized gated residual added to the raw-box MLP. If its
  predeclared development-drive gates fail, retain the raw-box MLP.
- Every correction candidate receives a fresh covariance model fitted from that candidate's
  whole-drive out-of-fold residuals. Never reuse one candidate's `R` for another.
- `R_i(p, psi)` may depend on position and heading because it describes the corrected
  measurement residual. This does not make heading an input to `q_i(p)`.
- Process-noise covariance `Q_k` is a fixed estimator/planner input frozen in the experiment
  configuration. Estimating, tuning, or rediscovering `Q` is not part of the thesis method.
- The planner is the existing IWAI EFE planner. Conditions change only the future `q` and `R`
  forecasts. They share dynamics, process noise, objective, optimizer, route candidates,
  feasibility rules, controller, estimator, tasks, and seeds.
- One robot filter consumes each physical camera frame once. Cascading a camera-filter
  posterior into another filter as a fresh independent measurement is prohibited.

## Evidence boundary

An artifact is current evidence only when an active thesis manifest names its exact path and
hash. All other results, plots, protocols, and run summaries are non-authoritative. Never use
an arbitrary `RESULTS.md`, a glob of run directories, or a value remembered from another chat.

For camera accuracy, localization error, belief error, RMSE, bias, NEES, coverage, or run
comparisons, also follow `docs/localization_metrics.md` and its registry. State the layer,
statistic, reference, run set, and sample unit. Score readings, fused corrections, and beliefs
at their own timestamps. Aggregate by complete drive before comparing drives.

## Working rules

- Ground truth is offline fitting/evaluation data only. It never enters the sensor gate,
  runtime correction, covariance query, fusion, estimator, planner, or stopping rule.
- Fit, development, and sealed-audit partitions contain complete drives.
- Preserve all expected opportunities and distinguish misses, refusals, admitted raw
  measurements, and corrected residuals.
- Search current source and active contracts before tests. Old names may survive in Git
  history but must not be revived.
- Before launching Gazebo or a campaign, run
  `pgrep -af "ros2 launch|ign gazebo|run_visibility_campaign"`. Do not start a second run.

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
