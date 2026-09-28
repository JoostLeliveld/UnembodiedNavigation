# Repository contract

This repository contains the final thesis implementation and its reproducibility
pipeline. Keep one canonical path from frozen data to reported results.

## Read first

1. `docs/METHOD.md` - scientific and experimental method.
2. `docs/PLANNER.md` - planner objective and locked parameters.
3. `docs/PROCESS_NOISE.md` - encoder-derived process covariance.
4. `docs/STATE.md` - final artifacts and repository state.
5. `pipeline/dataset_lock.json` and the final campaign manifest - exact
   evidence inputs and hashes.

## Active pipeline

```text
frozen camera data
  -> detector inference and deterministic gate
  -> learned camera-aware position correction
  -> global, per-camera and spatial residual covariance
  -> information-form camera fusion
  -> NIS-gated EKF belief update
  -> expected-free-energy route planning
  -> fixed-route execution with ff_fb tracking
  -> offline evidence and swept-footprint audit
```

Ground truth is used only for offline fitting and evaluation. It must never
enter the runtime gate, correction, fusion, estimator, planner, controller,
stopping rule or collision decision.

## Evidence rules

- Quote results only from artifacts named by the active locks and manifests.
- Preserve camera-frame identity and expected detector opportunities.
- Dataset partitions are by complete physical position.
- Navigation comparisons are matched by task and noise seed.
- A successful run requires complete evidence, true final distance below
  0.30 m and no swept-footprint departure from the driveable region.
- Do not pool the final thesis campaign with development or IWAI experiments.

## Repository rules

- Keep one implementation, configuration and generator for each thesis
  artifact. Use Git history for rejected alternatives.
- Do not add `_new`, `_final`, `_v2`, `copy` or similar parallel files.
- Version labels inside frozen data paths are provenance identifiers and may
  remain when manifests depend on them.
- Generated data and run logs belong under `logs/` and are not committed.
- Preserve unrelated user changes in a dirty worktree.
- Before a simulator run, check
  `pgrep -af "ros2 launch|ign gazebo|campaign_runner"`.

## TeX boundary

Do not create, edit, rename or delete TeX files unless the user explicitly
authorizes that TeX change in the current conversation.

## Verification

Run the smallest relevant tests while editing and the complete suite before a
submission release:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m pytest -q
```
