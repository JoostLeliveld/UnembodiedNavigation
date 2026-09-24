# State of the thesis pipeline

The one place to look for where the data and results are, what has been done, and what is
open. Method: `docs/METHOD.md`. Exact inputs: `pipeline/dataset_lock.json` and, once made,
the campaign manifest. Numbers here are pointers; quote results from the named artifacts.

## Where everything is (`logs/`, not in git)

| path | content |
|---|---|
| `logs/thesis/captures/v3_detector` | the v3 capture the frozen detector was trained on |
| `logs/thesis/captures/v5/{part1,part2}` | v5 reference capture (two passes over one plan) + plan files |
| `logs/thesis/captures/v8/{supplement,topup,repair,repair2}` | v8 additions and the repair of the broken v5 part-1 poses, with pose files, top-up plan and capture logs |
| `logs/thesis/detector/{dataset,training}` | frozen YOLO11n detector and its training provenance |
| `logs/thesis/fits/` | inference, gate, correction, corrected residuals, R0/R1/R2, D_dev evaluation, runtime package, planning precision (`pipeline/refit.sh`) |
| `logs/thesis/final_audit/`, `routes/`, `campaign_configs/`, `campaign/` | produced in the next steps |
| `logs/thesis/evidence/` | gate-variant reports and maps, rejected-v7 archive, v5-vs-v8 check, dataset audit, deleted broken-frame record |
| `logs/track_a_draft/` | only what today's draft figures read; deleted once they are rebuilt from `logs/thesis/` |
| `logs/thesis/pipeline.log` | one log for refit, routes and campaign |

## Current state (2026-09-24)

- **Dataset v8**: 2,619 positions, 52,380 camera opportunities; audit passes
  (`logs/thesis/evidence/dataset_audit.json`). See METHOD amendment A.
- **Gate**: `config/sensor_gate.yaml` (confidence and projection only). Amendment B.
- **Fits**: final deterministic refit into `logs/thesis/fits/` (see its manifests).
- **Final audit**: opened once, 2026-09-24 01:30 (`logs/thesis/final_audit/report.json`,
  `final_audit_fusion/report.json`; protocol `logs/thesis/final_audit_protocol.json`).
  150 positions, 3,000 opportunities, 1,633 admitted. Quote numbers from the reports.
- **Routes**: 30 solved and checked.
- **Campaign: done** (2026-09-24 03:38 to 07:33, commit 5e1e802b, manifest
  `logs/thesis/campaign/manifest.json`, no input drift afterwards). 90 of 90 cells have a
  valid outcome; 3 runs were infra_invalid on the first attempt (lockstep alignment step,
  fixed afterwards in c0c56b17) and valid on the retry. Analysis: `pipeline/analyze_campaign.py`
  -> `logs/thesis/analysis/` (`runs.csv`, `summary.json`, `trajectories.png`). Success
  per arm (of 15; re-read `summary.json` before quoting): spatial 15 intact / 14 removal,
  per-camera 14 / 11, global 15 / 8. The spatial model changes route under removal in four
  of five tasks (not task B).
- **Goal rule changed before the campaign** (agent, overnight): the template stopped at
  0.30 m on the belief, the same radius the success score uses, against the author's rerun
  decision of 2026-09-22. Now stop 0.10 m (belief), success < 0.30 m (ground truth), 90 s
  limit, `goal_loiter_timeout` terminator (METHOD amendment D). Five runs made with the old
  rule are in `logs/thesis/superseded_goal_radius_0p30/` and are not results.
- **Qualification with the new rule**: task C spatial_intact reached the goal (4.5 cm true);
  task E-long spatial_intact clipped an obstacle at the corner of its northern crossing
  (footprint 5 mm inside, then stuck) where the ideal replay had 5.2 cm clearance: a real
  outcome of belief error and actuation noise, not an infrastructure failure.
- **Repo**: lean layout, one version of every artifact, tests green; tag
  `pre-lean-20260924` holds everything removed during the lean-up.

## Open items, in order

0. For the author (from the overnight run): (a) review the goal-rule change and the task-B
   admission fix (both in git with reasons); (b) task E-long has a corner at about
   (1.3, 8.4) that two intact runs clipped (one in qualification, one per-camera intact in
   the campaign): a real outcome, but it may be worth a sentence; (c) the agent's tape-clear
   warning logs `reason=unspecified_call_site`, so the stop cause of `stuck` runs has to be
   inferred from belief sigma; (d) lockstep runs are not bitwise identical (see item 4).

1. Task B start: moved to (-7.6, -7.5), yaw 90 degrees (author's decision, 2026-09-24).
2. Final audit: done (see above).
3. Routes: 30 solved (`logs/thesis/routes/`); all 30 replayed through `ff_fb` arrive inside
   the driveable region (`follower_replay_check.json`); overview `routes_overview.png`.
4. Lockstep qualification (2026-09-24 night; runs in `logs/thesis/qualification/`, the
   pre-fix attempts under `superseded_before_lockstep_fix/`). Four defects found and fixed:
   the manager's decision timer ran on the simulation clock the scheduler holds (deadlock);
   the scheduler's camera phase depended on start-up timing (now aligned to the camera
   instant); the command barrier deadlocked before the first plan (now waits only while the
   controller publishes); after the terminal stop the perception barriers froze the clock
   so rest was never verified (now odometry only). Also: the logger wrote truth after
   closing its file, and had dropped `robot_collision_radius_m` from the run manifest.
   **Not bitwise identical**: three identical runs follow the same path (Hausdorff 5.5 to
   10.5 cm) but differ in timing (mission start is a wall timer; planner and command timers
   run on the simulation clock with start-up-dependent phase; actuation noise is drawn per
   message). Measured by resampling `ground_truth_pose.csv` on time since the first
   command and by path Hausdorff distance; re-measure before quoting. Wall time is about 2
   to 3 min per run.
5. Campaign: done (see Current state).
6. Figures: rebuild from `logs/thesis/`; `make_camera_views.py` must read through
   `pipeline/dataset.py`; maps draw the collision scene from the SDF (the zones omit the
   five loose objects).
7. After the campaign: remove the legacy planner objective and visibility-GP chain, the
   sensor-model v1 reader and the aws/full_4cam worlds (see memory note). (`pipeline_lock.json`,
   the v5 lock chain and the old alignment loader were removed in deletion round 2.)
8. Manuscript (TeX only with explicit approval): Table I's spatial row is the old
   shrink-to-R1 R2; capture statistics were inflated by the broken frames; every number
   must be regenerated from v8; `papers/Thesis/TODO.md` still describes the rejected v7 plan.

## Operating lessons

- Never commit while a campaign or check run is active: the runner freezes the whole-repo
  HEAD and aborts the seed on any commit (it happened once, 2026-09-24 03:02).
- The runner can survive SIGINT during a launch; stop it with SIGTERM, then stop any
  orphaned `ign gazebo`.

- The laptop lid stays open: after a sleep the capture ignores SIGTERM and needs `kill -9`.
- Never `pkill -f PATTERN` with the pattern in your own command line; use `[p]attern`.
- Run a disk guard; a capture writes about 3 MB per pose.
- Before any simulator run: `pgrep -af "ros2 launch|ign gazebo|campaign_runner"`.
- Collision scenes only via `profile_collision_scene` (the raw parser's defaults miss the
  loose objects).

## History (condensed; details in git)

- **2026-09-23 night (v7, rejected).** Thinned dense cells and retrained the detector on
  the thinned set; D_dev correction error rose on identical positions. Rejected; archived in
  `evidence/archive_rejected_20260923`. A finding from the same night: Table I's spatial row
  came from the old shrink-to-R1 R2, not the Bayesian R2 used for navigation.
- **2026-09-23 day (v8).** Uniform top-up (50 positions) and 12 camera-C positions added;
  refit; a first campaign launch failed on stripped planning artifacts and on ground-truth
  termination, both fixed in the config builders.
- **2026-09-23 afternoon (repair).** v5 part 1 was found to hold 4,182 empty frames (robot
  vanished in session 71daa8ab). Re-captured in two passes (a laptop sleep broke the first;
  the disk filled once and was cleared). A hard dataset audit now gates every fit.
- **2026-09-23 evening (gate).** Four gates refitted on identical inference; the author
  chose the gate without edge and size checks over the pre-declared rule's choice.
- **2026-09-23/24 lean-up.** Runtime change set reviewed and committed in six areas;
  1,100+ files outside the dependency closure removed (tag `pre-lean-20260924`); the tree
  renamed to `pipeline/`, `world/`, `figures/`; data moved to `logs/thesis/`; about 15 GB
  of superseded runs deleted.
- **Found and fixed during the lean-up:** GPU training of the correction was not
  reproducible (it flipped R2's selected K) — now CPU and deterministic, and R2's K and l are
  fixed; 12 top-up positions put the robot inside loose objects — dropped, with an audit
  check; the D_dev evaluation scored a second, unused covariance family — now one family; the
  runtime had its own copy of the correction network — now one definition; the follower
  swung about 0.5 m wide at corners, taking 10 of 30 previously solved routes out of the
  driveable region — speed now bounded by the safety margin; last session's "penetration
  with 0 contacts" was an artefact of the old centre-disc check (footprint clearance was
  25.5 cm); contact sensing removed in favour of the offline footprint score; lockstep
  execution adopted.
