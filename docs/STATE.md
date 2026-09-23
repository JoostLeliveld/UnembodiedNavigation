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
- **Final audit, routes, campaign**: not run yet.
- **Repo**: lean layout, one version of every artifact, tests green; tag
  `pre-lean-20260924` holds everything removed during the lean-up.

## Open items, in order

1. Task B start: (-7.6, -8.5) sees only camera B at the start heading; the belief needs two
   cameras. Proposal: move it to (-7.6, -7.5), yaw 90 degrees (cameras B, C, E), and shift
   the first waypoint of the eastbound seeds to y = -7.5. Needs the author's decision.
2. Final audit: `pipeline/audit_protocol.py`, `final_audit.py`, `final_audit_fusion.py`.
3. Routes: `pipeline/campaign_configs.py`, `routes.sh`; replay every solved route through
   `ff_fb` inside the driveable region; `plot_routes.py` for review.
4. Lockstep qualification: one route end to end; the same task and seed twice must give an
   identical trajectory; measure wall-clock per run; define the run time limit in
   simulated seconds.
5. Campaign manifest locking every input by path and hash; then the campaign, seed by seed,
   with a disk guard and keep-awake (laptop lid open).
6. Figures: rebuild from `logs/thesis/`; `make_camera_views.py` must read through
   `pipeline/dataset.py`; maps draw the collision scene from the SDF (the zones omit the
   five loose objects).
7. After the campaign: remove the legacy planner objective and visibility-GP chain, the
   sensor-model v1 reader and the aws/full_4cam worlds (see memory note), `pipeline_lock.json`,
   the v5 lock and its auditor.
8. Manuscript (TeX only with explicit approval): Table I's spatial row is the old
   shrink-to-R1 R2; capture statistics were inflated by the broken frames; every number
   must be regenerated from v8; `papers/Thesis/TODO.md` still describes the rejected v7 plan.

## Operating lessons

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
