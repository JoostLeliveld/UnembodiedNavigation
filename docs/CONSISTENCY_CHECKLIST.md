# Consistency Checklist: World ↔ YOLO ↔ GP ↔ Costmap ↔ Tasks

Four-way dependency chain. Editing any node without re-deriving downstream nodes
silently breaks the planner. The root `planner-diagnostician` and
`rollout-runner` agents use this checklist before a rollout is treated as more
than diagnostic.

```
              ┌────────────┐         ┌──────────┐
              │   World    │────────▶│  YOLO    │   (camera + lighting + obstacle
              │   SDF      │         │  model   │    geometry → detector training)
              └─────┬──────┘         └────┬─────┘
                    │                     │
                    ▼                     ▼
              ┌────────────────────────────────┐
              │     GP visibility artifact     │   geometry_json + P_*_map
              │  (logs/.../aws_gp_v*/*.npz)    │   captured by teleporting robot
              └─────┬──────────────────────────┘   through poses and scoring YOLO
                    │
                    ▼
              ┌────────────┐         ┌───────────────────┐
              │  Costmap   │◀────────│  world_profiles   │  (descriptive only,
              │  (nogo)    │         │  known_2d_regions │   not enforced; used for
              └────────────┘         └───────────────────┘   visualisation + sanity)
                    ▲
                    │
              ┌──────────┐
              │ tasks.   │   (start, goal must be in green region)
              │ yaml     │
              └──────────┘
```

---

## Hard checks (every one MUST pass before any new Gazebo campaign)

### C1 — GP geometry_json matches World SDF collision prisms

The GP artifact embeds a JSON snapshot of the world's collision boxes
(`geometry_json` in the `.npz`). If the world SDF has moved an obstacle since the GP was
captured, every plan against this GP plans against a phantom.

**Verify**:
```bash
python3 - <<'PY'
import json, re, numpy as np
gp = np.load('logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz')
geom = json.loads(str(gp['geometry_json']).strip("[]'"))
sdf = open('src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf').read()

mismatches = []
for p in geom['prisms']:
    name_short = p['name'].split('/')[-1].split(':')[0]
    cx = (p['xmin']+p['xmax'])/2; cy = (p['ymin']+p['ymax'])/2
    # SDF line should contain "<link name="<name_short>">...<pose>cx cy ...</pose>"
    m = re.search(rf'<link name="{re.escape(name_short)}"><pose>([^<]+)</pose>', sdf)
    if not m:
        mismatches.append(f'{name_short}: link missing in SDF')
        continue
    parts = m.group(1).split()
    sdf_cx = float(parts[0]); sdf_cy = float(parts[1])
    if abs(sdf_cx - cx) > 0.01 or abs(sdf_cy - cy) > 0.01:
        mismatches.append(f'{name_short}: GP=({cx:.2f},{cy:.2f}) SDF=({sdf_cx:.2f},{sdf_cy:.2f})')
print('OK' if not mismatches else 'STALE — recapture GP:\n  ' + '\n  '.join(mismatches))
PY
```

**Pass criterion**: prints `OK`. Anything else means the GP is stale and must be
recaptured before any new campaign.

**Historical stale case**: older AWS GP artifacts were captured against earlier
R4/high-stack geometries and earlier camera poses. The current paper GP is
`aws_gp_v7b` (camera z=4.8, y=-5.5); v5/v6/v6b are superseded. Any future AWS
geometry or camera change requires a fresh capture and GP fit.

### C2 — GP camera_pos equals World SDF camera pose

The capture was done from a specific camera position. If the SDF (or the launch tf) has
moved the camera, the GP is no longer valid.

**Verify**:
```bash
python3 - <<'PY'
import numpy as np
gp = np.load('logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz')
print('GP camera_pos:', gp['camera_pos'])
print('GP camera_pose (full RPY):', gp['camera_pose'])
PY
grep -A1 'camera\|cam_pos' src/experiments/config/world_profiles.yaml | head -20
```

**Pass criterion**: GP `camera_pos[:3]` matches the world's camera xyz to within 0.01 m.

### C3 — Costmap consistency reduces to GP consistency

In `base_planner.py:222–234`, the `NogoZoneCostModel` is built from the
`visibility_geometry_json` argument passed through launch params, which the launch graph
sources from the GP artifact (`geometry_json` field). Therefore: **if C1 passes, the
costmap is automatically consistent**. No separate check needed.

Code reference: `src/planning/planning/planners/base_planner.py:216–234` (NogoCostConfig
construction), `src/experiments/experiments/core/visibility_launch_common.py:498–525`
(camera_params + collision_geometry_json plumbing).

### C4 — world_profiles known_2d_regions consistent with the SDF

The green-driveable / red-staging rectangles in `world_profiles.yaml ::
known_2d_regions` are descriptive (not enforced by the planner). They are used for:
- visualisation overlay in `plot_world_geometry.py`,
- sanity checks on task start/goal placement (C5 below).

**Verify** (visually):
```bash
python3 scripts/visibility_comparison/plot_world_geometry.py \
  --out /tmp/world_geometry_check.png
```
Open the figure. Every rack body should be inside (or border) a green-outlined
traversable region. Every red-hatched staging pad should NOT overlap a traversable
rectangle except minimally at the boundary.

**Pass criterion**: no traversable region cuts through a rack body silently; every
shelf-end pad sits at the south face of its corresponding lower rack.

### C5 — Every task's start and goal lie inside a green region

For each task in `src/experiments/config/tasks.yaml`, the `start: {x, y}` and
`goal: {x, y}` must both lie inside at least one `known_2d_regions[type=traversable]`
rectangle and must NOT lie inside any `[type=non_driveable_staging]` rectangle.

**Verify**:
```bash
python3 - <<'PY'
import yaml
prof = yaml.safe_load(open('src/experiments/config/world_profiles.yaml'))
tasks = yaml.safe_load(open('src/experiments/config/tasks.yaml'))
for world, world_cfg in (prof['worlds'] or {}).items():
    regions = (world_cfg.get('known_2d_regions') or [])
    tr = [r for r in regions if 'traversable' in str(r.get('type',''))]
    nd = [r for r in regions if 'non_driveable' in str(r.get('type',''))]
    def in_rect(x, y, r):
        return r['xmin'] <= x <= r['xmax'] and r['ymin'] <= y <= r['ymax']
    for task in (tasks['tasks'] or {}).get(world, []) or []:
        name = task['name']
        for kind, pt in (('start', task.get('start',{})), ('goal', task.get('goal',{}))):
            x, y = float(pt.get('x', 1e9)), float(pt.get('y', 1e9))
            in_tr = any(in_rect(x, y, r) for r in tr)
            in_nd = any(in_rect(x, y, r) for r in nd)
            if (not in_tr) or in_nd:
                flag = 'NOT in green' if not in_tr else 'inside red staging'
                print(f'{world} :: {name} :: {kind} ({x}, {y}) — {flag}')
print('done')
PY
```

**Pass criterion**: only `done` is printed. Any other line is a misplaced task that must
be fixed (or the regions extended to cover it).

### C6 — YOLO model manifest matches the current world appearance

There is no programmatic check that the YOLO model
(`logs/perception_models/aws_yolo_simseg_v2/`) was trained on imagery from the current
world. **Manual gate**:

- If you change lighting, the rack colours, the floor texture, or add new visually
  distinct elements: assume YOLO is stale and verify on a fresh capture (run a short
  Gazebo session, see if YOLO's detection score / mask quality on the robot matches
  expectations).
- If you only moved an existing obstacle (e.g., the R4 stack move): YOLO is probably
  fine because the trained model recognises the robot, not the obstacles. The GP needs
  recapturing because the SHADOW PATTERN changed; YOLO does not.

Document any change in this manifest format:
```yaml
yolo_model_dir: logs/perception_models/aws_yolo_simseg_v2/
trained_against_world_sdf_sha: <sha256 of world SDF when training set was captured>
notes: |
  Trained on captures with R4 stack at y=0 (pre-fix). Robot detection unchanged
  by the R4-stack move; mask quality at the robot itself is geometry-independent.
```

Currently this manifest does not exist as a file in the repo. **Action item** (not in
this checklist's auto-verify): create `logs/perception_models/aws_yolo_simseg_v2/MANIFEST.yaml`
documenting the training-world correspondence.

---

## Recipe — re-deriving GP after a world geometry edit (~30–40 min)

```bash
# 0. Build/source workspace
cd /home/joostleliveld/Thesis/UnembodiedNavigation
source install/setup.bash

# 1. Capture (Gazebo runs; ~15–20 min for the default 24x20x4 grid)
python3 scripts/visibility_comparison/capture_visibility_samples.py \
  --world warehouse_aws.world.sdf \
  --out logs/visibility_comparison/aws_capture_v5 \
  --sample-nx 24 --sample-ny 20 --yaw-samples 4 --target-height-m 0.35

# 2. Run YOLO over the captured frames
python3 scripts/visibility_comparison/extract_perception_targets.py \
  --capture-dir logs/visibility_comparison/aws_capture_v5 \
  --out logs/visibility_comparison/aws_targets_v5 \
  --model logs/perception_models/aws_yolo_simseg_v2/model.pt --conf-threshold 0.10

# 3. Aggregate detections per pose into GP targets
python3 scripts/visibility_comparison/build_gp_targets.py \
  --perception-targets logs/visibility_comparison/aws_targets_v5/perception_targets.csv \
  --out logs/visibility_comparison/aws_gp_targets_v5

# 4. Fit the GP and write the .npz with embedded geometry_json.
#    Use the LOCKED params (length_scale 0.90, noise_var 0.05, beta 0.5) and that world's
#    own targets/capture. The active artifact aws_gp_v7b was fitted from the v7 aggregated
#    targets PLUS an added A0 west-corridor column (x=-4.61); see docs/decision_log.md
#    (2026-06-11). v5/v6/v6b/v7 capture+target dirs are archived under _archive_nonpaper/.
python3 scripts/visibility_comparison/fit_visibility_gps.py \
  --gp-targets logs/visibility_comparison/<world>_gp_targets/gp_targets_xy_aggregated.csv \
  --capture-manifest logs/visibility_comparison/<world>_capture/capture_manifest.json \
  --out logs/visibility_comparison/aws_gp_v7b \
  --grid-nx 220 --grid-ny 200 --gp-length-scale 0.90 --gp-noise-var 0.05 --beta 0.5

# 5. Point the configs at the new artifact
sed -i 's|OLD_AWS_GP_DIR|NEW_AWS_GP_DIR|g' \
  scripts/visibility_comparison/aws_smoke_config.yaml \
  scripts/visibility_comparison/aws_campaign_config.yaml

# 6. Re-run C1 (the GP↔SDF check above). Must print OK.
```

---

## What this checklist deliberately does NOT cover

- The numeric value of any planner hyperparameter — see `docs/PLANNER_HYPERPARAMETERS.md`.
- The pass/fail of any specific campaign result — see `docs/experiment_registry.md`
  and the registered run directory.
- The thesis claim being supported by the data — see
  `~/.claude/projects/-home-joostleliveld-Thesis/memory/project_thesis_stability_claim.md`.
