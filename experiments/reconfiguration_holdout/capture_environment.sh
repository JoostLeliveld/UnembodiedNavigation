#!/bin/bash
# Capture one environment of the reconfiguration holdout: real Gazebo, four
# cameras, one teleport grid, then the frozen detector scored offline on the saved
# frames.
#
#   capture_environment.sh <env_key> <world_file> [n_yaws] [position_list_json]
#
# The study environments differ only in the world file: identical camera poses,
# identical grid, identical detector and identical detector thresholds.  Those
# thresholds are the ones the L0 reference capture
# (logs/visibility_comparison/commissioning_grid_20260807) recorded in its own
# manifest, restated here as constants so a new environment cannot be scored under
# a different detector configuration than the environment it is compared against.
#
# No 'set -e/-u': sourcing the ROS setup scripts trips both.
cd "$(dirname "$0")/../.."
REPO="$(pwd)"

ENV_KEY="${1:?usage: capture_environment.sh <env_key> <world_file> [n_yaws] [pos_json]}"
WORLD="${2:?missing world file}"
NYAWS="${3:-4}"
POSJSON="${4:-}"

# Capture datasets belong under logs/visibility_comparison/ with the other capture
# datasets, not under logs/studies/: the capture tool enforces that root, and the
# repo convention is that raw captures are append-only there while a study folder
# holds only what analysis derives from them.
OUT="$REPO/logs/visibility_comparison/recfg_holdout_$ENV_KEY"
# Logs live OUTSIDE the capture directory.  capture_visibility_samples.py calls
# safe_reset_generated_dir on its --out, which deletes and recreates that directory
# after this script has already opened a log inside it: the file is unlinked, the
# shell keeps writing into a deleted inode, and the diagnostic for a failed capture
# is gone exactly when it is needed.  Learned the hard way on a 90-minute run.
LOGDIR="$REPO/logs/visibility_comparison/recfg_holdout_logs"
mkdir -p "$LOGDIR"
PROFILES="$REPO/experiments/reconfiguration_holdout/world_profiles_variants.yaml"
DETECTOR="$REPO/logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"
GZLOG="$LOGDIR/${ENV_KEY}_gz_bringup.log"
L2_CONTRACT="$REPO/experiments/reconfiguration_holdout/frozen/L2_capture_contract.json"
L2_CONTRACT_TOOL="$REPO/experiments/reconfiguration_holdout/freeze_l2_capture_contract.py"

# Frozen detector configuration, from the L0 capture's manifest.json.
CONF=0.01
IOU=0.45
IMGSZ=640

# Grid geometry, from the L0 capture's capture_manifest.json visibility_bounds.
# Overridable only so a pilot can run a coarse grid; the study captures use the
# defaults, which are the L0 reference grid exactly.
NX=${RH_NX:-46}
NY=${RH_NY:-36}
WALL_MARGIN=${RH_WALL_MARGIN:-0.45}

echo "[$ENV_KEY] === reconfiguration-holdout capture ==="
echo "[$ENV_KEY] world      : $WORLD"
echo "[$ENV_KEY] out        : ${OUT#$REPO/}"
echo "[$ENV_KEY] grid       : ${NX}x${NY}, wall margin ${WALL_MARGIN} m, ${NYAWS} yaws, 4 cameras"

# L2 is the prospective replication. Refuse any pilot override or drift before the
# capture tool resets its output directory, and verify every frozen input hash.
if [ "$ENV_KEY" = "L2" ]; then
  if [ "$WORLD" != "warehouse_full_4cam_recfg2.world.sdf" ] || \
     [ "$NYAWS" != "4" ] || [ "$NX" != "46" ] || [ "$NY" != "36" ] || \
     [ "$WALL_MARGIN" != "0.45" ] || [ -n "$POSJSON" ]; then
    echo "[L2] ABORT: frozen design requires recfg2 world, 46x36 grid, margin 0.45, four yaws, no position override"
    exit 3
  fi
  python3 "$L2_CONTRACT_TOOL" check-inputs --contract "$L2_CONTRACT"
  CONTRACT_RC=$?
  if [ "$CONTRACT_RC" != "0" ]; then
    echo "[L2] ABORT: frozen input contract failed before capture"; exit "$CONTRACT_RC"
  fi
fi

# A second simulator on the same partition corrupts both runs, so refuse rather
# than race.  Two traps, both hit here already: this repo runs Ignition Fortress
# ('ign gazebo', launched through a ruby wrapper), NOT 'gz sim', so a pattern that
# only looks for 'gz sim' finds nothing and leaves a live server behind; and a
# pattern loose enough to match this script's own shell command line reports a
# false positive.  Match the simulator's own argv, and exclude our own processes.
live_sim() {
  pgrep -af "ign gazebo -r|gz sim -r|run_visibility_campaign" \
    | grep -v "capture_environment" | grep -v "snapshot-bash" | grep -v "pgrep"
}
LIVE=$(live_sim)
if [ -n "$LIVE" ]; then
  echo "[$ENV_KEY] ABORT: a simulator is already running:"; echo "$LIVE" | cut -c1-140; exit 2
fi

# Teardown has to take the bridges down too.  Killing only the server leaves ~8
# ros_gz_bridge processes advertising the old world's topics, and the next run
# then sees stale /external_camera topics and captures frames from nothing.
teardown() {
  pkill -9 -f "ign gazebo" >/dev/null 2>&1
  pkill -9 -f "gz sim"     >/dev/null 2>&1
  pkill -9 -f "ros_gz_bridge/parameter_bridge" >/dev/null 2>&1
  # The sim package's own nodes outlive the launch too: a clock_throttle_node from a
  # finished capture was still running twenty minutes later.
  pkill -9 -f "sim/lib/sim/clock_throttle_node" >/dev/null 2>&1
  pkill -9 -f "robot_state_publisher" >/dev/null 2>&1
  kill "$LAUNCH_PID" >/dev/null 2>&1
  sleep 4
}

mkdir -p "$OUT"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID=91

echo "[$ENV_KEY] 1/4 bringing up Gazebo (headless, four camera bridges)"
ros2 launch sim bringup_sim.launch.py \
  world:="$WORLD" headless:=true show_pose_markers:=false use_lidar:=false \
  bridge_camera_a:=true bridge_camera_b:=true bridge_camera_c:=true bridge_camera_d:=true \
  > "$GZLOG" 2>&1 &
LAUNCH_PID=$!

ok=0
for i in $(seq 1 40); do
  a=0; b=0; c=0; d=0
  timeout 4 ros2 topic echo --once /external_camera/image_raw   >/dev/null 2>&1 && a=1
  timeout 4 ros2 topic echo --once /external_camera_b/image_raw >/dev/null 2>&1 && b=1
  timeout 4 ros2 topic echo --once /external_camera_c/image_raw >/dev/null 2>&1 && c=1
  timeout 4 ros2 topic echo --once /external_camera_d/image_raw >/dev/null 2>&1 && d=1
  if [ "$a$b$c$d" = "1111" ]; then ok=1; break; fi
  sleep 3
done
if [ "$ok" != "1" ]; then
  echo "[$ENV_KEY] ABORT: four camera streams did not all come up (a=$a b=$b c=$c d=$d)"
  tail -25 "$GZLOG"; teardown; exit 1
fi
echo "[$ENV_KEY] all four camera streams live"

echo "[$ENV_KEY] 2/4 capturing teleport grid"
POSARG=()
if [ -n "$POSJSON" ]; then POSARG=(--position-list-json "$POSJSON"); fi
python3 scripts/visibility_comparison/capture_visibility_samples.py \
  --world "$WORLD" --world-profiles "$PROFILES" --out "$OUT" \
  --sample-nx "$NX" --sample-ny "$NY" --wall-margin-m "$WALL_MARGIN" \
  --yaw-samples "$NYAWS" --no-previews \
  --extra-cameras "external_camera_b=/external_camera_b/image_raw,external_camera_c=/external_camera_c/image_raw,external_camera_d=/external_camera_d/image_raw" \
  "${POSARG[@]}" > "$LOGDIR/${ENV_KEY}_capture.log" 2>&1
CAP_RC=$?
NIMG=$(find "$OUT/images" -iname '*.jpg' 2>/dev/null | wc -l)
echo "[$ENV_KEY] capture exit=$CAP_RC, images=$NIMG"
teardown
if [ ! -f "$OUT/capture_manifest.json" ]; then
  echo "[$ENV_KEY] ABORT: no capture_manifest.json"; tail -25 "$LOGDIR/${ENV_KEY}_capture.log"; exit 1
fi
if [ "$ENV_KEY" = "L2" ] && [ "$CAP_RC" != "0" ]; then
  echo "[L2] ABORT: capture process exited $CAP_RC"; tail -25 "$LOGDIR/${ENV_KEY}_capture.log"; exit "$CAP_RC"
fi

# Fail closed before spending GPU time on detector inference.  A nearly complete
# capture is not a complete experimental unit, and stale camera frames cannot be
# repaired offline because the row-level source image is no longer trustworthy.
python3 - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = json.loads((out / "capture_manifest.json").read_text(encoding="utf-8"))
heading = manifest.get("heading_sampling") or {}
n_positions = int(heading.get("position_count", 0))
n_headings = int(heading.get("yaw_samples", heading.get("samples_per_xy", 0)))
n_cameras = 1 + len(manifest.get("extra_camera_frames") or [])
expected = n_positions * n_headings * n_cameras
sample_count = int(manifest.get("sample_count", -1))
n_stale = int(manifest.get("n_stale_views", len(manifest.get("stale_views") or [])))
samples_path = out / "samples.csv"
rows = -1
if samples_path.is_file():
    with samples_path.open(encoding="utf-8") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
problems = []
if expected <= 0:
    problems.append(f"invalid expected membership {expected}")
if sample_count != expected:
    problems.append(f"manifest sample_count {sample_count} != expected {expected}")
if rows != expected:
    problems.append(f"samples.csv rows {rows} != expected {expected}")
if n_stale != 0:
    problems.append(f"stale camera views {n_stale} != 0")
if problems:
    print("[capture gate] FAIL: " + "; ".join(problems), file=sys.stderr)
    raise SystemExit(3)
print(f"[capture gate] PASS: exact {expected} rows, {n_cameras} cameras, zero stale views")
PY
GATE_RC=$?
if [ "$GATE_RC" != "0" ]; then
  echo "[$ENV_KEY] ABORT: capture membership/staleness gate failed"; exit "$GATE_RC"
fi
if [ "$ENV_KEY" = "L2" ]; then
  python3 "$L2_CONTRACT_TOOL" validate-capture \
    --contract "$L2_CONTRACT" --capture-dir "$OUT" --stage samples
  CONTRACT_RC=$?
  if [ "$CONTRACT_RC" != "0" ]; then
    echo "[L2] ABORT: exact frozen sample membership failed"; exit "$CONTRACT_RC"
  fi
fi

echo "[$ENV_KEY] 3/4 scoring the saved frames with the frozen detector (offline)"
python3 scripts/visibility_comparison/extract_perception_targets.py \
  --capture-dir "$OUT" --model "$DETECTOR" --device 0 \
  --imgsz "$IMGSZ" --conf-threshold "$CONF" --iou-threshold "$IOU" \
  --target-class robot --class-id 0 --no-use-masks \
  --out "$OUT" > "$LOGDIR/${ENV_KEY}_detect.log" 2>&1
DET_RC=$?
echo "[$ENV_KEY] detector exit=$DET_RC"
if [ ! -f "$OUT/perception_targets.csv" ]; then
  echo "[$ENV_KEY] ABORT: no perception_targets.csv"; tail -25 "$LOGDIR/${ENV_KEY}_detect.log"; exit 1
fi
if [ "$ENV_KEY" = "L2" ]; then
  if [ "$DET_RC" != "0" ]; then
    echo "[L2] ABORT: detector process failed"; exit "$DET_RC"
  fi
  python3 "$L2_CONTRACT_TOOL" validate-capture \
    --contract "$L2_CONTRACT" --capture-dir "$OUT" --stage perception
  CONTRACT_RC=$?
  if [ "$CONTRACT_RC" != "0" ]; then
    echo "[L2] ABORT: exact frozen detector membership failed"; exit "$CONTRACT_RC"
  fi
fi

echo "[$ENV_KEY] 4/4 appearance features from the same frames"
python3 experiments/reconfiguration_holdout/appearance_features.py --capture-dir "$OUT" \
  > "$LOGDIR/${ENV_KEY}_appearance.log" 2>&1
APP_RC=$?
echo "[$ENV_KEY] appearance exit=$APP_RC"

python3 - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader((out / "perception_targets.csv").open()))
hit = sum(1 for r in rows if str(r.get("yolo_detected_after_threshold", "0")).strip() in ("1", "1.0", "True", "true"))
print(f"[{out.name}] {len(rows)} samples, detector hit rate {hit / max(len(rows), 1):.4f}")
PY
echo "[$ENV_KEY] DONE  (frames kept under ${OUT#$REPO/}/images -- prune_images.py releases the disk)"
