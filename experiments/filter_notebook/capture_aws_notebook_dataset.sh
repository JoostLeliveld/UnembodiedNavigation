#!/usr/bin/env bash
# Capture ONE single-camera drive in the AWS warehouse, for the filter notebooks.
#
# The four-camera counterpart is `capture_notebook_dataset.sh`. This is the same
# protocol in the world where method development belongs (research/06_world_camera_design.md),
# with one camera instead of four:
#
#   views/camera_A/   1280x720 PNG + index.csv                 (record_multicamera_views)
#   raw/              odom_noisy + camera_A_perception.csv      (record_demonstration_capture)
#   evaluation_only/  ground_truth.csv                          (same recorder)
#   raw/capture_manifest.json   world, camera, detector, route, seed, RTF
#
# One route per invocation, because the notebook needs several DIFFERENT drives: three to
# commission the observation noise on and a fourth, held out of that fit, to analyse.
#
#   ROUTE=aisle_east_north bash experiments/filter_notebook/capture_aws_notebook_dataset.sh
#
# Evidence role is `diagnostic`: a demonstration capture, never campaign evidence.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"

STUDY="$REPO/experiments/filter_notebook/config/aws_study.yaml"
ROUTE="${ROUTE:-aisle_east_north}"
SEED="${SEED:-20260813}"
SPEED="${SPEED:-0.15}"

# ---- hard rule: never launch on top of a live run.
#
# `pgrep -a PATTERN` matches the process NAME, and Ignition's `ign` launcher is a Ruby
# script, so a running simulator appears as comm=`ruby` and that guard passes vacuously
# while orphaned servers pile up at ~1.2 GiB of VRAM each. `-f` matches the full command
# line, which is what this always needed.
SIM_PATTERN="ign gazebo|gzserver|gz sim|ros2 launch|run_visibility_campaign"
live_processes() {
  pgrep -af "$SIM_PATTERN" 2>/dev/null | grep -vE "(^| )(bash|sh|pgrep|grep) " || true
}
if [ -n "$(live_processes)" ]; then
  echo "ABORT: something is already running:" >&2
  live_processes >&2
  exit 1
fi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 9999)
if [ "$FREE_MIB" -lt 2000 ]; then
  echo "ABORT: only ${FREE_MIB} MiB of VRAM free; need ~2000 (Gazebo + detector + headroom)" >&2
  nvidia-smi >&2
  exit 1
fi
echo "== ${FREE_MIB} MiB VRAM free"

# ---- route geometry comes from the study file, so the spawn and the driver cannot drift
read -r START_X START_Y START_YAW GOAL_X GOAL_Y LENGTH < <(python3 - "$STUDY" "$ROUTE" <<'PY'
import math, sys, yaml
study, name = sys.argv[1], sys.argv[2]
routes = yaml.safe_load(open(study))["collection"]["routes"]
route = next((r for r in routes if r["name"] == name), None)
if route is None:
    raise SystemExit(f"unknown route {name!r}; have: "
                     + ", ".join(r["name"] for r in routes))
s = route["start"]
# A route is either start+goal (a straight line) or start+waypoints (a polyline, used by
# the turning routes). Report the last point as the destination and the POLYLINE length,
# so the echoed distance is what the robot will actually drive.
if route.get("waypoints"):
    pts = [(s["x"], s["y"])] + [(w["x"], w["y"]) for w in route["waypoints"]]
else:
    g = route["goal"]
    pts = [(s["x"], s["y"]), (g["x"], g["y"])]
length = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
print(s["x"], s["y"], s["yaw"], pts[-1][0], pts[-1][1], length)
PY
)
echo "== route $ROUTE: ($START_X, $START_Y) -> ($GOAL_X, $GOAL_Y), ${LENGTH} m at ${SPEED} m/s"

RUN_TAG="${RUN_TAG:-aws_${ROUTE}_$(date +%Y%m%d_%H%M%S)}"
OUT="$REPO/logs/studies/filter_notebook/$RUN_TAG"

# The clean 2026-06-17 retrain, captured and trained in THIS world with an occlusion gate
# on the labels. NOT the four-camera v3 model: that one was trained on the flagship
# world's four viewpoints. imgsz must match training (960) or box placement degrades.
MODEL="$REPO/logs/perception_models/warehouse_yolo_detector_v1/model.pt"
IMGSZ=960
CONF=0.05

STARTUP_S="${STARTUP_S:-100}"
RECORD_S="${RECORD_S:-1800}"

mkdir -p "$OUT"/{raw,views,evaluation_only}
LOGS="$OUT/orchestration"; mkdir -p "$LOGS"

export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}" IGN_PARTITION="${IGN_PARTITION:-aws_notebook_capture}"

set +u                       # colcon's setup.bash reads COLCON_TRACE unguarded
# shellcheck disable=SC1091
source "$REPO/install/setup.bash"
set -u

echo "== launching the AWS world (headless), one camera, spawn at the route start"
setsid ros2 launch experiments warehouse_aws_notebook_capture.launch.py \
  world:=warehouse_aws.world.sdf world_name:=warehouse_aws headless:=true \
  spawn_x:=$START_X spawn_y:=$START_Y spawn_yaw:=$START_YAW \
  use_encoder_noise:=true encoder_noise_seed:=$SEED seed:=$SEED \
  yolo_model:="$MODEL" yolo_imgsz:=$IMGSZ yolo_device:=0 \
  yolo_conf_threshold:=$CONF yolo_iou_threshold:=0.45 \
  camera_observation_r_visible_uv:=2.5 camera_observation_r_miss_uv:=40.0 \
  >"$LOGS/launch.log" 2>&1 &
LAUNCH_PGID=$!

# The simulator ignores SIGINT and has outlived SIGTERM on every earlier abort, which is
# how orphaned servers got created. Escalate, then verify the VRAM came back.
teardown() {
  echo "== tearing down"
  local pgid
  pgid=$(ps -o pgid= -p "$LAUNCH_PGID" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && pkill -TERM -g "$pgid" 2>/dev/null || true
  for _ in $(seq 1 8); do
    [ -z "$(live_processes)" ] && break
    sleep 1
  done
  if [ -n "$(live_processes)" ]; then
    echo "   SIGTERM was not enough; killing:"; live_processes
    [ -n "$pgid" ] && pkill -KILL -g "$pgid" 2>/dev/null || true
    pkill -KILL -f "ign gazebo" 2>/dev/null || true
    sleep 3
  fi
  echo "   VRAM free after teardown: $(nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader 2>/dev/null || echo unknown)"
}
trap teardown EXIT

echo "== waiting ${STARTUP_S}s for the simulator and the detector"
for _ in $(seq 1 "$STARTUP_S"); do sleep 1; done
ros2 topic list >"$LOGS/topics.txt" 2>&1 || true
if ! grep -q "camera_observation" "$LOGS/topics.txt"; then
  echo "ABORT: /perception/camera_observation/camera_A never appeared; see launch.log" >&2
  tail -30 "$LOGS/launch.log" >&2
  exit 1
fi
echo "   observation topic is up"

# ---- buy observation rate by slowing the simulator, not the detector
#
# The filter consumes observations per SIMULATED second, and
#     obs per sim second = (inferences per wall second) / RTF
# so throttling raises it, capped by the camera's own 5 Hz. `max_step_size` stays at
# 0.001 s, so the physics integration is identical step for step and only wall pacing
# changes; it is a runtime service call, so the world file is untouched. The cost is wall
# time, which for an offline dataset is free.
#
# One camera renders and one image per inference, against four of each in the flagship
# world, so this needs far less throttling than that capture did. Set from the pilot.
RTF="${RTF:-0.5}"
echo "== throttling the simulator to ${RTF}x real time"
ign service -s "/world/warehouse_aws/set_physics" \
  --reqtype ignition.msgs.Physics --reptype ignition.msgs.Boolean --timeout 5000 \
  --req "max_step_size: 0.001, real_time_factor: $RTF" 2>&1 | head -2

# Written BEFORE the drive, not after: a run that aborts half way still has to be
# identifiable, and the notebook loader reads this file to decide which world -- and
# therefore which cameras -- a capture belongs to.
python3 - "$OUT" "$ROUTE" "$MODEL" "$IMGSZ" "$CONF" "$SEED" "$SPEED" "$RTF" <<'PY'
import json, sys
from pathlib import Path
out, route, model, imgsz, conf, seed, speed, rtf = sys.argv[1:9]
Path(out, "raw", "capture_manifest.json").write_text(json.dumps({
    "world": "warehouse_aws.world.sdf",
    "world_name": "warehouse_aws",
    "cameras": ["camera_A"],
    "camera_world_includes": {"camera_A": "external_camera"},
    "image_topics": {"camera_A": "/external_camera/image_raw"},
    "route": route,
    "detector_model": model,
    "detector_imgsz": int(imgsz),
    "detector_conf": float(conf),
    "encoder_noise_seed": int(seed),
    "speed_mps": float(speed),
    "real_time_factor": float(rtf),
    "evidence_role": "diagnostic",
}, indent=2) + "\n", encoding="utf-8")
PY

echo "== starting the frame recorder"
python3 scripts/reliability/record_multicamera_views.py \
  --out-dir "$OUT/views" \
  --camera camera_A=/external_camera/image_raw \
  --every "${VIEW_EVERY:-2}" --duration-s "$RECORD_S" \
  >"$LOGS/views.log" 2>&1 &
VIEWS=$!

echo "== starting the demonstration recorder (observations + odometry + truth)"
python3 experiments/filter_notebook/record_demonstration_capture.py \
  --out-dir "$OUT" --duration-s "$RECORD_S" --cameras camera_A \
  --odom-origin-x-m=$START_X --odom-origin-y-m=$START_Y --odom-origin-yaw-rad=$START_YAW \
  >"$LOGS/operational.log" 2>&1 &
OPERATIONAL=$!

sleep 5
echo "== driving $ROUTE"
# Two things about this driver, both measured rather than assumed:
#
#  * Its default sim deadline is checked against `now_sim - started_at_s`, but it takes
#    `started_at_s` from the first odometry callback, which fires before /clock arrives,
#    so it records 0.0 and the check silently becomes one on ABSOLUTE sim time. Startup
#    already eats most of a default budget, so the limits are passed explicitly.
#  * **It does not exit after the route completes.** It logs `route_complete`, writes the
#    completion artifact, and then keeps spinning, so running it in the foreground hangs
#    this script until the outer timeout fires. Run it in the background, wait for the
#    artifact it promises, and then kill it. SIGTERM is not enough -- it needs SIGKILL.
COMPLETION="$OUT/raw/route_completion.json"
rm -f "$COMPLETION"
python3 experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py \
  --study "$STUDY" --route "$ROUTE" --speed-mps $SPEED --start-hold-s 5 \
  --max-sim-runtime-s 900 --max-wall-runtime-s 2400 \
  --manifest-path "$OUT/raw/route_manifest.json" \
  --completion-path "$COMPLETION" \
  >"$LOGS/drive.log" 2>&1 &
DRIVER=$!

DRIVE_DEADLINE=$((SECONDS + ${DRIVE_TIMEOUT_S:-1500}))
while [ ! -f "$COMPLETION" ]; do
  if ! kill -0 $DRIVER 2>/dev/null; then
    echo "   driver exited without writing a completion artifact; see drive.log"
    break
  fi
  if [ "$SECONDS" -ge "$DRIVE_DEADLINE" ]; then
    echo "   ABORT: route did not complete within ${DRIVE_TIMEOUT_S:-1500}s of wall time"
    break
  fi
  sleep 2
done
if [ -f "$COMPLETION" ]; then
  echo "   $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['completion_reason'])" "$COMPLETION")"
fi
kill -KILL $DRIVER 2>/dev/null || true
wait $DRIVER 2>/dev/null || true

# The driver logs completion, writes its artifact and then does NOT exit, so an unbounded
# wait here idles forever. Close the recorders as soon as the drive returns.
echo "== closing the recorders"
kill -INT $OPERATIONAL $VIEWS 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 $OPERATIONAL 2>/dev/null || break
  sleep 1
done
kill -TERM $OPERATIONAL $VIEWS 2>/dev/null || true
wait $OPERATIONAL 2>/dev/null || echo "   operational recorder exited non-zero"
wait $VIEWS 2>/dev/null || true


echo
echo "== capture written to $OUT"
echo "   frames: $(ls "$OUT/views/camera_A"/*.png 2>/dev/null | wc -l)"
wc -l "$OUT"/raw/*.csv "$OUT"/evaluation_only/*.csv 2>/dev/null || true
python3 - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = list(csv.DictReader((out / "raw" / "camera_A_perception.csv").open()))
hit = sum(1 for r in rows if r["detected"] == "1")
stamps = [float(r["diag_stamp"]) for r in rows if r["diag_stamp"]]
span = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0
print(f"   camera_A: {hit} detections in {len(rows)} messages "
      f"({100 * hit / max(len(rows), 1):.0f}%), "
      f"{len(rows) / span if span else 0:.2f} messages per simulated second")
done = out / "raw" / "route_completion.json"
print(f"   route: {json.loads(done.read_text())['completion_reason'] if done.is_file() else 'DID NOT COMPLETE'}")
PY
