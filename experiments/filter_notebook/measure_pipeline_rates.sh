#!/usr/bin/env bash
# Measure the rates the pipeline actually achieves on this machine. No drive, no
# dataset, no truth -- just: how fast does the clock advance, how fast do the
# cameras render, and how many of those frames become observations?
#
# Two launch defaults are suspected of throttling the detector when it runs on CPU:
#   yolo_max_pending_wall_s = 0.50  vs  1.1 s measured CPU inference per batch
#   yolo_max_batch_stamp_skew_s = 0.10  vs  cameras at 5 Hz (200 ms apart)
# Pass SKEW/PENDING to compare settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$PWD"

# -f, not -a: `ign` is a Ruby script, so a live simulator has comm=`ruby` and the
# name-matching form of this guard never fired. See capture_notebook_dataset.sh.
live_processes() {
  pgrep -af "ign gazebo|gzserver|gz sim|ros2 launch" 2>/dev/null \
    | grep -vE "(^| )(bash|sh|pgrep|grep) " || true
}
if [ -n "$(live_processes)" ]; then
  echo "ABORT: something is already running:" >&2; live_processes >&2; exit 1
fi

SKEW="${SKEW:-0.10}"
PENDING="${PENDING:-0.50}"
WINDOW="${WINDOW:-40}"
MODE="${MODE:-batched}"   # batched | separate
STARTUP_S="${STARTUP_S:-130}"
OUT="$REPO/logs/studies/filter_notebook/rate_probe_${MODE}_skew${SKEW}_pending${PENDING}"
mkdir -p "$OUT"

MODEL="$REPO/logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"
export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-71}" IGN_PARTITION="${IGN_PARTITION:-rate_probe}"

set +u; source "$REPO/install/setup.bash"; set -u

if [ "$MODE" = "separate" ]; then
  # One detector per camera. Keeps the no-reuse and strictly-newer-stamp guards,
  # drops only the four-way simultaneity requirement -- which a filter that
  # consumes detections sequentially, in stamp order, never uses. 12 cores, so
  # 2 threads each leaves headroom for Gazebo and the bridges.
  MODE_ARGS="yolo_batched_four_camera:=false yolo_device:=cpu \
  yolo_cpu_num_threads_camera_a:=2 yolo_cpu_num_threads_camera_b:=2 \
  yolo_cpu_num_threads_camera_c:=2 yolo_cpu_num_threads_camera_d:=2"
else
  MODE_ARGS="yolo_batched_four_camera:=true yolo_batched_device:=cpu \
  yolo_batched_cpu_num_threads:=4 \
  yolo_max_batch_stamp_skew_s:=$SKEW yolo_max_pending_wall_s:=$PENDING"
fi

echo "== launching mode=$MODE (skew=$SKEW, pending=$PENDING, CPU, imgsz 960)"
setsid ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  world:=warehouse_full_4cam.world.sdf world_name:=warehouse_full_4cam \
  headless:=true spawn_x:=-1.5 spawn_y:=-7.2 spawn_yaw:=1.5708 \
  yolo_model:="$MODEL" yolo_imgsz:=960 \
  yolo_conf_threshold:=0.05 yolo_iou_threshold:=0.45 \
  yolo_use_masks:=false yolo_use_torchscript:=false \
  $MODE_ARGS \
  >"$OUT/launch.log" 2>&1 &
LAUNCH_PID=$!
trap 'pkill -TERM -g "$(ps -o pgid= -p $LAUNCH_PID | tr -d " ")" 2>/dev/null || true' EXIT

echo "== waiting ${STARTUP_S}s for startup"
for _ in $(seq 1 "$STARTUP_S"); do sleep 1; done

echo "== measuring for ${WINDOW}s"
python3 - "$OUT" "$WINDOW" <<'PY'
import sys, time, json
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String

out, window = Path(sys.argv[1]), float(sys.argv[2])
IMAGES = {"camera_A": "/external_camera/image_raw", "camera_B": "/external_camera_b/image_raw",
          "camera_C": "/external_camera_c/image_raw", "camera_D": "/external_camera_d/image_raw"}

rclpy.init()
node = Node("rate_probe")
qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
counts = {f"image::{k}": 0 for k in IMAGES}
counts.update({f"obs::{k}": 0 for k in IMAGES})
counts.update({f"obs_valid::{k}": 0 for k in IMAGES})
sim = {"first": None, "last": None}

def on_clock(msg):
    t = msg.clock.sec + msg.clock.nanosec * 1e-9
    if sim["first"] is None:
        sim["first"] = t
    sim["last"] = t

node.create_subscription(Clock, "/clock", on_clock, qos)
for cam, topic in IMAGES.items():
    node.create_subscription(Image, topic,
                             lambda _m, _c=cam: counts.__setitem__(f"image::{_c}", counts[f"image::{_c}"] + 1), qos)

def on_obs(msg, cam):
    counts[f"obs::{cam}"] += 1
    try:
        if json.loads(msg.data).get("detection_valid"):
            counts[f"obs_valid::{cam}"] += 1
    except Exception:
        pass

for cam in IMAGES:
    node.create_subscription(String, f"/perception/camera_observation/{cam}",
                             lambda m, _c=cam: on_obs(m, _c), qos)

start = time.time()
while rclpy.ok() and time.time() - start < window:
    rclpy.spin_once(node, timeout_sec=0.1)
wall = time.time() - start
span = (sim["last"] - sim["first"]) if sim["first"] is not None else 0.0
rtf = span / wall if wall else 0.0

report = {"wall_s": round(wall, 1), "sim_span_s": round(span, 1), "rtf": round(rtf, 3)}
print(f"\n  wall {wall:.0f} s | sim {span:.1f} s | RTF {rtf:.2f}\n")
print(f"  {'camera':10s}{'image Hz sim':>14s}{'obs Hz sim':>12s}{'valid obs':>11s}")
for cam in IMAGES:
    ihz = counts[f"image::{cam}"] / span if span else 0.0
    ohz = counts[f"obs::{cam}"] / span if span else 0.0
    print(f"  {cam.replace('camera_',''):10s}{ihz:>14.2f}{ohz:>12.2f}{counts[f'obs_valid::{cam}']:>11d}")
    report[cam] = {"image_hz_sim": round(ihz, 3), "obs_hz_sim": round(ohz, 3),
                   "images": counts[f"image::{cam}"], "observations": counts[f"obs::{cam}"],
                   "valid": counts[f"obs_valid::{cam}"]}
(out / "rates.json").write_text(json.dumps(report, indent=2))
node.destroy_node(); rclpy.shutdown()
PY

echo
echo "== expired-pending warnings in the detector log:"
grep -c "async_pending_expired\|expired pending" "$OUT/launch.log" || echo "  0"
echo "== wrote $OUT/rates.json"
