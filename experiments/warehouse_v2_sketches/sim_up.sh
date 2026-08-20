#!/bin/bash
# Bring warehouse_v2 up headless with all five camera bridges + the overview
# camera, wait for every stream, grab one frame each, build a contact sheet,
# then tear everything down.  Usage: sim_up.sh <tag>
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
TAG="${1:-iter}"
WORLD="${2:-warehouse_v2.world.sdf}"
WORLD_NAME="${WORLD%.world.sdf}"
WORLD_NAME="${WORLD_NAME%.sdf}"
LOG_DIR="${WAREHOUSE_V2_LOG_DIR:-/tmp/warehouse_v2_sim}"
LOG="${LOG_DIR}/wh_v2_${TAG}.log"
mkdir -p "$(dirname "$LOG")"

# This repo runs Ignition Fortress ('ign gazebo'), not 'gz sim'.  Match the
# simulator's own argv and exclude our own processes, or a loose pattern reports
# a false positive against this script's command line.
LIVE=$(pgrep -af "ign gazebo -r|gz sim -r|run_visibility_campaign" | grep -v sim_up | grep -v pgrep)
if [ -n "$LIVE" ]; then echo "ABORT: simulator already running:"; echo "$LIVE" | cut -c1-140; exit 2; fi

teardown() {
  pkill -9 -f "ign gazebo" >/dev/null 2>&1
  pkill -9 -f "gz sim" >/dev/null 2>&1
  pkill -9 -f "ros_gz_bridge/parameter_bridge" >/dev/null 2>&1
  pkill -9 -f "sim/lib/sim/clock_throttle_node" >/dev/null 2>&1
  pkill -9 -f "robot_state_publisher" >/dev/null 2>&1
  kill "$PLAN_PID" >/dev/null 2>&1
  kill "$LAUNCH_PID" >/dev/null 2>&1
  sleep 3
}
trap teardown EXIT

source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID=91

echo "[$TAG] bringing up $WORLD headless"
ros2 launch sim bringup_sim.launch.py \
  world:="$WORLD" world_name:="$WORLD_NAME" headless:=true show_pose_markers:=false \
  spawn_x:=${SPAWN_X:-0.55} spawn_y:=${SPAWN_Y:--7.50} spawn_z:=0.05 spawn_yaw:=1.5708 \
  use_lidar:=false bridge_scan:=false bridge_contacts:=false \
  bridge_camera_a:=true bridge_camera_b:=true bridge_camera_c:=true \
  bridge_camera_d:=true bridge_camera_e:=true bridge_overview_camera:=true \
  > "$LOG" 2>&1 &
LAUNCH_PID=$!

# The plan-view checking camera is not in the launch's camera registry on
# purpose: it is a verification aid, not a sixth localisation camera. Bridge it
# here so it lives and dies with this script.
sleep 8
ros2 run ros_gz_bridge parameter_bridge \
  /plan_view_camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image \
  >> "$LOG" 2>&1 &
PLAN_PID=$!

ok=0
for i in $(seq 1 30); do
  n=0
  for t in /external_camera/image_raw /external_camera_b/image_raw /external_camera_c/image_raw \
           /external_camera_d/image_raw /external_camera_e/image_raw \
           /presentation_overview_camera/image_raw /plan_view_camera/image_raw; do
    timeout 3 ros2 topic echo --once "$t" >/dev/null 2>&1 && n=$((n+1))
  done
  echo "[$TAG] streams up: $n/7"
  if [ "$n" = "7" ]; then ok=1; break; fi
  sleep 3
done
if [ "$ok" != "1" ]; then echo "[$TAG] ABORT: not all streams came up"; tail -30 "$LOG"; exit 1; fi

python3 experiments/warehouse_v2_sketches/grab_frames.py \
  --out experiments/warehouse_v2_sketches/frames --tag "$TAG" --timeout 60
python3 experiments/warehouse_v2_sketches/sheet.py "$TAG"
