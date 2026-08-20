#!/bin/bash
# Capture a front/rear marker keypoint dataset by teleporting the robot on a grid.
#
# The robot must be spawned WITH the two marker disks (show_pose_markers:=true),
# otherwise the images contain no markers and every label points at nothing.
# Labels are the analytic projection of the marker centres through the camera
# model that world_profiles.yaml resolves for this world, so the camera pose in
# the world SDF is the definition of the label. Re-capture whenever it moves.
#
# NOTE: no 'set -e' / 'set -u' because ROS setup.bash can trip them.

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/roslog_keypoint_capture}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib_keypoint_capture}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export IGN_IP="${IGN_IP:-127.0.0.1}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
mkdir -p "$ROS_LOG_DIR" "$MPLCONFIGDIR"

WORLD="${WORLD:-warehouse_aws.world.sdf}"
# Every camera is read at the same teleported pose, so the four-camera world
# costs the same teleports as the one-camera world.
CAMERAS="${CAMERAS:-external_camera}"
DATASET="${DATASET:-logs/perception_datasets/projected_keypoint_dataset_aws_v4}"
SAMPLE_NX="${SAMPLE_NX:-14}"
SAMPLE_NY="${SAMPLE_NY:-10}"
YAW_SAMPLES="${YAW_SAMPLES:-12}"
REPEATS="${REPEATS:-1}"
POSITION_JITTER_M="${POSITION_JITTER_M:-0.0}"
YAW_JITTER_RAD="${YAW_JITTER_RAD:-0.0}"
CAPTURE_SESSION_ID="${CAPTURE_SESSION_ID:-session_0}"
PLAN_SEED="${PLAN_SEED:-0}"
# Keep independent restarts from discovering a stale Gazebo server.  Fortress
# consumes IGN_PARTITION; setting the GZ spelling too keeps the script safe if
# the installed ros_gz stack is upgraded later.
export IGN_PARTITION="${IGN_PARTITION:-keypoint_capture_${CAPTURE_SESSION_ID}_$$}"
export GZ_PARTITION="${GZ_PARTITION:-$IGN_PARTITION}"
VAL_FRACTION="${VAL_FRACTION:-0.20}"
SPLIT_MODE="${SPLIT_MODE:-spatial_yaw_bucket}"
SPATIAL_BLOCK_SIZE="${SPATIAL_BLOCK_SIZE:-2}"
ROBOT_Z="${ROBOT_Z:-0.0}"
# 18 px (the script default) throws away everything past ~5.7 m. This world's
# single camera sees 1.4-10 m and the far half is the interesting half, so the
# floor is set to what is still resolvable instead. The box is also tall enough
# to contain the marker disks at z=0.21.
MIN_HITS="${MIN_HITS:-2}"
MIN_BBOX_PX="${MIN_BBOX_PX:-8.0}"
BOX_HEIGHT="${BOX_HEIGHT:-0.23}"
GZLOG="${GZLOG:-/tmp/keypoint_capture_gazebo.log}"
CAPLOG="${CAPLOG:-/tmp/keypoint_capture.log}"

case "$DATASET" in
  /*) DATASET_ABS="$DATASET" ;;
  *)  DATASET_ABS="$ROOT/$DATASET" ;;
esac

if [ -e "$DATASET_ABS" ]; then
  echo "[kp-capture] refusing to overwrite existing dataset: $DATASET_ABS"
  exit 1
fi

echo "[kp-capture] clearing any live simulator"
pkill -9 -f "ign gazebo" >/dev/null 2>&1
pkill -9 -f "gz sim" >/dev/null 2>&1
pkill -9 -f "ruby.*ign" >/dev/null 2>&1
sleep 3

# Cameras b/c/d are unbridged by default, so a four-camera capture sees nothing
# from them unless each is switched on explicitly.
BRIDGE_ARGS=""
for cam in $(echo "$CAMERAS" | tr ',' ' '); do
  case "$cam" in
    external_camera)   BRIDGE_ARGS="$BRIDGE_ARGS bridge_camera_a:=true" ;;
    external_camera_b) BRIDGE_ARGS="$BRIDGE_ARGS bridge_camera_b:=true" ;;
    external_camera_c) BRIDGE_ARGS="$BRIDGE_ARGS bridge_camera_c:=true" ;;
    external_camera_d) BRIDGE_ARGS="$BRIDGE_ARGS bridge_camera_d:=true" ;;
    *) echo "[kp-capture] unknown camera '$cam'"; exit 1 ;;
  esac
done

echo "[kp-capture] bringing up $WORLD with the pose markers ON, cameras: $CAMERAS"
ros2 launch sim bringup_sim.launch.py \
  world:="$WORLD" \
  $BRIDGE_ARGS \
  show_pose_markers:=true \
  use_lidar:=false \
  bridge_scan:=false \
  bridge_contacts:=false \
  bridge_segmentation:=false \
  headless:=true > "$GZLOG" 2>&1 &
GZ=$!

TOPICS=""
for cam in $(echo "$CAMERAS" | tr ',' ' '); do
  TOPICS="$TOPICS /$cam/image_raw"
done
echo "[kp-capture] waiting for$TOPICS (up to 180 s: discovery + first render)"
ok_rgb=0
all_up=1
# Keep one subscriber alive through Gazebo's slow first OGRE render.  Replacing
# it every three seconds can repeatedly miss that first frame even though both
# the native sensor and bridge are healthy.  Image transports are conventionally
# sensor-data QoS, so state it explicitly at this commissioning gate.
for topic in $TOPICS; do
  advertised=0
  for i in $(seq 1 60); do
    ros2 topic list --no-daemon 2>/dev/null | rg -qx "$topic" \
      && advertised=1 && break
    sleep 1
  done
  if [ "$advertised" != "1" ]; then
    all_up=0
    continue
  fi
  timeout 120 ros2 topic echo --once --no-daemon --spin-time 5 \
    --qos-profile sensor_data --field header "$topic" >/dev/null 2>&1 \
    || all_up=0
done
if [ "$all_up" = "1" ]; then ok_rgb=1; fi
if [ "$ok_rgb" != "1" ]; then
  echo "[kp-capture] not every camera published; see $GZLOG"
  kill -9 $GZ >/dev/null 2>&1
  pkill -9 -f "ign gazebo" >/dev/null 2>&1
  exit 1
fi

echo "[kp-capture] capturing ${SAMPLE_NX}x${SAMPLE_NY}x${YAW_SAMPLES} poses -> $DATASET_ABS"
python3 scripts/perception/capture_projected_keypoint_dataset.py \
  --world "$WORLD" \
  --cameras "$CAMERAS" \
  --out "$DATASET_ABS" \
  --sample-nx "$SAMPLE_NX" \
  --sample-ny "$SAMPLE_NY" \
  --yaw-samples "$YAW_SAMPLES" \
  --repeats "$REPEATS" \
  --position-jitter-m "$POSITION_JITTER_M" \
  --yaw-jitter-rad "$YAW_JITTER_RAD" \
  --session-id "$CAPTURE_SESSION_ID" \
  --plan-seed "$PLAN_SEED" \
  --val-fraction "$VAL_FRACTION" \
  --split-mode "$SPLIT_MODE" \
  --spatial-block-size "$SPATIAL_BLOCK_SIZE" \
  --robot-z "$ROBOT_Z" \
  --min-bbox-size-px "$MIN_BBOX_PX" --min-marker-hits "$MIN_HITS" \
  --box-height "$BOX_HEIGHT" \
  --preview-count 120 > "$CAPLOG" 2>&1
RC=$?

echo "[kp-capture] capture exit $RC; tearing down"
kill -9 $GZ >/dev/null 2>&1
pkill -9 -f "ign gazebo" >/dev/null 2>&1
pkill -9 -f "gz sim" >/dev/null 2>&1
tail -20 "$CAPLOG"
exit $RC
