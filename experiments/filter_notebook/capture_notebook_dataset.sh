#!/usr/bin/env bash
# Capture the notebook dataset: camera FRAMES alongside detections, odometry and truth.
#
# The three existing filter captures logged pixels but not images, so the
# perception stage could never be shown. This records all four streams against one
# Gazebo session, on one clock:
#
#   views/            1280x720 PNG per camera + index.csv   (record_multicamera_views)
#   raw/              odom_noisy + per-camera obs_u/obs_v    (record_operational_logs)
#   evaluation_only/  ground_truth.csv                      (record_evaluation_truth)
#   evaluation_inputs/  the joined layout rcond_common reads (attach_evaluation_truth)
#
# Frozen flagship world, one full central-aisle traverse through the A/C -> B/D
# handover band. Evidence role is `diagnostic`: this is a demonstration capture and
# must never be presented as campaign evidence.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"

# ---- hard rule: never launch on top of a live run
#
# `pgrep -a PATTERN` matches the process NAME, and Ignition's `ign` launcher is a
# Ruby script, so a running simulator appears as comm=`ruby`. The pattern below
# never matched it: the guard silently passed while three orphaned servers
# accumulated, 1255 MiB of VRAM each, filling a 4 GiB card. That is what caused the
# detector's CUDA OOM and Gazebo's own fall back to software rendering. `-f`
# matches the full command line, which is what this always needed.
SIM_PATTERN="ign gazebo|gzserver|gz sim|ros2 launch|run_visibility_campaign"
live_processes() {
  pgrep -af "$SIM_PATTERN" 2>/dev/null | grep -vE "(^| )(bash|sh|pgrep|grep) " || true
}
if [ -n "$(live_processes)" ]; then
  echo "ABORT: something is already running:" >&2
  live_processes >&2
  echo "Free the GPU first, or these will starve the detector of VRAM." >&2
  exit 1
fi

# A 4 GiB card cannot host the desktop, four 720p Gazebo renders and the detector at
# once. Refuse to start unless most of it is free, rather than discover it in a
# warmup traceback 130 s later.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 9999)
if [ "$FREE_MIB" -lt 2600 ]; then
  echo "ABORT: only ${FREE_MIB} MiB of VRAM free; need ~2600 (Gazebo 1255 + detector ~200 + headroom)" >&2
  nvidia-smi >&2
  exit 1
fi
echo "== ${FREE_MIB} MiB VRAM free"

RUN_TAG="${RUN_TAG:-notebook_$(date +%Y%m%d_%H%M%S)}"
OUT="$REPO/logs/studies/filter_notebook/$RUN_TAG"
MODEL="$REPO/logs/perception_models/warehouse_yolo_detector_4cam_v3_960/model.pt"
IMGSZ=960                                   # from the model manifest, not the README
SEED=20260811

# south_to_north_handover, from config/study.yaml
START_X=-1.5; START_Y=-7.2; START_YAW=1.5708
ROUTE=south_to_north_handover
# Settings from measurement (rate probes 2026-08-11), and from one correction:
#
#   * The detector runs on the GPU, as the frozen config and the historical
#     captures did: 0.082 s per four-image batch (12.2 Hz wall) using 194 MiB.
#     An earlier CUDA OOM led to a CPU detour; that reading was taken seconds
#     after a failed teardown while dying processes still held VRAM, and was
#     wrong. On CPU the pipeline yielded 0.33 Hz per camera against the 4.99 Hz
#     the historical captures achieved, because Gazebo also lost hardware
#     rendering (`libEGL: failed to create dri2 screen`) and four software 720p
#     renders starved everything of CPU.
#   * The reference: smoke1/smoke2 produced observation messages at 4.99-5.00 Hz
#     simulated -- one per rendered frame, 1:1 with the 5 Hz camera config. That
#     is the rate this capture should reproduce.
#   * Batcher defaults (skew 0.10 s, pending 0.50 s) are right again once
#     inference is 0.08 s rather than 1.1 s, so they are left frozen.
SPEED=0.15
YOLO_DEVICE=0
CPU_THREADS=4

STARTUP_S="${STARTUP_S:-130}"               # detectors need ~90 s to come up
# The traverse is 14.4 m. The robot tracked ~0.074 m/s against 0.15 commanded in the
# 2026-07-21 runs (a ~1 Hz control loop), so allow for the slow case: 195 s of sim,
# which at RTF 0.15 is ~1300 s of wall. Generous, because the recorders are stopped
# as soon as the drive returns rather than running their duration out.
RECORD_S="${RECORD_S:-1800}"

mkdir -p "$OUT"/{raw,views,evaluation_only}
LOGS="$OUT/orchestration"; mkdir -p "$LOGS"

export ROS_LOCALHOST_ONLY=1 IGN_IP=127.0.0.1 GZ_IP=127.0.0.1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-71}" IGN_PARTITION="${IGN_PARTITION:-notebook_capture}"

# colcon's setup.bash reads COLCON_TRACE unguarded, so -u has to stand down here
set +u
# shellcheck disable=SC1091
source "$REPO/install/setup.bash"
set -u

echo "== launching the frozen four-camera world (headless), spawn at the route start"
setsid ros2 launch experiments warehouse_full4cam_commissioning.launch.py \
  world:=warehouse_full_4cam.world.sdf world_name:=warehouse_full_4cam \
  headless:=true \
  spawn_x:=$START_X spawn_y:=$START_Y spawn_yaw:=$START_YAW \
  use_encoder_noise:=true encoder_noise_seed:=$SEED \
  yolo_model:="$MODEL" yolo_imgsz:=$IMGSZ \
  yolo_conf_threshold:=0.05 yolo_iou_threshold:=0.45 \
  yolo_use_masks:=false yolo_use_torchscript:=false \
  yolo_batched_four_camera:=true \
  yolo_batched_device:=$YOLO_DEVICE \
  yolo_batched_cpu_num_threads:=$CPU_THREADS \
  camera_observation_r_visible_uv:=2.5 camera_observation_r_miss_uv:=40.0 \
  >"$LOGS/launch.log" 2>&1 &
LAUNCH_PGID=$!

# The simulator ignored SIGINT and outlived SIGTERM on every earlier abort, which is
# how the orphans above were created. Escalate, then verify the VRAM came back.
teardown() {
  echo "== tearing down"
  local pgid
  pgid=$(ps -o pgid= -p "$LAUNCH_PGID" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && pkill -TERM -g "$pgid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8; do
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

echo "== waiting ${STARTUP_S}s for the simulator and four detectors"
for _ in $(seq 1 "$STARTUP_S"); do sleep 1; done
ros2 topic list >"$LOGS/topics.txt" 2>&1 || true
echo "   camera observation topics seen:"
grep -c "camera_observation" "$LOGS/topics.txt" || true

COMPLETION="$OUT/raw/route_completion.json"

# ---- buy back observation rate by slowing the simulator, not the detector
#
# The detector is inference-bound: one four-image batch costs ~0.8-0.9 s of WALL
# time, whatever else happens (measured 2026-07-21 at imgsz 640 and again today at
# 960). At RTF ~0.7 that is 1.1 observations per SIMULATED second per camera, and
# the earlier session established there is no way round it at full quality --
# imgsz 416 reaches 2.42 Hz but detection collapses to one camera of four.
#
# The rate the filter actually cares about is per simulated second, and that equals
# (batches per wall second) / RTF. So throttling the simulator raises it, capped by
# the 5 Hz cameras. Measured live today: RTF 0.244 -> 3.14 Hz sim on all four
# cameras, a 2.9x gain with the detector, model, imgsz and world all unchanged.
#
# This does not alter the data. `max_step_size` stays 0.001 s, so the physics
# integration is identical step for step; only wall-clock pacing changes. It is a
# runtime service call, so the frozen world file is untouched. The cost is wall
# time, which for an offline dataset is free.
# 0.25 gives the measured 3.14 Hz sim per camera, which is the rate chosen for the
# notebook: ~3x the unthrottled rate, and a ~13 min capture rather than ~22. Going to
# 0.15 would reach the 5 Hz camera ceiling if a denser dataset were ever wanted.
RTF="${RTF:-0.25}"
echo "== throttling the simulator to ${RTF}x real time (raises observations per simulated second)"
ign service -s "/world/warehouse_full_4cam/set_physics" \
  --reqtype ignition.msgs.Physics --reptype ignition.msgs.Boolean --timeout 5000 \
  --req "max_step_size: 0.001, real_time_factor: $RTF" 2>&1 | head -2

echo "== starting the frame recorder"
python3 scripts/reliability/record_multicamera_views.py \
  --out-dir "$OUT/views" \
  --camera camera_A=/external_camera/image_raw \
  --camera camera_B=/external_camera_b/image_raw \
  --camera camera_C=/external_camera_c/image_raw \
  --camera camera_D=/external_camera_d/image_raw \
  --every 4 --duration-s "$RECORD_S" \
  >"$LOGS/views.log" 2>&1 &
VIEWS=$!

echo "== starting the demonstration recorder (observations + odometry)"
python3 experiments/filter_notebook/record_demonstration_capture.py \
  --out-dir "$OUT" --duration-s "$RECORD_S" \
  --odom-origin-x-m=$START_X --odom-origin-y-m=$START_Y --odom-origin-yaw-rad=$START_YAW \
  >"$LOGS/operational.log" 2>&1 &
OPERATIONAL=$!

TRUTH=""   # truth is written by the demonstration recorder above

sleep 5
echo "== driving $ROUTE at ${SPEED} m/s"
# The driver's default sim deadline is route length / speed + 30 s, compared against
# `now_sim - started_at_s`. It sets `started_at_s` from the first odometry callback,
# which fires before the node has received `/clock`, so it records 0.0 and the check
# silently becomes one on ABSOLUTE simulated time. We spend ~99 s of sim waiting for
# the detectors to come up, so only ~32 s of the 131 s budget was left and the route
# aborted at 4.2 m of 14.4 m. Passing the limits explicitly, sized for the absolute
# clock and for RTF ~0.7, is the fix that does not reach into the shared driver.
python3 experiments/multicamera_commissioning_bigwarehouse/tools/drive_study_route.py \
  --route "$ROUTE" --speed-mps $SPEED --start-hold-s 5 \
  --max-sim-runtime-s 900 --max-wall-runtime-s 2400 \
  --manifest-path "$OUT/raw/route_manifest.json" \
  --completion-path "$COMPLETION" \
  >"$LOGS/drive.log" 2>&1 || echo "   driver exited non-zero; see drive.log"

# The recorders are given a duration longer than the drive can possibly need, so
# that a slow traverse is never truncated. Once the drive returns there is nothing
# left to record, so ask them to finish rather than sitting out the remainder --
# an unbounded `wait` here is what left the previous run idling for minutes after
# the route had already aborted.
echo "== closing the recorders"
kill -INT $OPERATIONAL $VIEWS 2>/dev/null || true
for _ in $(seq 1 30); do
  kill -0 $OPERATIONAL 2>/dev/null || break
  sleep 1
done
kill -TERM $OPERATIONAL $VIEWS 2>/dev/null || true
wait $OPERATIONAL 2>/dev/null || echo "   operational recorder exited non-zero"
wait $VIEWS 2>/dev/null || true


echo "== joining truth onto the operational log"
python3 experiments/multicamera_commissioning_bigwarehouse/tools/attach_evaluation_truth.py \
  --raw-dir "$OUT/raw" \
  --truth-csv "$OUT/evaluation_only/ground_truth.csv" \
  --out-dir "$OUT/evaluation_inputs" \
  >"$LOGS/attach.log" 2>&1 || echo "   attach exited non-zero; see attach.log"

echo
echo "== capture written to $OUT"
for c in A B C D; do
  n=$(ls "$OUT/views/camera_$c"/*.png 2>/dev/null | wc -l)
  echo "   camera_$c: $n frames"
done
wc -l "$OUT"/raw/*.csv "$OUT"/evaluation_only/*.csv 2>/dev/null || true
