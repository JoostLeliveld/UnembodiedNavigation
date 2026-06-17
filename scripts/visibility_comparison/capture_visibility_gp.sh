#!/bin/bash
# Capture the GP calibration set with the selected detector so the C2
# reliability field matches the detector used in runtime. Capture is light
# (Gazebo + camera + teleport; the detector is scored OFFLINE on the saved images), so it
# also produces the GP-capture-video frames in $CAP/images/.
# No 'set -e/-u' (ROS setup.bash trips them).
cd "$(dirname "$0")/../.."
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID=88
DETECTOR=logs/perception_models/warehouse_yolo_detector_v1/model.pt
CAP=logs/visibility_comparison/warehouse_gp_capture_v1
TGT=logs/visibility_comparison/warehouse_gp_targets_v1
GP=logs/visibility_comparison/warehouse_gp_v1
rm -rf "$CAP" "$TGT" "$GP"

echo "[gp] 1/5 bringing up Gazebo (warehouse_aws, camera + set_pose)"
pkill -9 -f "ign gazebo" >/dev/null 2>&1; sleep 3
# headless:=true is REQUIRED: without it bringup_sim tries to open the Gazebo GUI
# and the camera sensor never renders / the bridge never publishes image_raw.
ros2 launch sim bringup_sim.launch.py world:=warehouse_aws.world.sdf headless:=true show_pose_markers:=false use_lidar:=false > /tmp/gp_gz.log 2>&1 &
ok=0
for i in $(seq 1 30); do
  if timeout 4 ros2 topic echo --once /external_camera/image_raw >/dev/null 2>&1; then ok=1; break; fi
  sleep 3
done
if [ "$ok" != "1" ]; then echo "[gp] ABORT: no /external_camera/image_raw. gz log:"; tail -15 /tmp/gp_gz.log; pkill -9 -f "ign gazebo"; exit 1; fi

echo "[gp] 2/5 capturing teleport grid (15x15x4)"
python3 scripts/visibility_comparison/capture_visibility_samples.py --world warehouse_aws.world.sdf --out "$CAP" > /tmp/gp_capture.log 2>&1
echo "[gp] capture exit=$?; images: $(find "$CAP/images" -iname '*.jpg' 2>/dev/null | wc -l)"
pkill -9 -f "ign gazebo" >/dev/null 2>&1; sleep 3
if [ ! -f "$CAP/capture_manifest.json" ]; then echo "[gp] ABORT: no capture_manifest.json"; tail -20 /tmp/gp_capture.log; exit 1; fi

echo "[gp] 3/5 scoring captured images with the detector (offline)"
python3 scripts/visibility_comparison/extract_perception_targets.py --capture-dir "$CAP" --model "$DETECTOR" --device 0 --out "$TGT" > /tmp/gp_targets.log 2>&1
echo "[gp] extract exit=$?"
if [ ! -f "$TGT/perception_targets.csv" ]; then echo "[gp] ABORT: no perception_targets.csv"; tail -20 /tmp/gp_targets.log; exit 1; fi

echo "[gp] 4/5 building gp targets"
python3 scripts/visibility_comparison/build_gp_targets.py --perception-targets "$TGT/perception_targets.csv" --out "$TGT" > /tmp/gp_build.log 2>&1
echo "[gp] build exit=$?"
if [ ! -f "$TGT/gp_targets_xy_aggregated.csv" ]; then echo "[gp] ABORT: no gp_targets_xy_aggregated.csv"; tail -20 /tmp/gp_build.log; exit 1; fi

echo "[gp] 5/5 fitting GP"
python3 scripts/visibility_comparison/fit_visibility_gps.py \
  --gp-targets "$TGT/gp_targets_xy_aggregated.csv" --capture-manifest "$CAP/capture_manifest.json" \
  --out "$GP" --grid-nx 220 --grid-ny 200 --gp-length-scale 0.90 --gp-noise-var 0.05 --beta 0.5 > /tmp/gp_fit.log 2>&1
echo "[gp] fit exit=$?; artifact: $(ls "$GP"/*.npz 2>/dev/null)"
echo "[gp] DONE"
