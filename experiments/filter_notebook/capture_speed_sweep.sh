#!/usr/bin/env bash
# Speed sweep: the same two routes driven at warehouse speeds instead of 0.15 m/s.
#
# Every capture in `logs/studies/filter_notebook/aws_*` so far ran at 0.15 m/s, which is
# a walking-pace crawl no real AMR does. A warehouse AMR cruises at 0.5 to 1.5 m/s. Speed
# changes two things that matter and nothing else:
#
#   * how many looks the camera gets per metre driven -- the detector is capped at 5 Hz,
#     so at 1.5 m/s the robot travels 30 cm between consecutive sightings;
#   * how much the robot moves between the image being formed and the observation being
#     consumed, which turns any latency into an along-track displacement.
#
# It does NOT introduce motion blur: Gazebo renders instantaneous frames with no shutter
# model, so that error source stays zero by construction here and has to be stated as a
# limit rather than measured.
#
# Runs strictly sequentially: `capture_aws_notebook_dataset.sh` aborts if a simulator is
# already up, and two Gazebos on one GPU corrupt both captures.
#
#   bash experiments/filter_notebook/capture_speed_sweep.sh
set -uo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
CAPTURE="$REPO/experiments/filter_notebook/capture_aws_notebook_dataset.sh"
LEDGER="$REPO/logs/studies/filter_notebook/SPEED_SWEEP_LEDGER.md"

# route:speed pairs. 0.15 m/s already exists for both routes and is not repeated.
RUNS=(
  "aisle_east_north:0.50"
  "aisle_east_north:1.00"
  "aisle_east_north:1.50"
  "apron_west_to_east:0.50"
  "apron_west_to_east:1.00"
)

mkdir -p "$(dirname "$LEDGER")"
{
  echo "# Speed sweep ledger"
  echo
  echo "Started $(date -Is). Protocol: \`capture_aws_notebook_dataset.sh\`, unchanged."
  echo "Only \`SPEED\` varies. Same world, camera, detector weights and seed as the"
  echo "0.15 m/s captures, so speed is the only thing that differs."
  echo
  echo "| run tag | route | speed m/s | outcome | detections | messages | rate/sim s |"
  echo "|---|---|---|---|---|---|---|"
} >"$LEDGER"

for entry in "${RUNS[@]}"; do
  route="${entry%%:*}"
  speed="${entry##*:}"
  tag="aws_${route}_v$(printf '%03d' "$(echo "$speed * 100" | bc | cut -d. -f1)")"

  echo
  echo "############################################################"
  echo "## $tag  --  $route at $speed m/s"
  echo "############################################################"

  if [ -d "$REPO/logs/studies/filter_notebook/$tag" ]; then
    echo "   already captured; skipping"
    continue
  fi

  ROUTE="$route" SPEED="$speed" RUN_TAG="$tag" bash "$CAPTURE"
  status=$?

  python3 - "$REPO/logs/studies/filter_notebook/$tag" "$tag" "$route" "$speed" "$status" \
    >>"$LEDGER" <<'PY'
import csv, json, sys
from pathlib import Path
root, tag, route, speed, status = Path(sys.argv[1]), *sys.argv[2:6]
det = msg = 0
rate = 0.0
outcome = "FAILED (capture script exit %s)" % status
perception = root / "raw" / "camera_A_perception.csv"
if perception.is_file():
    rows = list(csv.DictReader(perception.open()))
    msg = len(rows)
    det = sum(1 for r in rows if r.get("detected") == "1")
    stamps = [float(r["diag_stamp"]) for r in rows if r.get("diag_stamp")]
    span = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0
    rate = (msg / span) if span else 0.0
done = root / "raw" / "route_completion.json"
if done.is_file():
    outcome = json.loads(done.read_text())["completion_reason"]
elif perception.is_file():
    outcome = "no completion artifact"
print(f"| `{tag}` | {route} | {speed} | {outcome} | {det} | {msg} | {rate:.2f} |")
PY
done

echo
echo "== speed sweep done; ledger at $LEDGER"
cat "$LEDGER"
