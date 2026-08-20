#!/usr/bin/env bash
# Occlusion captures: the case where a detection ARRIVES but is displaced.
#
# Occlusion is usually treated as a coverage problem -- the camera either sees the robot
# or it does not. That is the harmless half. The half that corrupts a belief is PARTIAL
# occlusion: a rack hides where the robot touches the floor while its top is still in
# view, so the detector still returns a box, the box bottom is the bottom of the VISIBLE
# part, and the measurement is displaced with nothing in the data to say so.
#
# Routes come from `check_route_clearance.py`, which reads all 22 collision boxes from the
# world SDF and ray-tests the camera against the robot's contact point and its 0.191 m top:
#
#   graze_aisle_*      68% clean / 32% partial /  0% hidden  -- pure displacement, no dropout
#   cross_aisle_full_* 50% clean / 12% partial / 38% hidden  -- dropout AND displacement
#
# Each is driven in both directions on the SAME line, so the occlusion pattern, ranges and
# bearings are identical to the metre and only the robot's heading changes.
#
# Sequential: `capture_aws_notebook_dataset.sh` aborts if a simulator is already up.
#
#   bash experiments/filter_notebook/capture_occlusion_set.sh
set -uo pipefail

cd "$(dirname "$0")/../.."
REPO="$PWD"
CAPTURE="$REPO/experiments/filter_notebook/capture_aws_notebook_dataset.sh"
LEDGER="$REPO/logs/studies/filter_notebook/OCCLUSION_SET_LEDGER.md"

# Driven at 0.15 m/s to match the four analysis drives, so occlusion is the only thing
# that differs from them. Speed is varied separately, by capture_speed_sweep.sh.
RUNS=(
  "graze_aisle_north"
  "graze_aisle_south"
  "cross_aisle_full_east"
  "cross_aisle_full_west"
)

mkdir -p "$(dirname "$LEDGER")"
{
  echo "# Occlusion set ledger"
  echo
  echo "Started $(date -Is). Protocol: \`capture_aws_notebook_dataset.sh\`, unchanged,"
  echo "0.15 m/s, same world / camera / detector weights / seed as the four analysis"
  echo "drives. Routes chosen by \`check_route_clearance.py\` from the world SDF."
  echo
  echo "| run tag | route | outcome | detections | messages | detection rate | rate/sim s |"
  echo "|---|---|---|---|---|---|---|"
} >"$LEDGER"

for route in "${RUNS[@]}"; do
  tag="aws_${route}"
  echo
  echo "############################################################"
  echo "## $tag"
  echo "############################################################"

  if [ -d "$REPO/logs/studies/filter_notebook/$tag" ]; then
    echo "   already captured; skipping"
    continue
  fi

  ROUTE="$route" SPEED="0.15" RUN_TAG="$tag" bash "$CAPTURE"
  status=$?

  python3 - "$REPO/logs/studies/filter_notebook/$tag" "$tag" "$route" "$status" \
    >>"$LEDGER" <<'PY'
import csv, json, sys
from pathlib import Path
root, tag, route, status = Path(sys.argv[1]), *sys.argv[2:5]
det = msg = 0
rate = pct = 0.0
outcome = "FAILED (capture script exit %s)" % status
perception = root / "raw" / "camera_A_perception.csv"
if perception.is_file():
    rows = list(csv.DictReader(perception.open()))
    msg = len(rows)
    det = sum(1 for r in rows if r.get("detected") == "1")
    pct = 100 * det / msg if msg else 0.0
    stamps = [float(r["diag_stamp"]) for r in rows if r.get("diag_stamp")]
    span = (max(stamps) - min(stamps)) if len(stamps) > 1 else 0.0
    rate = (msg / span) if span else 0.0
done = root / "raw" / "route_completion.json"
if done.is_file():
    outcome = json.loads(done.read_text())["completion_reason"]
elif perception.is_file():
    outcome = "no completion artifact"
print(f"| `{tag}` | {route} | {outcome} | {det} | {msg} | {pct:.0f}% | {rate:.2f} |")
PY
done

echo
echo "== occlusion set done; ledger at $LEDGER"
cat "$LEDGER"
