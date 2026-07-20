#!/usr/bin/env bash
# Clean-CPU staleness test.
#
# PURPOSE: re-measure the camera->belief staleness (corr_age) on an UNLOADED
# machine, to test whether the ~0.8s staleness is just CPU contention from the
# IDE/browser/dev-env rather than an algorithmic bug.
#
# HOW TO USE (do NOT run this from inside VSCode):
#   1. Save your work and CLOSE VSCode (this ends the Claude session - that's expected).
#   2. CLOSE the browser (firefox) and any other heavy apps.
#   3. Open a plain terminal (Ctrl+Alt+T) and run:
#         bash ~/Thesis/UnembodiedNavigation/scripts/visibility_comparison/run_clean_cputest.sh
#   4. Wait ~5 min. It prints the verdict at the end (corr_age clean vs contended).
#   5. Reopen VSCode/Claude afterwards if you want me to dig further.

set +e
ROOT="/home/joostleliveld/Thesis/UnembodiedNavigation"
cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }

echo "================================================================"
echo " CLEAN-CPU STALENESS TEST"
echo "================================================================"
echo "Current load average (should be LOW now that the IDE is closed):"
uptime
echo
echo "Heaviest processes right now (sanity-check the IDE/browser are gone):"
top -b -n1 2>/dev/null | head -12 | tail -6
echo

# --- source ROS + workspace ---
source /opt/ros/humble/setup.bash 2>/dev/null
source "$ROOT/install/setup.bash" 2>/dev/null

# --- clean any stragglers ---
pkill -9 -f "ign gazebo" 2>/dev/null; pkill -9 -f parameter_bridge 2>/dev/null
pkill -9 -f run_visibility_campaign 2>/dev/null; sleep 2

echo "Launching one C2 run (route_apron_to_a3_mid, seed 1) on the clean machine..."
echo "(this is the same config as the contended baseline, for apples-to-apples)"
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config scripts/visibility_comparison/_timingfix.yaml \
  --log-root logs/visibility_comparison/_cleancpu \
  --run-timeout 320 --first-cmd-timeout 280

# --- analyze corr_age (camera->belief staleness) ---
echo
echo "================================================================"
echo " VERDICT"
echo "================================================================"
python3 - <<'PY'
import pandas as pd, numpy as np, glob
def corr_age(pat):
    fs=glob.glob(pat)
    if not fs: return None,0
    e=pd.read_csv(sorted(fs)[-1], na_values=['NaN','nan',''], low_memory=False)
    c=e[e.get('planner_pixel_correction_available',0)==1]['planner_pixel_correction_age_s'].dropna()
    b=e[e['stamp']>= (e[(e['cmd_v'].abs()>1e-4)]['stamp'].iloc[0] if (e['cmd_v'].abs()>1e-4).any() else e['stamp'].iloc[0])]
    return (float(c.median()) if len(c) else None,
            float(b['truth_belief_error_m'].median()) if 'truth_belief_error_m' in b else None,
            len(c))
clean=corr_age(f"/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/_cleancpu/route_apron_to_a3_mid/C2/seed1/experiment_*/experiment.csv")
cont =corr_age(f"/home/joostleliveld/Thesis/UnembodiedNavigation/logs/visibility_comparison/_validate_c2/route_apron_to_a3_mid/C2/seed1/experiment_*/experiment.csv")
print(f"  CONTENDED (IDE+browser open) baseline: corr_age med = {cont[0]}s   belief_err = {cont[1]}m")
print(f"  CLEAN     (this run)                 : corr_age med = {clean[0]}s   belief_err = {clean[1]}m")
print()
if clean[0] is not None and cont[0] is not None:
    if clean[0] < 0.6*cont[0]:
        print("  => CPU contention WAS the dominant cause. Run real experiments on a clean machine.")
    elif clean[0] < 0.85*cont[0]:
        print("  => CPU contention contributes, but staleness remains high -> also needs the GPU-render / node fixes.")
    else:
        print("  => staleness barely changed -> NOT mainly CPU contention; it's the render/transport path itself.")
else:
    print("  (run did not produce corrections - check the run completed)")
PY
echo "Done. Reopen VSCode/Claude and share this output if you want the next step."
