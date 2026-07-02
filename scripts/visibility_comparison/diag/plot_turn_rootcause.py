#!/usr/bin/env python3
"""Why the robot loses track in a turn: BOTH the measurement and the heading degrade.

Pools the 5 west_upper C2 runs. Left: camera (detector) error vs belief error by
turn rate -- the detector is ~7x worse in turns AND the belief diverges. Right:
the heading dead-reckons in turns (no camera yaw correction), which compounds.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

ROOT = 'logs/visibility_comparison/robustness_keepin_clean_20260619/route_west_to_a1_upper/C2'
def n(d, c): return pd.to_numeric(d.get(c), errors='coerce')
bins = [0, 0.2, 0.5, 0.7, 1.01]; lab = ['straight\n<0.2', '0.2-0.5', '0.5-0.7', 'hard turn\n>=0.7']

DET = []; BEL = []
for seed in range(5):
    ef = glob.glob(f'{ROOT}/seed{seed}/experiment_*/experiment.csv'); pf = glob.glob(f'{ROOT}/seed{seed}/experiment_*/perception.csv')
    if not ef or not pf: continue
    e = pd.read_csv(ef[0], low_memory=False); p = pd.read_csv(pf[0], low_memory=False)
    ct = n(e, 'first_crash_stamp').dropna(); ct = float(ct.iloc[0]) if len(ct) else np.inf
    et = n(e, 'stamp'); ew = n(e, 'cmd_w').abs()
    m = (et <= ct) & n(e, 'belief_error_odom_m').notna() & ew.notna()
    BEL.append(pd.DataFrame({'w': ew[m], 'be': n(e, 'belief_error_odom_m')[m],
                             'yaw': n(e, 'yaw_error_odom_map_vs_belief_rad').abs()[m] * 180/np.pi}))
    ej = pd.DataFrame({'t': et, 'w': ew}).dropna().sort_values('t')
    cap = pd.DataFrame({'t': n(p, 'log_stamp') - n(p, 'pixel_pose_age_s'), 'tl': n(p, 'log_stamp'),
                        'locerr': n(p, 'localization_error_captime_m')}).dropna(subset=['t', 'locerr']).sort_values('t')
    j = pd.merge_asof(cap, ej, on='t', direction='nearest', tolerance=0.25).dropna(subset=['w'])
    DET.append(j[j.tl <= ct])

BEL = pd.concat(BEL); DET = pd.concat(DET)
BEL['b'] = pd.cut(BEL.w, bins, right=False, labels=lab); DET['b'] = pd.cut(DET.w, bins, right=False, labels=lab)
x = np.arange(len(lab))
cam_med = [DET[DET.b == l].locerr.median() for l in lab]; cam_p95 = [DET[DET.b == l].locerr.quantile(.95) for l in lab]
bel_med = [BEL[BEL.b == l].be.median() for l in lab]; bel_p95 = [BEL[BEL.b == l].be.quantile(.95) for l in lab]
yaw_med = [BEL[BEL.b == l].yaw.median() for l in lab]; yaw_p95 = [BEL[BEL.b == l].yaw.quantile(.95) for l in lab]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
w = 0.36
axL.bar(x - w/2, cam_med, w, color="#ff7f0e", label="camera measurement err (median)")
axL.bar(x + w/2, bel_med, w, color="#1f77b4", label="belief err (median)")
axL.plot(x - w/2, cam_p95, "o--", color="#cc5500", label="camera p95")
axL.plot(x + w/2, bel_p95, "s--", color="#0b3d66", label="belief p95")
axL.set_xticks(x); axL.set_xticklabels(lab); axL.set_ylabel("error (m)")
axL.set_title("Cause 1: the MEASUREMENT degrades in turns (~7x)\n(YOLO still detects confidently — it's a projection error, not a miss)")
axL.legend(fontsize=8); axL.grid(alpha=.3, axis='y')

axR.bar(x, yaw_med, 0.5, color="#2ca02c", label="heading err (median)")
axR.plot(x, yaw_p95, "o--", color="#145a14", label="heading err p95")
axR.set_xticks(x); axR.set_xticklabels(lab); axR.set_ylabel("belief heading error (deg)")
axR.set_title("Cause 2: HEADING dead-reckons in turns\n(camera_xy_only: heading is never measured -> drifts, compounds via lever arm)")
axR.legend(fontsize=8); axR.grid(alpha=.3, axis='y')

fig.suptitle("Why it loses track in a turn: measurement degrades AND heading dead-reckons — they compound (5 west_upper C2 runs)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(ROOT).parents[1] / "figures" / "turn_rootcause_measurement_and_heading.png"
fig.savefig(out, dpi=130); print("wrote", out)
print("camera err  straight %.3f -> hardturn %.3f (%.1fx)" % (cam_med[0], cam_med[-1], cam_med[-1]/cam_med[0]))
print("belief err  straight %.3f -> hardturn %.3f (%.1fx)" % (bel_med[0], bel_med[-1], bel_med[-1]/bel_med[0]))
print("heading err straight %.1f  -> hardturn %.1f deg" % (yaw_med[0], yaw_med[-1]))
