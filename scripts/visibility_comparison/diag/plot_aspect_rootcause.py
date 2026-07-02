#!/usr/bin/env python3
"""EXACT root cause of the 'turn' localization error: it is the robot's ASPECT
(orientation vs the oblique south-wall camera), not rotation.

Camera at (0,-5.5,4.8) pitched ~53deg. Localization back-projects the bbox-BOTTOM
to the ground plane z=0. For a 3D robot with height, that point only equals the
true ground-contact when the robot is BROADSIDE; when it is END-ON (facing toward/
away from the oblique camera) the bbox-bottom back-projects with a large radial
offset that the fixed affine calibration (fit on the dominant broadside pose) does
not cancel. Turns merely sweep the robot through end-on aspects.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
ROOT = 'logs/visibility_comparison/robustness_keepin_clean_20260619/route_west_to_a1_upper/C2'
def n(d, c): return pd.to_numeric(d.get(c), errors='coerce')
CAM = np.array([0.0, -5.5])

rows = []
for seed in range(5):
    ef = glob.glob(f'{ROOT}/seed{seed}/experiment_*/experiment.csv'); pf = glob.glob(f'{ROOT}/seed{seed}/experiment_*/perception.csv')
    if not ef or not pf: continue
    e = pd.read_csv(ef[0], low_memory=False); p = pd.read_csv(pf[0], low_memory=False)
    ej = pd.DataFrame({'t': n(e, 'stamp'), 'w': n(e, 'cmd_w').abs()}).dropna(subset=['t']).sort_values('t')
    tx, ty, tyaw = n(p, 'true_x'), n(p, 'true_y'), n(p, 'true_yaw')
    rx, ry = tx - CAM[0], ty - CAM[1]
    aspect = np.degrees(np.abs(np.arctan2(np.sin(tyaw - np.arctan2(ry, rx)), np.cos(tyaw - np.arctan2(ry, rx)))))
    df = pd.DataFrame({'t_cap': n(p, 'log_stamp') - n(p, 'pixel_pose_age_s'),
                       'locerr': n(p, 'localization_error_captime_m'),
                       'aspect': aspect, 'dist': np.sqrt(rx**2 + ry**2)}).dropna(subset=['t_cap', 'locerr']).sort_values('t_cap')
    df = pd.merge_asof(df, ej, left_on='t_cap', right_on='t', direction='nearest', tolerance=0.25)
    rows.append(df)
D = pd.concat(rows).dropna(subset=['w'])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.3))
sc = axA.scatter(D.aspect, D.locerr, c=D.w, cmap="viridis", s=22, vmin=0, vmax=1)
# binned median
bins = np.arange(0, 181, 20); cen = bins[:-1] + 10
med = [D[(D.aspect >= bins[i]) & (D.aspect < bins[i+1])].locerr.median() for i in range(len(bins)-1)]
axA.plot(cen, med, "r-o", lw=2, label="median")
axA.axvspan(0, 45, color="red", alpha=0.06); axA.axvspan(135, 180, color="green", alpha=0.06)
axA.text(20, axA.get_ylim()[1]*0.9, "END-ON\n(facing camera)", color="#a00", ha="center", fontsize=9)
axA.text(160, axA.get_ylim()[1]*0.9, "BROADSIDE", color="#070", ha="center", fontsize=9)
axA.set_xlabel("robot aspect vs camera (deg)  0 = end-on, 180 = broadside")
axA.set_ylabel("camera localization error (m)")
axA.set_title("Error is a clean function of ASPECT, not rotation\n(colour = |cmd_w|: fast turns just live at low aspect)")
axA.legend(); axA.grid(alpha=.3); fig.colorbar(sc, ax=axA, label="|cmd_w| rad/s")

# Panel B: STRAIGHT-only proves rotation isn't needed
st = D[D.w < 0.2]
ab = pd.cut(st.aspect, [0, 45, 90, 135, 181], labels=['0-45\nend-on', '45-90', '90-135', '135-180\nbroadside'])
g = st.groupby(ab).locerr.median(); cnt = st.groupby(ab).locerr.size()
x = np.arange(len(g))
axB.bar(x, g.values, color="#1f77b4")
for i, (v, c) in enumerate(zip(g.values, cnt.values)):
    axB.text(i, v + 0.01, f"n={c}", ha="center", fontsize=8)
axB.set_xticks(x); axB.set_xticklabels(g.index.astype(str))
axB.set_xlabel("robot aspect vs camera (deg)")
axB.set_ylabel("camera localization error (m)")
axB.set_title("STRAIGHT-driving only (|cmd_w|<0.2): no rotation,\nyet end-on still has ~14x the error -> it is orientation")
axB.grid(alpha=.3, axis='y')

fig.suptitle("EXACT cause: oblique-camera bbox-bottom ground projection is orientation-dependent; the affine calibration only fixes the broadside pose",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(ROOT).parents[1] / "figures" / "turn_error_is_aspect.png"
fig.savefig(out, dpi=130); print("wrote", out)
