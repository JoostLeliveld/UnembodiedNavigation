#!/usr/bin/env python3
"""De-confounded detector-score audit on honest_campaign_v2 (refresh of exp0).

For successful detections, is the raw YOLO score a usable predictor of projected
localization error? Reports the Simpson-confound structure (pooled vs per-range-band
vs dwell/moving Spearman), tail exceedance by score tercile, and detection rate.
Uses canonical scripts/shared/metrics.py.
"""
import csv, glob, os, sys, statistics
import numpy as np
sys.path.insert(0,"scripts/shared"); import metrics as M
CAMP="logs/visibility_comparison/honest_campaign_v2"
cam=np.load("paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz",allow_pickle=True)["camera_pos"]
cx,cy=float(cam[0]),float(cam[1])

rows=[]  # per detection: score, err, range, dwell(bool), detected
allframes=0; hits=0
for pc in glob.glob(f"{CAMP}/**/perception.csv",recursive=True):
    rr=list(csv.DictReader(open(pc)))
    if not rr: continue
    ts=[float(r.get('log_stamp') or 'nan') for r in rr]
    t0=min(t for t in ts if np.isfinite(t))
    for r,t in zip(rr,ts):
        allframes+=1
        det=str(r.get('detected','')).strip() in ('1','1.0','True','true')
        if det: hits+=1
        try:
            sc=float(r['yolo_score_raw']); err=float(r['localization_error_captime_m'])
            x=float(r['true_x']); y=float(r['true_y'])
        except (KeyError,ValueError): continue
        if not det or not (np.isfinite(sc) and np.isfinite(err) and sc>0): continue
        rng=float(np.hypot(x-cx,y-cy))
        dwell = np.isfinite(t) and (t-t0)<40.0
        rows.append((sc,err,rng,dwell))

sc=np.array([r[0] for r in rows]); err=np.array([r[1] for r in rows])
rng=np.array([r[2] for r in rows]); dwell=np.array([r[3] for r in rows])
print(f"=== detector-score audit (honest_campaign_v2) ===")
print(f"frames={allframes}  detections(scored)={len(rows)}  overall detection rate={hits/allframes:.3f}")
print(f"pooled Spearman(score,err) = {M.spearman(sc,err)[0]:+.3f}   (confounded)")
for lo,hi in [(0,5),(5,7),(7,20)]:
    m=(rng>=lo)&(rng<hi)
    if m.sum()>10: print(f"  range[{lo},{hi}) m: rho={M.spearman(sc[m],err[m])[0]:+.3f}  n={int(m.sum())}")
print(f"  start-dwell (t<40s): rho={M.spearman(sc[dwell],err[dwell])[0]:+.3f} n={int(dwell.sum())}  |  moving: rho={M.spearman(sc[~dwell],err[~dwell])[0]:+.3f} n={int((~dwell).sum())}")
# terciles
q1,q2=np.quantile(sc,[1/3,2/3])
lowm=sc<=q1; highm=sc>=q2
print(f"  score terciles: low<= {q1:.3f}  high>= {q2:.3f}")
for thr in (0.15,0.30,0.50):
    print(f"  P(err>{thr:.2f}m): low-tercile={ (err[lowm]>thr).mean():.3f}  high-tercile={ (err[highm]>thr).mean():.3f}")
print(f"  BEV err low-tercile p50/p95 = {np.percentile(err[lowm],50):.3f}/{np.percentile(err[lowm],95):.3f} | high-tercile p50/p95 = {np.percentile(err[highm],50):.3f}/{np.percentile(err[highm],95):.3f}")
