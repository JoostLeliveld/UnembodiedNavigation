#!/usr/bin/env python3
"""Why does a camera correction land ~8 cm from the truth when commissioning measured 1.5 cm?

    python3 experiments/fusion_on_fixed_routes/latency_decomposition.py

Offline, on the 24 drives already on disk. No simulator, no new data.

The answer, in one line: the correction is a GOOD measurement of a STALE pose. It is about
350-400 ms old by its own timestamp and is applied as if it were current, which at 0.22 m/s
is most of the error. Scored against where the robot actually was when the frame was taken,
the same corrections are accurate to 2.8 cm.

Six tests, each one able to kill the explanation:

  1  motion state      is the error there when the robot is standing still?
  2  along travel      does the error point BEHIND the robot, and grow with speed?
  3  message age       do the timestamps themselves say the correction is old?
  4  lag sweep         does scoring against an earlier pose collapse the error?
  5  odometry control  or is GROUND TRUTH simply logged late? (it is not)
  6  what is left      per camera and per range, after the lag is accounted for
  7  is it constant?   the same lag on every route and every arm, or does it move?
  8  jitter            after compensating a constant lag, does anything grow with speed?
  9  camera count      does waiting for more cameras make the correction staler?

Writes logs/studies/fusion_on_fixed_routes/latency/{numbers.json,01_the_correction_is_stale.png}.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "deck_figures"))
sys.path.insert(0, str(HERE.parents[1] / "warehouse_v2_sketches"))
import style as D                                        # noqa: E402

OUT = D.REPO / "logs/studies/fusion_on_fixed_routes/latency"
#: the hull arms only. O1 and O2 carry a 20 cm observation-model bias by construction, which
#: would swamp every effect measured here.
DRIVE_GLOB = "logs/studies/fusion_on_fixed_routes/drives/*/[F]*/seed0/experiment_*"
LAGS = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6)
CRUISE_MPS = 0.22


def _col(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out)


def _drives():
    for d in sorted(glob.glob(str(D.REPO / DRIVE_GLOB))):
        path = Path(d) / "experiment.csv"
        if path.exists():
            yield Path(d), list(csv.DictReader(open(path)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    findings = {}

    # ---- 1  motion state -----------------------------------------------------
    buckets = {"stopped": [], "turning in place": [], "driving": []}
    for _d, rows in _drives():
        v, w = np.abs(_col(rows, "cmd_v")), np.abs(_col(rows, "cmd_w"))
        e = _col(rows, "state_error_gt_m") * 100
        ok = np.isfinite(e)
        buckets["stopped"].append(e[ok & (v < 0.02) & (w < 0.05)])
        buckets["turning in place"].append(e[ok & (v < 0.02) & (w >= 0.05)])
        buckets["driving"].append(e[ok & (v >= 0.18)])
    findings["by_motion_state_cm"] = {
        k: {"n": int(len(np.concatenate(v))),
            "median": round(float(np.median(np.concatenate(v))), 2)}
        for k, v in buckets.items() if len(np.concatenate(v)) > 20}

    # ---- 2  along travel -----------------------------------------------------
    along, speed = [], []
    for _d, rows in _drives():
        t, gx, gy = _col(rows, "stamp"), _col(rows, "gt_x"), _col(rows, "gt_y")
        sx, sy = _col(rows, "state_x"), _col(rows, "state_y")
        ok = np.isfinite(t) & np.isfinite(gx) & np.isfinite(sx)
        t, gx, gy, sx, sy = (a[ok] for a in (t, gx, gy, sx, sy))
        if len(t) < 50:
            continue
        dx, dy = np.gradient(gx, t), np.gradient(gy, t)
        n = np.hypot(dx, dy)
        moving = n > 0.05
        along.append(((sx - gx)[moving] * (dx[moving] / n[moving])
                      + (sy - gy)[moving] * (dy[moving] / n[moving])) * 100)
        speed.append(n[moving])
    along, speed = np.concatenate(along), np.concatenate(speed)
    findings["along_travel"] = {
        "median_cm": round(float(np.median(along)), 2),
        "note": "negative means the correction sits BEHIND the robot",
        "implied_lag_ms_by_speed": {
            f"{lo:.2f}-{hi:.2f}": round(float(-np.median(along[(speed >= lo) & (speed < hi)])
                                              / 100 / ((lo + hi) / 2) * 1000))
            for lo, hi in ((0.05, 0.12), (0.12, 0.18), (0.18, 0.21), (0.21, 0.30))
            if ((speed >= lo) & (speed < hi)).sum() > 50}}

    # ---- 3  message age ------------------------------------------------------
    ages = np.concatenate([_col(rows, "state_age_s")[np.isfinite(_col(rows, "state_age_s"))]
                           for _d, rows in _drives()])
    findings["correction_age_ms"] = {
        "median": round(float(np.median(ages)) * 1000),
        "p95": round(float(np.percentile(ages, 95)) * 1000),
        "travel_at_cruise_cm": round(float(np.median(ages)) * CRUISE_MPS * 100, 1)}

    # ---- 4  lag sweep + 5  odometry control ---------------------------------
    sweep, control = {lag: [] for lag in LAGS}, {lag: [] for lag in LAGS}
    for _d, rows in _drives():
        t, gx, gy = _col(rows, "stamp"), _col(rows, "gt_x"), _col(rows, "gt_y")
        sx, sy = _col(rows, "state_x"), _col(rows, "state_y")
        ox, oy = _col(rows, "odom_map_x"), _col(rows, "odom_map_y")
        ok = np.isfinite(t) & np.isfinite(gx) & np.isfinite(sx)
        okc = np.isfinite(t) & np.isfinite(gx) & np.isfinite(ox)
        for lag in LAGS:
            if ok.sum() > 50:
                gxl = np.interp(t[ok] - lag, t[ok], gx[ok])
                gyl = np.interp(t[ok] - lag, t[ok], gy[ok])
                sweep[lag].append(np.hypot(sx[ok] - gxl, sy[ok] - gyl) * 100)
            if okc.sum() > 200:
                early = (t[okc] - t[okc][0]) < 30.0     # before odometry drift dominates
                gxl = np.interp(t[okc] - lag, t[okc], gx[okc])
                gyl = np.interp(t[okc] - lag, t[okc], gy[okc])
                control[lag].append((np.hypot(ox[okc] - gxl, oy[okc] - gyl) * 100)[early])
    sweep_med = {lag: float(np.median(np.concatenate(v))) for lag, v in sweep.items() if v}
    sweep_p95 = {lag: float(np.percentile(np.concatenate(v), 95)) for lag, v in sweep.items() if v}
    control_med = {lag: float(np.median(np.concatenate(v))) for lag, v in control.items() if v}
    best = min(sweep_med, key=sweep_med.get)
    findings["lag_sweep_cm"] = {str(k): round(v, 2) for k, v in sweep_med.items()}
    findings["lag_sweep_p95_cm"] = {str(k): round(v, 2) for k, v in sweep_p95.items()}
    findings["odometry_control_cm"] = {str(k): round(v, 2) for k, v in control_med.items()}
    findings["best_lag_s"] = best
    findings["odometry_best_lag_s"] = min(control_med, key=control_med.get)

    # ---- 6  what is left, per camera and range ------------------------------
    lay = D.layout()
    campos = {c.name: (c.x, c.y) for c in lay.cameras}
    per = {}
    for d, _rows in _drives():
        path = d / "fusion_observations.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            try:
                gx, gy = float(r["gt_x"]), float(r["gt_y"])
                ox, oy = float(r["obs_x"]), float(r["obs_y"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(gx) and np.isfinite(ox)):
                continue
            cx, cy = campos[r["camera"]]
            per.setdefault(r["camera"], []).append(
                (float(np.hypot(gx - cx, gy - cy)), float(np.hypot(ox - gx, oy - gy)) * 100))
    findings["per_camera_reading_error_cm"] = {}
    for cam, vals in sorted(per.items()):
        a = np.array(vals)
        entry = {"n": len(a), "median": round(float(np.median(a[:, 1])), 2), "by_range": {}}
        for lo, hi in ((0, 6), (6, 10), (10, 14), (14, 20), (20, 30)):
            s = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if s.sum() > 20:
                entry["by_range"][f"{lo}-{hi} m"] = {
                    "n": int(s.sum()), "median": round(float(np.median(a[s, 1])), 2)}
        findings["per_camera_reading_error_cm"][cam] = entry

    # ---- 7  is the lag constant across routes and arms? ---------------------
    fine = np.arange(0.15, 0.61, 0.05)
    per_drive = {}
    for d, rows in _drives():
        t, gx, gy = _col(rows, "stamp"), _col(rows, "gt_x"), _col(rows, "gt_y")
        sx, sy = _col(rows, "state_x"), _col(rows, "state_y")
        ok = np.isfinite(t) & np.isfinite(gx) & np.isfinite(sx)
        t, gx, gy, sx, sy = (a[ok] for a in (t, gx, gy, sx, sy))
        if len(t) < 100:
            continue
        meds = [float(np.median(np.hypot(sx - np.interp(t - l, t, gx),
                                         sy - np.interp(t - l, t, gy)) * 100)) for l in fine]
        key = f"{d.parts[-4]}/{d.parts[-3]}"
        per_drive[key] = {"best_lag_s": round(float(fine[int(np.argmin(meds))]), 2),
                          "residual_cm": round(min(meds), 2)}
    findings["per_drive_best_lag"] = per_drive
    lags = [v["best_lag_s"] for v in per_drive.values()]
    findings["lag_is_constant"] = {"min": min(lags), "max": max(lags),
                                   "n_drives": len(lags),
                                   "note": "one number compensates every drive" if
                                   min(lags) == max(lags) else "varies between drives"}

    # ---- 8  does anything grow with speed once the lag is removed? ----------
    resid = []
    for _d, rows in _drives():
        t, gx, gy = _col(rows, "stamp"), _col(rows, "gt_x"), _col(rows, "gt_y")
        sx, sy = _col(rows, "state_x"), _col(rows, "state_y")
        ok = np.isfinite(t) & np.isfinite(gx) & np.isfinite(sx)
        t, gx, gy, sx, sy = (a[ok] for a in (t, gx, gy, sx, sy))
        if len(t) < 100:
            continue
        v = np.hypot(np.gradient(gx, t), np.gradient(gy, t))
        e = np.hypot(sx - np.interp(t - best, t, gx), sy - np.interp(t - best, t, gy)) * 100
        resid.append(np.column_stack([v, e]))
    resid = np.vstack(resid)
    slow = resid[resid[:, 0] < 0.03][:, 1]
    fast = resid[resid[:, 0] >= 0.20][:, 1]
    findings["residual_after_compensation_cm"] = {
        "standing_still": round(float(np.median(slow)), 2),
        "at_cruise": round(float(np.median(fast)), 2),
        "note": "flat with speed means the lag is a constant bias, not jitter; no extra "
                "speed-dependent covariance term is needed"}

    # ---- 9  does waiting for more cameras cost latency? ---------------------
    by_n = {}
    for _d, rows in _drives():
        a, n = _col(rows, "state_age_s"), _col(rows, "fusion_candidates_n")
        ok = np.isfinite(a) & np.isfinite(n)
        for k in (1, 2, 3, 4):
            sel = ok & (n == k)
            if sel.sum():
                by_n.setdefault(k, []).append(a[sel])
    findings["age_ms_by_camera_count"] = {
        str(k): {"n": int(len(np.concatenate(v))),
                 "median_ms": round(float(np.median(np.concatenate(v))) * 1000)}
        for k, v in sorted(by_n.items())}

    (OUT / "numbers.json").write_text(json.dumps(findings, indent=2) + "\n")

    # ---- the figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.6), constrained_layout=True)
    ax = axes[0]
    lags = sorted(sweep_med)
    ax.plot(lags, [sweep_med[k] for k in lags], "-o", color=D.BAD, lw=2.6, ms=9,
            label="the camera correction")
    ax.plot(lags, [control_med[k] for k in lags], "-o", color=D.MUTED, lw=2.0, ms=7,
            label="odometry (no camera pipeline)")
    ax.axvline(best, color=D.INK, lw=1.4, ls=(0, (5, 3)))
    ax.text(best, max(sweep_med.values()) * 0.96, f"  best at {best:.2f} s",
            fontsize=12, color=D.INK, va="top")
    ax.set_xlabel("scored against where the robot was this many seconds earlier", fontsize=12)
    ax.set_ylabel("error (cm)", fontsize=12.5)
    ax.grid(True, color="#eeede8", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11.5)
    ax.set_title(f"The correction is right — about {best*1000:.0f} ms ago",
                 loc="left", fontsize=15, color=D.INK)

    ax = axes[1]
    names = list(findings["by_motion_state_cm"])
    vals = [findings["by_motion_state_cm"][k]["median"] for k in names]
    ax.bar(range(len(names)), vals, color=[D.GOOD, D.OLD, D.BAD], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.1f} cm", ha="center", fontsize=12.5, color=D.INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" in place", "\nin place") for n in names], fontsize=12)
    ax.set_ylabel("correction error, median (cm)", fontsize=12.5)
    ax.grid(True, axis="y", color="#eeede8", lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("It appears when the robot moves", loc="left", fontsize=15, color=D.INK)

    ax = axes[2]
    for cam, entry in findings["per_camera_reading_error_cm"].items():
        xs, ys = [], []
        for rng, cell in entry["by_range"].items():
            lo, hi = rng.split(" ")[0].split("-")
            xs.append((float(lo) + float(hi)) / 2)
            ys.append(cell["median"])
        if xs:
            ax.plot(xs, ys, "-o", color=D.CAM_COLOUR[cam], lw=2.2, ms=8, label=f"camera {cam}")
    ax.set_yscale("log")
    ax.set_xlabel("range from that camera (m)", fontsize=12)
    ax.set_ylabel("that camera's reading error (cm, log)", fontsize=12.5)
    ax.grid(True, which="both", color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11, ncol=2)
    ax.set_title("What the lag does not explain", loc="left", fontsize=15, color=D.INK)

    fig.suptitle("The correction is a good measurement of a stale pose",
                 x=0.004, ha="left", fontsize=20, color=D.INK)
    fig.text(0.004, -0.05,
             f"Left: scoring the same corrections against an earlier ground-truth pose collapses "
             f"the median error from {sweep_med[0.0]:.2f} cm to {sweep_med[best]:.2f} cm at "
             f"{best:.2f} s, and rises again after — a lag, not noise. Odometry, which has no "
             f"camera pipeline, bottoms out at "
             f"{findings['odometry_best_lag_s']:.1f} s, so ground truth is not the thing that is late.\n"
             f"Middle: the error is largely absent at a standstill. Right: what remains is "
             f"per-camera geometry — camera C, and camera A closer than 6 m — and it is "
             f"untouched by the lag (p95 stays near "
             f"{sweep_p95[0.0]:.0f} cm at every lag).\n"
             f"24 drives, hull arms only. Ground truth scores these; it is never an input.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "01_the_correction_is_stale.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"correction age: {findings['correction_age_ms']['median']} ms median "
          f"= {findings['correction_age_ms']['travel_at_cruise_cm']} cm at cruise")
    print(f"lag sweep: {sweep_med[0.0]:.2f} cm at 0 s -> {sweep_med[best]:.2f} cm at {best} s")
    print(f"odometry control bottoms out at {findings['odometry_best_lag_s']} s")
    print(f"by motion state: " + ", ".join(
        f"{k} {v['median']} cm" for k, v in findings["by_motion_state_cm"].items()))
    print(f"lag is the same on every drive: {findings['lag_is_constant']}")
    print(f"residual after compensation: still {findings['residual_after_compensation_cm']['standing_still']} cm, "
          f"cruise {findings['residual_after_compensation_cm']['at_cruise']} cm")
    print(f"age by camera count: " + ", ".join(
        f"{k}:{v['median_ms']}ms" for k, v in findings['age_ms_by_camera_count'].items()))
    print(f"wrote {OUT.relative_to(D.REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
