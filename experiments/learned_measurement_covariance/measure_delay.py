"""How much of the shared, along-track error survives once the reading is scored against
the truth at the moment the CAMERA saw it, rather than the moment the log wrote it.

Every previous measurement compared a reading to the truth at logging time, which is later
by the whole detector-and-manager delay. That delay is identical on every camera, so it
looks exactly like measurement error and gets charged to each camera's own noise. With the
capture time now recorded, the two can finally be separated:

  * scored at LOG time    -- what the old numbers measured: sensor error PLUS pipeline delay
  * scored at CAPTURE time -- the sensor's own error, with the delay taken out

The difference between the two is the delay, in centimetres, and it is the term the
covariance model has no parameter for.
"""
import csv, math, sys
from pathlib import Path
import numpy as np


def truth_track(run: Path):
    """Ground truth as a function of time, for interpolation. EVALUATION ONLY."""
    t, x, y = [], [], []
    for row in csv.DictReader(open(run / "experiment.csv")):
        try:
            if float(row["gt_available"]) != 1.0:
                continue
            t.append(float(row["stamp"])); x.append(float(row["gt_x"])); y.append(float(row["gt_y"]))
        except (KeyError, ValueError):
            continue
    order = np.argsort(t)
    return np.asarray(t)[order], np.asarray(x)[order], np.asarray(y)[order]


def load(run: Path):
    """Every reading ONCE, with the delay it actually arrived with.

    The manager republishes each detection on about four consecutive decisions, so a row
    is a republish and not a reading. Kept as rows, the delay distribution measured here
    is the distribution over republishes -- each detection contributing one fresh entry
    and three progressively staler copies -- which drags the median up by roughly half a
    decision period. The first row for each (camera, capture time) is the one that
    describes when the reading actually landed.
    """
    rows = []
    seen = set()
    for row in csv.DictReader(open(run / "fusion_observations.csv")):
        key = (row.get("camera"), row.get("obs_stamp"))
        if key in seen:
            continue
        seen.add(key)
        try:
            rec = dict(
                stamp=float(row["stamp"]), obs_stamp=float(row["obs_stamp"]),
                camera=row["camera"],
                xy=np.array([float(row["obs_x"]), float(row["obs_y"])]),
                gt_log=np.array([float(row["gt_x"]), float(row["gt_y"])]),
                cov=np.array([[float(row["obs_cov_xx"]), float(row["obs_cov_xy"])],
                              [float(row["obs_cov_xy"]), float(row["obs_cov_yy"])]]),
                conf=float(row.get("conf", "nan") or "nan"),
                rng=float(row.get("range_m", "nan") or "nan"))
        except (KeyError, ValueError):
            continue
        if not (np.isfinite(rec["xy"]).all() and np.isfinite(rec["cov"]).all()):
            continue
        if np.linalg.det(rec["cov"]) <= 0.0 or not math.isfinite(rec["obs_stamp"]):
            continue
        rows.append(rec)
    return rows


def report(run: Path, label: str):
    tt, tx, ty = truth_track(run)
    rows = load(run)
    if not rows:
        print(f"{label}: no usable observations (is obs_stamp logged?)")
        return None
    for r in rows:
        r["gt_cap"] = np.array([np.interp(r["obs_stamp"], tt, tx),
                                np.interp(r["obs_stamp"], tt, ty)])
        r["lat"] = r["stamp"] - r["obs_stamp"]

    lat = np.array([r["lat"] for r in rows])
    e_log = np.array([np.linalg.norm(r["gt_log"] - r["xy"]) for r in rows]) * 100
    e_cap = np.array([np.linalg.norm(r["gt_cap"] - r["xy"]) for r in rows]) * 100
    print(f"\n=== {label} ===")
    print(f"observations {len(rows)},  capture-to-log delay median {np.median(lat)*1000:.0f} ms "
          f"(p95 {np.percentile(lat,95)*1000:.0f} ms)")
    print(f"reading's error scored at LOG time     : median {np.median(e_log):5.2f} cm")
    print(f"reading's error scored at CAPTURE time : median {np.median(e_cap):5.2f} cm"
          f"   <- the sensor's own error")

    # shared vs private, at capture time
    groups = {}
    for r in rows:
        groups.setdefault(round(r["stamp"], 4), []).append(r)
    shared, private = [], {}
    for members in groups.values():
        if len(members) < 2:
            continue
        errs = np.array([m["gt_cap"] - m["xy"] for m in members])
        mean = errs.mean(axis=0)
        shared.append(np.linalg.norm(mean) * 100)
        for m, e in zip(members, errs):
            private.setdefault(m["camera"], []).append(np.linalg.norm(e - mean) * 100)
    allp = np.concatenate([np.asarray(v) for v in private.values()]) if private else np.array([0.0])
    s = np.asarray(shared) if shared else np.array([0.0])
    print(f"error every camera makes together   : median {np.median(s):5.2f} cm")
    print(f"error each camera makes on its own  : median {np.median(allp):5.2f} cm")
    print(f"share of squared error that is COMMON: "
          f"{np.mean(s**2)/(np.mean(s**2)+np.mean(allp**2))*100:.0f}%")

    # is what remains still along the direction of travel?
    st = sorted(groups)
    along, across = [], []
    for i in range(1, len(st) - 1):
        a, b = groups[st[i - 1]][0], groups[st[i + 1]][0]
        v = b["gt_cap"] - a["gt_cap"]
        if np.linalg.norm(v) < 1e-4 or len(groups[st[i]]) < 2:
            continue
        u = v / np.linalg.norm(v); n = np.array([-u[1], u[0]])
        e = np.mean([m["gt_cap"] - m["xy"] for m in groups[st[i]]], axis=0)
        along.append(float(e @ u) * 100); across.append(float(e @ n) * 100)
    if along:
        print(f"what remains, along travel: {np.mean(along):+5.2f} cm   across: {np.mean(across):+5.2f} cm")

    # per camera, on the lag-free part
    print(f"\n  {'camera':8} {'n':>6} {'its own error':>14} {'stated 1sig':>12} {'off by':>8}")
    for c in sorted(private):
        v = [r for r in rows if r["camera"] == c]
        nis = []
        for m in groups.values():
            if len(m) < 2:
                continue
            mean = np.mean([q["gt_cap"] - q["xy"] for q in m], axis=0)
            for q in m:
                if q["camera"] != c:
                    continue
                d = (q["gt_cap"] - q["xy"]) - mean
                nis.append(float(d @ np.linalg.solve(q["cov"], d)))
        if not nis:
            continue
        st1 = np.median([math.sqrt(np.trace(q["cov"]) / 2) * 100 for q in v])
        print(f"  {c:8} {len(nis):6d} {np.median(private[c]):11.2f} cm {st1:9.2f} cm "
              f"{np.median(nis)/1.3863:7.1f}x")
    return dict(e_cap=float(np.median(e_cap)), e_log=float(np.median(e_log)),
                lat=float(np.median(lat)), shared=float(np.median(s)))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        report(p, p.parent.parent.name)
