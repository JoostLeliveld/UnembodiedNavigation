#!/usr/bin/env python3
"""Show the detector's own view of the readings that were a metre wrong.

    python3 experiments/anomaly_investigation/show_the_view.py

A statistic cannot tell you whether a 130 cm reading is a broken sensor or a broken pipeline.
The frames can. These are real commissioning frames at the geometry where the drives produced
their worst readings, with the box the detector drew, the box the robot's own shape predicts,
and the verdict of the admission check that the live pipeline never ran.

Writes logs/studies/anomaly_investigations/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "deck_figures"))
sys.path.insert(0, str(HERE.parents[1] / "deck_figures/observation"))
sys.path.insert(0, str(HERE.parents[1] / "measurement_commissioning"))
import style as D                                            # noqa: E402
from _common import CAMS, DATA, index                        # noqa: E402
from observation import predicted_box, h, jacobian           # noqa: E402
from admission import gate                                   # noqa: E402

OUT = D.REPO / "logs/studies/anomaly_investigations"
DETECTOR_CSV = "detector_readings_halfopen_detect_20260825.csv"


def detections():
    out = {}
    for cam in CAMS:
        for r in csv.DictReader(open(DATA / cam / DETECTOR_CSV)):
            if r["detected"] == "1":
                out[(cam, r["image"])] = r
    return out


def sightings():
    det = detections()
    rows = []
    for r in index():
        if not r["image"]:
            continue
        d = det.get((r["camera_id"], r["image"]))
        if d is None:
            continue
        cam = CAMS[r["camera_id"]]
        x, y, yaw = float(r["robot_x"]), float(r["robot_y"]), float(r["robot_yaw"])
        pb, pred = predicted_box(cam, x, y, yaw), h(cam, x, y, yaw)
        if pb is None or pred is None:
            continue
        db = (float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]))
        z = np.array([0.5 * (db[0] + db[2]), db[3]])
        step = np.linalg.inv(jacobian(cam, x, y, yaw)) @ (z - np.array(pred))
        ok, why = gate(pb, db)
        rows.append({"camera": r["camera_id"], "image": r["image"], "cam": cam,
                     "x": x, "y": y, "yaw": yaw, "pred_box": pb, "det_box": db,
                     "error_cm": float(np.linalg.norm(step)) * 100,
                     "admitted": ok, "why": why,
                     "range_m": float(np.hypot(x - cam.cam_pos[0], y - cam.cam_pos[1]))})
    return rows


def panel(ax, s, tag):
    im = cv2.imread(str(DATA / s["camera"] / s["image"]))[:, :, ::-1]
    p, q = s["pred_box"], s["det_box"]
    cx = 0.5 * (p[0] + p[2]); cy = 0.5 * (p[1] + p[3])
    half = max(p[2] - p[0], p[3] - p[1], 45.0) * 1.5
    x0 = int(max(cx - half, 0)); x1 = int(min(cx + half, im.shape[1]))
    y0 = int(max(cy - half, 0)); y1 = int(min(cy + half, im.shape[0]))
    ax.imshow(im[y0:y1, x0:x1])
    ax.add_patch(plt.Rectangle((p[0] - x0, p[1] - y0), p[2] - p[0], p[3] - p[1],
                               fill=False, edgecolor=D.ROBOT, lw=2.4, ls=(0, (5, 3))))
    col = D.GOOD if s["admitted"] else D.BAD
    ax.add_patch(plt.Rectangle((q[0] - x0, q[1] - y0), q[2] - q[0], q[3] - q[1],
                               fill=False, edgecolor=col, lw=2.8))
    # the two bottom edges, which is what the check is about
    ax.plot([p[0] - x0, p[2] - x0], [p[3] - y0, p[3] - y0], color=D.ROBOT, lw=2.0, ls=(0, (2, 2)))
    ax.plot([q[0] - x0, q[2] - x0], [q[3] - y0, q[3] - y0], color=col, lw=2.6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2.6)
    verdict = "kept" if s["admitted"] else "REFUSED: " + ", ".join(
        w.replace("_", " ") for w in s["why"])
    ax.set_title(f"{tag} · camera {s['camera'][-1]} at {s['range_m']:.1f} m",
                 loc="left", fontsize=13, color=D.INK, pad=5)
    ax.set_xlabel(f"{verdict}\nthis reading is {s['error_cm']:.0f} cm from the truth",
                  fontsize=11.5, color=col if not s["admitted"] else D.INK2, labelpad=6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sightings()
    bad = sorted([s for s in rows if not s["admitted"]], key=lambda s: -s["error_cm"])
    good = sorted([s for s in rows if s["admitted"]], key=lambda s: s["error_cm"])

    # three worst refusals from three different cameras, and three ordinary kept readings
    picked, seen = [], set()
    for s in bad:
        if s["camera"] in seen:
            continue
        picked.append(s); seen.add(s["camera"])
        if len(picked) == 3:
            break
    keepers = [good[len(good) // 2], good[len(good) // 3], good[2 * len(good) // 3]]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.4), constrained_layout=True)
    for ax, s in zip(axes[0], picked):
        panel(ax, s, "the pipeline kept this")
    for ax, s in zip(axes[1], keepers):
        panel(ax, s, "an ordinary reading")
    fig.suptitle("What a 130 cm reading actually looks like", x=0.004, ha="left",
                 fontsize=21, color=D.INK)
    fig.text(0.004, -0.035,
             "Dashed blue is the box the robot's own shape predicts at its true pose; solid is "
             "what YOLO drew. The horizontal bars are the two bottom edges — the contact point, "
             "which is what the reading is made from.\n"
             "Top row: the detector found the robot, but its feet are hidden, so the box bottom "
             "sits high and the reading is thrown along the viewing ray. The admission check "
             "refuses exactly these — and the live pipeline never called it.\n"
             "Bottom row: the same detector, the same cameras, readings the check keeps.\n"
             "Real commissioning frames. Ground truth places the predicted box; it is never an "
             "input at runtime.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "01_what_a_metre_looks_like.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    ke = np.array([s["error_cm"] for s in rows if s["admitted"]])
    re_ = np.array([s["error_cm"] for s in rows if not s["admitted"]])
    summary = {
        "admitted": {"n": len(ke), "median_cm": round(float(np.median(ke)), 2),
                     "p95_cm": round(float(np.percentile(ke, 95)), 1),
                     "max_cm": round(float(ke.max()), 1)},
        "refused": {"n": len(re_), "median_cm": round(float(np.median(re_)), 2),
                    "p95_cm": round(float(np.percentile(re_, 95)), 1),
                    "max_cm": round(float(re_.max()), 1)},
        "fraction_refused": round(len(re_) / (len(ke) + len(re_)), 3),
        "runtime_calls_the_gate": False,
        "note": "plausibility_reasons() in reliability/silhouette_observation.py has the same "
                "thresholds as the commissioning gate, is unit-tested, and has no caller in "
                "src/. The camera manager imports only equivalent_position_measurement.",
    }
    (OUT / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(11.0, 5.6), constrained_layout=True)
    bins = np.logspace(-0.7, 2.2, 60)
    ax.hist(ke, bins=bins, color=D.GOOD, alpha=0.65, label=f"kept by the check (n={len(ke)})")
    ax.hist(re_, bins=bins, color=D.BAD, alpha=0.6, label=f"refused (n={len(re_)})")
    ax.set_xscale("log")
    ax.axvline(float(np.median(ke)), color=D.GOOD, lw=2.4)
    ax.axvline(float(np.median(re_)), color=D.BAD, lw=2.4)
    ax.text(float(np.median(ke)), ax.get_ylim()[1] * 0.95, f" {np.median(ke):.1f} cm",
            color=D.GOOD, fontsize=12.5, va="top")
    ax.text(float(np.median(re_)), ax.get_ylim()[1] * 0.95, f" {np.median(re_):.0f} cm",
            color=D.BAD, fontsize=12.5, va="top")
    ax.set_xlabel("how far that single reading is from the truth (cm, log)", fontsize=12.5)
    ax.set_ylabel("readings", fontsize=12.5)
    ax.grid(True, color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=12)
    ax.set_title("The check separates a 1.4 cm sensor from a 24 cm one — and it was not running",
                 loc="left", fontsize=16, color=D.INK)
    fig.text(0.004, -0.06,
             f"Every detection in the commissioning capture, scored the way the runtime scores "
             f"one. {summary['fraction_refused']*100:.0f}% fail the admission check; those are "
             f"the readings the live pipeline has been fusing.\n"
             f"Kept: median {summary['admitted']['median_cm']} cm, worst "
             f"{summary['admitted']['max_cm']:.0f} cm. Refused: median "
             f"{summary['refused']['median_cm']} cm, worst {summary['refused']['max_cm']:.0f} cm "
             f"— which is the size of the worst readings seen on the drives.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(OUT / "02_the_check_separates_two_sensors.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"kept    n={len(ke)} median {np.median(ke):.2f} cm max {ke.max():.0f}")
    print(f"refused n={len(re_)} median {np.median(re_):.2f} cm max {re_.max():.0f}")
    print(f"wrote {OUT.relative_to(D.REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
