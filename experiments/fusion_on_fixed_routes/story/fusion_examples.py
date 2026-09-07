#!/usr/bin/env python3
"""How one arm actually fused, shown on real moments from its own drive.

    python3 experiments/fusion_on_fixed_routes/story/fusion_examples.py F4

  05_how_it_fused.png            five worked examples: every camera's own answer at that
                                 instant, which ones the rule used, and what came out
  06_where_observations_landed.png  every camera observation of the whole drive -- on the
                                 floor, and as an offset cloud per camera around the truth

Reads fusion_observations.csv, which the experiment logger writes from the manager's own
decision messages: one row per camera per correction. Ground truth is used to place and score
these, never as an input.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "deck_figures"))
sys.path.insert(0, str(HERE.parents[1]))
import aligned as A                                    # noqa: E402
import style as D                                      # noqa: E402
from score import FOLDER, TASKS, showcase_run, story_dir   # noqa: E402
from arm import ARM_TITLE                              # noqa: E402


def load(arm, task=TASKS[0], *, max_reference_gap_s=None):
    run = showcase_run(arm, task)
    start,stop=A.mission_interval(run)
    readings=A.readings(run,admitted_only=False,max_reference_gap_s=max_reference_gap_s)
    groups={}
    for r in readings:groups.setdefault(r['source_batch_id'],[]).append(r)
    moments=[]
    for event in A.fused_answers(run,max_reference_gap_s=max_reference_gap_s):
        if not start<=event['fused_stamp']<=stop:continue
        members=groups.get(event['source_batch_id'],[])
        if not members:continue
        A.validate_covariances(event['fused_cov'][None])
        # This panel displays residual vectors in common world axes. Each camera
        # is centred on its own capture-time truth; the fused answer on fused-time
        # truth. Translated points are residual illustrations, not simultaneous poses.
        gt=np.asarray(event['truth'])
        obs=[]
        for r in members:
            A.validate_covariances(r['cov'][None])
            obs.append(dict(camera=r['camera'],used=r['used'],xy=gt+r['error'],
                            world_xy=np.array([r['obs_x'],r['obs_y']]),truth_at_obs=r['truth'],
                            cov=r['cov'],obs_stamp=r['obs_stamp']))
        moments.append(dict(stamp=event['fused_stamp'],source_batch_id=event['source_batch_id'],
            obs=obs,cameras={r['camera'] for r in members},fused=event['fused_xy'],
            fused_cov=event['fused_cov'],gt=gt,n=event['n_candidates'],n_used=event['n_used'],
            error_cm=event['error_cm'],coordinates='own-time residuals in world axes'))
    if not moments:raise SystemExit('no reference-supported fusion examples; supply an explicit reference gap policy')
    return run,moments


def ellipse(cov, scale=1.0, n=180):
    w, V = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * math.pi, n)
    return (V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.array([np.cos(t), np.sin(t)]))).T * scale


def pick_examples(moments):
    """One worked example per camera-count regime, plus the worst moment of the drive."""
    chosen = []
    for n in (1, 2, 3, 4, 5):
        same = [m for m in moments if m["n"] == n]
        if same:
            chosen.append((f"{n} camera{'s' if n > 1 else ''} available", same[len(same) // 2]))
    worst = max(moments, key=lambda m: m["error_cm"])
    chosen.append(("the worst correction of the drive", worst))
    return chosen


def draw_moment(ax, label, m, travelled_m=None):
    gt = np.asarray(m["gt"])
    def cm(p):
        return (np.asarray(p) - gt) * 100.0
    for o in m["obs"]:
        col = D.CAM_COLOUR[o["camera"]]
        e = ellipse(o["cov"], 100.0) + cm(o["xy"])
        if o["used"]:
            ax.fill(e[:, 0], e[:, 1], color=col, alpha=0.13, lw=0, zorder=3)
            ax.plot(e[:, 0], e[:, 1], color=col, lw=2.0, zorder=4)
            ax.plot(*cm(o["xy"]), "o", ms=9, color=col, mec="white", mew=1.4, zorder=6)
        else:
            ax.plot(e[:, 0], e[:, 1], color=col, lw=1.4, ls=(0, (3, 3)), alpha=0.85, zorder=4)
            ax.plot(*cm(o["xy"]), "x", ms=9, color=col, mew=2.0, zorder=6)
        ax.annotate(o["camera"], xy=cm(o["xy"]), xytext=(4, 4), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=col, zorder=7)
    fe = ellipse(m["fused_cov"], 100.0) + cm(m["fused"])
    ax.plot(fe[:, 0], fe[:, 1], color=D.INK, lw=2.6, zorder=8)
    # hollow, so the camera it came from stays visible underneath
    ax.plot(*cm(m["fused"]), "D", ms=15, markerfacecolor="none", markeredgecolor=D.INK,
            mew=2.4, zorder=9)
    ax.plot(0, 0, "*", ms=22, color=D.GOOD, mec="white", mew=1.2, zorder=10)
    span = max(np.abs(np.concatenate([cm(o["xy"]) for o in m["obs"]] + [cm(m["fused"])])).max()
               * 1.6, 6.0)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span); ax.set_aspect("equal")
    ax.grid(True, color="#f2f1ec", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # If a camera that was available sat much closer to the truth than the answer the rule
    # produced, say so on the figure. This is scoring, not an input: the rule could not see it.
    best = min(m["obs"], key=lambda o: float(np.linalg.norm(o["xy"] - gt)))
    best_cm = float(np.linalg.norm(best["xy"] - gt)) * 100.0
    if m["error_cm"] > max(3.0 * best_cm, 5.0):
        note = (f"camera {best['camera']} was {best_cm:.1f} cm from the truth — unused"
                if not best["used"] else
                f"camera {best['camera']} was {best_cm:.1f} cm from the truth and WAS used, "
                f"and the answer still came out {m['error_cm']:.0f} cm off")
        ax.annotate(note,
                    xy=cm(best["xy"]), xytext=(0.5, 0.03), textcoords="axes fraction",
                    ha="center", va="bottom", fontsize=10.5, color=D.BAD, fontweight="bold",
                    zorder=11, wrap=True,
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=D.BAD, shrinkB=8))
    used = "".join(o["camera"] for o in m["obs"] if o["used"]) or "none"
    fused_sigma = math.sqrt(np.trace(m["fused_cov"]) / 2) * 100
    ax.set_title(label, loc="left", fontsize=13.5, color=D.INK, pad=5)
    ax.set_xlabel(f"Own-time residuals; used {used}\n{m['error_cm']:.1f} cm out, claimed ±{fused_sigma:.1f} cm",
                  fontsize=11.5, color=D.INK2, labelpad=6)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',nargs='?',default='F4',choices=list(FOLDER))
    parser.add_argument('--task',default=TASKS[0],choices=TASKS)
    parser.add_argument('--max-reference-gap-s',type=float)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    arm,task=args.arm,args.task
    out = args.out or story_dir(task, arm)
    run, moments = load(arm, task,max_reference_gap_s=args.max_reference_gap_s)
    protocol=dict(task=task,arm=arm,run=str(run),max_reference_gap_s=args.max_reference_gap_s,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(A.__file__),run/'run_manifest.json',run/'fusion_observations.csv']})
    A.freeze_analysis_protocol(out,'fusion_examples_protocol.json',protocol,
        owned_outputs=['05_how_it_fused.png','06_where_observations_landed.png','07_every_camera_along_the_drive.png'])
    title = f"{ARM_TITLE[arm]} — {task.replace('fusion_', '').replace('_', ' ')}"

    # ---------------- 05 worked examples ----------------
    examples = pick_examples(moments)
    fig, axes = plt.subplots(1, len(examples), figsize=(3.9 * len(examples) + 1.0, 5.8),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (label, m) in zip(axes, examples):
        draw_moment(ax, label, m)
    axes[0].set_ylabel("centimetres from the true position", fontsize=12)
    axes[0].plot([], [], "o", ms=9, color=D.MUTED, label="a camera the rule used")
    axes[0].plot([], [], "x", ms=9, color=D.MUTED, mew=2.0, label="a camera it did not use")
    axes[0].plot([], [], "D", ms=11, markerfacecolor="none", markeredgecolor=D.INK, mew=2.0,
                 label="what the rule produced")
    axes[0].plot([], [], "*", ms=15, color=D.GOOD, label="the truth")
    axes[0].legend(frameon=False, fontsize=10.5, loc="upper left", ncol=1)
    fig.suptitle(f"{title}: how it fused, at {len(examples)} selected moments of its drive",
                 x=0.004, ha="left", fontsize=19, color=D.INK)
    fig.text(0.004, -0.06,
             "Each dot is one camera residual scored at its own capture time, with its own "
             "ellipse; a cross and a dashed ellipse mean the rule did not use that camera. The "
             "black diamond and ellipse are what the rule handed the filter, and the green star "
             "is zero residual. Camera and fused residuals have different reference times.\n"
             "Ellipses are 1σ. Ground truth places and scores these; it is never an input to "
             "the manager, the filter or the planner.\nOne explicitly frozen showcase drive.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(out / "05_how_it_fused.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 06 where the observations landed ----------------
    lay = D.layout()
    fig = plt.figure(figsize=(15.6, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    D.draw_warehouse(ax, lay)
    gts = np.array([m["gt"] for m in moments])
    ax.plot(gts[:, 0], gts[:, 1], color=D.MUTED, lw=2.0, zorder=5)
    for cam in sorted({o["camera"] for m in moments for o in m["obs"]}):
        pts = np.array([o["world_xy"] for m in moments for o in m["obs"] if o["camera"] == cam])
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=D.CAM_COLOUR[cam], alpha=0.55, lw=0,
                   zorder=6, label=f"camera {cam} — {len(pts)} readings")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=11, frameon=False, ncol=2)
    ax.set_title("Recorded camera readings paired with scored fused events", loc="left", fontsize=15,
                 color=D.INK, pad=18)

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color="#eceae4", lw=1); ax.axvline(0, color="#eceae4", lw=1)
    lines = []
    for cam in sorted({o["camera"] for m in moments for o in m["obs"]}):
        off = np.array([(o["xy"] - np.asarray(m["gt"])) * 100.0
                        for m in moments for o in m["obs"] if o["camera"] == cam])
        ax.scatter(off[:, 0], off[:, 1], s=7, color=D.CAM_COLOUR[cam], alpha=0.45, lw=0)
        mean = off.mean(axis=0)
        ax.plot(*mean, "o", ms=13, color=D.CAM_COLOUR[cam], mec="white", mew=2.0, zorder=8)
        ax.annotate(cam, xy=mean, xytext=(6, 6), textcoords="offset points", fontsize=13,
                    fontweight="bold", color=D.CAM_COLOUR[cam], zorder=9)
        lines.append(f"{cam}: {np.linalg.norm(mean):.1f} cm signed-bias vector magnitude, "
                     f"{off.shape[0]} readings")
    ax.plot(0, 0, "*", ms=24, color=D.GOOD, mec="white", mew=1.2, zorder=10)
    ax.set_aspect("equal"); ax.grid(True, color="#f2f1ec", lw=0.7); ax.set_axisbelow(True)
    ax.set_xlabel("centimetres east of the truth", fontsize=12)
    ax.set_ylabel("centimetres north of the truth", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("The same readings, stacked on the truth", loc="left", fontsize=15,
                 color=D.INK, pad=18)
    fig.suptitle(f"{title}: every camera reading of the drive", x=0.004, y=1.06,
                 ha="left", fontsize=19, color=D.INK)
    fig.text(0.004, -0.03,
             "Left: each reading in the warehouse, in its camera's colour, over the driven "
             "path. Right: the same readings as offsets from where the robot really was, so a "
             "camera that is consistently off shows as a cloud away from the star.\n"
             + "   ·   ".join(lines) + "\nOne explicitly frozen showcase drive.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(out / "06_where_observations_landed.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 07 every camera, along the whole route ----------------
    cams = sorted({o["camera"] for m in moments for o in m["obs"]})
    stamps = np.array([m["stamp"] for m in moments])
    t0 = A.mission_interval(run)[0]
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 8.4), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax = axes[0]
    for cam in cams:
        xs, ys = [], []
        for m in moments:
            for o in m["obs"]:
                if o["camera"] == cam:
                    xs.append(o["obs_stamp"] - t0)
                    ys.append(float(np.linalg.norm(o["xy"] - np.asarray(m["gt"]))) * 100.0)
        ax.plot(xs, ys, ".", ms=4, alpha=0.55, color=D.CAM_COLOUR[cam], label=f"camera {cam}")
    fused_err = np.array([m["error_cm"] for m in moments])
    ax.plot(stamps - t0, fused_err, color=D.INK, lw=1.8, zorder=6,
            label="what the rule produced")
    ax.set_yscale("log")
    ax.set_ylabel("how far each reading was\nfrom the truth (cm, log)", fontsize=12)
    ax.grid(True, which="both", color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=11, ncol=6, loc="upper left")
    ax.set_title("Every camera's own answer, and the one the rule made from them",
                 loc="left", fontsize=16, color=D.INK)

    ax = axes[1]
    order = {c: i for i, c in enumerate(cams)}
    for cam in cams:
        on_used, on_seen = [], []
        for m in moments:
            for o in m["obs"]:
                if o["camera"] != cam:
                    continue
                (on_used if o["used"] else on_seen).append(o["obs_stamp"] - t0)
        y = order[cam]
        ax.plot(on_seen, [y] * len(on_seen), "|", ms=11, color=D.CAM_COLOUR[cam], alpha=0.30)
        ax.plot(on_used, [y] * len(on_used), "|", ms=13, color=D.CAM_COLOUR[cam])
    ax.set_yticks(range(len(cams)))
    ax.set_yticklabels([f"camera {c}" for c in cams], fontsize=12)
    ax.set_xlabel("seconds into the drive", fontsize=12)
    ax.grid(True, axis="x", color="#eeede8", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Which cameras were available (faint) and which the rule used (solid)",
                 loc="left", fontsize=16, color=D.INK)

    fig.suptitle(f"{title}: the whole drive, camera by camera", x=0.004, y=1.03, ha="left",
                 fontsize=19, color=D.INK)
    fig.text(0.004, -0.03,
             "Top: one dot per camera per correction, on a log scale because the readings span "
             "centimetres to metres. The black line is what the rule handed the filter.\n"
             "Bottom: availability against use — a faint tick means that camera had a usable "
             "reading, a solid tick means the rule actually used it.\nOne explicitly frozen showcase drive.",
             fontsize=11.5, color=D.INK2, va="top", linespacing=1.5)
    fig.savefig(out / "07_every_camera_along_the_drive.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"{task}/{arm}: {len(moments)} corrections, {sum(len(m['obs']) for m in moments)} camera "
          f"readings -> 3 figures in {out.relative_to(D.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
