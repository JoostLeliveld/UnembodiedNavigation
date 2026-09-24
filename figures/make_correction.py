#!/usr/bin/env python3
"""What the learned correction removes, on the development set D_dev (the population the
correction variants are compared on; covariance and fusion use the sealed final audit).

(a) Residuals in each camera's ray frame (along the ray, across it), raw projection against
    the full correction. The raw cloud sits a quarter metre short along the ray.
(b) Distribution of position error for the raw projection, the structured branch alone and
    the full correction, with the position-balanced RMSE (mean over positions of the
    per-position RMSE) in the legend.

The structured-only positions are rebuilt from the stored branch prediction and checked
against the D_dev evaluation manifest before anything is drawn.

    python3 figures/make_correction.py
"""
from __future__ import annotations

import collections
import json

import numpy as np

import paper as P

VARIANTS = (("raw", "raw projection", "#b3b3b3", "-"),
            ("structured_only", "structured branch", "#5f5f5f", (0, (3, 1.5))),
            ("structured_plus_visibility", "full correction", P.INK, "-"))


def load():
    a = np.load(P.THESIS / "fits/correction/predictions.npz", allow_pickle=False)
    m = a["role"] == "D_dev"
    raw, cor, truth = a["raw_xy_m"][m], a["corrected_xy_m"][m], a["truth_xy_m"][m]
    delta = cor - raw
    ang = np.arctan2(delta[:, 1], delta[:, 0]) - np.arctan2(a["correction_ray_m"][m][:, 1], a["correction_ray_m"][m][:, 0])
    c, s = np.cos(ang), np.sin(ang)
    to_world = lambda v: np.c_[c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]]
    to_ray = lambda v: np.c_[c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]]
    xy = {"raw": raw, "structured_only": raw + to_world(a["base_prediction_ray_m"][m]),
          "structured_plus_visibility": cor}
    return a["position_key"][m], truth, xy, to_ray


def balanced_rmse(keys, err):
    groups = collections.defaultdict(list)
    for k, e in zip(keys, err):
        groups[k].append(e)
    return float(np.mean([np.sqrt(np.mean(np.square(v))) for v in groups.values()]))


def main():
    keys, truth, xy, to_ray = load()
    manifest = json.loads((P.THESIS / "fits/ddev_evaluation/manifest.json").read_text())["D_dev_correction_metrics"]
    err = {k: np.linalg.norm(v - truth, axis=1) for k, v in xy.items()}
    rmse = {k: balanced_rmse(keys, e) for k, e in err.items()}
    for k, value in rmse.items():   # the instrument must reproduce the manifest exactly
        if abs(value - manifest[k]["equal_position_rmse_m"]) > 1e-9:
            raise RuntimeError(f"{k}: rebuilt RMSE {value} != manifest {manifest[k]['equal_position_rmse_m']}")

    fig, (ax, bx) = P.plt.subplots(1, 2, figsize=(P.COLUMN, 1.75), gridspec_kw=dict(width_ratios=(1, 1.25), wspace=0.42))
    for key, label, colour, _ in (VARIANTS[0], VARIANTS[2]):
        r = 100.0 * to_ray(xy[key] - truth)
        ax.scatter(r[:, 0], r[:, 1], s=0.6, lw=0, color=colour, alpha=0.5 if key == "raw" else 0.35,
                   rasterized=True, zorder=2 if key == "raw" else 3)
        mean = r.mean(axis=0)
        ax.plot(*mean, "+", color=colour if key != "raw" else "#6f6f6f", ms=7, mew=1.3, zorder=5)
    ax.axhline(0, color=P.MUTED, lw=0.4, zorder=1); ax.axvline(0, color=P.MUTED, lw=0.4, zorder=1)
    ax.set_xlim(-60, 30); ax.set_ylim(-45, 45); ax.set_aspect("equal")
    ax.set_xlabel("along ray (cm)", labelpad=1); ax.set_ylabel("across ray (cm)", labelpad=1)
    bias = 100 * np.asarray(manifest["raw"]["equal_position_signed_bias_ray_m"])
    ax.annotate(f"raw bias\n{bias[0]:+.0f} cm", xy=(bias[0], bias[1]), xytext=(-58, 30), fontsize=6.3,
                color="#5f5f5f", arrowprops=dict(arrowstyle="-", lw=0.5, color="#8f8f8f"))
    fig.text(0.0, 0.97, "(a)", fontweight="bold", va="top")

    grid = np.logspace(np.log10(0.3), np.log10(80), 300)
    for key, label, colour, ls in VARIANTS:
        e = np.sort(100.0 * err[key])
        bx.plot(grid, np.searchsorted(e, grid) / e.size, color=colour, ls=ls, lw=1.3,
                label=f"{label}  {100 * rmse[key]:.1f}")
    bx.set_xscale("log"); bx.set_xlim(0.3, 80); bx.set_ylim(0, 1.0)
    bx.set_xticks([1, 3, 10, 30]); bx.set_xticklabels(["1", "3", "10", "30"])
    bx.set_xlabel("position error (cm)", labelpad=1); bx.set_ylabel("fraction of observations", labelpad=1)
    bx.legend(loc="upper left", fontsize=6, handlelength=1.6, title="RMSE (cm)", title_fontsize=6,
              borderaxespad=0.1)
    fig.text(0.47, 0.97, "(b)", fontweight="bold", va="top")
    P.save(fig, "correction")
    print({k: round(100 * v, 2) for k, v in rmse.items()}, "observations", len(keys), "positions", len(set(keys)))


if __name__ == "__main__":
    main()
