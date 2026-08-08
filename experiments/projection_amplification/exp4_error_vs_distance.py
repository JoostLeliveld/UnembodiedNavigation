#!/usr/bin/env python3
"""Is measurement uncertainty constant across a camera footprint? No -- and not simply.

This is a component figure, not a contribution. The paper's subject is correlated error
and belief honesty; this establishes one prerequisite that a reviewer will ask about:
whether a single per-camera covariance is defensible, or whether the measurement
uncertainty varies with where the robot is.

It answers with the deployed baseline (plain IPM, zero parameters) so the statement is
about the measurement path actually in use, not about a candidate.

The finding splits in two, and the split matters because a single "error vs distance"
curve hides it:

  * ACCURACY improves with distance. Median error falls from ~83 mm near the camera to
    ~55 mm at 11-13 m. Steep viewing angles turn the silhouette offset into a large
    ground error.
  * CONSISTENCY degrades with distance. The spread roughly doubles, ~23 mm to ~45 mm.
    Projection amplifies pixel noise as range grows.

Near the camera is precise but biased; far away is unbiased but noisy. Both trends hold
on all four cameras independently, which is what makes them a property of the geometry
rather than of one mount.

Run:  python3 experiments/projection_amplification/exp4_error_vs_distance.py
Out:  logs/studies/projection_amplification/exp4_error_vs_distance/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
DEFAULT_RESIDUALS = (
    REPO / "logs/visibility_comparison/commissioning_grid_20260807/grid_residuals_raw_ipm.csv"
)
OUT = REPO / "logs/studies/projection_amplification/exp4_error_vs_distance"

CAMERA_COLORS = {"camera_A": "#0072B2", "camera_B": "#E69F00",
                 "camera_C": "#009E73", "camera_D": "#CC79A7"}
#: Wide enough that every bin clears the minimum on every camera.
BIN_WIDTH_M = 2.0
MIN_BIN_N = 25


def load(path: Path, clear_only: bool = True) -> dict[str, dict[str, np.ndarray]]:
    per: dict[str, dict[str, list]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if clear_only and row.get("oracle_visible") != "1":
                continue
            bucket = per.setdefault(row["camera"], {"d": [], "e": []})
            bucket["d"].append(float(row["range_m"]))
            bucket["e"].append(1000.0 * float(row["cor_norm"]))
    return {c: {k: np.asarray(v, dtype=float) for k, v in b.items()} for c, b in per.items()}


def binned_stats(d: np.ndarray, e: np.ndarray) -> list[dict]:
    edges = np.arange(np.floor(d.min()), np.ceil(d.max()) + 0.01, BIN_WIDTH_M)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d < hi)
        if mask.sum() < MIN_BIN_N:
            continue
        values = e[mask]
        out.append({
            "lo_m": float(lo), "hi_m": float(hi), "mid_m": float(0.5 * (lo + hi)),
            "n": int(mask.sum()),
            "median_mm": float(np.median(values)),
            "q25_mm": float(np.percentile(values, 25)),
            "q75_mm": float(np.percentile(values, 75)),
            "std_mm": float(values.std()),
        })
    return out


#: Mount height of every camera in this world, for the elevation-angle axis.
CAMERA_HEIGHT_M = 6.1


def _along_by_range(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d, a = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("oracle_visible") != "1":
                continue
            d.append(float(row["range_m"]))
            a.append(1000.0 * float(row["cor_along"]))
    return np.asarray(d, dtype=float), np.asarray(a, dtype=float)


def fig_e2(bottom_csv: Path, centre_csv: Path, out_dir: Path) -> dict:
    """Why no single fixed-plane pixel statistic works across a footprint.

    Both statistics assume the detected pixel corresponds to a known height. Neither
    assumption survives the full range of viewing angles, and they fail at OPPOSITE ends:

      * the BOTTOM edge is a poor stand-in for the contact point seen from above, where
        the silhouette's lower boundary is the far side of the robot rather than its feet;
      * the BOX CENTRE relies on an assumed height, and the ground error from a height
        error scales as 1/tan(elevation), so it blows up at shallow angles.

    Plotted against elevation angle rather than range, because the angle is the mechanism
    and the range is only its proxy at this mount height.
    """

    series = {"box bottom edge  ($z=0$)": _along_by_range(bottom_csv),
              "box centre  ($z^{*}=0.085$ m)": _along_by_range(centre_csv)}
    colours = {"box bottom edge  ($z=0$)": "#0072B2",
               "box centre  ($z^{*}=0.085$ m)": "#009E73"}
    edges = np.arange(2.0, 17.1, 2.0)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.grid(True, alpha=0.3, zorder=0)
    ax.axhline(0.0, color="#222222", lw=1.4, zorder=3)
    out: dict = {}
    for label, (d, a) in series.items():
        mids, bias, elev = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (d >= lo) & (d < hi)
            if mask.sum() < MIN_BIN_N:
                continue
            mid = 0.5 * (lo + hi)
            mids.append(mid)
            bias.append(float(a[mask].mean()))
            elev.append(float(np.degrees(np.arctan2(CAMERA_HEIGHT_M, mid))))
        ax.plot(elev, bias, marker="o", ms=6, lw=2.0, color=colours[label], zorder=5,
                label=label)
        out[label] = {"elevation_deg": elev, "along_bias_mm": bias, "range_mid_m": mids}

    ax.invert_xaxis()  # steep (near) on the left, shallow (far) on the right
    ax.set_xlabel("camera elevation angle to the robot [deg]   "
                  "(steep / near  $\\leftarrow$   $\\rightarrow$  shallow / far)")
    ax.set_ylabel("along-bearing bias [mm]")
    ax.set_title("Each pixel statistic fails at the OPPOSITE end of the footprint\n"
                 "bottom edge is biased looking down; box centre is biased looking flat",
                 fontweight="bold", fontsize=11)
    ax.set_ylim(-62.0, 46.0)
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate("bottom edge: the silhouette base\nis not the contact point\nwhen seen from above",
                xy=(56, -50), fontsize=8.5, color="#0072B2", ha="center", va="top",
                bbox=dict(fc="white", ec="#0072B2", alpha=0.9, pad=2.5), zorder=8)
    ax.annotate("box centre: assumed-height error\namplified by 1/tan(elevation)",
                xy=(32, 40), fontsize=8.5, color="#009E73", ha="center", va="top",
                bbox=dict(fc="white", ec="#009E73", alpha=0.9, pad=2.5), zorder=8)
    ax.axvline(40.0, color="#777777", ls=":", lw=1.2, zorder=2)
    ax.text(40.0, -58, "crossover ~40 deg", fontsize=8, color="#555555", ha="center")
    fig.text(0.5, -0.02,
             "Same clear detections through both statistics, so this is paired. Elevation is "
             "computed from the 6.1 m mount height. Neither curve is fitted -- each point is "
             "the mean along-bearing bias in a 2 m range bin. The crossing near 40 deg is why "
             "a single fixed-plane statistic cannot be unbiased across a whole footprint.",
             ha="center", va="top", fontsize=7.4, color="#333333", wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_e2_pixel_statistic_vs_elevation.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", default=str(DEFAULT_RESIDUALS))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    residuals = Path(args.residuals).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_camera = load(residuals)
    stats = {c: binned_stats(b["d"], b["e"]) for c, b in sorted(per_camera.items())}

    fig, (ax_acc, ax_con) = plt.subplots(1, 2, figsize=(13.0, 4.9))
    for ax in (ax_acc, ax_con):
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_xlabel("ground distance from camera [m]")

    # ---- (a) accuracy: median with inter-quartile band, pooled + per camera ----
    pooled_d = np.concatenate([b["d"] for b in per_camera.values()])
    pooled_e = np.concatenate([b["e"] for b in per_camera.values()])
    pooled = binned_stats(pooled_d, pooled_e)
    mid = [s["mid_m"] for s in pooled]
    ax_acc.fill_between(mid, [s["q25_mm"] for s in pooled], [s["q75_mm"] for s in pooled],
                        color="#888888", alpha=0.22, zorder=2, label="pooled inter-quartile")
    for camera, rows in stats.items():
        ax_acc.plot([s["mid_m"] for s in rows], [s["median_mm"] for s in rows],
                    marker="o", ms=4, lw=1.5, color=CAMERA_COLORS[camera], zorder=4,
                    label=camera.replace("camera_", ""))
    ax_acc.plot(mid, [s["median_mm"] for s in pooled], color="#222222", lw=2.6, zorder=5,
                label="pooled median")
    ax_acc.set_ylabel("position error [mm]")
    ax_acc.set_title("(a)  ACCURACY improves with distance\n"
                     "median error falls; near the camera is the most biased",
                     fontweight="bold", fontsize=10)
    ax_acc.legend(fontsize=7.5, ncol=2)
    ax_acc.set_ylim(bottom=0)

    # ---- (b) consistency: spread, the quantity a covariance has to get right ----
    for camera, rows in stats.items():
        ax_con.plot([s["mid_m"] for s in rows], [s["std_mm"] for s in rows],
                    marker="s", ms=4, lw=1.5, color=CAMERA_COLORS[camera], zorder=4,
                    label=camera.replace("camera_", ""))
    pooled_std = [s["std_mm"] for s in pooled]
    ax_con.plot(mid, pooled_std, color="#222222", lw=2.6, zorder=5, label="pooled")
    ax_con.axhline(float(np.mean(pooled_std)), color="#D55E00", ls="--", lw=1.5, zorder=3,
                   label=f"one constant ({np.mean(pooled_std):.0f} mm)")
    ax_con.set_ylabel("spread of the error, std [mm]")
    ax_con.set_title("(b)  CONSISTENCY degrades with distance\n"
                     "spread roughly doubles — a single constant fits neither end",
                     fontweight="bold", fontsize=10)
    ax_con.legend(fontsize=7.5, ncol=2)
    ax_con.set_ylim(bottom=0)

    fig.suptitle("Measurement uncertainty is not constant across a camera footprint",
                 fontweight="bold", fontsize=12.5, y=1.02)
    fig.text(0.5, -0.06,
             f"Baseline path: plain inverse perspective mapping, zero parameters "
             f"({len(pooled_e)} clear detections, four cameras, eight robot headings, "
             f"{BIN_WIDTH_M:g} m bins with at least {MIN_BIN_N} samples). Both trends hold on "
             "every camera independently, so they are properties of the viewing geometry "
             "rather than of one mount. Near the camera the steep view turns the silhouette "
             "offset into a large but repeatable ground error; far away the projection "
             "amplifies pixel noise into spread. Ground truth measures the error and never "
             "enters the projection.",
             ha="center", va="top", fontsize=7.4, color="#333333", wrap=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_e1_error_vs_distance.{ext}", bbox_inches="tight")
    plt.close(fig)

    # ---- fig_e2: the pixel statistic interacts with viewing angle -----------------
    candidate_csv = residuals.parent / "grid_residuals_candidate.csv"
    crossover = {}
    if candidate_csv.is_file():
        crossover = fig_e2(residuals, candidate_csv, out_dir)

    summary = {
        "pixel_statistic_vs_elevation": crossover,
        "residuals_csv": str(residuals.relative_to(REPO)),
        "bin_width_m": BIN_WIDTH_M, "min_bin_n": MIN_BIN_N,
        "n_clear_detections": int(len(pooled_e)),
        "pooled": pooled,
        "per_camera": stats,
        "spread_growth_near_to_peak": {
            "near_std_mm": pooled[0]["std_mm"],
            "peak_std_mm": max(s["std_mm"] for s in pooled),
            "factor": max(s["std_mm"] for s in pooled) / pooled[0]["std_mm"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"{'range':>10} {'n':>5} {'median':>9} {'std':>8}")
    for s in pooled:
        print(f"{s['lo_m']:>4.0f}-{s['hi_m']:<4.0f} {s['n']:>5} "
              f"{s['median_mm']:>8.1f}mm {s['std_mm']:>7.1f}mm")
    g = summary["spread_growth_near_to_peak"]
    print(f"\nspread {g['near_std_mm']:.0f} mm -> {g['peak_std_mm']:.0f} mm "
          f"({g['factor']:.1f}x)")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
