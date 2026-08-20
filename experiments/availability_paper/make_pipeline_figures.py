#!/usr/bin/env python3
"""Per-arm figures: how each planning map is BUILT, and what route it then picks.

One folder per arm under logs/studies/availability_paper/figures/:

    C1_blind/            C2_operational_gp/    C3_mono_depth/    C4_depth_plus_gp/
      01_map_construction.png    — the pipeline that produces the field
      02_routes.png              — the route that field selects, on hard tasks

ROUTE OBJECTIVE. These figures select routes under an explicitly availability-weighted
objective, `length + K * integral (1 - p_use) ds`. That is NOT the runtime EFE
objective, which figure 06 shows is dominated by goal-reaching risk and does not act
on availability at all. These panels therefore show what each MAP implies a planner
should do, not what the deployed planner does.

Run:
    python3 experiments/availability_paper/make_pipeline_figures.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import render_all as base  # noqa: E402

_spec = importlib.util.spec_from_file_location("e3", HERE / "e3_route_discrimination/run_experiment.py")
e3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e3)  # type: ignore

FIGURES = C.OUT_ROOT / "figures"
DPI = 165
#: Weight on unobserved exposure, in metres of path per metre of fully-unobserved
#: driving. Stated on every figure; not tuned to produce a result.
K_EXPOSURE = 3.0
LAMBDAS = (0.0, 0.5, 2.0, 5.0, 20.0, 50.0)

#: Deliberately harder than a straight corridor: multi-turn routes across the
#: warehouse, chosen by measured turn count and length-over-straight-line ratio.
HARD_TASKS = ("mc_wp1_to_wp3", "mc_tour_L", "rob_hardA", "mc_m3_sw2ne_diag")

DAYZERO = C.REPO / "logs/studies/multicamera_commissioning_bigwarehouse/actual_commissioning_20260715/analysis/final_01/inputs"
GP_ROOT = C.REPO / "logs/visibility_comparison/spawn_grid_20260727/gp"
MONO_MAPS = C.MONO_DEPTH_MAPS
MONO_PRED = (C.REPO / "experiments/usable_observation/supervisor_comparison/10_monocular_depth_results"
             / "four_camera/predictions/unidepth_v2_vits14/s01_t0000400ms_external_camera__unidepth_v2_vits14.npz")
RGB = C.REPO / "logs/studies/dynamic_world_oracle/s01_box_in_aisle/run01/rgb/external_camera/t0000400ms.png"

ARMS = {
    # Directory keys are retained for compatibility with existing references.
    # A-labels prevent these availability sources from being confused with the
    # registered E4 planner conditions C1/C2/C3.
    "C1_blind": dict(title="A0 — constant availability baseline", field=None, survey=False),
    "C2_operational_gp": dict(title="A1 — operational GP (needs a surveyed model)",
                              field=C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz",
                              survey=True),
    "C3_mono_depth": dict(title="A2 — monocular depth (no survey)",
                          field=C.OUT_ROOT / "mono_depth_planner_v1/fused_planner_four_camera.npz", survey=False),
    "C4_depth_plus_gp": dict(title="A3 — monocular depth + GP residual (no survey)",
                             field=C.OUT_ROOT / "depth_gp_planner_v1/fused_planner_four_camera.npz", survey=False),
}


def load_field(path: Path | None, like: np.ndarray) -> np.ndarray:
    if path is None:
        return np.full_like(like, 0.5, dtype=float)
    return np.asarray(np.load(path)["P_conservative_plan_map"], dtype=float)


def exposure_objective(path: np.ndarray, field: np.ndarray, xs, ys) -> tuple[float, float, float]:
    """(objective, length, exposure) for a dense path."""
    L = e3.path_length(path)
    n = max(2, int(L / 0.05))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, L, n)
    pts = np.column_stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])
    p = C.sample_field_at(field, xs, ys, pts)
    exposure = float(np.sum(1.0 - p) * (L / max(n - 1, 1)))
    return L + K_EXPOSURE * exposure, L, exposure


def draw(ax, field, ap, title, *, cmap_field=True):
    if cmap_field:
        base.draw_field(ax, field, ap.xs, ap.ys, ap.driveable, ap.prisms, title="")
        for t in list(ax.texts):
            t.set_clip_on(True)
    ax.set_title(title, fontsize=9.5, weight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(labelsize=7)


def map_construction(arm: str, ap, out: Path) -> None:
    """The pipeline that produces this arm's planning field."""
    meta = ARMS[arm]
    fused = load_field(meta["field"], ap.driveable.astype(float))
    cam = "camera_A"

    if arm == "C1_blind":
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        draw(ax, fused, ap, "One availability number, everywhere")
        ax.text(0.5, 0.06, "no spatial field is built:\nevery pose receives the same\navailability value",
                transform=ax.transAxes, ha="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        stages = "There is no spatial estimator pipeline. A0 is the field-source baseline."
    elif arm == "C2_operational_gp":
        fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.6))
        prior = np.load(DAYZERO / f"{cam}_dayzero_prior.npz")["P_mean_map"]
        draw(axes[0], prior, ap, "1. Geometric day-zero prior\n(raycast through the SURVEYED model)")
        draw(axes[1], prior, ap, "2. Detector outcomes\n2,202 spawn poses, camera A")
        ev = ap.events[cam]
        hit = ev["hit"] > 0.5
        axes[1].scatter(ev["xy"][~hit, 0], ev["xy"][~hit, 1], s=4, c="#d62728", alpha=.55, label="missed", zorder=7)
        axes[1].scatter(ev["xy"][hit, 0], ev["xy"][hit, 1], s=4, c="#1a7f37", alpha=.55, label="detected", zorder=7)
        axes[1].legend(fontsize=7, loc="lower left", framealpha=.9)
        post = np.load(GP_ROOT / cam / "det_hit_expected_kernel_gp.npz")["P_mean_map"]
        draw(axes[2], post, ap, "3. GP posterior for camera A\n(prior corrected by the outcomes)")
        draw(axes[3], fused, ap, "4. Fused over four cameras\n(noisy-OR) — used for planning")
        stages = "surveyed geometry -> detector outcomes -> per-camera GP -> noisy-OR fusion"
    elif arm == "C3_mono_depth":
        fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.6))
        if RGB.is_file():
            axes[0].imshow(plt.imread(RGB)); axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[0].set_title("1. The camera's own RGB frame\n(no survey, no extra hardware)", fontsize=9.5, weight="bold")
        if MONO_PRED.is_file():
            d = np.load(MONO_PRED)["depth"]
            im = axes[1].imshow(d, cmap="magma"); axes[1].set_xticks([]); axes[1].set_yticks([])
            plt.colorbar(im, ax=axes[1], fraction=.035, pad=.02, label="metres")
        axes[1].set_title("2. Monocular metric depth\n(floor-anchored via calibration)", fontsize=9.5, weight="bold")
        vis = np.load(MONO_MAPS)
        vx, vy = np.asarray(vis["xs"], float), np.asarray(vis["ys"], float)
        pv = C._resample_to_grid(vx, vy, np.asarray(vis["external_camera__0p4__p_visible"], float), ap.xs, ap.ys)
        draw(axes[2], pv, ap, "3. Depth-buffer raycast, camera A\nvisible fraction per floor cell")
        draw(axes[3], fused, ap, "4. Calibrated + fused over four cameras\n— used for planning")
        stages = "camera RGB -> monocular depth -> occlusion raycast -> calibration + noisy-OR fusion"
    else:  # C4
        fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.6))
        prior = np.load(C.OUT_ROOT / f"mono_depth_planner_v1/gp/{cam}/det_hit_expected_kernel_gp.npz")["P_mean_map"]
        draw(axes[0], prior, ap, "1. Monocular-depth prior, camera A\n(from C3 — still no survey)")
        draw(axes[1], prior, ap, "2. Detector outcomes\n2,202 spawn poses, camera A")
        ev = ap.events[cam]; hit = ev["hit"] > 0.5
        axes[1].scatter(ev["xy"][~hit, 0], ev["xy"][~hit, 1], s=4, c="#d62728", alpha=.55, zorder=7)
        axes[1].scatter(ev["xy"][hit, 0], ev["xy"][hit, 1], s=4, c="#1a7f37", alpha=.55, zorder=7)
        post = np.load(C.OUT_ROOT / f"depth_gp_planner_v1/gp/{cam}/det_hit_expected_kernel_gp.npz")["P_mean_map"]
        draw(axes[2], post, ap, "3. GP residual ON the depth prior\ncamera A posterior")
        draw(axes[3], fused, ap, "4. Fused over four cameras\n— used for planning")
        stages = "camera RGB -> monocular depth -> GP residual from outcomes -> noisy-OR fusion"

    fig.suptitle(f"{meta['title']}   —   how the availability field is built", fontsize=13.5, weight="bold")
    fig.text(0.5, 0.02, f"{stages}\n"
                        r"Output is $p_{use}$ only, on a fixed 0-1 scale. Planner conditions C1/C2/C3 and covariance "
                        r"maps are shown in separate figures. EXPLORATORY availability source.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(out / "01_map_construction.png", dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out}/01_map_construction.png")


def routes(arm: str, ap, drive, candidates: dict, out: Path) -> list[dict]:
    meta = ARMS[arm]
    field = load_field(meta["field"], ap.driveable.astype(float))
    ref = load_field(ARMS["C2_operational_gp"]["field"], ap.driveable.astype(float))

    fig, axes = plt.subplots(1, len(HARD_TASKS), figsize=(5.0 * len(HARD_TASKS), 5.9))
    rows = []
    for ax, task in zip(np.atleast_1d(axes), HARD_TASKS):
        cands = candidates[task]
        scored = [(exposure_objective(p, field, ap.xs, ap.ys), p) for p in cands]
        (obj, L, expo), best = min(scored, key=lambda t: t[0][0])
        shortest = min(cands, key=e3.path_length)
        _, Ls, _ = exposure_objective(shortest, field, ap.xs, ap.ys)
        _, _, expo_ref = exposure_objective(best, ref, ap.xs, ap.ys)
        _, _, expo_ref_short = exposure_objective(shortest, ref, ap.xs, ap.ys)

        draw(ax, field, ap, "")
        ax.plot(shortest[:, 0], shortest[:, 1], color="#111111", lw=2.2, ls=":", label="shortest route")
        ax.plot(best[:, 0], best[:, 1], color="#ff7f0e", lw=3.0, label="route this map picks")
        ax.plot(best[0, 0], best[0, 1], "o", color="#1a7f37", ms=10, zorder=9)
        ax.plot(best[-1, 0], best[-1, 1], "*", color="#b0271f", ms=15, zorder=9)
        detour = L - Ls
        ax.set_title(f"{task}\n{L:.1f} m ({detour:+.2f} m vs shortest)\n"
                     f"unobserved exposure {expo_ref:.1f} m (shortest {expo_ref_short:.1f} m)",
                     fontsize=9.5, weight="bold")
        rows.append(dict(arm=arm, task=task, length_m=f"{L:.3f}", detour_m=f"{detour:.3f}",
                         exposure_own=f"{expo:.3f}", exposure_reference=f"{expo_ref:.3f}",
                         exposure_reference_shortest=f"{expo_ref_short:.3f}"))
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left", framealpha=.92)

    fig.suptitle(f"{meta['title']}   —   the route this map selects on hard, multi-turn tasks",
                 fontsize=13.5, weight="bold")
    fig.text(0.5, 0.02,
             f"Route chosen by minimising  length + {K_EXPOSURE:.0f} x unobserved exposure  over a shared candidate set, "
             "on the mask eroded by the 0.25 m keep-in contract.\n"
             "This is an availability-weighted objective, NOT the runtime EFE objective — figure 06 shows the deployed "
             "planner does not act on availability. Exposure is scored on the A1 operational-GP field so sources are comparable.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(out / "02_routes.png", dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out}/02_routes.png")
    return rows


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.parse_args()
    ap = C.build_apparatus()
    cl = e3.clearance_grid(ap.xs, ap.ys, ap.ctx["driveable_prisms"])
    drive = ap.driveable & (cl >= e3.REQUIRED_CLEARANCE_M)

    all_fields = {k: load_field(v["field"], ap.driveable.astype(float)) for k, v in ARMS.items()}
    candidates: dict[str, list] = {}
    for task in HARD_TASKS:
        spec = ap.tasks[task]
        s, g = spec["start"], spec["goal"]
        S = (float(s["x"]), float(s["y"])); G = (float(g["x"]), float(g["y"]))
        seen, cands = set(), []
        for f in all_fields.values():
            for lam in LAMBDAS:
                p = e3.dense_route(f, drive, ap.xs, ap.ys, S, G, lam)
                key = np.round(p, 3).tobytes()
                if key not in seen:
                    seen.add(key); cands.append(p)
        candidates[task] = cands
        print(f"{task}: {len(cands)} unique candidates")

    rows = []
    for arm in ARMS:
        out = FIGURES / arm
        out.mkdir(parents=True, exist_ok=True)
        print(f"{arm}:")
        map_construction(arm, ap, out)
        rows += routes(arm, ap, drive, candidates, out)
    C.write_csv(FIGURES / "per_arm_routes.csv",
                ("arm","task","length_m","detour_m","exposure_own","exposure_reference","exposure_reference_shortest"),
                rows)
    print(f"\nwrote {FIGURES}/per_arm_routes.csv")


if __name__ == "__main__":
    main()
