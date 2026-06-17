#!/usr/bin/env python3
"""Create the warehouse GP pipeline figure without replacing the compact one.

Panels:
  (a) heading-aggregated YOLO-score training samples,
  (b) planner-facing conservative reliability rho_plan,
  (c) induced image-space covariance from the planner-facing covariance mapping.

This is a setup/method figure. The GP is not a traversability map, not a
visibility reward, and not a physical occlusion geometry estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
THESIS = REPO.parent / "thesis-report"

WORLD = "warehouse_aws.world.sdf"
DEFAULT_GP = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
DEFAULT_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
DEFAULT_OUT = THESIS / "figures/campaign/gp_pipeline_aws_v7.pdf"
DEFAULT_PREVIEW = REPO / "paper_artifacts/figures/gp_pipeline_aws.png"


COL = {
    "drive": "#cae8c8",
    "drive_edge": "#1b9850",
    "non": "#f2b8b5",
    "non_edge": "#d73027",
    "rack": "#f2cf23",
    "rack_edge": "#0b6f8a",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gp", type=Path, default=DEFAULT_GP)
    p.add_argument("--world-profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    p.add_argument("--r-visible-uv", type=float, default=2.5)
    p.add_argument("--r-miss-uv", type=float, default=40.0)
    p.add_argument("--min-prob", type=float, default=1e-4)
    return p.parse_args()


def load_profile(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["worlds"][WORLD]


def load_gp(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        required = ["xs", "ys", "X_train", "p_train", "P_conservative_plan_map"]
        missing = [key for key in required if key not in d.files]
        if missing:
            raise RuntimeError(f"GP artifact missing required fields: {missing}")
        payload = {
            "xs": np.asarray(d["xs"], dtype=float),
            "ys": np.asarray(d["ys"], dtype=float),
            "X_train": np.asarray(d["X_train"], dtype=float),
            "p_train": np.asarray(d["p_train"], dtype=float),
            "P_plan": np.asarray(d["P_conservative_plan_map"], dtype=float),
        }
        if "P_mean_map" in d.files:
            payload["P_mean"] = np.asarray(d["P_mean_map"], dtype=float)
        if "F_std_map" in d.files:
            payload["F_std"] = np.asarray(d["F_std_map"], dtype=float)
        return payload


def draw_regions(ax, profile: dict, *, alpha: float = 0.35, edge_alpha: float = 0.65) -> None:
    for region in profile.get("known_2d_regions", []):
        typ = str(region.get("type", ""))
        x0 = float(region["xmin"])
        y0 = float(region["ymin"])
        w = float(region["xmax"]) - x0
        h = float(region["ymax"]) - y0
        if typ == "traversable":
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=COL["drive"], edgecolor=COL["drive_edge"], lw=0.55, alpha=alpha, zorder=2))
        # non_driveable_staging (red) markings are intentionally NOT drawn: they sit
        # outside the driveable zone and are not relevant to the GP pipeline figure.


def draw_racks(ax) -> None:
    rack_xs = [-4.05, -2.00, 0.05, 2.00, 4.15]
    rack_w = 0.55
    # R1 (x=-4.05) is one CONTINUOUS shelf (its mid gap is filled); R2..R5 keep the
    # open mid gap. Match the actual world geometry.
    split_segments = [(-0.82, 1.20), (2.20, 4.25)]
    for x in rack_xs:
        segments = [(-0.82, 4.25)] if abs(x - (-4.05)) < 1e-6 else split_segments
        for y0, y1 in segments:
            ax.add_patch(Rectangle((x - rack_w / 2.0, y0), rack_w, y1 - y0, facecolor=COL["rack"], edgecolor=COL["rack_edge"], lw=0.65, zorder=6))


def style_axis(ax, title: str, *, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=6)
    ax.set_xlim(-5.55, 5.55)
    ax.set_ylim(-5.05, 5.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [m]", fontsize=11.0)
    if show_ylabel:
        ax.set_ylabel(r"$y$ [m]", fontsize=11.0)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xticks([-5, -3, -1, 1, 3, 5])
    ax.set_yticks([-5, -3, -1, 1, 3, 5])
    ax.grid(True, color="#d0d0d0", lw=0.32, alpha=0.42, zorder=1)
    ax.tick_params(labelsize=9.5, length=2)


def induced_log_camera_covariance(p_plan: np.ndarray, *, r_visible_uv: float, r_miss_uv: float, min_prob: float) -> np.ndarray:
    p_eff = np.clip(np.asarray(p_plan, dtype=float), min_prob, 1.0 - min_prob)
    visible_var = float(r_visible_uv) ** 2
    miss_var = float(r_miss_uv) ** 2
    plan_var = 1.0 / np.maximum(p_eff / visible_var + (1.0 - p_eff) / miss_var, 1e-12)
    return 0.5 * np.log(np.clip(plan_var * plan_var, 1e-12, None))


def main() -> int:
    args = parse_args()
    profile = load_profile(args.world_profile)
    gp = load_gp(args.gp)
    xs = gp["xs"]
    ys = gp["ys"]
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    log_camera_covariance = induced_log_camera_covariance(
        gp["P_plan"],
        r_visible_uv=args.r_visible_uv,
        r_miss_uv=args.r_miss_uv,
        min_prob=args.min_prob,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.17, wspace=0.25)

    ax = axes[0]
    # Driveable-floor cells intentionally not drawn: panel (a) focuses on rack geometry + YOLO samples.
    draw_racks(ax)
    # Plot-only cleanup (no GP refit): hide the two aggregated training dots that render
    # inside/on the R0 (leftmost) shelf footprint so the obstacle reads cleanly. The GP fit
    # and all downstream panels are unchanged — only these markers are not drawn.
    _X = gp["X_train"]
    _p = gp["p_train"]
    _r0_x0, _r0_x1, _r0_y0, _r0_y1 = -4.325, -3.775, -0.82, 4.25
    _on_r0 = (
        (_X[:, 0] >= _r0_x0 - 0.02)
        & (_X[:, 0] <= _r0_x1 + 0.08)
        & (_X[:, 1] >= _r0_y0)
        & (_X[:, 1] <= _r0_y1)
    )
    _keep = ~_on_r0
    sc = ax.scatter(
        _X[_keep, 0],
        _X[_keep, 1],
        c=_p[_keep],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=18,
        edgecolor="black",
        linewidth=0.12,
        zorder=8,
    )
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03, label="YOLO score")
    style_axis(ax, "(a) YOLO-score samples", show_ylabel=True)

    ax = axes[1]
    im = ax.imshow(gp["P_plan"], extent=extent, origin="lower", cmap="viridis", vmin=0.0, vmax=0.9, aspect="equal", zorder=0)
    draw_racks(ax)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label=r"$\rho_{\mathrm{plan}}$")
    style_axis(ax, r"(b) conservative planner reliability $\rho_{\mathrm{plan}}$", show_ylabel=False)

    ax = axes[2]
    im = ax.imshow(log_camera_covariance, extent=extent, origin="lower", cmap="magma", aspect="equal", zorder=0)
    draw_racks(ax)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label=r"$\frac{1}{2}\log|R(\mathbf{p})|$")
    style_axis(ax, "(c) induced image-space covariance", show_ylabel=False)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=COL["rack"], edgecolor=COL["rack_edge"], label="rack geometry"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#440154", markeredgecolor="black", markersize=5, label="training sample"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10.0, frameon=False, bbox_to_anchor=(0.5, 0.02))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.preview, dpi=240, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Warehouse learned-observation-reliability pipeline. Raw YOLO scores are "
        "aggregated by sampled ground-plane position, fit with a GP, and converted "
        "to conservative planner-facing reliability rho_plan. The final panel shows "
        "the log image-space covariance induced by mapping rho_plan to observation covariance "
        f"with r_visible={args.r_visible_uv:g}px and r_miss={args.r_miss_uv:g}px. "
        "The GP affects the planner-facing image-space observation covariance; "
        "the floor layer remains a separate known traversability/forbidden-zone layer."
    )
    caption_path = args.preview.with_name("gp_pipeline_aws_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")

    provenance = {
        "figure": "gp_pipeline_aws",
        "world": WORLD,
        "gp": str(args.gp),
        "world_profile": str(args.world_profile),
        "out": str(args.out),
        "preview": str(args.preview),
        "r_visible_uv": float(args.r_visible_uv),
        "r_miss_uv": float(args.r_miss_uv),
        "num_training_points": int(gp["X_train"].shape[0]),
        "rho_plan_min": float(np.nanmin(gp["P_plan"])),
        "rho_plan_max": float(np.nanmax(gp["P_plan"])),
        "rho_plan_mean": float(np.nanmean(gp["P_plan"])),
        "log_camera_covariance_min": float(np.nanmin(log_camera_covariance)),
        "log_camera_covariance_max": float(np.nanmax(log_camera_covariance)),
        "notes": [
            "Saved as gp_pipeline_aws_v7.pdf for the paper and gp_pipeline_aws.png as a preview.",
            "rho_plan is P_conservative_plan_map from the v7b GP artifact.",
            "Forbidden/staging zones are drawn only as floor-layer context; they are not the GP.",
        ],
    }
    provenance_path = args.preview.with_name("gp_pipeline_aws_provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"wrote {args.preview}")
    print(f"wrote {caption_path}")
    print(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
