#!/usr/bin/env python3
"""Generate tutorial figures used by the active README/docs surface."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _latest_matching_dir(root: Path, required_files: tuple[str, ...]) -> Path:
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if all((child / name).is_file() for name in required_files):
            candidates.append(child)
    if not candidates:
        raise RuntimeError(f"No matching capture/run directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str, default=np.nan) -> float:
    raw = str(row.get(key, "")).strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _plot_capture_tutorial(capture_dir: Path, output_path: Path) -> None:
    raw_rows = _load_csv(capture_dir / "raw_detection_samples.csv")
    agg_rows = _load_csv(capture_dir / "aggregated_detection_samples.csv")
    with np.load(capture_dir / "empirical_visibility_gp.npz", allow_pickle=False) as data:
        xs = np.asarray(data["xs"], dtype=float)
        ys = np.asarray(data["ys"], dtype=float)
        p_mean = np.asarray(data["P_mean_map"], dtype=float) if "P_mean_map" in data.files else np.asarray(data["P_map"], dtype=float)
        p_cons = np.asarray(data["P_conservative_map"], dtype=float) if "P_conservative_map" in data.files else np.asarray(data["P_map"], dtype=float)
        camera_pos = np.asarray(data["camera_pos"], dtype=float).reshape(-1)

    raw_x = np.asarray([_float(r, "state_x", _float(r, "x")) for r in raw_rows], dtype=float)
    raw_y = np.asarray([_float(r, "state_y", _float(r, "y")) for r in raw_rows], dtype=float)
    raw_usable = np.asarray([_float(r, "usable_label", _float(r, "detected_label", 0.0)) for r in raw_rows], dtype=float)

    agg_x = np.asarray([_float(r, "x_center", _float(r, "x")) for r in agg_rows], dtype=float)
    agg_y = np.asarray([_float(r, "y_center", _float(r, "y")) for r in agg_rows], dtype=float)
    agg_usable = np.asarray([_float(r, "usable_rate_mean", _float(r, "detected_rate_mean", 0.0)) for r in agg_rows], dtype=float)

    extent = [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5), constrained_layout=True, sharex=True, sharey=True)

    sc0 = axes[0].scatter(raw_x, raw_y, c=raw_usable, cmap="RdYlGn", vmin=0.0, vmax=1.0, s=10, alpha=0.65)
    axes[0].set_title("Raw Driving Labels")
    axes[0].set_xlabel("state x [m]")
    axes[0].set_ylabel("state y [m]")
    axes[0].scatter(camera_pos[0], camera_pos[1], c="cyan", marker="^", s=55, edgecolors="black", linewidths=0.4)

    sc1 = axes[1].scatter(agg_x, agg_y, c=agg_usable, cmap="viridis", vmin=0.0, vmax=1.0, s=28, edgecolors="black", linewidths=0.25)
    axes[1].set_title("Aggregated Usability Rate")
    axes[1].set_xlabel("state x [m]")
    axes[1].scatter(camera_pos[0], camera_pos[1], c="cyan", marker="^", s=55, edgecolors="black", linewidths=0.4)

    im = axes[2].imshow(p_cons, extent=extent, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    axes[2].set_title("Fitted GP Conservative Field")
    axes[2].set_xlabel("state x [m]")
    axes[2].scatter(camera_pos[0], camera_pos[1], c="cyan", marker="^", s=55, edgecolors="black", linewidths=0.4)

    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.03, label="usable label")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.03, label="usable rate")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03, label="p_vis(x,y)")
    fig.suptitle("Empirical Visibility Artifact: supporting samples, cell rates, and fitted GP field", fontsize=12)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_observation_model_tutorial(output_path: Path, *, r_visible_uv: float = 2.5, r_miss_uv: float = 420.0, visibility_power: float = 3.0) -> None:
    p_vis = np.linspace(0.0, 1.0, 401)
    p_vis_eff = np.clip(p_vis**visibility_power, 1e-4, 1.0 - 1e-4)
    r_plan_var = p_vis_eff * (r_visible_uv**2) + (1.0 - p_vis_eff) * (r_miss_uv**2)
    r_plan_std = np.sqrt(r_plan_var)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axes[0].plot(p_vis, p_vis, linewidth=2.0, linestyle="--", color="gray", label=r"$p_{\mathrm{vis}}$")
    axes[0].plot(p_vis, p_vis_eff, linewidth=2.5, color="tab:blue", label=r"$p_{\mathrm{vis,eff}} = p_{\mathrm{vis}}^\gamma$")
    axes[0].set_title("Visibility Compression")
    axes[0].set_xlabel(r"predicted visibility $p_{\mathrm{vis}}$")
    axes[0].set_ylabel("effective visibility weight")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")

    axes[1].plot(p_vis, r_plan_std, linewidth=2.5, color="tab:red")
    axes[1].axhline(r_visible_uv, linestyle="--", linewidth=1.5, color="tab:green", label=r"$r_{\mathrm{visible}}$")
    axes[1].axhline(r_miss_uv, linestyle="--", linewidth=1.5, color="tab:orange", label=r"$r_{\mathrm{miss}}$")
    axes[1].set_title(r"Planned Observation Noise $\sqrt{\mathrm{diag}(R_{\mathrm{plan}})}$")
    axes[1].set_xlabel(r"predicted visibility $p_{\mathrm{vis}}$")
    axes[1].set_ylabel("planned image std [px]")
    axes[1].legend(loc="best")

    fig.suptitle("How the GP Visibility Field Enters the Planner", fontsize=12)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _plot_state_pipeline_tutorial(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.5), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.04, 0.33, 0.2, 0.34, "camera image\n+ detector"),
        (0.30, 0.33, 0.2, 0.34, "pixel pose\n(u,v)"),
        (0.56, 0.33, 0.2, 0.34, "BEV state\n(x,y)"),
        (0.82, 0.33, 0.14, 0.34, "theta from\nodom fallback"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#eef4fb", edgecolor="#1f4e79", linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    for x0, x1 in ((0.24, 0.30), (0.50, 0.56), (0.76, 0.82)):
        ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50), arrowprops=dict(arrowstyle="->", linewidth=2.0, color="#1f4e79"))
    ax.text(0.5, 0.12, "Current estimator used in the thesis-facing runtime: camera-derived x,y with odometry-backed heading.", ha="center", va="center", fontsize=11)
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def _generate_run_figures(repo_root: Path, experiment_dir: Path, output_dir: Path) -> None:
    plot_script = repo_root / "scripts" / "plot_visibility_run.py"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-doc-figures")
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            str(experiment_dir),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=env,
    )
    rename_map = {
        "field_story.png": "planner_field_story.png",
        "run_timeseries.png": "planner_run_timeseries.png",
    }
    for old_name, new_name in rename_map.items():
        src = output_dir / old_name
        if src.is_file():
            dst = output_dir / new_name
            if dst.is_file():
                dst.unlink()
            shutil.move(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else repo_root / "docs" / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_dir = (
        args.capture_dir.expanduser().resolve()
        if args.capture_dir else
        _latest_matching_dir(repo_root / "logs" / "visibility_capture", ("raw_detection_samples.csv", "aggregated_detection_samples.csv", "empirical_visibility_gp.npz"))
    )
    experiment_dir = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir else
        _latest_matching_dir(repo_root / "logs" / "experiments", ("experiment.csv", "perception.csv", "run_manifest.json", "visibility_artifacts.npz"))
    )

    _plot_capture_tutorial(capture_dir, output_dir / "visibility_capture_tutorial.png")
    _plot_observation_model_tutorial(output_dir / "observation_model_tutorial.png")
    _plot_state_pipeline_tutorial(output_dir / "state_pipeline_tutorial.png")
    _generate_run_figures(repo_root, experiment_dir, output_dir)

    manifest = {
        "capture_dir": str(capture_dir),
        "experiment_dir": str(experiment_dir),
        "generated": [
            "visibility_capture_tutorial.png",
            "observation_model_tutorial.png",
            "state_pipeline_tutorial.png",
            "planner_field_story.png",
            "planner_run_timeseries.png",
        ],
    }
    (output_dir / "figure_sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
