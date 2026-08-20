#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-factorized-successor")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import common as C
import decision_planner as D


def apparatus():
    directory = C.REPO / "experiments/availability_paper"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    name = "availability_common_for_successor_figure"
    spec = importlib.util.spec_from_file_location(name, directory / "common.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.build_apparatus()


def ground_covariance(camera_model, point: np.ndarray, covariance_uv: np.ndarray) -> np.ndarray:
    u, v, _ = camera_model.world_to_pixel(float(point[0]), float(point[1]), 0.0)
    eps = 0.25
    up, um = camera_model.pixel_to_world(u + eps, v), camera_model.pixel_to_world(u - eps, v)
    vp, vm = camera_model.pixel_to_world(u, v + eps), camera_model.pixel_to_world(u, v - eps)
    if any(item is None for item in (up, um, vp, vm)):
        return np.full((2, 2), np.nan)
    jac = np.column_stack([(np.asarray(up) - np.asarray(um)) / (2 * eps), (np.asarray(vp) - np.asarray(vm)) / (2 * eps)])
    return jac @ covariance_uv @ jac.T


def draw_warehouse(ax, driveable: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    ax.contour(xs, ys, driveable.astype(float), levels=[0.5], colors="#30343b", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


def main() -> None:
    heldout = json.loads((C.OUT / "heldout_BC/result.json").read_text())
    r_summary = json.loads((C.OUT / "rcond/summary.json").read_text())
    p = C.load_p(); xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    fused = C.fused_p(p, C.HOLDOUT_CAMERAS)
    app = apparatus()
    task = "full_traverse_handover"
    row = next(item for item in heldout["tasks"] if item["task"] == task)
    shortest = np.asarray(row["shortest"]["path"], float)
    selected = np.asarray(row["ds_route"]["path"], float)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), constrained_layout=True)
    image = axes[0].imshow(fused, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="viridis", vmin=0, vmax=1, aspect="equal")
    draw_warehouse(axes[0], app.driveable, xs, ys)
    axes[0].set_title("(a) held-out B+C $p_{use}(x)$")
    fig.colorbar(image, ax=axes[0], fraction=0.046, label="$p_{use}$")

    r = np.load(C.OUT / "rcond/r_cond_uv.npz")
    ids = [str(value) for value in r["camera_ids"]]
    models = {camera: D.camera_model_from_world(C.WORLD, include_name=D.INCLUDE[camera]) for camera in C.HOLDOUT_CAMERAS}
    stride_y, stride_x = 8, 8
    xq, yq = xs[::stride_x], ys[::stride_y]
    sigma = np.full((len(yq), len(xq)), np.nan)
    choice = np.argmax(np.stack([p[f"P_{camera}_map"][::stride_y, ::stride_x] for camera in C.HOLDOUT_CAMERAS]), axis=0)
    for iy, y in enumerate(yq):
        for ix, x in enumerate(xq):
            camera = C.HOLDOUT_CAMERAS[int(choice[iy, ix])]
            ci = ids.index(camera); gy, gx = iy * stride_y, ix * stride_x
            cov_uv = np.asarray([[r["R_uu_px2"][ci, gy, gx], r["R_uv_px2"][ci, gy, gx]], [r["R_uv_px2"][ci, gy, gx], r["R_vv_px2"][ci, gy, gx]]])
            cov_xy = ground_covariance(models[camera], np.asarray([x, y]), cov_uv)
            if np.isfinite(cov_xy).all(): sigma[iy, ix] = math.sqrt(max(np.linalg.eigvalsh(cov_xy)))
    image = axes[1].imshow(sigma, origin="lower", extent=[xq[0], xq[-1], yq[0], yq[-1]], cmap="magma", aspect="equal")
    draw_warehouse(axes[1], app.driveable, xs, ys)
    axes[1].set_title("(b) induced ground uncertainty")
    fig.colorbar(image, ax=axes[1], fraction=0.046, label="major-axis $1\sigma$ [m]")

    # True-scale 95% ellipses.  Width and height are in metres; no display multiplier.
    ellipse_points = C.resample_path(selected, 1.3)
    for point in ellipse_points:
        values = [float(C.sample(np.asarray(p[f"P_{camera}_map"]), xs, ys, point[None, :])[0]) for camera in C.HOLDOUT_CAMERAS]
        camera = C.HOLDOUT_CAMERAS[int(np.argmax(values))]
        ci = ids.index(camera); ix = int(np.argmin(abs(xs-point[0]))); iy = int(np.argmin(abs(ys-point[1])))
        cov_uv = np.asarray([[r["R_uu_px2"][ci, iy, ix], r["R_uv_px2"][ci, iy, ix]], [r["R_uv_px2"][ci, iy, ix], r["R_vv_px2"][ci, iy, ix]]])
        cov_xy = ground_covariance(models[camera], point, cov_uv)
        if not np.isfinite(cov_xy).all(): continue
        values_e, vectors = np.linalg.eigh(cov_xy); order = np.argsort(values_e)[::-1]
        values_e, vectors = values_e[order], vectors[:, order]
        angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
        scale95 = math.sqrt(5.9914645471)
        axes[1].add_patch(Ellipse(point, 2*scale95*math.sqrt(values_e[0]), 2*scale95*math.sqrt(values_e[1]), angle=angle, fill=False, edgecolor="white", linewidth=0.65, alpha=0.9))

    axes[2].imshow(fused, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="Greys", vmin=0, vmax=1, alpha=0.38, aspect="equal")
    draw_warehouse(axes[2], app.driveable, xs, ys)
    axes[2].plot(shortest[:, 0], shortest[:, 1], "--", color="#d1495b", linewidth=1.5, label="shortest")
    axes[2].plot(selected[:, 0], selected[:, 1], "-", color="#00798c", linewidth=2.0, label="DS-Route")
    axes[2].scatter(shortest[0,0], shortest[0,1], s=28, c="#222", marker="o", zorder=4)
    axes[2].scatter(shortest[-1,0], shortest[-1,1], s=38, c="#222", marker="*", zorder=4)
    axes[2].set_title("(c) map changes the solved route")
    axes[2].legend(loc="lower right", frameon=True, fontsize=8)
    fig.suptitle(f"Factorized observation model — {task} (ellipses shown at true scale)", fontsize=11)

    out = C.OUT / "figures"; out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"factorized_fields_and_route.{ext}", dpi=220)
    plt.close(fig)

    dev = json.loads((C.OUT / "offline_development/gate.json").read_text())
    fig, ax = plt.subplots(figsize=(7.4, 3.5), constrained_layout=True)
    labels = ["blind L", "W→E", "handover", "tall shadow"]
    x = np.arange(4); width = 0.34
    dev_values = [100*item["expected_longest_miss_reduction_fraction"] for item in dev["tasks"]]
    hold_values = [100*item["operational_expected_gap_reduction_fraction"] for item in heldout["tasks"]]
    ax.bar(x-width/2, dev_values, width, label="development A+B", color="#5875a4")
    ax.bar(x+width/2, hold_values, width, label="held-out B+C", color="#2a9d58")
    ax.axhline(0, color="#333", linewidth=.7); ax.set_xticks(x, labels); ax.set_ylabel("expected longest blackout reduction [%]")
    ax.set_title("Route-level result; closed loop stopped by conditional-covariance gate")
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"route_gate_results.{ext}", dpi=220)
    plt.close(fig)

    meeting = C.REPO / "logs/studies/availability_paper/figures"
    meeting.mkdir(parents=True, exist_ok=True)
    for source, target in ((out / "factorized_fields_and_route.png", meeting / "13_factorized_fields_and_route.png"), (out / "route_gate_results.png", meeting / "14_successor_gate_results.png")):
        shutil.copyfile(source, target)
    provenance = {
        "inputs": [str((C.OUT / "rcond/summary.json").relative_to(C.REPO)), str((C.OUT / "heldout_BC/result.json").relative_to(C.REPO))],
        "rcond_selected_model": r_summary["selected_model"],
        "ellipse_scale": "true metric 95% covariance ellipse; no display enlargement",
    }
    C.write_json(out / "provenance.json", provenance)
    print(out)


if __name__ == "__main__":
    main()
