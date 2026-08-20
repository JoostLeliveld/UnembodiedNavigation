#!/usr/bin/env python3
"""Figure: the routes each availability field sends the robot along, before and after.

Left pair: the nominal warehouse, where the frozen learned field and the recomputed
field agree. Right pair: after the restock, where they do not -- and the ground the
stale field routes through is ground that has since gone dark.

Everything drawn here comes from e3_routes.csv / e3_route_geometry.json.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent


def _load_exact(name: str, path: Path):
    expected = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != expected:
            raise ImportError(f"{name} resolves to an unexpected module")
        return existing
    spec = importlib.util.spec_from_file_location(name, expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {expected}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


C = _load_exact("_reconfiguration_holdout_common", HERE / "common.py")
CL = _load_exact("_reconfiguration_holdout_choose_layout", HERE / "choose_layout.py")
IHF = _load_exact(
    "_reconfiguration_holdout_e3_fields",
    HERE / "e3_availability_routing/independent_heading_fields.py",
)
import numpy.ma as ma       # noqa: E402
ora = C.ora
CL.ora = ora

E3 = C.OUT_ROOT / "e3_availability_routing"
OUT = C.OUT_ROOT / "figures"

SHOW_ARMS = (("gp", "frozen learned field", "#111111"),
             ("mono_depth", "recomputed from the image", "#00b050"))
BUDGET = 0.20
SUBSET = "4"


def truth_field(env: str) -> np.ndarray:
    ev = IHF.load_evaluation_events(env, C.PRIMARY_THRESHOLD)
    any_hit = {}
    for cam in C.CAMERA_SUBSETS[SUBSET]:
        e = ev[cam]
        for i in range(len(e["hit"])):
            k = (round(e["xy"][i, 0], 3), round(e["xy"][i, 1], 3), round(e["theta"][i], 3))
            any_hit[k] = max(any_hit.get(k, 0.0), e["hit"][i])
    keys = {}
    for (x, y, _t), h in any_hit.items():
        keys.setdefault((x, y), []).append(h)
    pts = np.array(list(keys.keys()))
    vals = np.array([float(np.mean(v)) for v in keys.values()])
    xs, ys = C.working_grid()
    gx, gy = np.meshgrid(xs, ys)
    field = np.zeros_like(gx)
    for iy in range(gx.shape[0]):
        d2 = (pts[:, 0][None, :] - gx[iy][:, None]) ** 2 + \
             (pts[:, 1][None, :] - gy[iy][:, None]) ** 2
        field[iy] = vals[np.argmin(d2, axis=1)]
    return field


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((E3 / "e3_routes.csv").open()))
    geom = json.loads((E3 / "e3_route_geometry.json").read_text())
    xs, ys = C.working_grid()
    drive = CL.driveable_mask(C.floor_grid(), CL.lanes())
    segs = CL.rack_segments(C.WORLDS / f"{C.ENV_BY_KEY['L0'].world_name}.world.sdf")

    # the task with the largest disagreement after the restock leads the figure
    def blind(env, task, arm):
        for r in rows:
            if (r["environment"], r["subset"], r["task"], r["arm"]) == (env, SUBSET, task, arm) \
               and abs(float(r["budget"]) - BUDGET) < 1e-9:
                return float(r["blind_true_m"])
        return None

    tasks = sorted({r["task"] for r in rows})
    gaps = [(blind("L1", t, "gp") - blind("L1", t, "mono_depth"), t)
            for t in tasks if blind("L1", t, "gp") is not None]
    gaps.sort(reverse=True)
    task = gaps[0][1]
    print(f"[fig] leading task {task}; gap {gaps[0][0]:.2f} m")

    truth = {e: truth_field(e) for e in ("L0", "L1")}
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.62))
    for ax, env, title in zip(axes, ("L0", "L1"),
                              ("Nominal warehouse", "After the restock")):
        ax.set_facecolor("#ffffff")
        im = ax.pcolormesh(xs, ys, ma.masked_where(~drive, truth[env]),
                           cmap="RdYlBu", vmin=0, vmax=1, shading="nearest")
        for s in segs:
            ax.add_patch(Rectangle((s["xmin"], s["ymin"]), s["xmax"] - s["xmin"],
                                   s["ymax"] - s["ymin"], facecolor="#9a9a9a",
                                   edgecolor="#6b6b6b", lw=0.3, zorder=2))
        for arm, label, colour in SHOW_ARMS:
            key = f"{env}|{SUBSET}|{task}|{BUDGET}|{arm}"
            if key not in geom:
                continue
            p = np.array(geom[key])
            b = blind(env, task, arm)
            ax.plot(p[:, 0], p[:, 1], lw=1.7, color=colour, zorder=4,
                    label=label, solid_capstyle="round",
                    path_effects=[pe.Stroke(linewidth=3.0, foreground="white"),
                                  pe.Normal()])
        key = f"{env}|{SUBSET}|{task}|{BUDGET}|shortest"
        if key in geom:
            p = np.array(geom[key])
            ax.plot(p[:, 0], p[:, 1], lw=0.9, color="#ffffff", ls=(0, (4, 2)), zorder=3,
                    label="shortest route (no availability model)",
                    path_effects=[pe.Stroke(linewidth=1.9, foreground="#404040"),
                                  pe.Normal()])
        ax.set_title(title, fontsize=6.4)
        ax.set_xlabel("x (m)", fontsize=5.8)
        ax.set_aspect("equal"); ax.tick_params(labelsize=4.8)
        txt = (f"blind driving on this task\n"
               f"  shortest route:  {blind(env, task, 'shortest'):.1f} m\n"
               f"  frozen learned:  {blind(env, task, 'gp'):.1f} m\n"
               f"  recomputed:      {blind(env, task, 'mono_depth'):.1f} m")
        ax.text(0.985, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=4.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white", alpha=0.93,
                          edgecolor="#999999"))
    axes[0].set_ylabel("y (m)", fontsize=5.8)
    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02)
    cb.set_label("measured availability: fraction of headings at which\n"
                 "some camera actually detected the robot",
                 fontsize=4.6)
    cb.ax.tick_params(labelsize=4.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=5.0, frameon=False,
               bbox_to_anchor=(0.46, -0.045))
    fig.suptitle("Where each availability field sends the robot, at a 20% detour budget",
                 fontsize=6.8, y=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_routes.{ext}", dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {OUT/'fig3_routes.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
