#!/usr/bin/env python3
"""Zero-shot 'what-if': predict the observability impact of a LAYOUT CHANGE from
geometry alone — no new data collection.

This is the offline core of the transfer claim and a commissioning what-if tool:
edit the world (drop a tall pallet in an aisle) and the geometry prior instantly
re-predicts where the fixed camera goes blind. Validating the prediction against a
fresh capture needs the sim; the prediction itself is free and immediate.

Output: logs/geometry_visibility_prior/demo/whatif_layout_change.png
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
OUT = REPO / "logs" / "geometry_visibility_prior" / "demo"; OUT.mkdir(parents=True, exist_ok=True)
Z_MARKER, TAU_OCC = 0.35, 0.10
REL, CHG, INK, MUTED = "RdYlGn", "RdBu", "#1a1a1a", "#6b6b6b"

# the layout change: a 2.6 m pallet stack dropped mid-aisle A2 (currently open + well-seen)
NEW_STACK = gv.Prism("dropped_pallet_A2", xmin=-1.25, xmax=-0.70, ymin=-0.35, ymax=0.20, zmin=0.0, zmax=2.6)


def main():
    meta = gv.load_gp_artifact_geometry(ART)
    drive = gv.prisms_from_json(json.loads(__import__("yaml").safe_load(open(CFG))["driveable_geometry_json"]))
    xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    dmask = gv.in_any_prism(xs, ys, drive)
    fov = gv.fov_projection_grid(cam, xs, ys, Z_MARKER)
    jac = gv.projection_jacobian_scale(cam, xs, ys, Z_MARKER)
    emp = meta["P_mean_map"]; base = dmask & fov["fov_mask"] & np.isfinite(emp)
    occ = meta["prisms"]; ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    cam_xy = (float(meta["camera_pos"][0]), float(meta["camera_pos"][1]))

    def reliability(prisms):
        clear = gv.raycast_min_clearance(cam, xs, ys, prisms, Z_MARKER)
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clear,
                                     px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
                                     img_w=cam.img_width, img_h=cam.img_height,
                                     tau_clearance=TAU_OCC)["visibility_score"]

    base_score = reliability(occ)
    # calibrate baseline to detection-prob; apply the SAME map to the edited world
    A = np.vstack([base_score[base], np.ones(base.sum())]).T
    a_, b_ = np.linalg.lstsq(A, emp[base], rcond=None)[0]
    to_prob = lambda s: np.clip(a_ * s + b_, 0.02, 0.98)
    before = to_prob(base_score)
    after = to_prob(reliability(occ + [NEW_STACK]))
    change = np.where(base, after - before, np.nan)

    lost = base & (before - after > 0.15)
    cov_b = float((before[base] > 0.5).mean()); cov_a = float((after[base] > 0.5).mean())
    print(f"trackable driveable area: {cov_b*100:.0f}% -> {cov_a*100:.0f}%  "
          f"(new blind cells: {int(lost.sum())})")

    def scene(a2, extra=None, extra_ec="k"):
        for p in occ:
            a2.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin,
                                   fill=True, fc="#e8e8e8", ec="#9a9a9a", lw=0.5, zorder=3))
        if extra is not None:
            a2.add_patch(Rectangle((extra.xmin, extra.ymin), extra.xmax-extra.xmin, extra.ymax-extra.ymin,
                                   fill=True, fc="#8a1c1c", ec="k", lw=1.4, zorder=4))
        a2.plot([cam_xy[0]], [cam_xy[1]], "*", ms=15, color="#1f5fd0", mec="w", mew=0.8, zorder=6)
        a2.set_xlim(ext[0], ext[1]); a2.set_ylim(ext[2], ext[3])
        a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    scene(ax[0]); im0 = ax[0].imshow(np.where(base, before, np.nan), origin="lower", extent=ext,
                                     cmap=REL, vmin=0, vmax=1, zorder=2)
    ax[0].set_title("Before — current layout", fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=6)
    ax[0].set_xlabel("predicted reliability", fontsize=8.5, color=MUTED)
    fig.colorbar(im0, ax=ax[0], shrink=0.82, pad=0.02).set_label("P(reliable)", fontsize=8)

    scene(ax[1], extra=NEW_STACK); im1 = ax[1].imshow(np.where(base, after, np.nan), origin="lower", extent=ext,
                                                      cmap=REL, vmin=0, vmax=1, zorder=2)
    ax[1].set_title("After — 2.6 m pallet dropped in aisle A2", fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=6)
    ax[1].set_xlabel("predicted reliability (red box = new obstacle)", fontsize=8.5, color=MUTED)
    fig.colorbar(im1, ax=ax[1], shrink=0.82, pad=0.02).set_label("P(reliable)", fontsize=8)

    scene(ax[2], extra=NEW_STACK); m = float(np.nanmax(np.abs(change)))
    im2 = ax[2].imshow(change, origin="lower", extent=ext, cmap=CHG, vmin=-m, vmax=m, zorder=2)
    ax[2].set_title("Predicted change — the new blind spot", fontsize=12.5, fontweight="bold", color=INK, loc="left", pad=6)
    ax[2].set_xlabel(f"red = reliability lost · trackable area {cov_b*100:.0f}% → {cov_a*100:.0f}%",
                     fontsize=8.5, color=MUTED)
    fig.colorbar(im2, ax=ax[2], shrink=0.82, pad=0.02).set_label("after − before", fontsize=8)

    fig.suptitle("Zero-shot what-if: geometry predicts the observability impact of a layout change — no data collection",
                 fontsize=14, fontweight="bold", color=INK)
    fig.savefig(OUT / "whatif_layout_change.png", dpi=130, facecolor="white")
    print(f"wrote {OUT/'whatif_layout_change.png'}")


if __name__ == "__main__":
    main()
