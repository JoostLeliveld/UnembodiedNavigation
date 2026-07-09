#!/usr/bin/env python3
"""Operator / commissioning dashboard for the camera-observability GP.

Composes the whole system into one operational picture, from REAL data:
  Row 1 (live):   localization reliability | data confidence | robot routes
  Row 2 (setup):  day-one prior            | change vs prior | multi-camera coverage

Color means the same thing everywhere: reliability is a green(trackable)->red(blind)
status ramp; confidence is a single-hue magnitude ramp; change is diverging at 0
(red = reality worse than the geometry prior predicted -> investigate).
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
from stereo_online_showcase import load_events, deposit, RUNS, PRIOR_STRENGTH, Z_MARKER, TAU_OCC, L_BASE
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
OUT = REPO / "logs" / "geometry_visibility_prior" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

REL = "RdYlGn"      # reliability status ramp: 0 red (blind) -> 1 green (trackable)
CONF = "Blues"      # confidence magnitude ramp
CHG = "RdBu"        # change diverging (red = worse than predicted)
INK, MUTED = "#1a1a1a", "#6b6b6b"


def raycast_hm(cam, xs, ys, hm, n=40):
    gx, gy = np.meshgrid(xs, ys)
    tgt = np.stack([gx, gy, np.full_like(gx, Z_MARKER)], -1).reshape(-1, 3)
    cp = np.asarray(cam.cam_pos); t = np.linspace(0.02, 0.98, n).reshape(1, -1, 1)
    s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys) - 1)
    return (s[..., 2] - hm[iy, ix]).min(axis=1).reshape(gx.shape)


def visibility(cam, xs, ys, fov, jac, hm):
    return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=raycast_hm(cam, xs, ys, hm),
                                 px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
                                 img_w=cam.img_width, img_h=cam.img_height,
                                 tau_clearance=TAU_OCC)["visibility_score"]


def draw_scene(ax, occ, cams, ext):
    for p in occ:
        ax.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin,
                               fill=True, fc="#e8e8e8", ec="#9a9a9a", lw=0.5, zorder=3))
    for (cx, cy), lbl in cams:
        ax.plot([cx], [cy], marker="*", ms=15, color="#1f5fd0", mec="w", mew=0.8, zorder=6)
        ax.annotate(lbl, (cx, cy), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=7, color=INK, zorder=6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    meta = gv.load_gp_artifact_geometry(ART)
    drive = gv.prisms_from_json(json.loads(__import__("yaml").safe_load(open(CFG))["driveable_geometry_json"]))
    xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    dmask = gv.in_any_prism(xs, ys, drive)
    fov = gv.fov_projection_grid(cam, xs, ys, Z_MARKER)
    jac = gv.projection_jacobian_scale(cam, xs, ys, Z_MARKER)
    emp = meta["P_mean_map"]; base = dmask & fov["fov_mask"] & np.isfinite(emp)
    h_true = gv.build_height_map(xs, ys, meta["prisms"])["h_max"]
    occ = meta["prisms"]; ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    gx, gy = np.meshgrid(xs, ys)
    cam1_xy = (float(meta["camera_pos"][0]), float(meta["camera_pos"][1]))

    # cold-start STEREO prior -> calibrated reliability
    rng = np.sqrt((gx-cam.cam_pos[0])**2 + (gy-cam.cam_pos[1])**2 + cam.cam_pos[2]**2)
    h_st = np.clip(h_true + np.random.RandomState(0).normal(0, 1, h_true.shape) * (rng**2/(1.5*640)*0.4)*(h_true > 0), 0, None)
    score = visibility(cam, xs, ys, fov, jac, h_st)
    A = np.vstack([score[base], np.ones(base.sum())]).T
    a_, b_ = np.linalg.lstsq(A, emp[base], rcond=None)[0]
    prior = np.clip(a_*score + b_, 0.02, 0.98)
    a0, b0 = prior*PRIOR_STRENGTH, (1-prior)*PRIOR_STRENGTH

    # drive real runs -> refined GP + confidence + routes
    a, b = a0.copy(), b0.copy(); routes = []
    for run in RUNS:
        ev = load_events(run)
        routes.append(np.array([(e["xb"], e["yb"]) for e in ev]))
        for e in ev:
            deposit(a, b, xs, ys, e)
    gp = a/(a+b); conf = (a+b) - (a0+b0)
    visited = base & (conf > 0.15)
    residual = np.where(visited, gp - prior, np.nan)

    # multi-camera union (commissioning lever): add a mirrored north camera
    cam2 = gv.ObliqueCameraModel(cam_pos=(0.0, 5.6, 4.8), look_at=(0.0, 1.8, 0.0),
                                 img_width=cam.img_width, img_height=cam.img_height, fov_h_rad=cam.fov_h_rad)
    fov2 = gv.fov_projection_grid(cam2, xs, ys, Z_MARKER)
    jac2 = gv.projection_jacobian_scale(cam2, xs, ys, Z_MARKER)
    v1 = visibility(cam, xs, ys, fov, jac, h_true)
    v2 = visibility(cam2, xs, ys, fov2, jac2, h_true)
    union = 1 - (1-v1)*(1-v2)
    cov1 = float((v1[dmask] > 0.5).mean()); covU = float((union[dmask] > 0.5).mean())

    # ---------------- figure ----------------
    from matplotlib.colors import PowerNorm
    from matplotlib.cm import ScalarMappable
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    def panel(a_, field, cmap, title, sub, norm=None, vmin=0, vmax=1, cams=((cam1_xy, "CAM"),), cbar_lbl=""):
        draw_scene(a_, occ, cams, ext)
        kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
        im = a_.imshow(field, origin="lower", extent=ext, cmap=cmap, zorder=2, **kw)
        a_.set_title(title, fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=6)
        a_.set_xlabel(sub, fontsize=8.5, color=MUTED)
        cb = fig.colorbar(im, ax=a_, shrink=0.82, pad=0.02); cb.set_label(cbar_lbl, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return im

    # Row 1 — live operational picture
    panel(ax[0, 0], np.where(base, gp, np.nan), REL, "Localization reliability",
          "green = robot trackable · red = blind spot", cbar_lbl="P(reliable detection)")
    cmax = float(np.percentile(conf[base], 97))
    panel(ax[0, 1], np.where(base, conf, np.nan), CONF, "Data confidence",
          "dark = confirmed by real data · pale = prior only",
          norm=PowerNorm(0.5, vmin=0, vmax=cmax), cbar_lbl="evidence collected")
    draw_scene(ax[0, 2], occ, ((cam1_xy, "CAM"),), ext)
    for path in routes:
        if len(path) < 2: continue
        c = gp[np.clip(np.searchsorted(ys, path[:, 1]), 0, len(ys)-1),
               np.clip(np.searchsorted(xs, path[:, 0]), 0, len(xs)-1)]
        seg = np.concatenate([path[:-1, None, :], path[1:, None, :]], axis=1)
        lc = LineCollection(seg, cmap=REL, norm=plt.Normalize(0, 1), lw=3, zorder=5)
        lc.set_array(c[:-1]); ax[0, 2].add_collection(lc)
    ax[0, 2].set_title("Robot routes", fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=6)
    ax[0, 2].set_xlabel("driven paths, colored by reliability where they went", fontsize=8.5, color=MUTED)
    sm = ScalarMappable(cmap=REL, norm=plt.Normalize(0, 1)); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax[0, 2], shrink=0.82, pad=0.02)
    cb.set_label("P(reliable detection)", fontsize=8); cb.ax.tick_params(labelsize=7)

    # Row 2 — commissioning & health
    panel(ax[1, 0], np.where(base, prior, np.nan), REL, "Day-one prior (no driving)",
          "stereo geometry only — before any data", cbar_lbl="P(reliable detection)")
    rmax = float(np.nanmax(np.abs(residual)))
    panel(ax[1, 1], residual, CHG, "Change vs prior (health)",
          "red = reality worse than predicted → investigate", vmin=-rmax, vmax=rmax, cbar_lbl="observed − prior")
    panel(ax[1, 2], np.where(dmask, union, np.nan), REL, "Multi-camera coverage",
          f"2nd camera → trackable area {cov1*100:.0f}% → {covU*100:.0f}%",
          cams=(((0.0, -5.5), "CAM 1"), ((0.0, 5.6), "CAM 2")), cbar_lbl="P(reliable), union")

    fig.suptitle("Camera-observability GP — operational dashboard    ·    warehouse_aws, real runs",
                 fontsize=15, fontweight="bold", color=INK)
    fig.savefig(OUT / "operator_dashboard.png", dpi=130, facecolor="white")
    print(f"wrote {OUT/'operator_dashboard.png'}")
    print(f"coverage 1-cam {cov1*100:.0f}% -> 2-cam {covU*100:.0f}%; "
          f"visited cells {int(visited.sum())}; residual |max| {np.nanmax(np.abs(residual)):.2f}")


if __name__ == "__main__":
    main()
