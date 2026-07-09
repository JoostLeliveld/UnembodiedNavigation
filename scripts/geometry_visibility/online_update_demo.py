#!/usr/bin/env python3
"""(1) GP after different initializations, and (2) a video of the GP updating as
the robot drives REAL logged trajectories, with each datapoint TRUSTED by the
robot's own EKF position certainty (gated) vs weighted equally (naive).

All trajectories, detections, and covariances are from real honest_campaign_v1
runs on warehouse_aws. The learned field is scored against the empirical YOLO GP.
"""
from __future__ import annotations
import csv, glob, json, pathlib, sys
from collections import deque
import numpy as np
from scipy.ndimage import gaussian_filter

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(REPO / "scripts" / "geometry_visibility"))
import geometry_visibility as gv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
CFG = REPO / "scripts/visibility_comparison/warehouse_visibility_campaign.yaml"
CAMP = REPO / "logs/visibility_comparison/honest_campaign_v1"
OUT = REPO / "logs" / "geometry_visibility_prior" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
Z_MARKER, TAU_OCC, H0 = 0.35, 0.10, 1.5
L_KERNEL = 0.55          # RBF lengthscale for online updates [m]
TAU_TRUST = 0.12         # certainty scale: sigma_major beyond this -> distrust [m]
PRIOR_STRENGTH = 2.5     # geometry prior pseudo-count

# curated real runs: cover A2/A3/west + span low->high localization certainty
RUNS = [
    "route_apron_to_a3_mid/C2/seed0",
    "route_apron_to_a2_mid/C2/seed0",
    "control_west_to_a1_low/C1/seed0",
    "route_west_to_a1_upper/C1/seed1",   # high drift (sigma_major up to 0.45 m), 51 misses
]


# --------------------------------------------------------------------------- utils
def interior_holes(free_mask):
    nd = ~free_mask; ny, nx = nd.shape
    seen = np.zeros_like(nd); dq = deque()
    for i in range(ny):
        for j in (0, nx - 1):
            if nd[i, j] and not seen[i, j]: seen[i, j] = True; dq.append((i, j))
    for j in range(nx):
        for i in (0, ny - 1):
            if nd[i, j] and not seen[i, j]: seen[i, j] = True; dq.append((i, j))
    while dq:
        i, j = dq.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx and nd[ni, nj] and not seen[ni, nj]:
                seen[ni, nj] = True; dq.append((ni, nj))
    return nd & ~seen


def raycast_hm(cam, xs, ys, hm, n=40):
    gx, gy = np.meshgrid(xs, ys)
    tgt = np.stack([gx, gy, np.full_like(gx, Z_MARKER)], -1).reshape(-1, 3)
    cp = np.asarray(cam.cam_pos)
    t = np.linspace(0.02, 0.98, n).reshape(1, -1, 1)
    s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
    ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys) - 1)
    return (s[..., 2] - hm[iy, ix]).min(axis=1).reshape(gx.shape)


def affine_to_prob(score, emp, base):
    A = np.vstack([score[base], np.ones(base.sum())]).T
    a, b = np.linalg.lstsq(A, emp[base], rcond=None)[0]
    return np.clip(a * score + b, 0.02, 0.98)


def load_events(run):
    d = glob.glob(str(CAMP / run / "*"))[0]
    perc = list(csv.DictReader(open(pathlib.Path(d) / "perception.csv")))
    exp = list(csv.DictReader(open(pathlib.Path(d) / "experiment.csv")))
    est = np.array([float(r["stamp"]) for r in exp])
    esig = np.array([float(r["state_sigma_major_m"]) if r["state_sigma_major_m"] not in ("", "nan")
                     else np.nan for r in exp])
    ok = ~np.isnan(esig)
    est, esig = est[ok], esig[ok]
    ev = []
    for r in perc:
        if r.get("detected") not in ("0", "1"):
            continue
        if r.get("state_available") != "1" or r.get("state_x") in ("", "nan"):
            continue
        if r.get("true_x") in ("", "nan"):
            continue
        t = float(r["log_stamp"])
        sig = float(np.interp(t, est, esig)) if len(est) else 0.05
        ev.append(dict(t=t, xb=float(r["state_x"]), yb=float(r["state_y"]),
                       xt=float(r["true_x"]), yt=float(r["true_y"]),
                       det=int(r["detected"]), sig=sig))
    return ev


def add_event(a, b, xs, ys, e, w, l=L_KERNEL):
    """Add one detection with weight w and kernel lengthscale l (m)."""
    xb, yb = e["xb"], e["yb"]
    ix0, ix1 = np.searchsorted(xs, xb - 3 * l), np.searchsorted(xs, xb + 3 * l)
    iy0, iy1 = np.searchsorted(ys, yb - 3 * l), np.searchsorted(ys, yb + 3 * l)
    sx, sy = slice(max(ix0, 0), ix1), slice(max(iy0, 0), iy1)
    KX, KY = np.meshgrid(xs[sx], ys[sy])
    k = np.exp(-((KX - xb) ** 2 + (KY - yb) ** 2) / (2 * l * l))
    if e["det"]:
        a[sy, sx] += w * k
    else:
        b[sy, sx] += w * k


# --------------------------------------------------------------------------- main
def main():
    meta = gv.load_gp_artifact_geometry(ART)
    drive = gv.prisms_from_json(json.loads(__import__("yaml").safe_load(open(CFG))["driveable_geometry_json"]))
    xs, ys = meta["xs"], meta["ys"]
    cam = gv.make_camera(meta)
    dmask = gv.in_any_prism(xs, ys, drive)
    fov = gv.fov_projection_grid(cam, xs, ys, Z_MARKER)
    jac = gv.projection_jacobian_scale(cam, xs, ys, Z_MARKER)
    emp = meta["P_mean_map"]
    base = dmask & fov["fov_mask"] & np.isfinite(emp)
    h_true = gv.build_height_map(xs, ys, meta["prisms"])["h_max"]
    holes = interior_holes(dmask)
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    occ = meta["prisms"]

    def vis(hm):
        return gv.compute_visibility(
            fov_mask=fov["fov_mask"], min_clearance=raycast_hm(cam, xs, ys, hm),
            px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
            img_w=cam.img_width, img_h=cam.img_height, tau_clearance=TAU_OCC)["visibility_score"]

    # ---- (1) init comparison -------------------------------------------------
    gx, gy = np.meshgrid(xs, ys)
    rng_map = np.sqrt((gx - cam.cam_pos[0]) ** 2 + (gy - cam.cam_pos[1]) ** 2 + cam.cam_pos[2] ** 2)
    sig_stereo = (rng_map ** 2) / (1.5 * 640.0) * 0.4          # triangulation error grows with range^2
    h_stereo = h_true + np.random.RandomState(0).normal(0, 1, h_true.shape) * sig_stereo * (h_true > 0)
    inits = {
        "SDF (true geometry)": vis(h_true),
        "FREESPACE holes": vis(holes.astype(float) * H0),
        "MONOCULAR (OOD)": vis(gaussian_filter(0.6 * h_true, 1.5)),
        "STEREO": vis(np.clip(h_stereo, 0, None)),
    }
    prob_inits = {k: affine_to_prob(v, emp, base) for k, v in inits.items()}
    panels = list(prob_inits.items()) + [("EMPIRICAL YOLO GP (target)", emp)]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.6))
    for a_, (name, f) in zip(ax, panels):
        im = a_.imshow(np.where(base, f, np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        for p in occ:
            a_.add_patch(Rectangle((p.xmin, p.ymin), p.xmax-p.xmin, p.ymax-p.ymin, fill=False, ec="k", lw=0.6))
        a_.plot([cam.cam_pos[0]], [cam.cam_pos[1]], "*", color="r", ms=11)
        a_.set_title(name, fontsize=10); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("GP detection-reliability map after different initializations (before any driving)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(detect)")
    fig.savefig(OUT / "gp_after_initializations.png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote gp_after_initializations.png")

    # ---- (2) online update from real runs ------------------------------------
    events = []
    for run in RUNS:
        ev = load_events(run)
        events += ev
        print(f"  {run:38s} {len(ev):4d} events  "
              f"(misses {sum(1 for e in ev if e['det']==0)}, sigMax {max(e['sig'] for e in ev):.2f})")
    print(f"total events: {len(events)}")

    p0 = prob_inits["SDF (true geometry)"]
    a0 = (p0 * PRIOR_STRENGTH).copy(); b0 = ((1 - p0) * PRIOR_STRENGTH).copy()
    # three strategies for using the datapoint given its position certainty:
    aN, bN = a0.copy(), b0.copy()   # NAIVE   : full weight, fixed kernel
    aG, bG = a0.copy(), b0.copy()   # GATED   : down-weight uncertain points (discard info)
    aS, bS = a0.copy(), b0.copy()   # SPREAD  : full weight, widen kernel by position uncertainty

    stride = max(1, len(events) // 110)
    snaps = []
    visited = np.zeros_like(dmask)
    for i, e in enumerate(events):
        sig = e["sig"]
        w = float(np.exp(-0.5 * (sig / TAU_TRUST) ** 2))          # trust from real EKF certainty
        l_eff = float(np.sqrt(L_KERNEL ** 2 + sig ** 2))          # widen influence by uncertainty
        add_event(aN, bN, xs, ys, e, 1.0, L_KERNEL)
        add_event(aG, bG, xs, ys, e, w, L_KERNEL)
        add_event(aS, bS, xs, ys, e, 1.0, l_eff)
        ix = int(np.clip(np.searchsorted(xs, e["xb"]), 0, len(xs)-1))
        iy = int(np.clip(np.searchsorted(ys, e["yb"]), 0, len(ys)-1))
        visited[max(iy-6,0):iy+6, max(ix-6,0):ix+6] = True
        if i % stride == 0 or i == len(events) - 1:
            mN, mG, mS = aN/(aN+bN), aG/(aG+bG), aS/(aS+bS)
            vm = visited & base
            rmse = lambda m: float(np.sqrt(np.mean((m[vm]-emp[vm])**2))) if vm.any() else np.nan
            snaps.append(dict(mS=mS.astype(np.float32), mN=mN.astype(np.float32),
                              st=(aS+bS).astype(np.float32),
                              rN=rmse(mN), rG=rmse(mG), rS=rmse(mS),
                              xb=e["xb"], yb=e["yb"], xt=e["xt"], yt=e["yt"],
                              w=w, sig=sig, n=i + 1))
    vm = visited & base
    r_prior = float(np.sqrt(np.mean(((a0/(a0+b0))[vm] - emp[vm]) ** 2)))
    s = snaps[-1]
    print(f"RMSE vs empirical (visited cells): prior-only {r_prior:.3f} | "
          f"naive {s['rN']:.3f} | gated/downweight {s['rG']:.3f} | spread/input-noise {s['rS']:.3f}")

    # ---- animation -----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    rNs=[s["rN"] for s in snaps]; rGs=[s["rG"] for s in snaps]; rSs=[s["rS"] for s in snaps]
    nn=[s["n"] for s in snaps]; ymax=max(max(rNs),max(rGs),max(rSs))*1.1

    def draw(fr):
        s = snaps[fr]
        for a_ in ax.ravel():
            a_.clear()
        # SPREAD (recommended) mean + robot; dot color=trust, size=uncertainty
        ax[0,0].imshow(np.where(base, s["mS"], np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        for p in occ:
            ax[0,0].add_patch(Rectangle((p.xmin,p.ymin),p.xmax-p.xmin,p.ymax-p.ymin,fill=False,ec="k",lw=0.5))
        ax[0,0].plot([s["xt"]],[s["yt"]], "x", color="w", ms=7, mew=2)
        ax[0,0].plot([s["xb"]],[s["yb"]], "o", color=plt.cm.RdYlGn(s["w"]),
                     ms=8 + 40*s["sig"], mec="k")
        ax[0,0].set_title(f"SPREAD (input-noise) GP mean | events={s['n']} | σ={s['sig']:.2f}m trust={s['w']:.2f}", fontsize=10)
        # NAIVE mean
        ax[0,1].imshow(np.where(base, s["mN"], np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        for p in occ:
            ax[0,1].add_patch(Rectangle((p.xmin,p.ymin),p.xmax-p.xmin,p.ymax-p.ymin,fill=False,ec="k",lw=0.5))
        ax[0,1].set_title("NAIVE GP mean (all points, fixed kernel)", fontsize=10)
        # confidence
        ax[1,0].imshow(np.where(base, np.clip(s["st"]/8,0,1), np.nan), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
        for p in occ:
            ax[1,0].add_patch(Rectangle((p.xmin,p.ymin),p.xmax-p.xmin,p.ymax-p.ymin,fill=False,ec="w",lw=0.5))
        ax[1,0].set_title("Accumulated evidence (confidence)", fontsize=10)
        # convergence: three strategies
        ax[1,1].plot(nn[:fr+1], rNs[:fr+1], "-", color="#3a6ea5", lw=2, label="naive (equal trust)")
        ax[1,1].plot(nn[:fr+1], rGs[:fr+1], "-", color="#d1495b", lw=2, label="gated (down-weight uncertain)")
        ax[1,1].plot(nn[:fr+1], rSs[:fr+1], "-", color="#2a9d3a", lw=2.4, label="spread (widen by uncertainty)")
        ax[1,1].axhline(r_prior, ls="--", color="gray", lw=1, label="geometry prior only")
        ax[1,1].set_xlim(0, nn[-1]); ax[1,1].set_ylim(0, ymax)
        ax[1,1].set_xlabel("detections processed"); ax[1,1].set_ylabel("RMSE vs empirical GP (visited)")
        ax[1,1].legend(loc="upper right", fontsize=8); ax[1,1].set_title("Convergence to true reliability", fontsize=10)
        for a_ in (ax[0,0],ax[0,1],ax[1,0]):
            a_.plot([cam.cam_pos[0]],[cam.cam_pos[1]],"*",color="r",ms=10)
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle("Online GP update from real driven trajectories — using EKF position certainty per datapoint",
                     fontsize=13)

    anim = FuncAnimation(fig, draw, frames=len(snaps), interval=120)
    anim.save(OUT / "gp_online_update.gif", writer=PillowWriter(fps=8), dpi=90)
    print(f"wrote gp_online_update.gif  ({len(snaps)} frames)")
    for fr, tag in [(int(0.93 * (len(snaps) - 1)), "westdrift"), (len(snaps) - 1, "final")]:
        draw(fr); fig.savefig(OUT / f"gp_update_{tag}.png", dpi=95)
    print("wrote gp_update_westdrift.png, gp_update_final.png")


if __name__ == "__main__":
    main()
