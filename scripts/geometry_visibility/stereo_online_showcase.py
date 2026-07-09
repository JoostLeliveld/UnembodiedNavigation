#!/usr/bin/env python3
"""Stereo-initialized reliability GP, refined by driving around collecting datapoints.

Cold start: a STEREO camera reconstructs the height map -> geometry visibility ->
calibrated detection-reliability prior (the GP init, before any driving).

Online: the robot drives REAL logged trajectories. Each detection is applied AT
THE BELIEF position and SHAPED BY THE REAL EKF COVARIANCE, with evidence mass
conserved per datapoint:
  * well-localized (small covariance)  -> sharp, tall deposit -> lots of confidence at one spot
  * poorly-localized (large covariance)-> diffuse, low deposit spread over the belief ellipse
This is the input-noise GP update: 1/peak-confidence grows with position uncertainty.

Outputs (logs/geometry_visibility_prior/demo/):
  stereo_online_showcase.gif      the driving/refinement video
  stereo_online_mechanism.png     well- vs poorly-localized datapoint, side by side
"""
from __future__ import annotations
import csv, glob, json, pathlib, sys
import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
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

Z_MARKER, TAU_OCC = 0.35, 0.10
L_BASE = 0.30       # GP smoothness floor (kernel width for a perfectly-localized point) [m]
EVID_MASS = 0.6     # evidence deposited per detection (conserved; spread by covariance)
PRIOR_STRENGTH = 2.0
RUNS = [
    "route_apron_to_a3_mid/C2/seed0",
    "route_apron_to_a2_mid/C2/seed0",
    "control_west_to_a1_low/C1/seed0",
    "route_west_to_a1_upper/C1/seed1",   # real belief drift: cov up to (0.45 m)^2, 51 misses
]


def nearest(stamps, t):
    return int(np.argmin(np.abs(stamps - t)))


def load_events(run):
    d = glob.glob(str(CAMP / run / "*"))[0]
    perc = list(csv.DictReader(open(pathlib.Path(d) / "perception.csv")))
    exp = list(csv.DictReader(open(pathlib.Path(d) / "experiment.csv")))

    def fcol(rows, k):
        return np.array([float(r[k]) if r.get(k, "") not in ("", "nan") else np.nan for r in rows])

    est = fcol(exp, "stamp")
    # planner BELIEF covariance (planner_cov_*) reflects real position uncertainty
    # (major axis up to 0.45 m, tracks actual drift); state_cov_* is the tight,
    # overconfident EKF covariance and must NOT be used for trust here.
    cxx, cxy, cyy = fcol(exp, "planner_cov_x"), fcol(exp, "planner_cov_xy"), fcol(exp, "planner_cov_y")
    ev = []
    for r in perc:
        if r.get("detected") not in ("0", "1") or r.get("state_available") != "1":
            continue
        if r.get("state_x") in ("", "nan") or r.get("true_x") in ("", "nan"):
            continue
        t = float(r["log_stamp"]); j = nearest(est, t)
        Cxx = cxx[j] if not np.isnan(cxx[j]) else 4e-4
        Cxy = cxy[j] if not np.isnan(cxy[j]) else 0.0
        Cyy = cyy[j] if not np.isnan(cyy[j]) else 4e-4
        ev.append(dict(xb=float(r["state_x"]), yb=float(r["state_y"]),
                       xt=float(r["true_x"]), yt=float(r["true_y"]),
                       det=int(r["detected"]), C=np.array([[Cxx, Cxy], [Cxy, Cyy]])))
    return ev


def deposit(a, b, xs, ys, e):
    """Add one detection: normalized anisotropic Gaussian (mass EVID_MASS) at the
    BELIEF position, covariance = EKF covariance + base smoothing. Mass-conserving,
    so a wider (more uncertain) footprint has a proportionally lower peak."""
    xb, yb = e["xb"], e["yb"]
    S = e["C"] + (L_BASE ** 2) * np.eye(2)
    evals = np.linalg.eigvalsh(S)
    reach = 3.0 * float(np.sqrt(evals.max()))
    ix0, ix1 = np.searchsorted(xs, xb - reach), np.searchsorted(xs, xb + reach)
    iy0, iy1 = np.searchsorted(ys, yb - reach), np.searchsorted(ys, yb + reach)
    sx, sy = slice(max(ix0, 0), ix1), slice(max(iy0, 0), iy1)
    KX, KY = np.meshgrid(xs[sx], ys[sy])
    inv = np.linalg.inv(S); detS = np.linalg.det(S)
    dx, dy = KX - xb, KY - yb
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    g = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(detS))     # integrates to 1
    contrib = EVID_MASS * g
    if e["det"]:
        a[sy, sx] += contrib
    else:
        b[sy, sx] += contrib


def cov_ellipse(ax, x, y, C, n=2.0, **kw):
    vals, vecs = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(vecs[1, np.argmax(vals)], vecs[0, np.argmax(vals)]))
    w, h = 2 * n * np.sqrt(np.maximum(vals[::-1], 1e-9))
    ax.add_patch(Ellipse((x, y), w, h, angle=ang, fill=False, **kw))


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
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    occ = meta["prisms"]
    gx, gy = np.meshgrid(xs, ys)

    def vis(hm):
        # raycast against a height map via nearest-cell lookup
        tgt = np.stack([gx, gy, np.full_like(gx, Z_MARKER)], -1).reshape(-1, 3)
        cp = np.asarray(cam.cam_pos); t = np.linspace(0.02, 0.98, 40).reshape(1, -1, 1)
        s = cp.reshape(1, 1, 3) + t * (tgt[:, None, :] - cp.reshape(1, 1, 3))
        ix = np.clip(np.searchsorted(xs, s[..., 0]), 0, len(xs) - 1)
        iy = np.clip(np.searchsorted(ys, s[..., 1]), 0, len(ys) - 1)
        clear = (s[..., 2] - hm[iy, ix]).min(axis=1).reshape(gx.shape)
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=clear,
                                     px_per_m_min=jac["px_per_m_min"], u=fov["u"], v=fov["v"],
                                     img_w=cam.img_width, img_h=cam.img_height,
                                     tau_clearance=TAU_OCC)["visibility_score"]

    # --- STEREO cold-start init ------------------------------------------------
    rng_map = np.sqrt((gx - cam.cam_pos[0]) ** 2 + (gy - cam.cam_pos[1]) ** 2 + cam.cam_pos[2] ** 2)
    sig_stereo = (rng_map ** 2) / (1.5 * 640.0) * 0.4
    h_stereo = np.clip(h_true + np.random.RandomState(0).normal(0, 1, h_true.shape) * sig_stereo * (h_true > 0), 0, None)
    score = vis(h_stereo)
    A = np.vstack([score[base], np.ones(base.sum())]).T
    aa, bb = np.linalg.lstsq(A, emp[base], rcond=None)[0]
    p0 = np.clip(aa * score + bb, 0.02, 0.98)
    a0, b0 = p0 * PRIOR_STRENGTH, (1 - p0) * PRIOR_STRENGTH

    # --- stream real detections ------------------------------------------------
    events = []
    for run in RUNS:
        ev = load_events(run); events += ev
    sig_of = lambda e: float(np.sqrt(np.linalg.eigvalsh(e["C"]).max()))
    print(f"{len(events)} events; sigma_major range {min(map(sig_of,events)):.3f}..{max(map(sig_of,events)):.3f} m")

    a, b = a0.copy(), b0.copy()
    S0 = a0 + b0
    stride = max(1, len(events) // 110)
    snaps, path = [], []
    for i, e in enumerate(events):
        deposit(a, b, xs, ys, e)
        path.append((e["xb"], e["yb"]))
        if i % stride == 0 or i == len(events) - 1:
            m = a / (a + b)
            vm = base & ((a + b - S0) > 0.15)
            rmse = float(np.sqrt(np.mean((m[vm] - emp[vm]) ** 2))) if vm.any() else np.nan
            snaps.append(dict(m=m.astype(np.float32), conf=(a + b - S0).astype(np.float32),
                              e=e, sig=sig_of(e), rmse=rmse, n=i + 1,
                              path=np.array(path)))
    r_prior = float(np.sqrt(np.mean(((a0 / (a0 + b0))[base & ((a + b - S0) > 0.15)] - emp[base & ((a + b - S0) > 0.15)]) ** 2)))
    print(f"RMSE vs empirical (visited): stereo-prior {r_prior:.3f} -> after driving {snaps[-1]['rmse']:.3f}")

    # --- mechanism figure: well- vs poorly-localized datapoint -----------------
    # Rendered on a LOCAL grid centred on each datapoint (grid-independent physics),
    # so the true extremes render fully with no warehouse-edge clipping.
    e_good = min(events, key=sig_of); e_bad = max(events, key=sig_of)
    print(f"mechanism examples: well σ={sig_of(e_good):.3f} | poor σ={sig_of(e_bad):.3f}")
    pw = 1.6
    lx = np.linspace(-pw, pw, 200); LX, LY = np.meshgrid(lx, lx)
    peak_ref = EVID_MASS / (2 * np.pi * L_BASE ** 2)

    def local_deposit(C):
        S = C + (L_BASE ** 2) * np.eye(2); inv = np.linalg.inv(S)
        q = inv[0, 0]*LX*LX + 2*inv[0, 1]*LX*LY + inv[1, 1]*LY*LY
        return EVID_MASS * np.exp(-0.5*q) / (2*np.pi*np.sqrt(np.linalg.det(S)))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (e, title) in enumerate([(e_good, "well-localized"), (e_bad, "poorly-localized")]):
        im = ax[k].imshow(local_deposit(e["C"]), origin="lower", extent=[-pw, pw, -pw, pw],
                          cmap="magma", vmin=0, vmax=peak_ref)
        cov_ellipse(ax[k], 0, 0, e["C"] + L_BASE**2*np.eye(2), n=2, ec="cyan", lw=2)
        ax[k].plot([0], [0], "o", color="cyan", ms=7, label="belief pos")
        dxt, dyt = e["xt"]-e["xb"], e["yt"]-e["yb"]
        if abs(dxt) < pw and abs(dyt) < pw:
            ax[k].plot([dxt], [dyt], "x", color="w", ms=10, mew=2.5, label="true pos")
        ax[k].set_xlim(-pw, pw); ax[k].set_ylim(-pw, pw)
        ax[k].set_title(f"{title}:  σ = {sig_of(e):.3f} m", fontsize=11)
        ax[k].set_xlabel("Δx from belief [m]"); ax[k].set_ylabel("Δy [m]"); ax[k].legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax[:2], shrink=0.7, label="confidence deposited")
    for e, c, lab in [(e_good, "#2a9d3a", "well-localized"), (e_bad, "#d1495b", "poorly-localized")]:
        S = e["C"] + L_BASE**2*np.eye(2); peak = EVID_MASS/(2*np.pi*np.sqrt(np.linalg.det(S)))
        r = np.linspace(-1.5, 1.5, 200); sig1 = np.sqrt(np.linalg.eigvalsh(S).max())
        ax[2].plot(r, peak*np.exp(-0.5*(r/sig1)**2), color=c, lw=2.2, label=f"{lab} (σ={sig1:.2f}m)")
    ax[2].set_title("evidence profile through the datapoint\n(same total mass; confident = tall/narrow)")
    ax[2].set_xlabel("distance from belief [m]"); ax[2].set_ylabel("confidence deposited"); ax[2].legend(fontsize=8)
    fig.suptitle("How one datapoint updates the GP: well-localized concentrates confidence; poorly-localized spreads it over the belief", fontsize=12)
    fig.savefig(OUT / "stereo_online_mechanism.png", dpi=125, bbox_inches="tight"); plt.close(fig)
    print("wrote stereo_online_mechanism.png")

    # --- animation -------------------------------------------------------------
    conf_max = float(np.percentile(snaps[-1]["conf"][base], 99)) or 1.0
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
    rr = [s["rmse"] for s in snaps]; nn = [s["n"] for s in snaps]

    def draw(fr):
        s = snaps[fr]; e = s["e"]
        for a_ in ax: a_.clear()
        # GP mean + path + current datapoint + belief ellipse
        ax[0].imshow(np.where(base, s["m"], np.nan), origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
        for p in occ:
            ax[0].add_patch(Rectangle((p.xmin,p.ymin),p.xmax-p.xmin,p.ymax-p.ymin,fill=False,ec="k",lw=0.5))
        if len(s["path"]) > 1:
            ax[0].plot(s["path"][:,0], s["path"][:,1], "-", color="w", lw=0.7, alpha=0.6)
        col = "#2a9d3a" if e["det"] else "#d1495b"
        cov_ellipse(ax[0], e["xb"], e["yb"], e["C"] + L_BASE**2*np.eye(2), n=2, ec=col, lw=2)
        ax[0].plot([e["xb"]],[e["yb"]], "o", color=col, ms=6)
        ax[0].plot([cam.cam_pos[0]],[cam.cam_pos[1]], "*", color="r", ms=10)
        ax[0].set_title(f"GP reliability mean (stereo init → driving)\nevents={s['n']}  σ={s['sig']:.2f}m  "
                        + ("DETECT" if e['det'] else "MISS"), fontsize=10)
        # confidence gained
        ax[1].imshow(np.where(base, np.clip(s["conf"],0,conf_max), np.nan), origin="lower", extent=ext,
                     cmap="magma", vmin=0, vmax=conf_max)
        for p in occ:
            ax[1].add_patch(Rectangle((p.xmin,p.ymin),p.xmax-p.xmin,p.ymax-p.ymin,fill=False,ec="w",lw=0.5))
        ax[1].set_title("confidence gained (evidence)\nconfident fixes = sharp/bright, uncertain = diffuse", fontsize=10)
        # convergence
        ax[2].plot(nn[:fr+1], rr[:fr+1], "-", color="#2a9d3a", lw=2.2)
        ax[2].axhline(r_prior, ls="--", color="gray", lw=1, label="stereo prior only")
        ax[2].set_xlim(0, nn[-1]); ax[2].set_ylim(0, max(rr)*1.15)
        ax[2].set_xlabel("datapoints collected"); ax[2].set_ylabel("RMSE vs empirical GP (visited)")
        ax[2].legend(loc="upper right", fontsize=9); ax[2].set_title("refinement toward true reliability", fontsize=10)
        for a_ in (ax[0], ax[1]): a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle("Stereo-initialized reliability GP + driving: each datapoint applied at belief, shaped by EKF certainty", fontsize=13)

    anim = FuncAnimation(fig, draw, frames=len(snaps), interval=120)
    anim.save(OUT / "stereo_online_showcase.gif", writer=PillowWriter(fps=8), dpi=90)
    print(f"wrote stereo_online_showcase.gif ({len(snaps)} frames)")
    for fr, tag in [(int(0.5*(len(snaps)-1)), "mid"), (len(snaps)-1, "final")]:
        draw(fr); fig.savefig(OUT / f"stereo_online_{tag}.png", dpi=95)
    print("wrote stereo_online_mid.png, stereo_online_final.png")


if __name__ == "__main__":
    main()
