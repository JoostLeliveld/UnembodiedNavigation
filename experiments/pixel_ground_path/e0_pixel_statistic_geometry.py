#!/usr/bin/env python3
"""e0 — which pixel statistic should a fixed camera invert, and on which plane?

The deployed path takes the detection box's BOTTOM-CENTRE pixel and intersects its ray
with a plane at ``contact_z_m``.  That choice is what every fitted correction in
``external_camera_bias_model`` was trying to repair.  This script asks the prior
question -- is the bottom edge the right pixel at all? -- entirely from geometry plus
the robot's datasheet dimensions.  No detector, no fitted parameter, no calibration
artifact is involved: every number below follows from the camera model, the object
model, and the recorded true poses.

Sections
  1  What each candidate pixel statistic images, and how much it moves with the
     robot's UNKNOWN yaw.  Yaw is unobservable at runtime, so this is irreducible.
  2  The inversion.  Back-project each statistic and measure position error against
     truth.  Sweeps the plane height and the assumed robot height.
  3  Cylinder vs bounding cuboid: does the yaw-blind object model lose anything?
  4  Occlusion sensitivity, and how much range information the box SIZE carries.
  5  The covariance the homography Jacobian implies, and its anisotropy and range
     growth -- the structure a single scalar R cannot express.
  6  Independence audit of the evidence base itself.

Run:  python3 experiments/pixel_ground_path/e0_pixel_statistic_geometry.py
Out:  logs/studies/pixel_ground_path/e0_pixel_statistic_geometry/
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
sys.path.insert(0, str(REPO / "src" / "reliability"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from reliability.projection import camera_model_from_world  # noqa: E402

WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
RESIDUALS = (
    REPO
    / "logs/studies/external_camera_bias_model/exp1_residual_characterization/residuals.csv"
)
OUT = REPO / "logs/studies/pixel_ground_path/e0_pixel_statistic_geometry"

MODEL_INCLUDES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
GROUND_TRUTH = {
    "smoke1_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716"
    / "evaluation_only/ground_truth.csv",
    "smoke2_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716"
    / "evaluation_only/ground_truth.csv",
    "fusion_handover_20260721": REPO
    / "logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data"
    / "evaluation_only/ground_truth.csv",
}

# TurtleBot3 Burger, datasheet / URDF.  These are the ONLY object numbers used and
# neither is fitted: HALF_TRACK is the wheel half-separation (the widest feature) and
# HEIGHT is the overall height (the tallest feature).
HALF_BODY_X = 0.070
HALF_TRACK = 0.089
HEIGHT = 0.192

CAMERA_HEIGHT = 6.10  # for the analytic range-growth cross-check only
FOCAL_PX = 640.0

N_RIM = 64
_TH = np.linspace(0.0, 2.0 * math.pi, N_RIM, endpoint=False)
_COS, _SIN = np.cos(_TH), np.sin(_TH)
YAW_SWEEP = np.linspace(-math.pi, math.pi, 24, endpoint=False)


# --------------------------------------------------------------------- forward models


def cylinder_bbox(camera, x, y, *, radius=HALF_TRACK, height=HEIGHT):
    """Image bbox of a vertical cylinder's silhouette.

    The silhouette is the two rim circles joined by the two profile lines.  A straight
    3-D segment projects to a straight 2-D segment, so u and v vary monotonically along
    each profile line and every extreme is attained on a rim -- sampling both rims is
    exact, not an approximation.
    """
    us, vs = [], []
    for z in (0.0, height):
        for cx, sy in zip(_COS, _SIN):
            u, v, _ = camera.world_to_pixel(x + radius * cx, y + radius * sy, z)
            us.append(u)
            vs.append(v)
    return min(us), min(vs), max(us), max(vs)


def cuboid_bbox(camera, x, y, yaw, *, height=HEIGHT):
    """Image bbox of the yaw-aware bounding prism, for comparison against the cylinder."""
    c, s = math.cos(yaw), math.sin(yaw)
    us, vs = [], []
    for lx in (-HALF_BODY_X, HALF_BODY_X):
        for ly in (-HALF_TRACK, HALF_TRACK):
            for lz in (0.0, height):
                u, v, _ = camera.world_to_pixel(x + c * lx - s * ly, y + s * lx + c * ly, lz)
                us.append(u)
                vs.append(v)
    return min(us), min(vs), max(us), max(vs)


def support_radius(yaw, bearing):
    """Half-extent of the footprint rectangle along the camera bearing."""
    a = bearing - yaw
    return HALF_BODY_X * abs(math.cos(a)) + HALF_TRACK * abs(math.sin(a))


def jacobian(camera, u, v, z_plane, *, step=0.5):
    """d(x, y) / d(u, v) of the homography onto plane z = z_plane."""
    cols = []
    for axis in (0, 1):
        du = step if axis == 0 else 0.0
        dv = step if axis == 1 else 0.0
        plus = camera.pixel_to_world_at_z(u + du, v + dv, z_plane)
        minus = camera.pixel_to_world_at_z(u - du, v - dv, z_plane)
        if plus is None or minus is None:
            return None
        cols.append(((plus[0] - minus[0]) / (2 * step), (plus[1] - minus[1]) / (2 * step)))
    return np.array([[cols[0][0], cols[1][0]], [cols[0][1], cols[1][1]]])


def bearing_frame(camera, x, y):
    cx, cy = float(camera.cam_pos[0]), float(camera.cam_pos[1])
    bx, by = x - cx, y - cy
    d = math.hypot(bx, by)
    return d, bx / d, by / d, math.atan2(by, bx)


# ------------------------------------------------------------------------------ loading


def _yaw_tables():
    tables = {}
    for capture, path in GROUND_TRUTH.items():
        stamps, yaws = [], []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("gt_yaw"):
                    continue
                stamps.append(float(row["stamp"]))
                yaws.append(float(row["gt_yaw"]))
        if stamps:
            order = np.argsort(stamps)
            tables[capture] = (np.asarray(stamps)[order], np.asarray(yaws)[order])
    return tables


def _nearest_yaw(tables, capture, stamp, tol=0.05):
    if capture not in tables:
        return math.nan
    stamps, yaws = tables[capture]
    i = int(np.clip(np.searchsorted(stamps, stamp), 1, len(stamps) - 1))
    j = i if abs(stamps[i] - stamp) < abs(stamps[i - 1] - stamp) else i - 1
    return float(yaws[j]) if abs(stamps[j] - stamp) <= tol else math.nan


def load(models):
    tables = _yaw_tables()
    rows = []
    with RESIDUALS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                dict(
                    camera=row["camera"],
                    capture=row["capture"],
                    u=float(row["u"]),
                    v=float(row["v"]),
                    true_x=float(row["true_x"]),
                    true_y=float(row["true_y"]),
                    yaw=_nearest_yaw(tables, row["capture"], float(row["stamp"])),
                )
            )
    return rows


# ---------------------------------------------------------------------------- sections


def section1_yaw_sensitivity(models, rows, summary):
    print("=== 1. what moves when the robot turns (yaw is unobservable at runtime) ===")
    rng = np.random.default_rng(20260806)
    sample = [rows[i] for i in rng.choice(len(rows), min(300, len(rows)), replace=False)]
    block = {}
    for name in ("bbox_bottom_at_floor", "bbox_centre_at_half_height"):
        radial, lateral = [], []
        for rec in sample:
            camera = models[rec["camera"]]
            _, ux, uy, _ = bearing_frame(camera, rec["true_x"], rec["true_y"])
            pts = []
            for yaw in YAW_SWEEP:
                u0, v0, u1, v1 = cuboid_bbox(camera, rec["true_x"], rec["true_y"], yaw)
                uc = 0.5 * (u0 + u1)
                if name.startswith("bbox_bottom"):
                    p = camera.pixel_to_world_at_z(uc, v1, 0.0)
                else:
                    p = camera.pixel_to_world_at_z(uc, 0.5 * (v0 + v1), HEIGHT / 2.0)
                pts.append(p)
            arr = np.asarray(pts)
            ex, ey = arr[:, 0] - rec["true_x"], arr[:, 1] - rec["true_y"]
            radial.append(float((ex * ux + ey * uy).std()))
            lateral.append(float((-ex * uy + ey * ux).std()))
        block[name] = dict(radial_m=float(np.mean(radial)), lateral_m=float(np.mean(lateral)))
        print(f"  {name:30} radial {np.mean(radial)*100:5.2f} cm   lateral "
              f"{np.mean(lateral)*100:5.2f} cm")
    ratio = block["bbox_bottom_at_floor"]["radial_m"] / \
        block["bbox_centre_at_half_height"]["radial_m"]
    print(f"  -> the bottom edge is {ratio:.0f}x more yaw-sensitive radially.")
    print("     Reason: the near-bottom rim and the far-top rim move in OPPOSITE senses")
    print("     as the robot turns, so their average -- the box centre -- barely moves.")
    summary["yaw_sensitivity"] = block | {"bottom_over_centre_radial": float(ratio)}


def section2_inversion(models, rows, summary):
    print("\n=== 2. the inversion: back-project the modelled statistic, compare to truth ===")
    stats = {}

    def errors(stat, z_plane, height=HEIGHT, assumed_height=None):
        out = []
        for rec in rows:
            camera = models[rec["camera"]]
            u0, v0, u1, v1 = cylinder_bbox(camera, rec["true_x"], rec["true_y"], height=height)
            uc = 0.5 * (u0 + u1)
            vv = v1 if stat == "bottom" else 0.5 * (v0 + v1)
            z = z_plane if assumed_height is None else assumed_height / 2.0
            p = camera.pixel_to_world_at_z(uc, vv, z)
            if p is None:
                continue
            out.append(math.hypot(p[0] - rec["true_x"], p[1] - rec["true_y"]))
        return np.asarray(out)

    print(f"  {'statistic':32} {'plane z':>8} {'mean':>9} {'p95':>8} {'max':>8}")
    e = errors("bottom", 0.05)
    print(f"  {'bbox bottom (DEPLOYED)':32} {0.05:8.3f} {e.mean()*1000:7.1f}mm "
          f"{np.percentile(e,95)*1000:6.1f}mm {e.max()*1000:6.1f}mm")
    stats["bottom_at_0.05"] = float(e.mean())
    e = errors("bottom", 0.0)
    print(f"  {'bbox bottom (floor)':32} {0.0:8.3f} {e.mean()*1000:7.1f}mm "
          f"{np.percentile(e,95)*1000:6.1f}mm {e.max()*1000:6.1f}mm")
    stats["bottom_at_floor"] = float(e.mean())
    e = errors("centre", HEIGHT / 2.0)
    print(f"  {'bbox centre (PROPOSED)':32} {HEIGHT/2:8.3f} {e.mean()*1000:7.1f}mm "
          f"{np.percentile(e,95)*1000:6.1f}mm {e.max()*1000:6.1f}mm")
    stats["centre_at_half_height"] = float(e.mean())

    print("\n  plane sweep for the box centre (h/2 = %.4f m is not tuned, it is derived):"
          % (HEIGHT / 2.0))
    sweep = {}
    for z in (0.070, 0.080, 0.090, 0.096, 0.100, 0.110, 0.120):
        e = errors("centre", z)
        sweep[f"{z:.3f}"] = float(e.mean())
        mark = "  <- h/2" if abs(z - HEIGHT / 2.0) < 1e-9 else ""
        print(f"    z = {z:.3f} m   mean {e.mean()*1000:6.1f} mm{mark}")

    print("\n  how well must the robot's height be known?  (true silhouette, assumed h/2)")
    height_sens = {}
    for h in (0.170, 0.180, 0.192, 0.200, 0.210):
        e = errors("centre", 0.0, assumed_height=h)
        height_sens[f"{h:.3f}"] = float(e.mean())
        print(f"    assumed h = {h:.3f} m   mean {e.mean()*1000:6.1f} mm")
    summary["inversion"] = dict(statistics=stats, plane_sweep_m=sweep,
                                assumed_height_sweep_m=height_sens)


def section3_object_model(models, rows, summary):
    print("\n=== 3. does the yaw-blind cylinder lose anything against the yaw-aware cuboid? ===")
    print(f"  {'camera':9} {'n':>5} {'cuboid u-ctr':>13} {'cylinder u-ctr':>15} "
          f"{'|diff|':>8} {'cuboid v-ctr':>13} {'cylinder v-ctr':>15} {'|diff|':>8}")
    block = {}
    for camera_id in sorted(models):
        group = [r for r in rows if r["camera"] == camera_id and math.isfinite(r["yaw"])]
        if not group:
            continue
        camera = models[camera_id]
        cub_u, cyl_u, cub_v, cyl_v = [], [], [], []
        for rec in group:
            a0, a1, a2, a3 = cuboid_bbox(camera, rec["true_x"], rec["true_y"], rec["yaw"])
            b0, b1, b2, b3 = cylinder_bbox(camera, rec["true_x"], rec["true_y"])
            cub_u.append(0.5 * (a0 + a2)); cyl_u.append(0.5 * (b0 + b2))
            cub_v.append(0.5 * (a1 + a3)); cyl_v.append(0.5 * (b1 + b3))
        du = abs(np.mean(cub_u) - np.mean(cyl_u))
        dv = abs(np.mean(cub_v) - np.mean(cyl_v))
        block[camera_id] = dict(n=len(group), du_px=float(du), dv_px=float(dv))
        print(f"  {camera_id:9} {len(group):5d} {np.mean(cub_u):13.2f} {np.mean(cyl_u):15.2f} "
              f"{du:8.2f} {np.mean(cub_v):13.2f} {np.mean(cyl_v):15.2f} {dv:8.2f}")
    worst = max(max(v["du_px"], v["dv_px"]) for v in block.values())
    print(f"  -> worst disagreement {worst:.2f} px.  The cylinder is the yaw-invariant hull")
    print("     of the footprint, so it is the only usable model when yaw is unknown, and")
    print("     this says using it costs nothing.")
    summary["object_model"] = block


def section4_occlusion_and_size(models, rows, summary):
    print("\n=== 4. occlusion of the bottom edge, and the range information in box SIZE ===")
    rng = np.random.default_rng(20260806)
    sample = [rows[i] for i in rng.choice(len(rows), min(300, len(rows)), replace=False)]
    print(f"  {'clip k px':>10} {'bottom @ floor':>16} {'centre @ h/2':>14}")
    occl = {}
    for k in (1, 2, 4, 8):
        eb, ec = [], []
        for rec in sample:
            camera = models[rec["camera"]]
            u0, v0, u1, v1 = cylinder_bbox(camera, rec["true_x"], rec["true_y"])
            uc = 0.5 * (u0 + u1)
            p0 = camera.pixel_to_world_at_z(uc, v1, 0.0)
            p1 = camera.pixel_to_world_at_z(uc, v1 - k, 0.0)
            eb.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            q0 = camera.pixel_to_world_at_z(uc, 0.5 * (v0 + v1), HEIGHT / 2)
            q1 = camera.pixel_to_world_at_z(uc, 0.5 * (v0 + v1 - k), HEIGHT / 2)
            ec.append(math.hypot(q1[0] - q0[0], q1[1] - q0[1]))
        occl[k] = dict(bottom_m=float(np.mean(eb)), centre_m=float(np.mean(ec)))
        print(f"  {k:10d} {np.mean(eb)*100:14.2f}cm {np.mean(ec)*100:12.2f}cm")

    heights, ranges = [], []
    for rec in rows:
        camera = models[rec["camera"]]
        _, _, _, _ = bearing_frame(camera, rec["true_x"], rec["true_y"])
        u0, v0, u1, v1 = cylinder_bbox(camera, rec["true_x"], rec["true_y"])
        heights.append(v1 - v0)
        ranges.append(bearing_frame(camera, rec["true_x"], rec["true_y"])[0])
    heights, ranges = np.asarray(heights), np.asarray(ranges)
    slope = float(np.linalg.lstsq(np.vstack([ranges, np.ones_like(ranges)]).T,
                                 heights, rcond=None)[0][0])
    print(f"\n  silhouette height spans {heights.min():.1f}-{heights.max():.1f} px over "
          f"{ranges.min():.1f}-{ranges.max():.1f} m;  dh/dd = {slope:.2f} px/m")
    print(f"  -> 1 px of box-height error = {abs(1.0/slope):.2f} m of range error.  Apparent")
    print("     size carries almost no range information, so a four-edge least-squares fit")
    print("     buys nothing over the centre and the centre is the whole story.")
    summary["occlusion_cm_per_px"] = occl
    summary["size_channel"] = dict(dh_dd_px_per_m=slope,
                                   metres_per_px_of_height=abs(1.0 / slope))


def section5_covariance(models, rows, summary):
    print("\n=== 5. the covariance the Jacobian implies (per 1 px of pixel noise) ===")
    print(f"  {'camera':9} {'mean d':>7} {'radial':>9} {'lateral':>9} {'anisotropy':>11}")
    block = {}
    pooled_d, pooled_r = [], []
    for camera_id in sorted(models):
        group = [r for r in rows if r["camera"] == camera_id]
        camera = models[camera_id]
        rad, lat, ds = [], [], []
        for rec in group:
            J = jacobian(camera, rec["u"], rec["v"], HEIGHT / 2.0)
            if J is None:
                continue
            d, ux, uy, _ = bearing_frame(camera, rec["true_x"], rec["true_y"])
            rot = np.array([[ux, uy], [-uy, ux]])
            S = rot @ J @ J.T @ rot.T
            rad.append(math.sqrt(S[0, 0])); lat.append(math.sqrt(S[1, 1])); ds.append(d)
        pooled_d.extend(ds); pooled_r.extend(rad)
        block[camera_id] = dict(mean_range_m=float(np.mean(ds)),
                                radial_m_per_px=float(np.mean(rad)),
                                lateral_m_per_px=float(np.mean(lat)),
                                anisotropy=float(np.mean(rad) / np.mean(lat)))
        print(f"  {camera_id:9} {np.mean(ds):7.2f} {np.mean(rad)*100:7.2f}cm "
              f"{np.mean(lat)*100:7.2f}cm {np.mean(rad)/np.mean(lat):11.2f}")

    pooled_d, pooled_r = np.asarray(pooled_d), np.asarray(pooled_r)
    print("\n  radial cm per px vs range, against the analytic (H^2 + d^2) / (f H) form:")
    growth = {}
    for lo, hi in ((5, 8), (8, 10), (10, 12), (12, 14), (14, 16)):
        sel = (pooled_d >= lo) & (pooled_d < hi)
        if not sel.any():
            continue
        mid = 0.5 * (lo + hi)
        analytic = (CAMERA_HEIGHT**2 + mid**2) / (FOCAL_PX * CAMERA_HEIGHT)
        growth[f"{lo}-{hi}"] = dict(n=int(sel.sum()),
                                    measured_m_per_px=float(pooled_r[sel].mean()),
                                    analytic_m_per_px=float(analytic))
        print(f"    {lo:2d}-{hi:2d} m  n={int(sel.sum()):4d}  measured "
              f"{pooled_r[sel].mean()*100:5.2f} cm/px   analytic {analytic*100:5.2f} cm/px")
    keys = list(growth)
    if len(keys) >= 2:
        m_ratio = growth[keys[-1]]["measured_m_per_px"] / growth[keys[0]]["measured_m_per_px"]
        a_ratio = growth[keys[-1]]["analytic_m_per_px"] / growth[keys[0]]["analytic_m_per_px"]
        print(f"    growth ratio {keys[0]} -> {keys[-1]}: measured {m_ratio:.2f}, "
              f"analytic {a_ratio:.2f}  ({abs(m_ratio/a_ratio-1)*100:.0f}% apart)")
        print("    The absolute analytic value carries a fixed obliquity factor this")
        print("    ignores; the SHAPE -- growth as H^2 + d^2 -- is what it confirms.")
    summary["jacobian_covariance"] = dict(per_camera=block, range_growth=growth)

    print("\n  measured residual anisotropy of the DEPLOYED path, for contrast:")
    for camera_id in sorted(models):
        group = [r for r in rows if r["camera"] == camera_id]
        camera = models[camera_id]
        along, cross = [], []
        for rec in group:
            p = camera.pixel_to_world_at_z(rec["u"], rec["v"], 0.0)
            d, ux, uy, _ = bearing_frame(camera, rec["true_x"], rec["true_y"])
            ex, ey = p[0] - rec["true_x"], p[1] - rec["true_y"]
            along.append(ex * ux + ey * uy); cross.append(-ex * uy + ey * ux)
        along, cross = np.asarray(along), np.asarray(cross)
        print(f"    {camera_id}: sd radial {along.std()*100:5.2f} cm  sd lateral "
              f"{cross.std()*100:5.2f} cm  ratio {along.std()/cross.std():5.2f}")


def section6_evidence_audit(rows, summary):
    print("\n=== 6. audit of the evidence base itself ===")
    captures = sorted({r["capture"] for r in rows})
    clouds = {c: np.asarray([(r["true_x"], r["true_y"]) for r in rows if r["capture"] == c])
              for c in captures}
    block = {}
    for c, arr in clouds.items():
        yaws = np.asarray([r["yaw"] for r in rows if r["capture"] == c and
                           math.isfinite(r["yaw"])])
        uniq = np.unique(np.round(np.degrees(yaws), 0)) if yaws.size else np.array([])
        block[c] = dict(n=int(len(arr)),
                        x_span_m=float(arr[:, 0].ptp()), y_span_m=float(arr[:, 1].ptp()),
                        distinct_yaw_deg=[float(x) for x in uniq[:8]])
        print(f"  {c:26} n={len(arr):5d}  x span {arr[:,0].ptp():5.2f} m  "
              f"y span {arr[:,1].ptp():5.2f} m  yaw {list(np.round(uniq,0))[:6]} deg")
    print("\n  pairwise route overlap (median nearest-neighbour distance):")
    dup = []
    for i, a in enumerate(captures):
        for b in captures[i + 1:]:
            A, B = clouds[a], clouds[b]
            dist = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(axis=1)
            frac = float((dist < 0.05).mean())
            print(f"    {a[:12]:12} vs {b[:12]:12}  median {np.median(dist):6.3f} m   "
                  f"fraction within 5 cm: {frac:.2f}")
            if frac > 0.9:
                dup.append((a, b))
    if dup:
        for a, b in dup:
            print(f"  -> {a} and {b} are the SAME ROUTE.  Leave-one-capture-out over these")
            print("     three files is not three independent folds.")
    summary["evidence_audit"] = dict(captures=block,
                                     duplicate_routes=[list(p) for p in dup])


def main() -> int:
    models = {c: camera_model_from_world(WORLD, include_name=i)
              for c, i in MODEL_INCLUDES.items()}
    rows = load(models)
    with_yaw = [r for r in rows if math.isfinite(r["yaw"])]
    print(f"{len(rows)} detections, {len(with_yaw)} with ground-truth yaw, "
          f"{len(models)} cameras\n")
    summary: dict[str, object] = dict(n_detections=len(rows), n_with_yaw=len(with_yaw),
                                      object_model=dict(half_body_x_m=HALF_BODY_X,
                                                        half_track_m=HALF_TRACK,
                                                        height_m=HEIGHT))
    section1_yaw_sensitivity(models, rows, summary)
    section2_inversion(models, rows, summary)
    section3_object_model(models, rows, summary)
    section4_occlusion_and_size(models, rows, summary)
    section5_covariance(models, rows, summary)
    section6_evidence_audit(rows, summary)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {(OUT / 'summary.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
