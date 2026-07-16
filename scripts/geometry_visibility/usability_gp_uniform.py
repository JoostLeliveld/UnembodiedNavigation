#!/usr/bin/env python3
"""[REAL, uniform] Validate the occlusion prior on the GP-FITTING samples — a uniform
teleport grid over the whole driveable region (X_train/p_train in the artifact), where
mean detection is 0.60 and ~40% of positions are blind. This is the right distribution
(NOT route-biased driving, which was 92% visible and hid the problem).

p_train = measured per-position detection rate (uniform). Predict it with:
  camera-only (FOV+range+boundary, no occlusion)
  camera + COMPLETE-CAD occlusion (clean world: racks + walls + shelf boxes)   [ground-truth geom]
Metrics: Spearman (continuous rate) and AUROC (rate<0.5 = unreliable). CAD = eval-only ref
for the geometry; deployment senses it (depth). No stack here (clean-world GP).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
from scipy.stats import rankdata

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))
import geometry_visibility as gv
from unav_common.occlusion_geometry import parse_occlusion_scene_from_world

ART = REPO / "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_aws.world.sdf"  # clean (GP world)
Z = 0.35


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y); pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return np.nan
    r = rankdata(s); return float((r[y == 1].sum() - pos*(pos+1)/2) / (pos*neg))


def main():
    d = np.load(ART, allow_pickle=True)
    Xt = np.asarray(d["X_train"], float); pt = np.asarray(d["p_train"], float)
    meta = gv.load_gp_artifact_geometry(ART); xs, ys = meta["xs"], meta["ys"]; cam = gv.make_camera(meta)
    print(f"[REAL uniform GP samples] {len(pt)} positions; detection rate mean {pt.mean():.2f}, "
          f"{100*(pt<0.5).mean():.0f}% unreliable (<0.5)")

    sc = parse_occlusion_scene_from_world(str(WORLD), model_name=("warehouse_walls", "warehouse_rack_occluders"),
                                          geometry_tags=("visual",))
    cad = [gv.Prism(p.name, p.xmin, p.xmax, p.ymin, p.ymax, p.zmin, p.zmax) for p in sc.prisms]
    fov = gv.fov_projection_grid(cam, xs, ys, Z); jac = gv.projection_jacobian_scale(cam, xs, ys, Z)
    def vis(mc):
        return gv.compute_visibility(fov_mask=fov["fov_mask"], min_clearance=mc, px_per_m_min=jac["px_per_m_min"],
                                     u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height, tau_clearance=0.10)["visibility_score"]
    cam_only = vis(np.full((len(ys), len(xs)), 10.0))
    cad_occ = vis(gv.raycast_min_clearance(cam, xs, ys, cad, Z))
    at = lambda f: f[np.clip(np.searchsorted(ys, Xt[:, 1]), 0, len(ys)-1), np.clip(np.searchsorted(xs, Xt[:, 0]), 0, len(xs)-1)]
    y = (pt >= 0.5).astype(int)   # reliable(1)/unreliable(0)
    print(f"\npredicting the UNIFORM detection rate ({len(pt)} positions, {int((y==0).sum())} unreliable):")
    for name, s in [("camera-only (no occlusion)", at(cam_only)), ("camera + complete-CAD occlusion", at(cad_occ))]:
        print(f"  {name:32s}  Spearman {spearman(s, pt):+.3f}   AUROC {auroc(s, y):.3f}")


if __name__ == "__main__":
    main()
