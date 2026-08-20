#!/usr/bin/env python3
"""P from the camera's own RGB: monocular depth -> where the robot can be seen.

The geometry prior needs a CAD model of the racks. This one does not: it takes one frame
from the fixed camera, predicts metric depth with Depth Anything V2 (Metric-Indoor-Large),
anchors that prediction to the floor -- the depth of every pixel whose ray lands on the
drivable floor is known analytically from the calibration, so a RANSAC affine fit puts the
whole image on metres -- back-projects it into a height map, and raycasts the camera's line
of sight to every cell. Deployment inputs only: calibration, the drivable map, and the
camera's own picture.

It then does the two things that make the field comparable with the surveyed GP:

  * calibrates its visibility score onto the detection-rate scale with the same affine fit
    the geometry prior uses, and reports how well it agrees with the survey;
  * fuses with the GP in logit space with the same prior width, so `P + GP` means the same
    operation for both priors.

Everything is reused: the height-map/raycast/visibility chain from
scripts/geometry_visibility, the depth backend from experiments/monocular_depth_adapter.

Run: python3 experiments/localization_reading_story/build_mono_depth_prior.py <frame.png>
Outputs -> logs/studies/localization_reading_story/availability_fields/mono_depth_prior.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_data as rd  # noqa: E402

REPO = rd.REPO_ROOT
sys.path.insert(0, str(REPO / 'scripts/geometry_visibility'))
sys.path.insert(0, str(REPO / 'src/unav_common'))
sys.path.insert(0, str(REPO / 'experiments/monocular_depth_adapter'))

import geometry_visibility as gv  # noqa: E402
from depth_occlusion_prior import _raycast_hm  # noqa: E402
from mono_depth_occlusion_prior import (backproject, floor_calibration_mask,  # noqa: E402
                                        ransac_affine)

GP_ARTIFACT = REPO / 'paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz'
CAMPAIGN = REPO / 'scripts/visibility_comparison/warehouse_visibility_campaign.yaml'
OUT = REPO / 'logs/studies/localization_reading_story/availability_fields'
MODEL = 'dav2_metric_indoor_large'
Z_MARKER = 0.35          # the height the visibility chain scores the robot at
SIGMA_PRIOR = 1.5        # logit-space prior width, as in build_geometry_visibility_prior.py


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def affine_to_detection_rate(score, target, mask):
    """Put a visibility score on the detection-rate scale, the way stage 1 does."""
    a_matrix = np.vstack([score[mask], np.ones(int(mask.sum()))]).T
    slope, intercept = np.linalg.lstsq(a_matrix, target[mask], rcond=None)[0]
    fitted = np.clip(slope * score + intercept, 0.0, 1.0)
    resid = target[mask] - (slope * score[mask] + intercept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((target[mask] - target[mask].mean()) ** 2).sum())
    return fitted, float(slope), float(intercept), 1.0 - ss_res / max(ss_tot, 1e-12)


def rank_agreement(a, b, mask):
    from scipy import stats as sps
    x, y = a[mask], b[mask]
    return float(sps.pearsonr(x, y)[0]), float(sps.spearmanr(x, y)[0])


def main() -> None:
    frame_path = Path(sys.argv[1]).expanduser().resolve()
    meta = gv.load_gp_artifact_geometry(str(GP_ARTIFACT))
    xs, ys = meta['xs'], meta['ys']
    camera = gv.make_camera(meta)
    drivable = gv.prisms_from_json(json.loads(
        yaml.safe_load(CAMPAIGN.read_text())['driveable_geometry_json']))

    import cv2
    image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
    print(f'frame {frame_path.name}  {image.shape[1]}x{image.shape[0]}')

    from monodepth.backends import create_backend
    from monodepth.types import CameraIntrinsics
    backend = create_backend(MODEL, device='cuda')
    backend.load()
    intr = CameraIntrinsics(fx=float(camera.K[0, 0]), fy=float(camera.K[1, 1]),
                            cx=float(camera.K[0, 2]), cy=float(camera.K[1, 2]),
                            width=image.shape[1], height=image.shape[0])
    depth_raw = backend.infer_batch([image], [intr])[0].depth.astype(float)
    backend.unload()
    print(f'raw monocular depth: {depth_raw.min():.2f}-{depth_raw.max():.2f} m')

    # --- anchor it to the floor: the one ruler a deployed camera already has
    u_idx, v_idx, d_floor = floor_calibration_mask(camera, depth_raw.shape, drivable)
    slope, offset, inlier = ransac_affine(depth_raw[v_idx, u_idx], d_floor)
    depth = slope * depth_raw + offset
    print(f'floor-anchored: d_true = {slope:.3f} * d_pred + {offset:.3f} '
          f'({100 * inlier:.0f}% inliers over {len(d_floor)} floor pixels)')

    # --- the same visibility chain the geometry prior uses
    fov = gv.fov_projection_grid(camera, xs, ys, Z_MARKER)
    jac = gv.projection_jacobian_scale(camera, xs, ys, Z_MARKER)
    height = gv.height_map_from_points(backproject(camera, depth), xs, ys)['h_max']
    clearance = _raycast_hm(camera, xs, ys, height)
    score = gv.compute_visibility(
        fov_mask=fov['fov_mask'], min_clearance=clearance,
        px_per_m_min=jac['px_per_m_min'], u=fov['u'], v=fov['v'],
        img_w=camera.img_width, img_h=camera.img_height,
        tau_clearance=0.10)['visibility_score']

    # --- agreement with the survey, and the same fusion
    prior_npz = np.load(OUT / 'geometry_visibility_prior.npz', allow_pickle=True)
    drive_mask = prior_npz['driveable_mask'].astype(bool)
    gp = np.load(GP_ARTIFACT, allow_pickle=True)
    gp_prob = np.asarray(gp['P_mean_map'], float)
    mask = drive_mask & fov['fov_mask'].astype(bool) & np.isfinite(score) & np.isfinite(gp_prob)
    calibrated, slope_p, intercept_p, r2 = affine_to_detection_rate(score, gp_prob, mask)
    pearson, spearman = rank_agreement(score, gp_prob, mask)
    print(f'against the surveyed GP over {int(mask.sum())} cells: '
          f'Pearson {pearson:.3f}, Spearman {spearman:.3f}; '
          f'calibration p = {slope_p:.3f}*score + {intercept_p:.3f}, R2 {r2:.3f}')

    tau_prior = 1.0 / SIGMA_PRIOR ** 2
    emp_std = np.clip(np.asarray(gp['F_std_map'], float), 1e-3, None)
    tau_emp = 1.0 / emp_std ** 2
    post_logit = (tau_prior * _logit(calibrated) + tau_emp * np.asarray(gp['F_mean_map'], float))
    post_logit = post_logit / (tau_prior + tau_emp)
    posterior = 1.0 / (1.0 + np.exp(-post_logit))
    weight = tau_prior / (tau_prior + tau_emp)
    print(f'fusion: median geometry-side weight {np.nanmedian(np.where(mask, weight, np.nan)):.2f}')

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / 'mono_depth_prior.npz', xs=xs, ys=ys,
             visibility_score_map=score, calibrated_mono_prob=calibrated,
             fused_posterior_prob=posterior, prior_weight_map=weight,
             height_map=height, min_clearance_map=clearance,
             empirical_gp_prob=gp_prob, driveable_mask=drive_mask,
             frame=str(frame_path.relative_to(REPO)), model=MODEL,
             floor_affine=np.array([slope, offset, inlier]),
             agreement=np.array([pearson, spearman, slope_p, intercept_p, r2]))
    print(f'wrote {OUT / "mono_depth_prior.npz"}')


if __name__ == '__main__':
    main()
