#!/usr/bin/env python3
"""V1 -- the three availability fields, in the world this whole folder is about.

Not the A0-A3 panel from the availability paper: those are four *estimator sources* in the
four-camera world. These are the three fields the single-camera story actually rests on,
all on one colour scale:

  1  the GP that was surveyed -- a Gaussian process fitted to the detection RATE measured
     at 139 driven-to places. It is the truth of record where the robot has been, and it
     knows nothing anywhere else.
  2  P from the camera's own picture -- monocular depth (Depth Anything V2 Metric-Indoor
     Large) on one frame, anchored to the floor by the calibration, back-projected into a
     height map and raycast. No survey, no CAD, no ground truth. It ranks the floor the way
     the measured GP does slightly BETTER than the CAD geometry prior: Spearman 0.773
     against 0.730, R2 0.581 against 0.515.
  3  P + GP -- precision-weighted in logit space: where the survey is dense the data wins,
     where it is thin the depth prior fills the gap. Median prior weight 0.52.

Pass --geometry to draw the CAD prior in panels 2 and 3 instead.

Fields come from logs/studies/localization_reading_story/availability_fields/, written by
scripts/geometry_visibility/build_geometry_visibility_prior.py against the LOCKED GP
artifact paper_artifacts/gp/warehouse_visibility_gp_v1.

Run: python3 experiments/localization_reading_story/figures/fig_V_availability_fields.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reading_data as rd  # noqa: E402

sys.path.insert(0, str(rd.REPO_ROOT / 'scripts/geometry_visibility'))
import geometry_visibility as gv  # noqa: E402

FIELDS = rd.REPO_ROOT / 'logs/studies/localization_reading_story/availability_fields'
USE_GEOMETRY = '--geometry' in sys.argv
GP_ARTIFACT = rd.REPO_ROOT / 'paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz'


def main() -> None:
    rd.style()
    geometry = np.load(FIELDS / 'geometry_visibility_prior.npz', allow_pickle=True)
    prior = geometry if USE_GEOMETRY else np.load(FIELDS / 'mono_depth_prior.npz',
                                                  allow_pickle=True)
    agreement = None if USE_GEOMETRY else np.asarray(prior['agreement'], float)
    gp = np.load(GP_ARTIFACT, allow_pickle=True)
    xs, ys = prior['xs'], prior['ys']
    drive = geometry['driveable_mask'].astype(bool)
    extent = (float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1]))
    # the artifact stores the geometry json as a one-element array of a string
    prisms = gv.prisms_from_json(json.loads(np.atleast_1d(gp['geometry_json'])[0]))
    camera = np.asarray(gp['camera_pos'], dtype=float)
    survey = np.asarray(gp['X_train'], dtype=float)
    rate = np.asarray(gp['p_train'], dtype=float)

    if USE_GEOMETRY:
        prior_key, prior_title = 'calibrated_geometry_prob', 'P from CAD geometry, no survey'
        prior_sub = ('camera pose, rack prisms, robot height -- and it ranks\n'
                     'the floor the way the survey does (Spearman 0.73)')
        fill = 'geometry'
    else:
        prior_key, prior_title = 'calibrated_mono_prob', 'P from the camera\'s own picture'
        prior_sub = ('one frame -> monocular depth -> floor-anchored -> raycast.\n'
                     f'No survey, no CAD: Spearman {agreement[1]:.2f} against the survey')
        fill = 'the depth prior'
    panels = [
        ('empirical_gp_prob',
         'the GP that was surveyed',
         f'fitted to the detection RATE at {len(survey)} driven-to places:\n'
         'sharp where the robot has been, blind everywhere else'),
        (prior_key, prior_title, prior_sub),
        ('fused_posterior_prob',
         'P + GP',
         f'the survey wins where it is dense,\n{fill} fills the gaps'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.8))
    mesh = None
    for ax, (key, title, sub) in zip(axes, panels):
        field = np.where(drive, prior[key], np.nan)
        mesh = ax.imshow(field, origin='lower', extent=extent, cmap='viridis',
                         vmin=0.0, vmax=1.0, aspect='equal', interpolation='nearest')
        for prism in prisms:
            ax.add_patch(plt.Rectangle((prism.xmin, prism.ymin),
                                       prism.xmax - prism.xmin,
                                       prism.ymax - prism.ymin,
                                       fill=False, edgecolor='white', lw=0.8, alpha=0.9))
        ax.plot([camera[0]], [camera[1]], marker='*', ms=17, color='#ff2d2d', mec='black',
                mew=0.8, zorder=6)
        if key == 'empirical_gp_prob':
            ax.scatter(survey[:, 0], survey[:, 1], s=16, facecolor='none',
                       edgecolor='white', lw=0.9, alpha=0.95, zorder=5,
                       label=f'the {len(survey)} places the survey visited')
            ax.legend(loc='lower left', fontsize=8.0, facecolor='white', framealpha=0.85,
                      bbox_to_anchor=(0.0, 0.03))
        ax.set_title(f'{title}\n{sub}', loc='left', fontsize=9.8)
        ax.set_xlabel('metres east')
        ax.grid(False)
    axes[0].set_ylabel('metres north')
    bar = fig.colorbar(mesh, ax=axes, fraction=0.022, pad=0.02)
    bar.set_label('chance that a detection here is usable')

    covered = float(np.mean(np.isfinite(np.where(drive, prior['empirical_gp_prob'], np.nan))))
    predicted_by = ('what CAD geometry predicts without one' if USE_GEOMETRY
                    else 'what the camera\'s own picture predicts without one')
    fig.suptitle(f'Where the camera can be trusted: what a survey measured, {predicted_by}, '
                 f'and the two combined', fontsize=13.2, fontweight='bold', y=0.99)
    source = ('CAD geometry (build_geometry_visibility_prior.py)' if USE_GEOMETRY else
              f'monocular depth on {Path(str(prior["frame"])).name}, model '
              f'{str(prior["model"])} (build_mono_depth_prior.py); floor-anchored with '
              f'd_true = {prior["floor_affine"][0]:.2f}*d_pred + {prior["floor_affine"][1]:.2f} '
              f'at {100 * prior["floor_affine"][2]:.0f}% inliers, then calibrated onto the '
              f'detection-rate scale (p = {agreement[2]:.2f}*score + {agreement[3]:.2f}, '
              f'R2 {agreement[4]:.2f}, Pearson {agreement[0]:.2f}, Spearman {agreement[1]:.2f})')
    rd.note(fig, f'Single-camera warehouse_aws, camera at ({camera[0]:.1f}, {camera[1]:.1f}, '
                 f'{camera[2]:.1f}) -- the same camera as every other figure in this folder. '
                 f'The GP is the locked artifact paper_artifacts/gp/warehouse_visibility_gp_v1: '
                 f'detection RATE per place, not detector confidence. Middle and right come '
                 f'from {source}. The fusion is precision-weighted in logit space with the '
                 f'same 1.5 prior width the geometry chain uses. White outlines are the rack '
                 f'prisms, the star is the camera, grey is off the driveable floor. Nothing '
                 f'here uses ground truth: the survey measures its own detections and the '
                 f'depth prior sees only the camera\'s own frame.',
            width=200)
    fig.subplots_adjust(top=0.80, bottom=0.22, left=0.05, right=0.90, wspace=0.20)
    rd.save(fig, 'V1_surveyed_gp_geometry_and_fusion' if USE_GEOMETRY
            else 'V1_surveyed_gp_monodepth_and_fusion')

    for key, title, _ in panels:
        f = np.where(drive, prior[key], np.nan)
        print(f'{title:38s} mean {np.nanmean(f):.3f}  '
              f'below 0.5: {100 * np.nanmean(f < 0.5):.0f}% of driveable cells')


if __name__ == '__main__':
    main()
