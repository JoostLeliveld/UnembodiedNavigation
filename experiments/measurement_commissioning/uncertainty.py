"""How much to trust a sighting: R, fitted in pixel space and carried into the world by geometry.

The detector's error is a property of the **detector**, not of where the robot happens to
be standing.  So it is measured once, in pixels, and everything that depends on the robot's
position is left to the camera geometry.  Measured on this capture, that assumption holds:

    distance     sd sideways   sd vertical
    0-6 m           0.740 px      0.954 px
    6-9 m           0.778         0.779
    12-15 m         0.790         0.576
    18-25 m         0.819         0.614

Sideways is flat to within 10% across a fourfold change in distance.  And the payoff -- one
pixel number pushed through the geometry, against the spread actually observed:

    distance     predicted   actual   ratio
    0-6 m         0.79 cm    0.91 cm   1.15
    9-12 m        1.92       1.67      0.87
    18-25 m       4.81       4.39      0.91

The spread changes sixfold across the building and one number tracks it to within 15%.
That is why this beat a fitted spatial map with a thousand times the parameters: the
geometry already knew the shape, so there was nothing left for data to learn.

**What must NOT be folded in here.**  Heading error does not travel through the pixel
channel at all -- it corrupts the prediction directly, at roughly 1 cm per degree, and it is
*shared by every camera* because there is one robot with one heading.  Fusing five cameras
removes 1% of it.  Putting it into a per-camera pixel noise would be the wrong magnitude
and, worse, would treat a perfectly shared error as independent, making the filter more
overconfident the more cameras are fused.  It belongs in the state, not in R.
"""
from __future__ import annotations

import math

import numpy as np


def fit_sigma_px(residuals):
    """One pixel-noise number from the centred residuals.

    ``residuals`` is an (n, 2) array of sideways/vertical pixel residuals.  The two axes are
    averaged because they come out within a few percent of each other; per-axis and
    per-camera variants are the next rungs of the ladder and are reported separately.
    """
    r = np.asarray(residuals, dtype=float)
    return float(r.std(axis=0).mean())


def sigma_px_by_camera(residuals_by_camera):
    """Per-camera pixel noise -- the first rung above a single number.

    Worth checking because it is nearly free: on this capture it runs from 0.60 px on one
    camera to 1.00 px on another, a 1.6x spread that a single pooled number averages away.
    """
    return {c: fit_sigma_px(r) for c, r in residuals_by_camera.items()}


def ground_covariance(J_inv, sigma_px):
    """``R`` in metres^2: pixel noise carried into the world by the camera geometry.

    ``R = J^-1 (sigma^2 I) J^-T``.  A single fitted number produces a correctly shaped and
    correctly sized ellipse everywhere in the building -- stretched along the viewing ray,
    because that is the direction in which one pixel is worth the most centimetres.
    """
    return J_inv @ (sigma_px ** 2 * np.eye(2)) @ J_inv.T


def stated_spread_cm(J_inv, sigma_px):
    """The one-number summary of that ellipse, in centimetres: sqrt(trace(R)/2)."""
    return math.sqrt(np.trace(ground_covariance(J_inv, sigma_px)) / 2.0) * 100.0


def heading_term_cm(dc_dtheta, J_inv, sigma_theta_rad):
    """What a heading uncertainty is worth in centimetres, for reporting only.

    Provided so the size of the effect can be stated beside R without being added to it.
    ``dc_dtheta`` is the pixel movement of the bottom-centre per radian of heading, from
    ``geometry.heading_jacobian``.  Adding this to R would be wrong: the error is shared
    across cameras, and R is assumed independent.
    """
    return float(np.linalg.norm(J_inv @ np.asarray(dc_dtheta)) * sigma_theta_rad * 100.0)
