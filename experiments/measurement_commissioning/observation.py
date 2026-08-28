"""Turning a detector's box into a position: the box is not the robot.

**The problem.**  The bottom edge of a detector's box is the robot's nearest point to the
camera and its horizontal middle is the midpoint of its widest part.  Neither is the middle
of the robot.  Back-project the bottom-centre through the floor and it lands **24 to 35 cm**
from the truth, always toward the camera, and that gap **swings 11 cm as the robot turns**,
because a 0.80 x 0.55 m footprint presents 27.5 to 48.5 cm of half-extent along the viewing
ray depending which way it faces.

**The wrong answer is to convert.**  Recovering the robot's centre from the measured pixel
needs the heading, which one box does not give you.  It is ill-posed, and it costs 30 cm.

**The right answer is to predict.**  Put a candidate pose into the robot's own shape,
project it, box it, take the bottom-centre.  Prediction and measurement are then the same
physical quantity and the gap never exists.  Residual: about half a centimetre.  This is the
ordinary filter rule -- never invert the measurement function, evaluate it forward.

**What that leaves.**  ``h`` needs a heading, so heading error becomes position error at
about 0.23 cm per degree -- and that lands directly in any error measured against the true
robot centre.  Worse, it barely moves the *pixel* ``h`` predicts (0.03 px per degree at
close range, less further out), so a heading error does not look like a disagreement and
``admission`` cannot catch it.  The box **width** moves 10 to 20 times more with heading;
that information is discarded today, and using it is what would let the cameras correct
heading instead of depending on it.

Nothing here is fitted.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src/unav_common") not in sys.path:
    sys.path.insert(0, str(REPO / "src/unav_common"))

from unav_common.robot_hull import VISUAL_HULL, silhouette_box  # noqa: E402


def predicted_box(cam, x, y, yaw):
    """The robot's outline in the image: rotate the hull, move it, project it, box it."""
    return silhouette_box(cam, x, y, yaw, VISUAL_HULL)


def h(cam, x, y, yaw, edge_offset=(0.0, 0.0)):
    """``h(x)``: where the detector's reported box bottom-centre should appear, in pixels.

    ``edge_offset`` reconciles two things that are not quite the same quantity -- the hull
    projection is a continuous image coordinate while the detector reports a half-open
    pixel extent.  It is measured with the detector absent (see ``data``) and passed in
    rather than hard-coded, so the constant in force is always the one recorded beside the
    results.  On this capture it is under 0.03 px: the hull matches the rendered outline
    almost exactly once the robot is unoccluded.
    """
    box = predicted_box(cam, x, y, yaw)
    if box is None:
        return None
    u0, _v0, u1, v1 = box
    return 0.5 * (u0 + u1) + edge_offset[0], v1 + edge_offset[1]


def jacobian(cam, x, y, yaw, eps=0.02):
    """How many pixels the prediction moves per metre the robot moves: ``d(pixel)/d(x,y)``.

    Its inverse turns a pixel error into a ground error, which is why the same pixel noise
    is worth a few millimetres near a camera and several centimetres far from it, and why
    the resulting uncertainty is an ellipse stretched along the viewing ray rather than a
    circle.  Computed by nudging the pose, so it stays correct if the hull ever changes.
    """
    a = np.array(h(cam, x + eps, y, yaw)); b = np.array(h(cam, x - eps, y, yaw))
    c = np.array(h(cam, x, y + eps, yaw)); d = np.array(h(cam, x, y - eps, yaw))
    return np.column_stack([(a - b) / (2 * eps), (c - d) / (2 * eps)])


def heading_jacobian(cam, x, y, yaw, eps=math.radians(0.5)):
    """``d(pixel)/d(heading)`` for the bottom-centre AND for the box width and height.

    Returns ``(d_bottom_centre, d_width, d_height)``, all in pixels per radian.  The first
    is tiny and the second is not: that asymmetry is why a heading error slips past the
    admission check while quietly moving the inferred position, and why the width is the
    thing to use if the cameras are ever to correct heading rather than depend on it.
    """
    b0 = predicted_box(cam, x, y, yaw - eps)
    b1 = predicted_box(cam, x, y, yaw + eps)
    if b0 is None or b1 is None:
        return None
    step = 2 * eps
    dc = np.array([(0.5 * (b1[0] + b1[2]) - 0.5 * (b0[0] + b0[2])) / step,
                   (b1[3] - b0[3]) / step])
    dw = ((b1[2] - b1[0]) - (b0[2] - b0[0])) / step
    dh = ((b1[3] - b1[1]) - (b0[3] - b0[1])) / step
    return dc, dw, dh


def bottom_centre_offset(cam, x, y, yaw):
    """How far the box bottom-centre lands from the robot's true centre, in metres.

    The quantity this whole module exists to avoid paying.  Useful for reporting: sweep the
    heading at a fixed place and the spread of this number is the heading-dependent part.
    """
    point = h(cam, x, y, yaw)
    if point is None:
        return None
    gx, gy = cam.pixel_to_world(point[0], point[1])[:2]
    return math.hypot(gx - x, gy - y)
