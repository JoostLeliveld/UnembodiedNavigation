"""The rendered-marker test is what keeps hidden robots out of the dataset.

A projected keypoint always lands on some pixel, so without this check a pose
behind a rack is written as a confident label of a robot in plain sight. Two
signals have to agree: the pixel carries the marker colour, and it differs from
a robot-free background frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / 'scripts' / 'perception'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from capture_projected_keypoint_dataset import _marker_hits  # noqa: E402

CYAN = (255, 242, 5)      # BGR, as the renderer draws the front disk
MAGENTA = (255, 5, 242)   # BGR, the rear disk
RACK_RAIL = (201, 149, 5)  # BGR, the blue shelf rails: close to cyan, and static
FLOOR = (140, 141, 143)


def _frame(colour=FLOOR) -> np.ndarray:
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, :] = colour
    return img


def _blob(img: np.ndarray, uv, colour, half=2) -> np.ndarray:
    u, v = int(uv[0]), int(uv[1])
    img[v - half:v + half + 1, u - half:u + half + 1] = colour
    return img


def test_visible_front_marker_is_counted() -> None:
    background = _frame()
    image = _blob(_frame(), (30, 30), CYAN)
    assert _marker_hits(image, background, np.array([30.0, 30.0]), 'front') >= 9


def test_visible_rear_marker_is_counted() -> None:
    background = _frame()
    image = _blob(_frame(), (30, 30), MAGENTA)
    assert _marker_hits(image, background, np.array([30.0, 30.0]), 'rear') >= 9


def test_marker_hidden_behind_a_rack_scores_zero() -> None:
    """The pose projects to (30, 30) but nothing of the robot renders there."""
    background = _frame()
    image = _frame()
    assert _marker_hits(image, background, np.array([30.0, 30.0]), 'front') == 0


def test_static_rack_rail_is_not_mistaken_for_a_cyan_marker() -> None:
    """The rails are blue enough to survive a loose colour test on their own.
    They are in the background frame too, so the change test removes them."""
    background = _blob(_frame(), (30, 30), RACK_RAIL)
    image = _blob(_frame(), (30, 30), RACK_RAIL)
    assert _marker_hits(image, background, np.array([30.0, 30.0]), 'front') == 0


def test_front_and_rear_are_not_interchangeable() -> None:
    background = _frame()
    image = _blob(_frame(), (30, 30), CYAN)
    assert _marker_hits(image, background, np.array([30.0, 30.0]), 'rear') == 0


def test_only_the_window_around_the_keypoint_is_searched() -> None:
    background = _frame()
    image = _blob(_frame(), (10, 10), CYAN)
    assert _marker_hits(image, background, np.array([40.0, 40.0]), 'front', half=4) == 0


def test_keypoint_at_the_image_edge_does_not_raise() -> None:
    background = _frame()
    image = _frame()
    assert _marker_hits(image, background, np.array([0.0, 0.0]), 'front') == 0
    assert _marker_hits(image, background, np.array([59.0, 59.0]), 'rear') == 0


def test_without_a_background_the_colour_test_stands_alone() -> None:
    image = _blob(_frame(), (30, 30), CYAN)
    assert _marker_hits(image, None, np.array([30.0, 30.0]), 'front') >= 9
