#!/usr/bin/env python3
"""Deterministic toy-scene tests for the geometry visibility prior.

Run standalone (no pytest, no ROS, no Gazebo):

    python3 scripts/geometry_visibility/test_geometry_visibility.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

# Put unav_common on the path BEFORE importing the module, so its camera-model
# import resolves. Then import the module under test from this directory.
_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src" / "unav_common"))
sys.path.insert(0, str(_HERE))

import geometry_visibility as gv  # noqa: E402


def _toy_camera():
    return gv.ObliqueCameraModel(
        cam_pos=(0.0, -5.0, 4.0),
        look_at=(0.0, 0.0, 0.0),
        img_width=1280,
        img_height=720,
        fov_h_rad=1.5708,
    )


def _toy_grid(lo=-2.0, hi=2.0, n=41):
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    return xs, ys


# ---------------------------------------------------------------------------
def test_height_map_single_box():
    xs, ys = _toy_grid()
    box = gv.Prism("box", xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, zmin=0.0, zmax=1.7)
    hm = gv.build_height_map(xs, ys, [box])
    gx, gy = np.meshgrid(xs, ys)
    inside = box.covers(gx, gy)
    assert np.allclose(hm["h_max"][inside], 1.7)
    assert np.allclose(hm["h_max"][~inside], 0.0)


def test_world_to_grid_roundtrip():
    xs, ys = _toy_grid()
    for (x, y) in [(-1.3, 0.4), (0.0, 0.0), (1.9, -1.75)]:
        iy, ix = gv.world_to_grid(xs, ys, x, y)
        assert abs(xs[ix] - x) <= (xs[1] - xs[0])
        assert abs(ys[iy] - y) <= (ys[1] - ys[0])


def test_empty_scene_visible():
    cam = _toy_camera()
    xs, ys = _toy_grid()
    fov = gv.fov_projection_grid(cam, xs, ys, z_marker=0.3)
    clear = gv.raycast_min_clearance(cam, xs, ys, prisms=[], z_marker=0.3)
    iy, ix = gv.world_to_grid(xs, ys, 0.0, 0.0)
    assert fov["fov_mask"][iy, ix]
    assert clear[iy, ix] > 0.0
    f_occ = gv._sigmoid(clear / 0.10)
    assert f_occ[iy, ix] > 0.9  # clear sightline -> high occlusion score


def test_wall_blocks_ray():
    cam = _toy_camera()
    xs, ys = _toy_grid()
    wall = gv.Prism("wall", xmin=-1.0, xmax=1.0, ymin=-2.7, ymax=-2.3, zmin=0.0, zmax=3.0)
    clear = gv.raycast_min_clearance(cam, xs, ys, prisms=[wall], z_marker=0.3)
    iy, ix = gv.world_to_grid(xs, ys, 0.0, 0.0)  # target behind the wall
    assert clear[iy, ix] < 0.0
    f_occ = gv._sigmoid(clear / 0.10)
    assert f_occ[iy, ix] < 0.1


def test_side_obstacle_does_not_block():
    cam = _toy_camera()
    xs, ys = _toy_grid()
    empty = gv.raycast_min_clearance(cam, xs, ys, prisms=[], z_marker=0.3)
    side = gv.Prism("side", xmin=3.0, xmax=5.0, ymin=-2.7, ymax=-2.3, zmin=0.0, zmax=3.0)
    with_side = gv.raycast_min_clearance(cam, xs, ys, prisms=[side], z_marker=0.3)
    iy, ix = gv.world_to_grid(xs, ys, 0.0, 0.0)
    assert with_side[iy, ix] > 0.0
    assert np.isclose(with_side[iy, ix], empty[iy, ix])  # ray never enters the footprint


def test_outside_fov_zero_visibility():
    cam = _toy_camera()
    # A point well behind the camera cannot project into the image.
    u, v, in_front = gv.project_points(cam, np.array([0.0, -20.0, 0.3]))
    assert not bool(in_front)
    xs = np.array([-20.0]); ys = np.array([0.0])
    fov = gv.fov_projection_grid(cam, xs, ys, z_marker=0.3)
    comp = gv.compute_visibility(
        fov_mask=fov["fov_mask"],
        min_clearance=np.array([[1.0]]),
        px_per_m_min=np.array([[100.0]]),
        u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
    )
    assert comp["f_fov"][0, 0] == 0.0
    assert comp["visibility_score"][0, 0] == 0.0


def test_visibility_bounds():
    cam = _toy_camera()
    xs, ys = _toy_grid()
    box = gv.Prism("box", xmin=-0.5, xmax=0.5, ymin=-1.5, ymax=-1.0, zmin=0.0, zmax=2.0)
    fov = gv.fov_projection_grid(cam, xs, ys, z_marker=0.3)
    clear = gv.raycast_min_clearance(cam, xs, ys, prisms=[box], z_marker=0.3)
    jac = gv.projection_jacobian_scale(cam, xs, ys, z_marker=0.3)
    comp = gv.compute_visibility(
        fov_mask=fov["fov_mask"], min_clearance=clear, px_per_m_min=jac["px_per_m_min"],
        u=fov["u"], v=fov["v"], img_w=cam.img_width, img_h=cam.img_height,
    )
    s = comp["visibility_score"]
    assert np.all(np.isfinite(s))
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_r_plan_monotonic():
    trust = np.linspace(0.0, 1.0, 50)
    std, var = gv.trust_to_r_plan(trust, r_visible_uv=2.5, r_miss_uv=40.0)
    d = np.diff(std)
    assert np.all(d <= 1e-9)  # non-increasing in trust


def test_r_plan_endpoints():
    std1, _ = gv.trust_to_r_plan(1.0, r_visible_uv=2.5, r_miss_uv=40.0)
    std0, _ = gv.trust_to_r_plan(0.0, r_visible_uv=2.5, r_miss_uv=40.0)
    assert np.isclose(float(std1), 2.5, atol=1e-6)
    assert np.isclose(float(std0), 40.0, atol=1e-6)


def test_height_map_from_points():
    xs, ys = _toy_grid()
    pts = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.8], [1.0, 1.0, 0.4]])  # tallest wins per cell
    hm = gv.height_map_from_points(pts, xs, ys)
    iy0, ix0 = gv.world_to_grid(xs, ys, 0.0, 0.0)
    iy1, ix1 = gv.world_to_grid(xs, ys, 1.0, 1.0)
    assert np.isclose(hm["h_max"][iy0, ix0], 1.5)
    assert hm["observed"][iy0, ix0] and hm["observed"][iy1, ix1]
    assert not hm["observed"][0, 0]  # a cell with no returns stays unobserved


def test_points_visible_from_camera():
    cam = _toy_camera()
    wall = gv.Prism("wall", xmin=-1.0, xmax=1.0, ymin=-2.7, ymax=-2.3, zmin=0.0, zmax=3.0)
    open_pt = np.array([[0.0, 0.0, 0.3]])
    behind = np.array([[0.0, 0.0, 0.3]])  # same point, but wall between cam and it
    assert bool(gv.points_visible_from_camera(cam, open_pt, [])[0])
    assert not bool(gv.points_visible_from_camera(cam, behind, [wall])[0])


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
