from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

import common as C

C.add_import_paths()
from reliability.projection import camera_model_from_world  # noqa: E402

INCLUDE = {
    "camera_A": "external_camera", "camera_B": "external_camera_b",
    "camera_C": "external_camera_c", "camera_D": "external_camera_d",
}


@dataclass
class RouteScore:
    route_id: str
    path: np.ndarray
    length_m: float
    expected_longest_miss_steps: float
    conditional_information: float

    @property
    def expected_longest_miss_s(self) -> float:
        return self.expected_longest_miss_steps * C.DT_S


def load_rcond() -> tuple[dict, dict[str, int]]:
    path = C.OUT / "rcond/r_cond_uv.npz"
    if not path.is_file():
        raise RuntimeError("run commission_rcond.py first")
    z = np.load(path)
    value = {key: np.asarray(z[key]) for key in z.files}
    ids = [str(v) for v in value["camera_ids"]]
    return value, {camera: ids.index(camera) for camera in ids}


def ground_trace(camera_model, point: np.ndarray, covariance_uv: np.ndarray) -> float:
    u, v, _ = camera_model.world_to_pixel(float(point[0]), float(point[1]), 0.0)
    eps = 0.25
    up, um = camera_model.pixel_to_world(u + eps, v), camera_model.pixel_to_world(u - eps, v)
    vp, vm = camera_model.pixel_to_world(u, v + eps), camera_model.pixel_to_world(u, v - eps)
    if any(value is None for value in (up, um, vp, vm)):
        return float("inf")
    jac = np.column_stack([
        (np.asarray(up) - np.asarray(um)) / (2 * eps),
        (np.asarray(vp) - np.asarray(vm)) / (2 * eps),
    ])
    return float(np.trace(jac @ covariance_uv @ jac.T))


def score_route(route_id: str, path: np.ndarray, cameras: tuple[str, ...], p: dict[str, np.ndarray]) -> RouteScore:
    pts = C.resample_path(path)
    xs, ys = np.asarray(p["xs"], float), np.asarray(p["ys"], float)
    per_camera_p = {camera: C.sample(np.asarray(p[f"P_{camera}_map"], float), xs, ys, pts) for camera in cameras}
    fused = 1.0 - np.prod(np.stack([1.0 - per_camera_p[camera] for camera in cameras]), axis=0)
    rcond, indices = load_rcond()
    models = {camera: camera_model_from_world(C.WORLD, include_name=INCLUDE[camera]) for camera in cameras}
    info = 0.0
    for k, point in enumerate(pts):
        ix = int(np.argmin(np.abs(xs - point[0])))
        iy = int(np.argmin(np.abs(ys - point[1])))
        for camera in cameras:
            ci = indices[camera]
            cov = np.asarray([
                [rcond["R_uu_px2"][ci, iy, ix], rcond["R_uv_px2"][ci, iy, ix]],
                [rcond["R_uv_px2"][ci, iy, ix], rcond["R_vv_px2"][ci, iy, ix]],
            ])
            trace = ground_trace(models[camera], point, cov)
            if np.isfinite(trace):
                info += float(per_camera_p[camera][k]) / max(trace, 1e-9)
    return RouteScore(
        route_id=route_id, path=np.asarray(path, float), length_m=C.path_length(path),
        expected_longest_miss_steps=C.expected_longest_miss(fused),
        conditional_information=float(info / max(len(pts), 1)),
    )


def solve(task: str, cameras: tuple[str, ...], p: dict[str, np.ndarray] | None = None) -> dict:
    p = C.load_p() if p is None else p
    routes = C.route_library()[task]
    scores = [score_route(route_id, path, cameras, p) for route_id, path in routes]
    shortest = min(scores, key=lambda item: (item.length_m, item.route_id))
    budget = shortest.length_m * C.LENGTH_BUDGET_RATIO
    eligible = [item for item in scores if item.length_m <= budget + 1e-9]
    selected = min(
        eligible,
        key=lambda item: (
            round(item.expected_longest_miss_steps, 10),
            -round(item.conditional_information, 10),
            item.length_m,
            item.route_id,
        ),
    )
    return {"shortest": shortest, "selected": selected, "eligible": eligible, "all": scores, "budget_m": budget}


def serialise(score: RouteScore) -> dict:
    return {
        "route_id": score.route_id, "path": score.path.tolist(), "length_m": score.length_m,
        "expected_longest_miss_s": score.expected_longest_miss_s,
        "conditional_information": score.conditional_information,
    }

