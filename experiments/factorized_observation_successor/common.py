from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "logs/studies/factorized_observation_successor"
P_SOURCE = REPO / "logs/studies/availability_paper/depth_gp_planner_v1/fused_planner_four_camera.npz"
FROZEN_P = OUT / "frozen/p_use_depth_gp_four_camera.npz"
R_ROWS = REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation/detector_boxes.csv"
R_INDEX = REPO / "logs/perception_datasets/warehouse_yolo_dataset_4cam_v3_20260724/merged/localization_calibration_index.csv"
R_REGISTRY = REPO / "logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
ROUTES = REPO / "logs/studies/availability_paper/e3_route_discrimination/e3_selected_routes.json"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
DEV_CAMERAS = ("camera_A", "camera_B")
HOLDOUT_CAMERAS = ("camera_B", "camera_C")
TASKS = (
    "mc_blind_L",
    "mc_m2_w2e_traverse",
    "full_traverse_handover",
    "route_tall_shadow_west",
)
DT_S = 0.4
SPEED_MPS = 0.6
LENGTH_BUDGET_RATIO = 1.05


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_frozen(paths: tuple[Path, ...]) -> None:
    manifest_path = OUT / "frozen/manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("frozen input manifest missing; run freeze_inputs.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["inputs_sha256"]
    for path in paths:
        key = str(path.relative_to(REPO))
        expected = manifest.get(key)
        actual = sha256(path) if path.is_file() else "MISSING"
        if expected != actual:
            raise RuntimeError(f"frozen input changed: {key}; expected {expected}, found {actual}")


def load_p() -> dict[str, np.ndarray]:
    if not FROZEN_P.is_file():
        raise RuntimeError("frozen p_use copy missing; run freeze_inputs.py")
    frozen = json.loads((OUT / "frozen/manifest.json").read_text(encoding="utf-8"))["p_use"]
    if sha256(FROZEN_P) != frozen["sha256"]:
        raise RuntimeError("frozen p_use copy hash mismatch")
    z = np.load(FROZEN_P)
    return {key: np.asarray(z[key]) for key in z.files}


def fused_p(p: dict[str, np.ndarray], cameras: tuple[str, ...]) -> np.ndarray:
    miss = np.ones_like(np.asarray(p[f"P_{cameras[0]}_map"], dtype=float))
    for camera in cameras:
        miss *= 1.0 - np.clip(np.asarray(p[f"P_{camera}_map"], dtype=float), 0.0, 1.0)
    return 1.0 - miss


def sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    ix = np.abs(xs[None, :] - points[:, 0, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - points[:, 1, None]).argmin(axis=1)
    return np.asarray(field)[iy, ix]


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def resample_path(path: np.ndarray, step_m: float = SPEED_MPS * DT_S) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return path[:1]
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    distance = np.linspace(0.0, total, max(2, int(np.ceil(total / step_m)) + 1))
    return np.column_stack([
        np.interp(distance, cumulative, path[:, 0]),
        np.interp(distance, cumulative, path[:, 1]),
    ])


def expected_longest_miss(p_hit: np.ndarray) -> float:
    """Exact E[max consecutive misses] for independent non-identical Bernoulli trials."""
    probs = {(0, 0): 1.0}  # (current miss run, maximum miss run) -> mass
    for p in np.clip(np.asarray(p_hit, dtype=float), 0.0, 1.0):
        nxt: dict[tuple[int, int], float] = {}
        for (run, maximum), mass in probs.items():
            nxt[(0, maximum)] = nxt.get((0, maximum), 0.0) + mass * float(p)
            new_run = run + 1
            key = (new_run, max(maximum, new_run))
            nxt[key] = nxt.get(key, 0.0) + mass * float(1.0 - p)
        probs = nxt
    return float(sum(maximum * mass for (_, maximum), mass in probs.items()))


def route_library() -> dict[str, list[tuple[str, np.ndarray]]]:
    assert_frozen((ROUTES,))
    payload = json.loads(ROUTES.read_text(encoding="utf-8"))["routes"]
    result: dict[str, dict[bytes, tuple[str, np.ndarray]]] = {task: {} for task in TASKS}
    for subset, by_task in payload.items():
        for task in TASKS:
            for source, points in by_task[task].items():
                route = np.asarray(points, dtype=float)
                key = np.round(resample_path(route, 0.05), 3).tobytes()
                result[task].setdefault(key, (f"{subset}:{source}", route))
    return {task: list(routes.values()) for task, routes in result.items()}


def max_route_separation(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = resample_path(a, 0.05), resample_path(b, 0.05)
    da = np.min(np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2), axis=1)
    db = np.min(np.linalg.norm(bb[:, None, :] - aa[None, :, :], axis=2), axis=1)
    return float(max(da.max(), db.max()))


def add_import_paths() -> None:
    for rel in ("src/unav_common", "src/reliability"):
        value = str(REPO / rel)
        if value not in sys.path:
            sys.path.insert(0, value)
