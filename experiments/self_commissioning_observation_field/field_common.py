from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "logs/studies/self_commissioning_observation_field"

P_SOURCE = REPO / "logs/studies/availability_paper/depth_gp_planner_v1/fused_planner_four_camera.npz"
P_FROZEN = OUT / "frozen/a3_depth_gp_four_camera.npz"
R_ROWS = REPO / "logs/studies/pixel_ground_path/e2_detector_edge_characterisation/detector_boxes.csv"
R_INDEX = REPO / "logs/perception_datasets/warehouse_yolo_dataset_4cam_v3_20260724/merged/localization_calibration_index.csv"
R_REGISTRY = REPO / "logs/studies/pixel_ground_path/e7_ipm_zero_parameter/summary.json"
WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_full_4cam.world.sdf"
ROUTES = REPO / "logs/studies/availability_paper/e3_route_discrimination/e3_selected_routes.json"

CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
INCLUDE_NAMES = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}
TASKS = (
    "mc_blind_L",
    "mc_m2_w2e_traverse",
    "full_traverse_handover",
    "route_tall_shadow_west",
)

VISIBILITY_THRESHOLD = 0.80
DT_S = 0.4
SPEED_MPS = 0.6
LENGTH_BUDGET_RATIO = 1.05
PROCESS_SIGMA_M_PER_STEP = 0.02
INITIAL_SIGMA_M = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def freeze_inputs() -> dict:
    inputs = (P_SOURCE, R_ROWS, R_INDEX, R_REGISTRY, WORLD, ROUTES)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"required input missing: {missing}")
    P_FROZEN.parent.mkdir(parents=True, exist_ok=True)
    if not P_FROZEN.is_file() or sha256(P_FROZEN) != sha256(P_SOURCE):
        shutil.copyfile(P_SOURCE, P_FROZEN)
    payload = {
        "inputs_sha256": {str(path.relative_to(REPO)): sha256(path) for path in inputs},
        "frozen_a3": {
            "path": str(P_FROZEN.relative_to(REPO)),
            "sha256": sha256(P_FROZEN),
            "role": "monocular-depth geometry prior plus learned availability GP residual",
        },
    }
    write_json(OUT / "frozen/manifest.json", payload)
    return payload


def assert_frozen() -> None:
    path = OUT / "frozen/manifest.json"
    if not path.is_file():
        raise RuntimeError("frozen manifest missing; run freeze_inputs.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest["inputs_sha256"].items():
        source = REPO / relative
        actual = sha256(source) if source.is_file() else "MISSING"
        if actual != expected:
            raise RuntimeError(f"frozen input changed: {relative}; expected {expected}, found {actual}")
    if not P_FROZEN.is_file() or sha256(P_FROZEN) != manifest["frozen_a3"]["sha256"]:
        raise RuntimeError("frozen A3 artifact is missing or changed")


def load_p() -> dict[str, np.ndarray]:
    assert_frozen()
    archive = np.load(P_FROZEN)
    return {key: np.asarray(archive[key]) for key in archive.files}


def sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    ix = np.abs(xs[None, :] - points[:, 0, None]).argmin(axis=1)
    iy = np.abs(ys[None, :] - points[:, 1, None]).argmin(axis=1)
    return np.asarray(field)[iy, ix]


def visibility_mode(probability: float | np.ndarray) -> str | np.ndarray:
    values = np.asarray(probability)
    result = np.where(values >= VISIBILITY_THRESHOLD, "clear", "marginal")
    return str(result) if result.ndim == 0 else result


def fused_p(p: dict[str, np.ndarray], cameras: tuple[str, ...] = CAMERAS) -> np.ndarray:
    miss = np.ones_like(np.asarray(p[f"P_{cameras[0]}_map"], dtype=float))
    for camera in cameras:
        miss *= 1.0 - np.clip(np.asarray(p[f"P_{camera}_map"], dtype=float), 0.0, 1.0)
    return 1.0 - miss


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path, dtype=float), axis=0), axis=1).sum())


def resample_path(path: np.ndarray, step_m: float = SPEED_MPS * DT_S) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(segments.sum())
    if total <= 0.0:
        return path[:1]
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    distances = np.linspace(0.0, total, max(2, int(np.ceil(total / step_m)) + 1))
    return np.column_stack([
        np.interp(distances, cumulative, path[:, 0]),
        np.interp(distances, cumulative, path[:, 1]),
    ])


def expected_longest_miss(p_hit: np.ndarray) -> float:
    """Exact E[max consecutive misses] for independent non-identical trials."""
    state = {(0, 0): 1.0}
    for probability in np.clip(np.asarray(p_hit, dtype=float), 0.0, 1.0):
        nxt: dict[tuple[int, int], float] = {}
        for (run, maximum), mass in state.items():
            nxt[(0, maximum)] = nxt.get((0, maximum), 0.0) + mass * float(probability)
            new_run = run + 1
            key = (new_run, max(maximum, new_run))
            nxt[key] = nxt.get(key, 0.0) + mass * float(1.0 - probability)
        state = nxt
    return float(sum(maximum * mass for (_, maximum), mass in state.items()))


def route_library() -> dict[str, list[tuple[str, np.ndarray]]]:
    assert_frozen()
    payload = json.loads(ROUTES.read_text(encoding="utf-8"))["routes"]
    result: dict[str, dict[bytes, tuple[str, np.ndarray]]] = {task: {} for task in TASKS}
    for subset, by_task in payload.items():
        for task in TASKS:
            for source, points in by_task[task].items():
                route = np.asarray(points, dtype=float)
                key = np.round(resample_path(route, 0.05), 3).tobytes()
                result[task].setdefault(key, (f"{subset}:{source}", route))
    return {task: list(entries.values()) for task, entries in result.items()}


def max_route_separation(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = resample_path(a, 0.05), resample_path(b, 0.05)
    da = np.min(np.linalg.norm(aa[:, None, :] - bb[None, :, :], axis=2), axis=1)
    db = np.min(np.linalg.norm(bb[:, None, :] - aa[None, :, :], axis=2), axis=1)
    return float(max(da.max(), db.max()))


def add_import_paths() -> None:
    for relative in ("src/unav_common", "src/reliability"):
        value = str(REPO / relative)
        if value not in sys.path:
            sys.path.insert(0, value)
