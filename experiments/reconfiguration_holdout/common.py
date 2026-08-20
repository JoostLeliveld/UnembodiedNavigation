#!/usr/bin/env python3
"""Shared apparatus for the reconfiguration holdout.

One place for the things every experiment here must agree on, so two experiments can
never silently score different fields, different labels or different thresholds:

- the five environments and the world file each one is;
- the one working grid every availability field is evaluated on;
- how a capture's detector outcomes become labels, including the threshold;
- the per-environment fields (monocular depth, CAD raycast) and the frozen
  ``L0``-fitted fields (GP, hybrid);
- the calibration link, the spatial blocks and the scoring rules -- all imported from
  the availability study rather than reimplemented, so the two papers cannot diverge
  on what "Brier" or "leave-one-block-out" means.

Nothing here reads ground truth as a model input. ``oracle_visible`` and the CAD
prisms are loaded, and are used only for evaluation-side reporting and for the two
declared CAD reference arms.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
_REPO_PATH = HERE.parents[1]
for _rel in ("src/unav_common", "src/experiments", "src/reliability"):
    _p = str(_REPO_PATH / _rel)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_exact_module(name: str, path: Path):
    """Load a study dependency by file, immune to generic-name module caches."""

    expected = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        actual = Path(getattr(existing, "__file__", "")).resolve()
        if actual != expected:
            raise ImportError(f"module {name!r} already resolves to {actual}, expected {expected}")
        return existing
    spec = importlib.util.spec_from_file_location(name, expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name!r} from {expected}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution: dataclass and other decorators resolve their
    # defining module through sys.modules while the module body is executing.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


_paths = _load_exact_module(
    "_reconfiguration_holdout_paths", _REPO_PATH / "scripts/shared/paths.py"
)
M = _load_exact_module(
    "_reconfiguration_holdout_metrics", _REPO_PATH / "scripts/shared/metrics.py"
)
ora = _load_exact_module(
    "_reconfiguration_holdout_oracle",
    _REPO_PATH / "experiments/dynamic_world_oracle/oracle.py",
)
repo_root = _paths.repo_root

REPO = repo_root(HERE)
STUDY = "reconfiguration_holdout"
OUT_ROOT = REPO / "logs/studies" / STUDY
CAPTURE_ROOT = REPO / "logs/visibility_comparison"
WORLDS = REPO / "src/sim/gazebo_worlds/worlds"

#: The L0 reference capture: 942 positions x 8 headings x 4 cameras, already on disk.
L0_CAPTURE = CAPTURE_ROOT / "commissioning_grid_20260807"

CAMERAS = ("external_camera", "external_camera_b", "external_camera_c", "external_camera_d")
SHORT = {"external_camera": "A", "external_camera_b": "B",
         "external_camera_c": "C", "external_camera_d": "D"}

#: Detector confidence threshold for every headline number.  Chosen in the
#: preregistration as the middle of the 0.05-0.50 plateau over which the nominal and
#: changed-lighting environments agree; at the L0 reference capture's own 0.01 the
#: detector fires at 60% of changed-lighting poses with no sight-line at all.
PRIMARY_THRESHOLD = 0.25

#: The four headings the new captures use.  A subset of the L0 reference's eight, so
#: every paired comparison can be made on identical poses.
THETAS = (0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469)
THETA_TOL = 1e-3

TARGET_HEIGHT_M = 0.35


@dataclass(frozen=True)
class Environment:
    key: str
    world_name: str
    layout: str
    lighting: str
    capture: Path
    #: True for the environment every arm is fitted on.  Exactly one is.
    is_development: bool = False


ENVIRONMENTS: tuple[Environment, ...] = (
    Environment("L0", "warehouse_full_4cam", "nominal", "nominal",
                L0_CAPTURE, is_development=True),
    Environment("L1", "warehouse_full_4cam_recfg", "restocked", "nominal",
                CAPTURE_ROOT / "recfg_holdout_L1"),
    Environment("L2", "warehouse_full_4cam_recfg2", "restocked_constrained_random", "nominal",
                CAPTURE_ROOT / "recfg_holdout_L2"),
    Environment("L0_lit", "warehouse_full_4cam_lit", "nominal", "changed",
                CAPTURE_ROOT / "recfg_holdout_L0_lit"),
    Environment("L1_lit", "warehouse_full_4cam_recfg_lit", "restocked", "changed",
                CAPTURE_ROOT / "recfg_holdout_L1_lit"),
)
ENV_BY_KEY = {e.key: e for e in ENVIRONMENTS}
DEVELOPMENT_ENV = "L0"


def working_grid() -> tuple[np.ndarray, np.ndarray]:
    """The one grid every field is evaluated on: 0.25 m over the operating floor.

    Chosen to match the dynamic-world oracle's grid so a visibility field computed
    here and an oracle grid computed there are directly comparable cell for cell.
    """
    grid = ora.FloorGrid(xmin=-11.75, xmax=11.75, ymin=-9.0, ymax=9.0, resolution_m=0.25)
    return grid.x_centres, grid.y_centres


def floor_grid() -> ora.FloorGrid:
    return ora.FloorGrid(xmin=-11.75, xmax=11.75, ymin=-9.0, ymax=9.0, resolution_m=0.25)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def _truthy(value: str) -> bool:
    return str(value).strip() in ("1", "1.0", "True", "true")


def load_events(env: Environment, *, threshold: float = PRIMARY_THRESHOLD,
                thetas: tuple[float, ...] | None = THETAS) -> dict[str, dict[str, np.ndarray]]:
    """Per-camera detector outcomes for one environment, at a stated threshold.

    The label is re-derived from ``yolo_raw_best_score`` rather than read from
    ``yolo_detected_after_threshold``, so every environment is thresholded
    identically no matter what threshold its capture happened to be scored at.  The
    L0 reference capture was scored at 0.01 and the new captures at 0.01 as well;
    re-thresholding offline is what makes them comparable at 0.25.

    ``thetas`` subsets the headings so the L0 reference's eight can be paired against
    the new captures' four.  Pass ``None`` to keep every heading.
    """
    path = env.capture / "perception_targets.csv"
    if not path.is_file():
        raise RuntimeError(f"{env.key}: no perception_targets.csv at {path}")
    per: dict[str, dict[str, list]] = {c: {"xy": [], "theta": [], "hit": [],
                                           "score": [], "oracle": []} for c in CAMERAS}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cam = str(row.get("camera_frame") or "")
            if cam not in per:
                continue
            theta = float(row["theta"])
            if thetas is not None and not any(abs(theta - t) < THETA_TOL for t in thetas):
                continue
            score = float(row.get("yolo_raw_best_score") or row.get("yolo_score_raw") or 0.0)
            per[cam]["xy"].append((float(row["x"]), float(row["y"])))
            per[cam]["theta"].append(theta)
            per[cam]["hit"].append(1.0 if score >= threshold else 0.0)
            per[cam]["score"].append(score)
            per[cam]["oracle"].append(1.0 if _truthy(row.get("oracle_visible", "")) else 0.0)
    out = {}
    for cam, d in per.items():
        if not d["xy"]:
            raise RuntimeError(f"{env.key}: no samples for {cam}")
        out[cam] = {
            "xy": np.asarray(d["xy"], dtype=float),
            "theta": np.asarray(d["theta"], dtype=float),
            "hit": np.asarray(d["hit"], dtype=float),
            "score": np.asarray(d["score"], dtype=float),
            "oracle": np.asarray(d["oracle"], dtype=float),
        }
    return out


def write_events_csv(events: dict[str, np.ndarray], path: Path) -> Path:
    """Write one camera's outcomes in the format ``fit_belief_aware_gp`` reads.

    The GP code is the repo's canonical implementation and takes an events CSV, so
    the way to reuse it rather than reimplement it is to hand it one.  The pose
    covariance is the same small fixed value the spawn-grid events carry: these are
    commanded teleports, not filtered estimates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["m_x", "m_y", "S_xx", "S_xy", "S_yy", "det_hit",
                    "yolo_score_raw", "run_id", "camera_id"])
        for (x, y), hit, score in zip(events["xy"], events["hit"], events["score"]):
            w.writerow([f"{x:.6f}", f"{y:.6f}", "0.01", "0.0", "0.01",
                        int(hit), f"{score:.6f}", "recfg_holdout", ""])
    return path


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------

def cad_field(world_name: str) -> dict[str, np.ndarray]:
    """Per-camera CAD raycast visibility on the working grid, for one world.

    EVALUATION REFERENCE, not a deployable arm: it needs a surveyed 3-D model of the
    building.  Two of them appear in the study -- the nominal world's, which is what
    a surveyed system still holds after the warehouse changed and nobody re-surveyed,
    and the changed world's, which is what it holds if somebody did.
    """
    scene = ora.OracleScene.from_world(WORLDS / f"{world_name}.world.sdf", list(CAMERAS))
    grids = ora.visibility_grids(scene.cameras, floor_grid(), scene.static_prisms, (),
                                target_height_m=TARGET_HEIGHT_M)
    # A visibility code becomes a score in [0, 1]: seen = 1, everything else 0. The
    # shared calibration link turns that into a probability, exactly as for the other
    # geometric arms, so no arm gets a free calibration the others do not.
    return {c: np.clip((grids[c] == ora.VISIBLE).astype(float), 1e-4, 1 - 1e-4)
            for c in CAMERAS}


def distance_field() -> dict[str, np.ndarray]:
    """Range-to-camera score, unfitted: ``1 - d / D`` with D the warehouse diagonal."""
    xs, ys = working_grid()
    gx, gy = np.meshgrid(xs, ys)
    poses = {"external_camera": (-6.0, -10.0), "external_camera_b": (-6.0, 10.0),
             "external_camera_c": (6.0, -10.0), "external_camera_d": (6.0, 10.0)}
    return {c: np.clip(1.0 - np.hypot(gx - p[0], gy - p[1]) / 30.0, 1e-4, 1 - 1e-4)
            for c, p in poses.items()}


def fov_range_field() -> dict[str, np.ndarray]:
    """In-image-and-in-range score: calibration plus the drivable map, no obstacles.

    Deliberately blind to occlusion, which is what makes it the control that says how
    much of any arm's skill is occlusion reasoning rather than framing and range.
    """
    scene = ora.OracleScene.from_world(WORLDS / "warehouse_full_4cam.world.sdf", list(CAMERAS))
    grids = ora.visibility_grids(scene.cameras, floor_grid(), (), (),
                                target_height_m=TARGET_HEIGHT_M)
    return {c: np.clip((grids[c] == ora.VISIBLE).astype(float), 1e-4, 1 - 1e-4)
            for c in CAMERAS}


def sample_at(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nearest-cell lookup of a (ny, nx) field at world points (N, 2).

    Nearest rather than bilinear on purpose: these fields carry occlusion edges, and
    interpolating across one invents visibility that no raycast produced.
    """
    xs, ys = working_grid()
    pts = np.asarray(pts, dtype=float)
    ix = np.clip(np.searchsorted(xs, pts[:, 0]), 0, len(xs) - 1)
    ix_lo = np.clip(ix - 1, 0, len(xs) - 1)
    ix = np.where(np.abs(xs[ix_lo] - pts[:, 0]) <= np.abs(xs[ix] - pts[:, 0]), ix_lo, ix)
    iy = np.clip(np.searchsorted(ys, pts[:, 1]), 0, len(ys) - 1)
    iy_lo = np.clip(iy - 1, 0, len(ys) - 1)
    iy = np.where(np.abs(ys[iy_lo] - pts[:, 1]) <= np.abs(ys[iy] - pts[:, 1]), iy_lo, iy)
    return np.asarray(field, dtype=float)[iy, ix]


def mono_depth_path(env_key: str) -> Path:
    return OUT_ROOT / "mono_depth" / f"{env_key}_visibility.npz"


def mono_depth_field(env_key: str) -> dict[str, np.ndarray]:
    """Per-camera monocular-depth visibility for one environment, on the working grid.

    ``p_unknown`` is folded in as a conservative fallback, matching the availability
    study: a cell the depth model could not resolve is not silently called visible.
    """
    path = mono_depth_path(env_key)
    if not path.is_file():
        raise RuntimeError(
            f"no monocular-depth field for {env_key}: {path}. "
            f"Run experiments/reconfiguration_holdout/mono_depth_field.py --env {env_key}")
    data = np.load(path)
    out = {}
    for c in CAMERAS:
        vis = np.asarray(data[f"{c}__p_visible"], dtype=float)
        unknown = np.asarray(data[f"{c}__p_unknown"], dtype=float)
        out[c] = np.clip(vis * (1.0 - np.clip(unknown, 0.0, 1.0)), 1e-4, 1 - 1e-4)
    return out


# --------------------------------------------------------------------------
# Link, blocks, scoring -- imported, never reimplemented
# --------------------------------------------------------------------------

def _logit(p, eps: float = 1e-4) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def fit_link(scores: np.ndarray, hits: np.ndarray, *, max_iter: int = 200) -> tuple[float, float]:
    """``P_D = sigmoid(a * logit(v) + b)`` by Newton-Raphson MLE, on training data only.

    Same two-parameter link for every arm, so a source that is already a calibrated
    probability recovers a ~ 1, b ~ 0 and no arm is advantaged by a free calibration
    the others do not get.  Kept identical to the availability study's
    implementation, which the tests compare against.
    """
    x = _logit(scores)
    y = np.asarray(hits, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2, dtype=float)
    w[1] = _logit(np.clip(np.mean(y), 1e-3, 1 - 1e-3))
    for _ in range(max_iter):
        eta = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))
        grad = X.T @ (y - p)
        s = np.clip(p * (1.0 - p), 1e-9, None)
        H = X.T @ (X * s[:, None])
        H[np.diag_indices_from(H)] += 1e-6
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return float(w[0]), float(w[1])


def apply_link(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    eta = a * _logit(scores) + b
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))


#: Three x-thirds by two y-halves, the six blocks the availability study froze.
BLOCK_X_EDGES = (-11.7, -3.9, 3.9, 11.7)
BLOCK_Y_EDGES = (-9.0, -0.25, 9.0)
N_BLOCKS = (len(BLOCK_X_EDGES) - 1) * (len(BLOCK_Y_EDGES) - 1)


def block_ids(pts: np.ndarray) -> np.ndarray:
    """Leave-one-block-out fold id for world points (N, 2).

    Spatial blocking, not random k-fold: on a dense pose grid a random split leaves
    held-out points centimetres from training points, so any smoother scores near
    perfectly and the comparison measures interpolation instead of transfer.
    """
    pts = np.asarray(pts, dtype=float)
    bx = np.clip(np.searchsorted(np.asarray(BLOCK_X_EDGES[1:-1]), pts[:, 0]),
                 0, len(BLOCK_X_EDGES) - 2)
    by = np.clip(np.searchsorted(np.asarray(BLOCK_Y_EDGES[1:-1]), pts[:, 1]),
                 0, len(BLOCK_Y_EDGES) - 2)
    return (bx * (len(BLOCK_Y_EDGES) - 1) + by).astype(int)


def score_predictions(hits: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """The registered scoring rules, from the canonical metrics module.

    ``brier_skill`` is reported alongside the raw Brier score because the raw score
    is not comparable across environments here.  The restock lowers the detection
    base rate from 0.313 to 0.270, and a rarer positive class makes Brier smaller for
    free: the constant-prevalence arm alone "improves" by 0.044 without knowing
    anything.  The skill score divides out each unit's own climatology --
    ``p(1 - p)`` at that unit's base rate -- so a change in skill is a change in what
    the estimator knew, not in how often the detector happened to fire.
    """
    y = np.asarray(hits, dtype=float)
    p = np.asarray(pred, dtype=float)
    base = float(np.mean(y))
    climatology = base * (1.0 - base)
    brier = float(M.brier(y, p))
    return {
        "brier": brier,
        "brier_climatology": climatology,
        "brier_skill": float(1.0 - brier / climatology) if climatology > 1e-9 else float("nan"),
        "logloss": float(M.logloss(y, p)),
        "auroc": float(M.auroc(y, p)),
        "ece": float(M.ece(y, p)),
        # The error that hurts a planner most: asserting the camera will see the
        # robot where it will not.
        "false_visible_rate": float(np.mean(p[y == 0] >= 0.5)) if np.any(y == 0) else float("nan"),
        "pred_mean": float(np.mean(p)),
        "target_mean": float(np.mean(y)),
    }


#: Camera subsets the study reports over.  Analysis-only: one capture serves all of
#: them.  Registered in advance because the availability study found the benefit of
#: availability modelling is largest at two cameras on opposite walls, not at four.
CAMERA_SUBSETS = {
    "4": ("external_camera", "external_camera_b", "external_camera_c", "external_camera_d"),
    "3": ("external_camera", "external_camera_b", "external_camera_c"),
    "2_opposite": ("external_camera", "external_camera_d"),
    "2_same_wall": ("external_camera", "external_camera_c"),
    "1": ("external_camera",),
}
