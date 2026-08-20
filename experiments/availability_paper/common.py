#!/usr/bin/env python3
"""Shared apparatus for the availability-aware planning paper (E0-E4).

One place for the things every experiment in this package must agree on, so that
two experiments can never silently score different fields, different splits, or
different link functions:

- the candidate-pose grid and the driveable mask (from ``render_all``);
- the six per-camera availability SOURCES, all resampled onto that one grid;
- the spatially-blocked leave-one-block-out split used for every held-out number;
- the two-parameter calibration LINK that turns a geometric visibility score
  into a detection probability, fitted on training folds only;
- input hashing for the manifest.

Nothing here reads ground truth. The detector-outcome events (``det_hit``) are
labels, not truth about the world: they are the estimand itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parents[1] / "scripts/shared"))
from paths import repo_root  # noqa: E402  depth-independent root (see scripts/shared/paths.py)

REPO = repo_root(HERE)
SUPERVISOR = REPO / "experiments/usable_observation/supervisor_comparison"

#: Study outputs live under logs/studies/<study>/<expN>/, never beside the code.
#: See CLAUDE.md, "Starting a new investigation study".
STUDY_NAME = "availability_paper"
OUT_ROOT = REPO / "logs/studies" / STUDY_NAME

for _source in (
    REPO / "src/unav_common",
    REPO / "src/reliability",
    REPO / "src/planning",
    REPO / "scripts/shared",
    str(SUPERVISOR),
):
    _p = str(_source)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics as M  # noqa: E402  canonical metric implementations (scripts/shared)
import render_all as base  # noqa: E402

CAMERAS: tuple[str, ...] = tuple(base.CAMERA_POSES)

MONO_DEPTH_MAP_DIR = (
    REPO
    / "experiments/usable_observation/supervisor_comparison"
    / "10_monocular_depth_results/four_camera/maps"
)
#: Default monocular model. The four-camera study selected UniDepthV2 ViT-S on
#: floor-anchored depth MAE (0.247 m, against 0.327 / 0.337 / 0.420 m for the
#: others), so it is the default rather than a free choice. Swap it with
#: ``set_mono_depth_model`` to test whether an availability result depends on which
#: depth model produced the field — a reviewer will ask.
MONO_DEPTH_MODEL = "unidepth_v2_vits14"
MONO_DEPTH_MAPS = MONO_DEPTH_MAP_DIR / f"visibility_maps__{MONO_DEPTH_MODEL}.npz"
MONO_DEPTH_RESULTS = MONO_DEPTH_MAP_DIR.parent / "results.json"

# Frozen E4 planner endpoints.  These are image-space standard deviations from
# the registered campaign, not camera-measurement calibration results.  Keeping
# the mapping here lets every availability figure render the same corresponding
# historical folded planner covariance.
R_VISIBLE_UV_PX = 2.5
R_MISS_UV_PX = 40.0


def folded_planner_sigma_px(p_use: np.ndarray | float) -> np.ndarray:
    """Historical precision blend from availability to isotropic planner R.

    This is the C2 planning adapter, expressed as its per-axis standard
    deviation.  It is deliberately not used for the explicit hit/miss model,
    where availability and conditional measurement covariance stay separate.
    """

    p = np.clip(np.asarray(p_use, dtype=float), 0.0, 1.0)
    precision = p / R_VISIBLE_UV_PX**2 + (1.0 - p) / R_MISS_UV_PX**2
    return np.sqrt(1.0 / np.maximum(precision, 1e-12))


def available_mono_depth_models() -> list[str]:
    """Every monocular model with a four-camera visibility map on disk."""

    return sorted(
        p.name.removeprefix("visibility_maps__").removesuffix(".npz")
        for p in MONO_DEPTH_MAP_DIR.glob("visibility_maps__*.npz")
    )


def set_mono_depth_model(name: str) -> None:
    """Point the monocular arm at a different depth model, before build_apparatus()."""

    global MONO_DEPTH_MODEL, MONO_DEPTH_MAPS
    options = available_mono_depth_models()
    if name not in options:
        raise RuntimeError(f"unknown monocular model {name!r}; available: {options}")
    MONO_DEPTH_MODEL = name
    MONO_DEPTH_MAPS = MONO_DEPTH_MAP_DIR / f"visibility_maps__{name}.npz"


#: The clear-frame timestamp. The 1p2 frame contains the added pallet and is a
#: *dynamic* regime; assumption A06 keeps the two regimes separate, so the
#: static availability comparison uses 0p4 only.
MONO_DEPTH_STAMP = "0p4"
MONO_DEPTH_CAMERA_KEY = {
    "camera_A": "external_camera",
    "camera_B": "external_camera_b",
    "camera_C": "external_camera_c",
    "camera_D": "external_camera_d",
}

EVENT_ROOT = base.EVENT_ROOT

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One availability estimator arm.

    ``needs_surveyed_model`` is the deployment question the paper actually asks:
    can the field be produced without an explicit surveyed 3-D obstacle model?
    ``is_reference`` marks arms that are evaluation references, not deployable
    methods, and which therefore may never be reported as a deployable result.
    """

    key: str
    label: str
    operational_inputs: tuple[str, ...]
    needs_surveyed_model: bool
    is_reference: bool = False
    color: str = "#444444"


SOURCES: tuple[Source, ...] = (
    Source(
        key="constant",
        label="Constant (train prevalence)",
        operational_inputs=("detector_outcomes",),
        needs_surveyed_model=False,
        color="#8a8a8a",
    ),
    # The standard "reliability decays with range" model, and the closest thing here
    # to an external baseline. Answers reviewer question RQ01 directly. It is scored
    # as a pure geometric quantity — range from the camera — with all calibration
    # left to the same two-parameter link every arm gets, so nothing is fitted on
    # the held-out block.
    Source(
        key="distance",
        label="Distance to camera only",
        operational_inputs=("camera_calibration",),
        needs_surveyed_model=False,
        color="#5875a4",
    ),
    Source(
        key="fov_range",
        label="FOV / range",
        operational_inputs=("camera_calibration", "drivable_map"),
        needs_surveyed_model=False,
        color="#00a6a6",
    ),
    Source(
        key="cad_reference",
        label="CAD raycast",
        operational_inputs=("camera_calibration", "surveyed_3d_model"),
        needs_surveyed_model=True,
        is_reference=True,
        color="#d94b4b",
    ),
    Source(
        key="mono_depth",
        label="Monocular depth raycast",
        operational_inputs=("camera_calibration", "drivable_map", "camera_rgb"),
        needs_surveyed_model=False,
        color="#d89000",
    ),
    # The two GP arms differ ONLY in their mean function. Keeping both is the
    # point: it separates what the detector outcomes contribute from what the
    # geometry contributes, which a single "GP" arm cannot show.
    Source(
        key="gp",
        label="GP, no geometric prior mean",
        operational_inputs=("camera_calibration", "detector_outcomes"),
        needs_surveyed_model=False,
        color="#7b53b5",
    ),
    Source(
        key="hybrid",
        label="GP on a geometric prior mean (operational)",
        operational_inputs=("camera_calibration", "surveyed_3d_model", "detector_outcomes"),
        needs_surveyed_model=True,
        color="#2a9d58",
    ),
)

SOURCE_BY_KEY = {s.key: s for s in SOURCES}


def _resample_to_grid(
    src_xs: np.ndarray,
    src_ys: np.ndarray,
    src_field: np.ndarray,
    dst_xs: np.ndarray,
    dst_ys: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour resample of a (ny, nx) field onto a coarser/finer grid.

    Nearest rather than bilinear on purpose: these fields contain occlusion
    edges, and bilinear would invent intermediate visibility on the shelf
    boundary that neither the depth buffer nor the CAD raycast produced.
    """

    ix = np.clip(np.searchsorted(src_xs, dst_xs) , 0, len(src_xs) - 1)
    ix_lo = np.clip(ix - 1, 0, len(src_xs) - 1)
    pick_lo = np.abs(src_xs[ix_lo] - dst_xs) <= np.abs(src_xs[ix] - dst_xs)
    ix = np.where(pick_lo, ix_lo, ix)

    iy = np.clip(np.searchsorted(src_ys, dst_ys), 0, len(src_ys) - 1)
    iy_lo = np.clip(iy - 1, 0, len(src_ys) - 1)
    pick_lo_y = np.abs(src_ys[iy_lo] - dst_ys) <= np.abs(src_ys[iy] - dst_ys)
    iy = np.where(pick_lo_y, iy_lo, iy)

    return np.asarray(src_field)[np.ix_(iy, ix)]


#: Normalising range for the distance-only arm: the warehouse's own diagonal, so the
#: score spans [0, 1] over reachable ground without a tuned constant.
DISTANCE_NORMALISER_M = 30.0


def distance_fields(xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    """Per-camera score that decreases with ground range to the camera.

    Deliberately unfitted: ``1 - d / D`` with ``D`` the warehouse diagonal. Every
    monotone rescaling of this is absorbed by the shared calibration link, so the
    arm tests whether *range alone* predicts detection, with no free parameters of
    its own and nothing fitted on held-out ground.
    """

    gx, gy = np.meshgrid(xs, ys)
    out: dict[str, np.ndarray] = {}
    for camera, pose in base.CAMERA_POSES.items():
        d = np.hypot(gx - float(pose[0]), gy - float(pose[1]))
        out[camera] = np.clip(1.0 - d / DISTANCE_NORMALISER_M, 1e-4, 1.0 - 1e-4)
    return out


def load_mono_depth_fields(xs: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
    """Per-camera monocular-depth visibility, resampled onto the working grid.

    ``p_unknown`` is folded in as an explicit conservative fallback (A05): a cell
    the depth model could not resolve is not silently called visible. The
    published field is ``p_visible`` on resolved cells and ``0`` where unknown
    dominates, which is the same fallback the operator dashboard uses.
    """

    if not MONO_DEPTH_MAPS.is_file():
        raise RuntimeError(
            f"Monocular-depth visibility maps not found: {MONO_DEPTH_MAPS}. "
            "Run experiments/usable_observation/supervisor_comparison/"
            "10_monocular_depth_results/four_camera_study.py first."
        )
    data = np.load(MONO_DEPTH_MAPS, allow_pickle=True)
    src_xs = np.asarray(data["xs"], dtype=float)
    src_ys = np.asarray(data["ys"], dtype=float)
    out: dict[str, np.ndarray] = {}
    for camera, prefix in MONO_DEPTH_CAMERA_KEY.items():
        vis = np.asarray(data[f"{prefix}__{MONO_DEPTH_STAMP}__p_visible"], dtype=float)
        unknown = np.asarray(data[f"{prefix}__{MONO_DEPTH_STAMP}__p_unknown"], dtype=float)
        resolved = np.clip(vis * (1.0 - np.clip(unknown, 0.0, 1.0)), 0.0, 1.0)
        out[camera] = _resample_to_grid(src_xs, src_ys, resolved, xs, ys)
    return out


@dataclass
class Apparatus:
    """Everything the offline experiments share, built once."""

    xs: np.ndarray
    ys: np.ndarray
    driveable: np.ndarray
    prisms: list
    tasks: dict
    events: dict[str, dict[str, np.ndarray]]
    fields: dict[str, dict[str, np.ndarray]]
    prevalence: dict[str, float]
    ctx: dict = dc_field(repr=False, default_factory=dict)

    def field(self, source_key: str, camera: str) -> np.ndarray:
        return self.fields[source_key][camera]


def build_apparatus() -> Apparatus:
    """Build the grid, the events and every per-camera availability field."""

    ctx = base.build_context()
    xs = np.asarray(ctx["xs"], dtype=float)
    ys = np.asarray(ctx["ys"], dtype=float)
    per_camera = ctx["per_camera"]

    fields: dict[str, dict[str, np.ndarray]] = {}
    for source in SOURCES:
        if source.key == "constant":
            continue  # fitted per fold, has no spatial field
        if source.key == "mono_depth":
            fields[source.key] = load_mono_depth_fields(xs, ys)
            continue
        if source.key == "distance":
            fields[source.key] = distance_fields(xs, ys)
            continue
        if source.key not in per_camera:
            raise RuntimeError(
                f"render_all.build_context() produced no '{source.key}' field; "
                f"available: {sorted(per_camera)}"
            )
        fields[source.key] = {
            camera: np.asarray(per_camera[source.key][camera], dtype=float)
            for camera in CAMERAS
        }

    events = ctx["events"]
    prevalence = {c: float(np.mean(events[c]["hit"])) for c in CAMERAS}

    return Apparatus(
        xs=xs,
        ys=ys,
        driveable=np.asarray(ctx["driveable"], dtype=bool),
        prisms=list(ctx["prisms"]),
        tasks=dict(ctx["tasks"]),
        events=events,
        fields=fields,
        prevalence=prevalence,
        ctx=ctx,
    )


def sample_field_at(field: np.ndarray, xs: np.ndarray, ys: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Nearest-cell lookup of a (ny, nx) field at world points ``pts`` (N, 2)."""

    pts = np.asarray(pts, dtype=float)
    ix = np.clip(np.searchsorted(xs, pts[:, 0]), 0, len(xs) - 1)
    ix_lo = np.clip(ix - 1, 0, len(xs) - 1)
    ix = np.where(np.abs(xs[ix_lo] - pts[:, 0]) <= np.abs(xs[ix] - pts[:, 0]), ix_lo, ix)
    iy = np.clip(np.searchsorted(ys, pts[:, 1]), 0, len(ys) - 1)
    iy_lo = np.clip(iy - 1, 0, len(ys) - 1)
    iy = np.where(np.abs(ys[iy_lo] - pts[:, 1]) <= np.abs(ys[iy] - pts[:, 1]), iy_lo, iy)
    return np.asarray(field, dtype=float)[iy, ix]


# --------------------------------------------------------------------------
# Spatially blocked split
# --------------------------------------------------------------------------

#: Three x-thirds by two y-halves over the warehouse extent. This reproduces the
#: six ``block_x{i}_y{j}`` groups already frozen in the 2026-07-27 spawn-grid
#: ``events_blocked`` capture; :func:`assert_blocks_match_capture` checks that.
BLOCK_X_EDGES = (-11.7, -3.9, 3.9, 11.7)
BLOCK_Y_EDGES = (-9.0, -0.25, 9.0)
N_BLOCKS = (len(BLOCK_X_EDGES) - 1) * (len(BLOCK_Y_EDGES) - 1)


def block_ids(pts: np.ndarray) -> np.ndarray:
    """Leave-one-block-out fold id in ``[0, N_BLOCKS)`` for world points (N, 2).

    Spatial blocking, not random k-fold. A random split over a dense pose grid
    puts held-out points a few centimetres from training points, so a smoother
    with any length scale at all scores near-perfectly and the comparison
    measures interpolation rather than the transfer the deployment needs.
    """

    pts = np.asarray(pts, dtype=float)
    bx = np.clip(np.searchsorted(np.asarray(BLOCK_X_EDGES[1:-1]), pts[:, 0]), 0, len(BLOCK_X_EDGES) - 2)
    by = np.clip(np.searchsorted(np.asarray(BLOCK_Y_EDGES[1:-1]), pts[:, 1]), 0, len(BLOCK_Y_EDGES) - 2)
    return (bx * (len(BLOCK_Y_EDGES) - 1) + by).astype(int)


def assert_blocks_match_capture(events: dict[str, dict[str, np.ndarray]]) -> dict[str, int]:
    """Check the geometric block rule reproduces the frozen capture's groups.

    Raises if the six blocks are not all populated, which is the failure mode
    that would silently turn leave-one-block-out into leave-nothing-out.
    """

    counts: dict[str, int] = {}
    for camera in CAMERAS:
        ids = block_ids(events[camera]["xy"])
        present = np.unique(ids)
        if len(present) != N_BLOCKS:
            raise RuntimeError(
                f"{camera}: spatial blocking produced {len(present)} of {N_BLOCKS} blocks "
                f"({present.tolist()}); the split rule and the capture disagree."
            )
        for b in present:
            counts[f"{camera}_block{int(b)}"] = int(np.sum(ids == b))
    return counts


# --------------------------------------------------------------------------
# Calibration link
# --------------------------------------------------------------------------


def _logit(p, eps: float = 1e-4) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def fit_link(scores: np.ndarray, hits: np.ndarray, *, max_iter: int = 200) -> tuple[float, float]:
    """Fit ``P_D = sigmoid(a * logit(v) + b)`` by Newton-Raphson MLE.

    This is the ``g(.)`` that turns a geometric visibility fraction into a
    detection probability. Every source gets the same two-parameter link, so no
    arm is advantaged by being handed a free calibration the others do not get:
    a source that is already a well-calibrated probability simply recovers
    ``a ~ 1, b ~ 0``.

    Fitted on TRAINING folds only. Returns ``(a, b)``.
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
        H[np.diag_indices_from(H)] += 1e-6  # ridge; the constant-score case is singular
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


def score_predictions(hits: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """The four registered scoring rules, from the canonical metrics module."""

    y = np.asarray(hits, dtype=float)
    p = np.asarray(pred, dtype=float)
    auc = M.auroc(y, p)
    return {
        "brier": M.brier(y, p),
        "logloss": M.logloss(y, p),
        "auroc": float(auc) if np.isfinite(auc) else float("nan"),
        "ece": M.ece(y, p),
        "pred_mean": float(np.mean(p)),
        "target_mean": float(np.mean(y)),
        "n": int(y.size),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_manifest(extra: Iterable[Path] = ()) -> dict[str, str]:
    """sha256 of every file the offline experiments consume."""

    paths = [EVENT_ROOT / f"{c}_events.csv" for c in CAMERAS]
    paths += [base.GP_ROOT / c / "det_hit_expected_kernel_gp.npz" for c in CAMERAS]
    paths += [base.WORLD, base.DAYZERO, MONO_DEPTH_MAPS, MONO_DEPTH_RESULTS]
    paths += list(extra)
    out: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        out[str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)] = (
            sha256(path) if path.is_file() else "MISSING"
        )
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    import csv

    columns = list(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
