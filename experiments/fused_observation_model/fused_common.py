#!/usr/bin/env python3
"""Shared helpers for the fused-observation-model study (availability half).

Scope
-----
The factorized external-camera observation model is

    p_use,c(x)  =  p_avail,c(x)  *  p_acc,c(x | detected)

This study nails down the AVAILABILITY factor for the four-camera warehouse
network: per-camera ``p_avail,c(x)``, whether the four cameras are conditionally
independent given position, and which fusion rule best predicts
``P(at least one camera usable | x)``.

Hard rules this module enforces for every caller
------------------------------------------------
* Scoring comes from ``scripts/shared/metrics.py`` ONLY (never hand-rolled).
* GP fitting comes from ``scripts/visibility_comparison/fit_belief_aware_gp.py``
  ONLY (never reimplemented) with the hyper-parameters frozen by the locked
  ``spawn_grid_20260727`` fits.
* Repo root via ``scripts/shared/paths.repo_root()``, never ``parents[N]``.
* Real captured events only. Nothing here interpolates, resamples or otherwise
  invents an observation.
* Ground truth / ``eval_*`` columns are evaluation-only. The spawn-grid event
  tables carry no GT columns at all; ``det_hit`` is the measured detector
  outcome and is both the fit target and the scored label.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------- paths
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
VIS_DIR = REPO / "scripts" / "visibility_comparison"
SHARED_DIR = REPO / "scripts" / "shared"
for _p in (str(VIS_DIR), str(SHARED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_belief_aware_gp as fbg  # noqa: E402  # canonical GP code — never reimplement
import metrics as M  # noqa: E402  # canonical scoring — never reimplement

CAMERAS = ("A", "B", "C", "D")

CAMPAIGN = REPO / "logs" / "studies" / "multicamera_commissioning_bigwarehouse"
EVENTS_DIR = CAMPAIGN / "spawn_grid_20260727" / "events"
PRIOR_DIR = (
    CAMPAIGN
    / "actual_commissioning_20260715"
    / "analysis"
    / "final_01"
    / "inputs"
)
LOCKED_GP_DIR = REPO / "logs" / "visibility_comparison" / "spawn_grid_20260727" / "gp"
LOCKED_VALIDATION_DIR = (
    REPO / "logs" / "visibility_comparison" / "spawn_grid_20260727" / "validation"
)
OUT_ROOT = REPO / "logs" / "studies" / "fused_observation_model"


def events_csv(cam: str) -> Path:
    return EVENTS_DIR / f"camera_{cam}_events.csv"


def prior_npz(cam: str) -> Path:
    return PRIOR_DIR / f"camera_{cam}_dayzero_prior.npz"


def locked_gp_npz(cam: str) -> Path:
    return LOCKED_GP_DIR / f"camera_{cam}" / "det_hit_expected_kernel_gp.npz"


# ------------------------------------------------------- frozen GP settings
#: Verbatim from logs/visibility_comparison/spawn_grid_20260727/gp/camera_A/
#: manifest.json -> "hyperparameters".  Refitting with anything else would make
#: the held-out numbers incomparable with the locked in-sample artifacts.
GP_HYPERPARAMS = dict(
    aggregate_resolution_m=0.30,
    max_bin_weight=20.0,
    gp_length_scale=1.20,
    gp_noise_var=0.05,
    beta=0.5,
    min_prob=1e-4,
    pose_length_scale=0.35,
    min_certainty=0.05,
    spread_scale=1.0,
    binary_target_pseudocount=0.0,
)
GP_MODE = "expected_kernel"
MIN_PROB = float(GP_HYPERPARAMS["min_prob"])


def gp_args(**overrides) -> SimpleNamespace:
    """Frozen hyper-parameter bundle in the shape ``fit_belief_aware_gp`` wants."""
    values = dict(GP_HYPERPARAMS)
    unknown = set(overrides) - set(values)
    if unknown:
        raise KeyError(f"unknown GP hyper-parameter(s): {sorted(unknown)}")
    values.update(overrides)
    return SimpleNamespace(**values)


# ------------------------------------------------------------- data loading
@dataclass(frozen=True)
class JointEvents:
    """One row per pose-synchronized four-camera observation event."""

    X: np.ndarray  # (n, 2) belief mean m_x, m_y
    cov: np.ndarray  # (n, 2, 2) belief covariance
    stamp: np.ndarray  # (n,) observation_stamp_s
    hits: dict[str, np.ndarray]  # camera -> (n,) det_hit in {0, 1}
    scores: dict[str, np.ndarray]  # camera -> (n,) yolo_score_raw
    run_id: np.ndarray  # (n,) source run identifier

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_hit(self) -> np.ndarray:
        return np.sum([self.hits[c] for c in CAMERAS], axis=0)

    @property
    def any_hit(self) -> np.ndarray:
        return (self.n_hit > 0).astype(float)


def load_joint_events() -> JointEvents:
    """Build the joint four-camera table, asserting pose+stamp synchronization.

    The four per-camera CSVs are separate files. They are only fusable if row i
    of every file is the SAME physical observation event. That is asserted here,
    not assumed: identical ``m_x``, ``m_y`` and ``observation_stamp_s`` on every
    row. A mismatch raises rather than silently pairing unrelated events.
    """
    per_cam: dict[str, list[dict[str, str]]] = {}
    for cam in CAMERAS:
        path = events_csv(cam)
        if not path.is_file():
            raise FileNotFoundError(f"missing captured event table: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            per_cam[cam] = list(csv.DictReader(handle))

    lengths = {cam: len(rows) for cam, rows in per_cam.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"event tables have different row counts: {lengths}")

    ref = per_cam[CAMERAS[0]]
    key = lambda r: (r["m_x"], r["m_y"], r["observation_stamp_s"])  # noqa: E731
    ref_keys = [key(r) for r in ref]
    for cam in CAMERAS[1:]:
        if [key(r) for r in per_cam[cam]] != ref_keys:
            raise RuntimeError(
                f"camera_{cam} events are not pose/stamp synchronized with "
                f"camera_{CAMERAS[0]}; refusing to build a joint table"
            )
    for cam in CAMERAS:
        ids = {r["camera_id"] for r in per_cam[cam]}
        if ids != {f"camera_{cam}"}:
            raise RuntimeError(f"camera_{cam} table carries camera_id values {ids}")

    def col(rows, name):
        return np.asarray([float(r[name]) for r in rows], dtype=float)

    n = len(ref)
    cov = np.zeros((n, 2, 2), dtype=float)
    cov[:, 0, 0] = col(ref, "S_xx")
    cov[:, 0, 1] = col(ref, "S_xy")
    cov[:, 1, 0] = col(ref, "S_xy")
    cov[:, 1, 1] = col(ref, "S_yy")

    hits = {}
    scores = {}
    for cam in CAMERAS:
        h = col(per_cam[cam], "det_hit")
        if not np.all(np.isin(h, (0.0, 1.0))):
            raise RuntimeError(f"camera_{cam} det_hit is not binary")
        hits[cam] = h
        scores[cam] = col(per_cam[cam], "yolo_score_raw")

    return JointEvents(
        X=np.column_stack([col(ref, "m_x"), col(ref, "m_y")]),
        cov=cov,
        stamp=col(ref, "observation_stamp_s"),
        hits=hits,
        scores=scores,
        run_id=np.asarray([r["run_id"] for r in ref], dtype=object),
    )


def event_data(joint: JointEvents, y: np.ndarray) -> "fbg.EventData":
    """Wrap the joint table as the canonical GP code's ``EventData``.

    ``y`` is whichever binary target is being fitted (a single camera's
    ``det_hit`` or the joint ``any_hit``). ``run_ids`` is unused downstream here
    because folds are supplied explicitly, but the dataclass requires it.
    """
    return fbg.EventData(
        X=np.asarray(joint.X, dtype=float),
        y=np.asarray(y, dtype=float),
        cov=np.asarray(joint.cov, dtype=float),
        run_ids=np.asarray(joint.run_id, dtype=np.str_),
        rows_used=int(joint.X.shape[0]),
        target_id="det_hit",
    )


# --------------------------------------------------------- spatial blocking
#: Strict scheme, byte-identical to the block IDs already used by the campaign's
#: own validation run (logs/.../spawn_grid_20260727/events_blocked/). Three
#: columns x two rows: a held-out fold is one contiguous ~7 m x ~8.5 m slab, so
#: interior test points sit several length-scales from any training point. This
#: is an EXTRAPOLATION test.
STRICT_X_EDGES = (-3.75, 3.75)
STRICT_Y_EDGES = (-0.25,)

#: Tile scheme: 2 m square tiles round-robined into 5 folds. No test point ever
#: shares a tile with a training point, so per-point leakage is still removed,
#: but neighbouring tiles are available for support. This is an INTERPOLATION
#: test and is the milder, more standard spatial block CV.
TILE_M = 2.0
TILE_FOLDS = 5


def strict_blocks(X: np.ndarray) -> np.ndarray:
    bx = np.digitize(X[:, 0], list(STRICT_X_EDGES))
    by = np.digitize(X[:, 1], list(STRICT_Y_EDGES))
    return np.asarray([f"block_x{a}_y{b}" for a, b in zip(bx, by)], dtype=object)


def tile_ids(X: np.ndarray, tile_m: float = TILE_M) -> np.ndarray:
    ix = np.floor(X[:, 0] / float(tile_m)).astype(int)
    iy = np.floor(X[:, 1] / float(tile_m)).astype(int)
    return np.asarray([f"tile_{a}_{b}" for a, b in zip(ix, iy)], dtype=object)


def tile_blocks(X: np.ndarray, folds: int = TILE_FOLDS, tile_m: float = TILE_M) -> np.ndarray:
    """Assign whole tiles to folds, deterministically, in raster order."""
    tiles = tile_ids(X, tile_m=tile_m)
    order = {t: i for i, t in enumerate(sorted(set(tiles)))}
    return np.asarray([f"tilefold_{order[t] % int(folds)}" for t in tiles], dtype=object)


SCHEMES = {
    "strict_block": strict_blocks,
    "tile_cv": tile_blocks,
}


# ---------------------------------------------------------------- GP driver
def load_prior(cam: str, args: SimpleNamespace) -> "fbg.PriorMap":
    return fbg._load_prior_map(prior_npz(cam), map_key="P_mean_map", min_prob=float(args.min_prob))


def noisy_or_prior(priors: dict[str, "fbg.PriorMap"], args: SimpleNamespace) -> "fbg.PriorMap":
    """Noisy-OR of the four day-zero geometric priors, as a prior mean function.

    Used as the joint-``any_hit`` GP's mean function so that the direct joint GP
    starts from exactly the same free (pre-driving) knowledge the noisy-OR
    fusion baseline starts from. Anything else would be an unfair comparison.
    """
    ref = priors[CAMERAS[0]]
    for cam in CAMERAS[1:]:
        p = priors[cam]
        if not (np.array_equal(p.xs, ref.xs) and np.array_equal(p.ys, ref.ys)):
            raise RuntimeError("day-zero priors are not on a common grid")
    mean = 1.0 - np.prod([1.0 - priors[c].p_mean for c in CAMERAS], axis=0)
    plan = 1.0 - np.prod([1.0 - priors[c].p_plan for c in CAMERAS], axis=0)
    eps = float(args.min_prob)
    digest = hashlib.sha256(
        b"noisy_or:" + b"|".join(priors[c].sha256.encode() for c in CAMERAS)
    ).hexdigest()
    return fbg.PriorMap(
        path=PRIOR_DIR / "DERIVED_noisy_or_of_four_dayzero_priors.npz",
        xs=ref.xs,
        ys=ref.ys,
        p_mean=np.clip(mean, eps, 1.0 - eps),
        p_plan=np.clip(plan, eps, 1.0 - eps),
        map_key="P_mean_map",
        sha256=digest,
    )


def fit_predict(
    data: "fbg.EventData",
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    prior: "fbg.PriorMap",
    args: SimpleNamespace,
    mode: str = GP_MODE,
) -> np.ndarray:
    """Fit the canonical GP on ``train_mask`` and predict at ``test_mask``.

    Pure delegation: aggregation, Beta smoothing, the expected-kernel solve and
    the prior-residual composition are all ``fit_belief_aware_gp`` code paths.
    """
    train = fbg._subset_events(data, np.asarray(train_mask, dtype=bool))
    test = fbg._subset_events(data, np.asarray(test_mask, dtype=bool))
    if train.rows_used == 0 or test.rows_used == 0:
        raise RuntimeError("fit_predict needs non-empty train and test sets")
    agg = fbg._aggregate_events(
        train,
        resolution_m=float(args.aggregate_resolution_m),
        max_bin_weight=float(args.max_bin_weight),
    )
    agg = fbg._smooth_binary_aggregate(
        agg, total_pseudocount=float(args.binary_target_pseudocount)
    )
    if agg.X.shape[0] < 4:
        raise RuntimeError(f"need >= 4 aggregate train points, got {agg.X.shape[0]}")
    return fbg._predict_mode_at_events(agg, test, mode=mode, prior=prior, args=args)


def prior_prob(prior: "fbg.PriorMap", X: np.ndarray, args: SimpleNamespace) -> np.ndarray:
    return M.clip_prob(fbg._prior_prob(prior, X), eps=float(args.min_prob))


def out_of_fold(
    data: "fbg.EventData",
    folds: np.ndarray,
    *,
    prior: "fbg.PriorMap",
    args: SimpleNamespace,
    mode: str = GP_MODE,
) -> np.ndarray:
    """Spatially-disjoint out-of-fold predictions for every event."""
    folds = np.asarray(folds, dtype=object)
    out = np.full(data.X.shape[0], np.nan, dtype=float)
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        out[test] = fit_predict(
            data, ~test, test, prior=prior, args=args, mode=mode
        )
    if not np.all(np.isfinite(out)):
        raise RuntimeError("out-of-fold prediction left gaps")
    return out


# ------------------------------------------------------------ recalibration
def fit_platt(p: np.ndarray, y: np.ndarray):
    """Platt scaling in logit space: p_cal = sigmoid(a * logit(p) + b)."""
    from sklearn.linear_model import LogisticRegression

    z = M.logit(M.clip_prob(p, MIN_PROB)).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    if len(set(y.tolist())) < 2:
        raise RuntimeError("Platt calibration needs both classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, (y >= 0.5))
    a = float(model.coef_[0, 0])
    b = float(model.intercept_[0])

    def apply(q: np.ndarray) -> np.ndarray:
        zq = M.logit(M.clip_prob(q, MIN_PROB))
        return M.clip_prob(M.sigmoid(a * zq + b), MIN_PROB)

    apply.params = (a, b)  # type: ignore[attr-defined]
    return apply


def fit_isotonic(p: np.ndarray, y: np.ndarray):
    """Isotonic regression calibrator (monotone, non-parametric)."""
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    )

    def apply(q: np.ndarray) -> np.ndarray:
        return M.clip_prob(model.predict(np.asarray(q, dtype=float)), MIN_PROB)

    return apply


# ------------------------------------------------------------------ scoring
SCORE_KEYS = ("brier", "logloss", "auroc", "auprc", "ece")


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Every number here comes from scripts/shared/metrics.py. No exceptions."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "brier": M.brier(y, p),
        "logloss": M.logloss(y, p, eps=MIN_PROB),
        "auroc": M.auroc(y, p),
        "auprc": M.auprc(y, p),
        "ece": M.ece(y, p),
        "pred_mean": float(np.mean(p)),
        "obs_mean": float(np.mean(y)),
        "n": int(y.size),
    }


def block_bootstrap_ci(
    stat_fn,
    tiles: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20260730,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI from resampling whole spatial tiles with replacement.

    Rows are NOT independent (two yaws per pose, and neighbouring poses share
    spatial structure), so an i.i.d. bootstrap would be far too tight. Tiles are
    the resampling unit.
    """
    tiles = np.asarray(tiles, dtype=object)
    uniq = sorted(set(tiles.tolist()))
    index = {t: np.flatnonzero(tiles == t) for t in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(n_boot)):
        picked = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([index[uniq[i]] for i in picked])
        value = stat_fn(idx)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 10):
    """Equal-count (quantile) reliability bins: mean predicted vs observed rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, int(bins))
    xs, ys, ns = [], [], []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        xs.append(float(np.mean(p[chunk])))
        ys.append(float(np.mean(y[chunk])))
        ns.append(int(chunk.size))
    return np.asarray(xs), np.asarray(ys), np.asarray(ns)


# ----------------------------------------------------------------- plotting
#: Fixed camera identity colors — IDENTICAL in every figure of this study.
#: Okabe-Ito subset; validated colorblind-safe (dataviz validator, light mode:
#: lightness band PASS, chroma PASS, CVD separation worst dE 11.0 PASS,
#: normal-vision floor 24.2 PASS; camera_B carries a contrast WARN which is
#: relieved by direct labels + the RESULTS.md tables).
CAM_COLOR = {
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
    "D": "#D55E00",
}

#: Model/variant colors (a different entity type than cameras; never mixed with
#: camera series in the same panel). Validated: ALL CHECKS PASS in this order.
class _ModelColors(dict):
    """Colour lookup that resolves suffixed variants to their family colour.

    Variant names carry a parameter suffix (``expected_kernel_pseudocount_0.5``)
    while the palette is keyed by family. Longest-prefix match keeps the two in
    step, so adding a PSEUDOCOUNTS value never needs a palette edit.
    """

    def __missing__(self, key: str) -> str:
        candidates = [k for k in self if key.startswith(k)]
        if candidates:
            return self[max(candidates, key=len)]
        return BASELINE_GRAY


MODEL_COLOR = _ModelColors({
    "noisy_or": "#3B5BC0",
    "best_single": "#C0392B",
    "joint_gp": "#8B3FA8",
    "noisy_or_recal": "#8A6D00",
    "prior_only": "#8A6D00",
    "expected_kernel": "#3B5BC0",
    "expected_kernel_platt": "#C0392B",
    "expected_kernel_isotonic": "#8B3FA8",
    "expected_kernel_pseudocount": "#009E73",
})
BASELINE_GRAY = "#8C8B85"

INK = "#111111"
INK2 = "#4A4A46"
GRIDC = "#BFBFBF"
SEQ_CMAP = "Blues"       # single hue, sequential — never jet/rainbow/turbo
DIV_CMAP = "RdBu_r"      # diverging, always used with symmetric limits


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9A9A94",
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9,
            "grid.color": GRIDC,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )


def recessive_grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRIDC, alpha=0.3, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig, out_dir: Path, stem: str) -> list[Path]:
    """Every figure ships as both .png (dpi 150) and .pdf, tight bbox."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, kwargs in (("png", {"dpi": 150}), ("pdf", {})):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def sym_limits(values) -> float:
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    return float(np.max(v)) if v.size else 1.0


# ------------------------------------------------------------------- output
def write_csv(path: Path, fieldnames, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
