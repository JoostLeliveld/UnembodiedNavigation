#!/usr/bin/env python3
"""Shared helpers for the Option-A commissioning experiment suite.

Option A: realistic commissioning of external-camera trust maps under
uncertain robot poses. Every experiment script in this folder imports from
here so that data loading, GP math, metrics, and plotting style stay
identical across experiments.

Data-hygiene rules inherited from the repo:
  * belief = planner_belief_x/y (+ planner_cov_*), never state_x/y
  * ground truth (gt_x/gt_y, eval_* columns, oracle teleport labels, CAD
    prisms) is EVALUATION-ONLY: it may score a map, never train one.
  * events.csv comes from scripts/visibility_comparison/build_belief_gp_events.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VIS_DIR = REPO_ROOT / "scripts" / "visibility_comparison"
GEO_DIR = REPO_ROOT / "scripts" / "geometry_visibility"
LOGS_VC = REPO_ROOT / "logs" / "visibility_comparison"
OUT_ROOT = REPO_ROOT / "logs" / "studies" / "optionA_commissioning"

LOCKED_GP = REPO_ROOT / "paper_artifacts" / "gp" / "warehouse_visibility_gp_v1" / "yolo_score_raw_gp.npz"
CALIBRATED_PRIOR = REPO_ROOT / "logs" / "studies" / "geometry_visibility_prior" / "calibrated_prior_v1" / "calibrated_prior.npz"
EVENTS_HONEST = LOGS_VC / "belief_gp_events" / "events.csv"
EVENTS_WHITENOISE = LOGS_VC / "optionA_whitenoise_events" / "events.csv"
GP_V7B = REPO_ROOT / "paper_artifacts" / "gp" / "archive" / "aws_gp_v7b_superseded" / "yolo_score_raw_gp.npz"
TELEPORT_TARGETS = LOGS_VC / "warehouse_visibility_targets_v1" / "gp_targets.csv"
TELEPORT_TARGETS_XY = LOGS_VC / "warehouse_visibility_targets_v1" / "gp_targets_xy_aggregated.csv"

SHARED_DIR = REPO_ROOT / "scripts" / "shared"
for _p in (str(VIS_DIR), str(GEO_DIR), str(SHARED_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# reuse the existing, validated GP machinery (do not reimplement)
import fit_belief_aware_gp as fbg  # noqa: E402
# canonical scoring functions live in scripts/shared/metrics.py
from metrics import (  # noqa: E402,F401
    brier, logloss, auroc, ece, fhtr, probit_prob,
    gaussian_nll_logit, coverage_logit, binned,
)

CAMERA_POS = np.array([0.0, -5.5, 4.8])

# ---------------------------------------------------------------- plotting style (mini_relifusion showcase)
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
BLUE, AQUA, YELLOW, GREEN, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"
RED, ORANGE = "#e34948", "#eb6834"
METHOD_COLORS = {
    "naive": BLUE,
    "point": BLUE,
    "tuned_point": VIOLET,
    "fixed_blur": MUTED,
    "uncertainty_weighted": YELLOW,
    "belief_spread": ORANGE,
    "expected_kernel": AQUA,
    "belief_aware": AQUA,
    "prior_only": BASE,
    "constant_train_mean": "#b3b2ac",
}
METHOD_LABELS = {
    "naive": "point GP",
    "point": "point GP",
    "tuned_point": "tuned point GP",
    "fixed_blur": "fixed blur",
    "uncertainty_weighted": "uncertainty weighting",
    "belief_spread": "belief spread",
    "expected_kernel": "belief-aware GP",
    "belief_aware": "belief-aware GP",
    "prior_only": "prior only",
    "constant_train_mean": "constant",
}
CMAP_TRUST = LinearSegmentedColormap.from_list("trust", [RED, "#f0efec", BLUE])
CMAP_INK = LinearSegmentedColormap.from_list("ink", [SURFACE, "#c9c8c2", "#52514e", INK])
CMAP_STD = LinearSegmentedColormap.from_list("std", [SURFACE, "#f6d8a8", ORANGE, "#8c2f10"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 9.5,
    "legend.frameon": False, "figure.dpi": 115,
})


def style_ax(ax, title=None, keep_ticks=True):
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, fontsize=8.8, color=INK2, pad=3)
    if not keep_ticks:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(labelsize=7.5)


def draw_warehouse(ax, alpha=1.0, camera=True):
    """Shelf-prism outlines + camera glyph (context only; CAD is eval-only)."""
    raw = np.load(LOCKED_GP)["geometry_json"]
    gj = json.loads(str(raw[0]) if raw.shape else str(raw))
    for pr in gj["prisms"]:
        ax.add_patch(Rectangle((pr["xmin"], pr["ymin"]), pr["xmax"] - pr["xmin"], pr["ymax"] - pr["ymin"],
                               fill=False, ec=INK2, lw=0.7, alpha=0.75 * alpha, zorder=6))
    if camera:
        ax.plot(CAMERA_POS[0], CAMERA_POS[1], marker="v", ms=8, color=INK, zorder=7)
        ax.annotate("camera", (CAMERA_POS[0], CAMERA_POS[1]), textcoords="offset points",
                    xytext=(6, 2), fontsize=7, color=INK2)


def badge(ax, text, loc="upper left"):
    xy = {"upper left": (0.02, 0.97), "upper right": (0.98, 0.97),
          "lower left": (0.02, 0.03), "lower right": (0.98, 0.03)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.2, color=INK,
            bbox=dict(fc=SURFACE, ec=GRID, boxstyle="round,pad=0.28"), zorder=10)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {path.relative_to(REPO_ROOT)}")
    return path


# ---------------------------------------------------------------- data loading
def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(row, key, default=math.nan) -> float:
    raw = row.get(key, "")
    if raw in (None, "", "nan", "NaN"):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def load_events(path: Path = EVENTS_HONEST) -> dict[str, np.ndarray]:
    """events.csv -> dict of arrays. Training inputs: m, S. Eval-only: eval_*."""
    rows = read_rows(path)
    def col(k):
        return np.array([fnum(r, k) for r in rows])
    ev = {
        "m": np.column_stack([col("m_x"), col("m_y")]),
        "S": np.stack([np.array([[a, b], [b, c]]) for a, b, c in
                       zip(col("S_xx"), np.nan_to_num(col("S_xy")), col("S_yy"))]),
        "sigma_major": col("sigma_major_m"),
        "trace_S": col("trace_S_xy"),
        "det_hit": col("det_hit"),
        "score": col("yolo_score_raw"),
        "route": np.array([r["route"] for r in rows]),
        "condition": np.array([r["condition"] for r in rows]),
        "seed": np.array([r["seed"] for r in rows]),
        "run": np.array([r["run_dir"] for r in rows]),
        "eval_gt": np.column_stack([col("eval_gt_x"), col("eval_gt_y")]),
        "eval_belief_err": col("eval_belief_error_gt_m"),
    }
    # score target: fill undetected frames with score 0 (same rule as fitter)
    s = ev["score"].copy()
    s[~np.isfinite(s) & (ev["det_hit"] <= 0)] = 0.0
    ev["score"] = np.clip(s, 0.0, 1.0)
    ok = np.isfinite(ev["m"]).all(axis=1) & np.isfinite(ev["det_hit"]) & np.isfinite(ev["score"])
    return {k: v[ok] for k, v in ev.items()}


ROUTES = ("control_west_to_a1_low", "route_apron_to_a2_mid",
          "route_apron_to_a3_mid", "route_west_to_a1_upper")
ROUTE_SHORT = {"control_west_to_a1_low": "west→A1 low", "route_apron_to_a2_mid": "apron→A2",
               "route_apron_to_a3_mid": "apron→A3", "route_west_to_a1_upper": "west→A1 up"}


# ---------------------------------------------------------------- GP wrappers (reuse fbg internals)
def make_event_data(m, y, S, run_ids, target_id="det_hit"):
    return fbg.EventData(X=np.asarray(m, float), y=np.asarray(y, float),
                         cov=np.asarray(S, float), run_ids=np.asarray(run_ids, dtype=np.str_),
                         rows_used=len(y), target_id=target_id)


def fit_predict(mode, agg, query_X, *, query_cov=None, length_scale=0.9, noise_var=0.05,
                pose_length_scale=0.35, min_certainty=0.05, spread_scale=1.0,
                prior_logit_fn=None):
    """Fit one mode on aggregated events, return (mu_logit, sigma_logit) at query points.

    prior_logit_fn: callable X->logit prior mean (residual fitting, same seam as fbg).
    Modes: naive | uncertainty_weighted | belief_spread | expected_kernel
           plus analysis-only baselines: tuned_point | fixed_blur
    """
    if prior_logit_fn is None:
        prior_logit_fn = lambda X: np.zeros(np.asarray(X).shape[0])
    query_X = np.asarray(query_X, float)

    if mode == "tuned_point":
        # point GP whose single global length scale is inflated by the mean pose var
        mean_tr = float(np.mean(np.trace(agg.cov, axis1=1, axis2=2)))
        ls = math.sqrt(length_scale ** 2 + mean_tr)
        return fit_predict("naive", agg, query_X, length_scale=ls, noise_var=noise_var,
                           prior_logit_fn=prior_logit_fn)
    if mode == "fixed_blur":
        # uncertain-input GP with one shared isotropic covariance = mean of all covs
        mean_tr = float(np.mean(np.trace(agg.cov, axis1=1, axis2=2)))
        shared = np.repeat((0.5 * mean_tr * np.eye(2))[None], agg.X.shape[0], axis=0)
        agg = fbg.AggregateData(agg.X, agg.y, shared, agg.count)
        mode = "expected_kernel"

    X_fit, y_fit, alpha, _ = fbg._fit_inputs(
        agg, mode=mode, noise_var=noise_var, gp_length_scale=length_scale,
        pose_length_scale=pose_length_scale, min_certainty=min_certainty,
        spread_scale=spread_scale)
    if mode == "expected_kernel":
        qc = np.zeros((query_X.shape[0], 2, 2)) if query_cov is None else np.asarray(query_cov, float)
        latent = fbg._logit(agg.y) - prior_logit_fn(agg.X)
        mu_r, sig, _j = fbg._fit_predict_expected_kernel_latent(
            agg.X, latent, agg.cov, alpha, query_X, query_cov=qc, length_scale=length_scale)
    else:
        latent = fbg._logit(y_fit) - prior_logit_fn(X_fit)
        mu_r, sig = fbg._fit_predict_latent_gp(X_fit, latent, alpha, query_X, length_scale=length_scale)
    return prior_logit_fn(query_X) + mu_r, sig


def aggregate(data, resolution_m=0.20, max_bin_weight=20.0):
    return fbg._aggregate_events(data, resolution_m=resolution_m, max_bin_weight=max_bin_weight)


def sigmoid(x):
    return fbg._sigmoid(x)


def logit(p):
    return fbg._logit(p)


# metrics (brier/logloss/auroc/ece/fhtr/probit_prob/gaussian_nll_logit/coverage_logit/binned)
# are imported from scripts/shared/metrics.py at the top of this module.


# ---------------------------------------------------------------- priors as logit functions
def prior_logit_from_artifact(path: Path, map_key="P_mean_map", inflate_to=None):
    """Return callable X -> prior logit mean from a planner-schema npz artifact."""
    with np.load(path, allow_pickle=False) as d:
        xs, ys = np.asarray(d["xs"], float), np.asarray(d["ys"], float)
        grid = np.asarray(d[map_key], float)
    grid = np.clip(grid, 1e-4, 1 - 1e-4)
    lg = logit(grid)

    def fn(X):
        return fbg._interp_grid(xs, ys, lg, np.asarray(X, float))
    fn.xs, fn.ys, fn.grid_logit = xs, ys, lg
    return fn


def prior_logit_calibration():
    """Calibration-only prior (geometry: clearance + px/m via logistic link)."""
    with np.load(CALIBRATED_PRIOR, allow_pickle=False) as d:
        xs, ys = np.asarray(d["xs"], float), np.asarray(d["ys"], float)
        lg = np.asarray(d["prior_logit_mean_map"], float)

    def fn(X):
        return fbg._interp_grid(xs, ys, lg, np.asarray(X, float))
    fn.xs, fn.ys, fn.grid_logit = xs, ys, lg
    return fn


def grid_query(xs=None, ys=None, nx=110, ny=100):
    if xs is None:
        xs = np.linspace(-5.5, 5.5, nx)
    if ys is None:
        ys = np.linspace(-5.0, 5.0, ny)
    Xg, Yg = np.meshgrid(xs, ys)
    return xs, ys, np.column_stack([Xg.ravel(), Yg.ravel()])


def write_md(out_dir: Path, name: str, text: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_text(text, encoding="utf-8")
    print(f"  doc: {p.relative_to(REPO_ROOT)}")
    return p
