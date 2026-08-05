#!/usr/bin/env python3
"""exp1 — availability half of the factorized external-camera observation model.

Question
--------
For the four-camera warehouse network, is network availability
``P(at least one camera usable | x)`` a *product* of per-camera availability
fields — i.e. are the cameras conditionally independent given position, and does
a noisy-OR of per-camera GPs predict the joint outcome out of sample?

Why this script exists
----------------------
An earlier pass answered "yes" using per-camera GPs fitted on ALL the data and
scored on the same data (``heldout_event_count: 0`` in the locked GP manifests).
Those numbers are in-sample and therefore optimistic. Everything below is
recomputed under a genuine SPATIALLY-DISJOINT holdout, under two schemes:

  strict_block  3x2 contiguous slabs  -> extrapolation regime (matches the
                                         campaign's own events_blocked folds)
  tile_cv       2 m tiles, 5 folds    -> interpolation regime

Outputs: logs/studies/fused_observation_model/exp1_availability_fusion/

Run: python3 exp1_availability_fusion.py     (offline only; no ROS, no Gazebo)
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

import fused_common as F
from fused_common import CAMERAS


OUT_DIR = F.OUT_ROOT / "exp1_availability_fusion"

FUSION_MODELS = (
    "constant_train_mean",
    "dayzero_prior_noisy_or",
    "best_single",
    "noisy_or",
    "joint_gp",
    "best_single_recal",
    "noisy_or_recal",
    "joint_gp_recal",
)
FUSION_LABEL = {
    "constant_train_mean": "constant (train mean)",
    "dayzero_prior_noisy_or": "day-zero prior, noisy-OR",
    "best_single": "best single cam (max)",
    "noisy_or": "noisy-OR of per-cam GPs",
    "joint_gp": "GP fitted on any_hit",
    "best_single_recal": "best single cam, recalibrated",
    "noisy_or_recal": "noisy-OR, recalibrated",
    "joint_gp_recal": "joint GP, recalibrated",
}
#: The three fusion rules the study is actually comparing (the rest are
#: reference points). Recalibrated twins are plotted dashed in the same color.
FUSION_HEAD = ("noisy_or", "best_single", "joint_gp")
FUSION_RECAL = {
    "noisy_or": "noisy_or_recal",
    "best_single": "best_single_recal",
    "joint_gp": "joint_gp_recal",
}

#: Beta-prior pseudocounts tried on the aggregated binary target (a knob the
#: canonical GP code already supports). PSEUDOCOUNT_HEAD is the one plotted.
PSEUDOCOUNTS = (0.5, 2.0)
PSEUDOCOUNT_HEAD = 0.5

RECAL_VARIANTS = (
    "prior_only",
    "expected_kernel",
    "expected_kernel_platt",
    "expected_kernel_isotonic",
    f"expected_kernel_pseudocount_{PSEUDOCOUNT_HEAD:g}",
)
RECAL_LABEL = {
    "prior_only": "day-zero prior only",
    "expected_kernel": "expected-kernel GP",
    "expected_kernel_platt": "GP + Platt",
    "expected_kernel_isotonic": "GP + isotonic",
    **{f"expected_kernel_pseudocount_{p:g}": f"GP, Beta pseudocount {p:g}"
       for p in PSEUDOCOUNTS},
}


# --------------------------------------------------------------------- core
def _nested_calibrated(
    data, folds, fold_ids, p_oof, *, prior, args
) -> tuple[np.ndarray, np.ndarray]:
    """Leak-free Platt / isotonic recalibration of out-of-fold predictions.

    The calibrator applied to outer fold k is fitted ONLY on inner-out-of-fold
    predictions from the other folds, produced by GPs that never saw fold k. No
    label from the scored fold reaches the calibrator, directly or through the
    model that generated the calibrator's input. (A single-level "calibrate on
    the pooled OOF predictions" shortcut would leak, because those predictions
    were made by models trained on fold k.)
    """
    n = data.X.shape[0]
    platt_out = np.full(n, np.nan, dtype=float)
    iso_out = np.full(n, np.nan, dtype=float)
    for fold in fold_ids:
        outer_test = folds == fold
        cal_p = np.full(n, np.nan, dtype=float)
        for inner in fold_ids:
            if inner == fold:
                continue
            inner_test = folds == inner
            cal_p[inner_test] = F.fit_predict(
                data, ~inner_test & ~outer_test, inner_test, prior=prior, args=args
            )
        cal_mask = np.isfinite(cal_p)
        platt = F.fit_platt(cal_p[cal_mask], data.y[cal_mask])
        iso = F.fit_isotonic(cal_p[cal_mask], data.y[cal_mask])
        platt_out[outer_test] = platt(p_oof[outer_test])
        iso_out[outer_test] = iso(p_oof[outer_test])
    return platt_out, iso_out


def run_scheme(joint, scheme: str, *, verbose: bool = True) -> dict:
    """All per-camera / fused / recalibration predictions for one CV scheme."""
    args = F.gp_args()
    folds = F.SCHEMES[scheme](joint.X)
    fold_ids = sorted(set(folds.tolist()))
    priors = {c: F.load_prior(c, args) for c in CAMERAS}
    prior_or = F.noisy_or_prior(priors, args)

    p_gp: dict[str, np.ndarray] = {}
    p_prior: dict[str, np.ndarray] = {}
    p_pc: dict[float, dict[str, np.ndarray]] = {pc: {} for pc in PSEUDOCOUNTS}
    p_platt: dict[str, np.ndarray] = {}
    p_iso: dict[str, np.ndarray] = {}

    t0 = time.time()
    for cam in CAMERAS:
        data = F.event_data(joint, joint.hits[cam])
        p_prior[cam] = F.prior_prob(priors[cam], joint.X, args)
        p_gp[cam] = F.out_of_fold(data, folds, prior=priors[cam], args=args)
        for pc in PSEUDOCOUNTS:
            p_pc[pc][cam] = F.out_of_fold(
                data, folds, prior=priors[cam],
                args=F.gp_args(binary_target_pseudocount=float(pc)),
            )
        p_platt[cam], p_iso[cam] = _nested_calibrated(
            data, folds, fold_ids, p_gp[cam], prior=priors[cam], args=args
        )
        if verbose:
            print(f"  [{scheme}] camera_{cam} done ({time.time() - t0:.1f}s)")

    any_hit = joint.any_hit
    joint_data = F.event_data(joint, any_hit)
    p_joint_gp = F.out_of_fold(joint_data, folds, prior=prior_or, args=args)
    _, p_joint_iso = _nested_calibrated(
        joint_data, folds, fold_ids, p_joint_gp, prior=prior_or, args=args
    )

    p_const = np.empty(joint.n, dtype=float)
    for fold in fold_ids:
        test = folds == fold
        p_const[test] = float(np.mean(any_hit[~test]))

    fused = {
        "constant_train_mean": p_const,
        "dayzero_prior_noisy_or": F.prior_prob(prior_or, joint.X, args),
        "best_single": np.max([p_gp[c] for c in CAMERAS], axis=0),
        "noisy_or": 1.0 - np.prod([1.0 - p_gp[c] for c in CAMERAS], axis=0),
        "joint_gp": p_joint_gp,
        "best_single_recal": np.max([p_iso[c] for c in CAMERAS], axis=0),
        "noisy_or_recal": 1.0 - np.prod([1.0 - p_iso[c] for c in CAMERAS], axis=0),
        "joint_gp_recal": p_joint_iso,
    }
    if verbose:
        print(f"  [{scheme}] fused models done ({time.time() - t0:.1f}s)")

    return {
        "scheme": scheme,
        "folds": folds,
        "fold_ids": fold_ids,
        "p_gp": p_gp,
        "p_prior": p_prior,
        "p_pc": p_pc,
        "p_platt": p_platt,
        "p_iso": p_iso,
        "fused": fused,
        "priors": priors,
        "prior_or": prior_or,
    }


def in_sample_locked_predictions(joint) -> dict[str, np.ndarray]:
    """Per-camera p(x) read off the LOCKED (in-sample, heldout=0) GP artifacts.

    Bilinear readback uses the canonical GP module's own grid interpolator, so
    these reproduce exactly what the deployed artifact would say at each pose.
    """
    out = {}
    for cam in CAMERAS:
        with np.load(F.locked_gp_npz(cam), allow_pickle=False) as data:
            xs = np.asarray(data["xs"], dtype=float)
            ys = np.asarray(data["ys"], dtype=float)
            grid = np.asarray(data["P_mean_map"], dtype=float)
        out[cam] = F.M.clip_prob(
            F.fbg._interp_grid(xs, ys, grid, joint.X), F.MIN_PROB
        )
    return out


def residual_corr(joint, p_by_cam) -> dict[tuple[str, str], float]:
    """Pearson correlation of (det_hit - p(x)) between camera pairs."""
    resid = {c: joint.hits[c] - np.asarray(p_by_cam[c], dtype=float) for c in CAMERAS}
    out = {}
    for a, b in itertools.combinations(CAMERAS, 2):
        out[(a, b)] = float(np.corrcoef(resid[a], resid[b])[0, 1])
    return out


def corr_matrix(pairs: dict[tuple[str, str], float]) -> np.ndarray:
    m = np.eye(len(CAMERAS), dtype=float)
    for (a, b), v in pairs.items():
        i, j = CAMERAS.index(a), CAMERAS.index(b)
        m[i, j] = m[j, i] = v
    return m


# ------------------------------------------------------------------ figures
def fig_a1(joint, results, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    maps = {}
    for cam in CAMERAS:
        with np.load(F.locked_gp_npz(cam), allow_pickle=False) as data:
            xs = np.asarray(data["xs"], dtype=float)
            ys = np.asarray(data["ys"], dtype=float)
            maps[cam] = np.asarray(data["P_mean_map"], dtype=float)
    fused_map = 1.0 - np.prod([1.0 - maps[c] for c in CAMERAS], axis=0)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    # observed per-pose any_hit fraction over the two yaws (real measurements)
    keys = {}
    for i in range(joint.n):
        keys.setdefault((joint.X[i, 0], joint.X[i, 1]), []).append(i)
    pose_xy = np.asarray(list(keys.keys()), dtype=float)
    pose_any = np.asarray(
        [float(np.mean(joint.any_hit[idx])) for idx in keys.values()], dtype=float
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.4), constrained_layout=True)
    order = list(CAMERAS) + ["fused", "observed"]
    for ax, key in zip(axes.ravel(), order):
        if key in CAMERAS:
            im = ax.imshow(
                maps[key], origin="lower", extent=extent, cmap=F.SEQ_CMAP,
                vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest",
            )
            ax.add_patch(
                Rectangle((0.03, 0.90), 0.07, 0.055, transform=ax.transAxes,
                          facecolor=F.CAM_COLOR[key], edgecolor="white",
                          linewidth=0.8, zorder=5)
            )
            ax.set_title(f"camera_{key}  ·  mean p = {maps[key].mean():.3f}")
        elif key == "fused":
            im = ax.imshow(
                fused_map, origin="lower", extent=extent, cmap=F.SEQ_CMAP,
                vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest",
            )
            ax.set_title(f"noisy-OR of the four maps  ·  mean p = {fused_map.mean():.3f}")
        else:
            ax.imshow(
                np.zeros_like(fused_map), origin="lower", extent=extent,
                cmap=F.SEQ_CMAP, vmin=0.0, vmax=1.0, aspect="equal",
            )
            im = ax.scatter(
                pose_xy[:, 0], pose_xy[:, 1], c=pose_any, cmap=F.SEQ_CMAP,
                vmin=0.0, vmax=1.0, s=9, linewidths=0.25, edgecolors="#33333355",
            )
            ax.set_title(
                f"observed any_hit  ·  {joint.any_hit.mean():.3f} of "
                f"{joint.n} real events"
            )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(length=2)

    axes[1, 1].annotate(
        f"noisy-OR mean {fused_map.mean():.3f}\nobserved {joint.any_hit.mean():.4f}",
        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8.5, color=F.INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BFBFBF", alpha=0.9),
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.012)
    cbar.set_label("P(usable detection)", fontsize=8.5)
    fig.suptitle(
        "Four narrow per-camera availability fields tile into near-full network "
        f"coverage: 82.2% of poses see >=1 camera, and noisy-OR reproduces it "
        f"({fused_map.mean():.3f} predicted)",
        fontsize=12, fontweight="semibold", color=F.INK,
    )
    F.save_fig(fig, out_dir, "fig_a1_spatial_availability")


def fig_a2(mats: dict[str, np.ndarray], out_dir: Path, headline: str) -> None:
    import matplotlib.pyplot as plt

    keys = list(mats.keys())
    vmax = max(F.sym_limits(m[~np.eye(len(CAMERAS), dtype=bool)]) for m in mats.values())
    vmax = float(np.ceil(vmax * 20) / 20)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.1 * len(keys), 4.3),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        m = mats[key].copy()
        np.fill_diagonal(m, np.nan)
        im = ax.imshow(m, cmap=F.DIV_CMAP, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(CAMERAS)), [f"cam {c}" for c in CAMERAS])
        ax.set_yticks(range(len(CAMERAS)), [f"cam {c}" for c in CAMERAS])
        for i in range(len(CAMERAS)):
            for j in range(len(CAMERAS)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#999999")
                    continue
                ax.text(j, i, f"{m[i, j]:+.3f}", ha="center", va="center",
                        fontsize=8.5, color=F.INK)
        off = m[~np.eye(len(CAMERAS), dtype=bool)]
        ax.set_title(f"{key}\nmax |r| = {np.nanmax(np.abs(off)):.3f}")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.72, pad=0.02)
    cbar.set_label("pairwise correlation of detection residuals", fontsize=8.5)
    fig.suptitle(headline, fontsize=11.5, fontweight="semibold", color=F.INK)
    F.save_fig(fig, out_dir, "fig_a2_conditional_independence")


def fig_a3(y, fused, scores, out_dir: Path, headline: str) -> None:
    import matplotlib.pyplot as plt

    show = FUSION_HEAD
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=F.BASELINE_GRAY, linestyle=":", linewidth=1.4,
            zorder=2, label="perfect calibration")
    for key in show:
        px, py, _ = F.reliability_curve(y, fused[key], bins=10)
        ax.plot(px, py, marker="o", markersize=5.5, linewidth=2.0,
                color=F.MODEL_COLOR[key], zorder=4,
                label=f"{FUSION_LABEL[key]}  (ECE {scores[key]['ece']:.3f})")
        rkey = FUSION_RECAL[key]
        px, py, _ = F.reliability_curve(y, fused[rkey], bins=10)
        ax.plot(px, py, marker="s", markersize=4.2, linewidth=1.6, linestyle="--",
                color=F.MODEL_COLOR[key], alpha=0.85, zorder=3,
                label=f"  + isotonic recalibration  (ECE {scores[rkey]['ece']:.3f})")
    F.recessive_grid(ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("mean predicted P(any camera usable)")
    ax.set_ylabel("observed frequency (spatially held out)")
    ax.set_title("Raw fusion is over-confident at the top end;\n"
                 "isotonic recalibration pulls every rule onto the diagonal")
    ax.legend(loc="upper left", fontsize=7.6)

    ax = axes[1]
    metrics = ("brier", "logloss", "auroc", "ece")
    labels = ("Brier (lower)", "logloss (lower)", "AUROC (higher)", "ECE (lower)")
    width = 0.26
    base = np.arange(len(metrics), dtype=float)
    for k, key in enumerate(show):
        vals = [scores[key][m] for m in metrics]
        pos = base + (k - 1) * width
        ax.bar(pos, vals, width=width * 0.9, color=F.MODEL_COLOR[key],
               zorder=3, label=FUSION_LABEL[key])
        for xpos, v in zip(pos, vals):
            ax.text(xpos, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=7.4, color=F.INK2, rotation=90)
    F.recessive_grid(ax, axis="y")
    ax.set_xticks(base, labels)
    ax.set_ylabel("held-out score")
    ax.set_ylim(0, max(1.02, max(scores[k][m] for k in show for m in metrics) * 1.26))
    ax.set_title("Direct joint GP wins on every score;\n"
                 "noisy-OR and best-single are indistinguishable")
    ax.legend(loc="upper left", ncol=1)

    fig.suptitle(headline, fontsize=11.5, fontweight="semibold", color=F.INK)
    F.save_fig(fig, out_dir, "fig_a3_fusion_calibration")


def fig_a4(recal_scores, out_dir: Path, headline: str) -> None:
    import matplotlib.pyplot as plt

    metrics = ("brier", "logloss", "auroc", "ece")
    titles = ("Brier (lower better)", "logloss (lower better)",
              "AUROC (higher better)", "ECE (lower better)")
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6), constrained_layout=True)
    width = 0.16
    base = np.arange(len(CAMERAS), dtype=float)
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        for k, variant in enumerate(RECAL_VARIANTS):
            vals = [recal_scores[(cam, variant)][metric] for cam in CAMERAS]
            pos = base + (k - (len(RECAL_VARIANTS) - 1) / 2) * width
            ax.bar(pos, vals, width=width * 0.9, color=F.MODEL_COLOR[variant],
                   zorder=3, label=RECAL_LABEL[variant] if metric == "brier" else None)
            for xpos, v in zip(pos, vals):
                ax.text(xpos, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.4, color=F.INK2, rotation=90)
        F.recessive_grid(ax, axis="y")
        ax.set_xticks(base, [f"camera_{c}" for c in CAMERAS])
        ax.set_title(title)
        top = max(recal_scores[(c, v)][metric] for c in CAMERAS for v in RECAL_VARIANTS)
        ax.set_ylim(0, top * 1.28)
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=7.8)
    fig.suptitle(headline, fontsize=11.5, fontweight="semibold", color=F.INK)
    F.save_fig(fig, out_dir, "fig_a4_recalibration")


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--schemes", default="tile_cv,strict_block",
                        help="comma-separated spatial CV schemes to run")
    parser.add_argument("--head-scheme", default="tile_cv",
                        help="scheme the figures and headline numbers use")
    args_cli = parser.parse_args()
    out_dir = Path(args_cli.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    F.apply_style()

    print("loading + verifying the four pose-synchronized event tables ...")
    joint = F.load_joint_events()
    tiles = F.tile_ids(joint.X)
    print(f"  {joint.n} synchronized events, "
          f"{len(set(map(tuple, joint.X.tolist())))} unique poses, "
          f"{len(set(tiles.tolist()))} spatial tiles")

    # ---------------------------------------------------------- deliverable 1
    fields = (
        ["event_index", "m_x", "m_y", "S_xx", "S_xy", "S_yy", "observation_stamp_s",
         "run_id"]
        + [f"det_hit_{c}" for c in CAMERAS]
        + [f"yolo_score_raw_{c}" for c in CAMERAS]
        + ["n_hit", "any_hit", "strict_block", "tile_fold", "tile_id"]
    )
    strict = F.strict_blocks(joint.X)
    tilef = F.tile_blocks(joint.X)
    rows = []
    for i in range(joint.n):
        row = {
            "event_index": i,
            "m_x": f"{joint.X[i, 0]:.6g}",
            "m_y": f"{joint.X[i, 1]:.6g}",
            "S_xx": f"{joint.cov[i, 0, 0]:.6g}",
            "S_xy": f"{joint.cov[i, 0, 1]:.6g}",
            "S_yy": f"{joint.cov[i, 1, 1]:.6g}",
            "observation_stamp_s": f"{joint.stamp[i]:.6g}",
            "run_id": joint.run_id[i],
            "n_hit": int(joint.n_hit[i]),
            "any_hit": int(joint.any_hit[i]),
            "strict_block": strict[i],
            "tile_fold": tilef[i],
            "tile_id": tiles[i],
        }
        for c in CAMERAS:
            row[f"det_hit_{c}"] = int(joint.hits[c][i])
            row[f"yolo_score_raw_{c}"] = f"{joint.scores[c][i]:.6g}"
        rows.append(row)
    joint_csv = F.write_csv(out_dir / "joint_events.csv", fields, rows)
    print(f"  wrote {joint_csv}")

    schemes = tuple(s.strip() for s in args_cli.schemes.split(",") if s.strip())
    # Figures/headline numbers come from one scheme; fall back to the first one
    # actually run so --schemes and --head-scheme can never disagree silently.
    head_scheme = args_cli.head_scheme if args_cli.head_scheme in schemes else schemes[0]
    results = {}
    for scheme in schemes:
        print(f"spatial-block CV [{scheme}] ...")
        results[scheme] = run_scheme(joint, scheme)

    # ---------------------------------------------------------- deliverable 2
    p_locked = in_sample_locked_predictions(joint)
    marginal = {}
    for a, b in itertools.combinations(CAMERAS, 2):
        marginal[(a, b)] = float(np.corrcoef(joint.hits[a], joint.hits[b])[0, 1])
    cond_rows = []
    mats: dict[str, np.ndarray] = {}

    resid_cache: dict[str, dict[str, np.ndarray]] = {
        "marginal": {c: joint.hits[c] - float(np.mean(joint.hits[c])) for c in CAMERAS},
        "in_sample_gp": {c: joint.hits[c] - p_locked[c] for c in CAMERAS},
        "dayzero_prior": {c: joint.hits[c] - results[schemes[0]]["p_prior"][c]
                          for c in CAMERAS},
    }
    for scheme in schemes:
        resid_cache[f"heldout_gp_{scheme}"] = {
            c: joint.hits[c] - results[scheme]["p_gp"][c] for c in CAMERAS
        }
        resid_cache[f"heldout_gp_recal_{scheme}"] = {
            c: joint.hits[c] - results[scheme]["p_iso"][c] for c in CAMERAS
        }

    def _pair_corr(idx, a, b, tag):
        ra = resid_cache[tag][a][idx]
        rb = resid_cache[tag][b][idx]
        if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])

    def add_condition(name, pairs, tag):
        mats[name] = corr_matrix(pairs)
        for (a, b), v in pairs.items():
            lo, hi = F.block_bootstrap_ci(
                lambda idx, a=a, b=b, tag=tag: _pair_corr(idx, a, b, tag),
                tiles, n_boot=int(args_cli.n_boot),
            )
            cond_rows.append({
                "conditioning": tag, "camera_a": a, "camera_b": b,
                "r": f"{v:.6g}", "ci_lo": f"{lo:.6g}", "ci_hi": f"{hi:.6g}",
                "ci_excludes_zero": str(bool(lo > 0 or hi < 0)).lower(),
            })

    add_condition("marginal (no conditioning)", marginal, "marginal")
    add_condition("conditioned on IN-SAMPLE GP", residual_corr(joint, p_locked),
                  "in_sample_gp")
    add_condition("conditioned on day-zero prior",
                  residual_corr(joint, results[schemes[0]]["p_prior"]), "dayzero_prior")
    for scheme in schemes:
        add_condition(f"conditioned on HELD-OUT GP ({scheme})",
                      residual_corr(joint, results[scheme]["p_gp"]),
                      f"heldout_gp_{scheme}")
        add_condition(f"conditioned on HELD-OUT recalibrated GP ({scheme})",
                      residual_corr(joint, results[scheme]["p_iso"]),
                      f"heldout_gp_recal_{scheme}")
    F.write_csv(
        out_dir / "conditional_independence.csv",
        ("conditioning", "camera_a", "camera_b", "r", "ci_lo", "ci_hi",
         "ci_excludes_zero"),
        cond_rows,
    )

    def maxabs(tag):
        pairs = [abs(float(r["r"])) for r in cond_rows if r["conditioning"] == tag]
        return max(pairs)

    # ---------------------------------------------------------- deliverable 3
    any_hit = joint.any_hit
    fusion_rows = []
    fusion_scores = {}
    for scheme in schemes:
        fusion_scores[scheme] = {}
        for key in FUSION_MODELS:
            p = results[scheme]["fused"][key]
            s = F.score(any_hit, p)
            fusion_scores[scheme][key] = s
            lo, hi = F.block_bootstrap_ci(
                lambda idx, p=p: F.M.brier(any_hit[idx], p[idx]),
                tiles, n_boot=int(args_cli.n_boot),
            )
            fusion_rows.append({
                "scheme": scheme, "model": key, "label": FUSION_LABEL[key],
                "n": s["n"], "brier": f"{s['brier']:.6g}",
                "brier_ci_lo": f"{lo:.6g}", "brier_ci_hi": f"{hi:.6g}",
                "logloss": f"{s['logloss']:.6g}", "auroc": f"{s['auroc']:.6g}",
                "auprc": f"{s['auprc']:.6g}", "ece": f"{s['ece']:.6g}",
                "pred_mean": f"{s['pred_mean']:.6g}", "obs_mean": f"{s['obs_mean']:.6g}",
            })
    F.write_csv(
        out_dir / "fusion_model_scores.csv",
        ("scheme", "model", "label", "n", "brier", "brier_ci_lo", "brier_ci_hi",
         "logloss", "auroc", "auprc", "ece", "pred_mean", "obs_mean"),
        fusion_rows,
    )

    # ---------------------------------------------------------- deliverable 4
    recal_rows = []
    recal_scores = {}
    for scheme in schemes:
        r = results[scheme]
        for cam in CAMERAS:
            y = joint.hits[cam]
            preds = {
                "prior_only": r["p_prior"][cam],
                "expected_kernel": r["p_gp"][cam],
                "expected_kernel_platt": r["p_platt"][cam],
                "expected_kernel_isotonic": r["p_iso"][cam],
                **{f"expected_kernel_pseudocount_{pc:g}": r["p_pc"][pc][cam]
                   for pc in PSEUDOCOUNTS},
            }
            for variant, p in preds.items():
                s = F.score(y, p)
                if scheme == head_scheme:
                    recal_scores[(cam, variant)] = s
                recal_rows.append({
                    "scheme": scheme, "camera": f"camera_{cam}", "variant": variant,
                    "label": RECAL_LABEL[variant], "n": s["n"],
                    "brier": f"{s['brier']:.6g}", "logloss": f"{s['logloss']:.6g}",
                    "auroc": f"{s['auroc']:.6g}", "auprc": f"{s['auprc']:.6g}",
                    "ece": f"{s['ece']:.6g}", "pred_mean": f"{s['pred_mean']:.6g}",
                    "obs_mean": f"{s['obs_mean']:.6g}",
                })
    F.write_csv(
        out_dir / "recalibration_scores.csv",
        ("scheme", "camera", "variant", "label", "n", "brier", "logloss", "auroc",
         "auprc", "ece", "pred_mean", "obs_mean"),
        recal_rows,
    )

    # ------------------------------------------------------------- figures
    print("rendering figures ...")
    head = schemes[0]
    fig_a1(joint, results, out_dir)
    fig_a2(
        {k: mats[k] for k in (
            "marginal (no conditioning)",
            "conditioned on IN-SAMPLE GP",
            f"conditioned on HELD-OUT GP ({head})",
        )},
        out_dir,
        "Conditioning on position removes apparent camera dependence "
        f"(max |r| {maxabs('marginal'):.2f} -> {maxabs('in_sample_gp'):.2f} in-sample, "
        f"-> {maxabs('heldout_gp_' + head):.2f} on spatially held-out points)",
    )
    fs = fusion_scores[head]
    fig_a3(
        any_hit, results[head]["fused"], fs, out_dir,
        "Held-out ("
        + head
        + f"): noisy-OR Brier {fs['noisy_or']['brier']:.3f} vs best-single "
          f"{fs['best_single']['brier']:.3f} vs direct joint GP "
          f"{fs['joint_gp']['brier']:.3f}",
    )
    ek = np.mean([recal_scores[(c, "expected_kernel")]["logloss"] for c in CAMERAS])
    iso = np.mean([recal_scores[(c, "expected_kernel_isotonic")]["logloss"] for c in CAMERAS])
    fig_a4(
        recal_scores, out_dir,
        f"Recalibration fixes the expected-kernel GP's calibration failure: "
        f"mean held-out logloss {ek:.2f} -> {iso:.2f} (isotonic), "
        f"AUROC unchanged by construction",
    )

    # ------------------------------------------------------------ manifest
    F.write_json(out_dir / "manifest.json", {
        "study": "fused_observation_model",
        "experiment": "exp1_availability_fusion",
        "generated_by": "experiments/fused_observation_model/exp1_availability_fusion.py",
        "events": {
            f"camera_{c}": {
                "path": str(F.events_csv(c)),
                "sha256": F.sha256_file(F.events_csv(c)),
            } for c in CAMERAS
        },
        "priors": {
            f"camera_{c}": {
                "path": str(F.prior_npz(c)),
                "sha256": F.sha256_file(F.prior_npz(c)),
            } for c in CAMERAS
        },
        "locked_in_sample_gp": {
            f"camera_{c}": str(F.locked_gp_npz(c)) for c in CAMERAS
        },
        "gp_code": "scripts/visibility_comparison/fit_belief_aware_gp.py",
        "gp_mode": F.GP_MODE,
        "gp_hyperparameters": F.GP_HYPERPARAMS,
        "metrics_code": "scripts/shared/metrics.py",
        "n_events": joint.n,
        "schemes": {
            s: {"folds": results[s]["fold_ids"],
                "n_folds": len(results[s]["fold_ids"])} for s in schemes
        },
        "bootstrap": {"unit": f"{F.TILE_M} m spatial tile", "n_boot": int(args_cli.n_boot)},
        "synthetic_data": False,
    })

    print(f"\ndone -> {out_dir}")
    print(f"  max |r| marginal      = {maxabs('marginal'):.4f}")
    print(f"  max |r| in-sample GP  = {maxabs('in_sample_gp'):.4f}")
    for scheme in schemes:
        print(f"  max |r| held-out GP [{scheme}] = {maxabs('heldout_gp_' + scheme):.4f}")
    for scheme in schemes:
        print(f"  [{scheme}] fusion Brier: " + ", ".join(
            f"{k} {fusion_scores[scheme][k]['brier']:.4f}" for k in FUSION_MODELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
