#!/usr/bin/env python3
"""Fit the two experience-based availability fields on L0 events, then freeze them.

Two arms, differing ONLY in their mean function, which is what separates what the
detector outcomes contribute from what the geometry contributes:

* ``gp``      — zero prior mean: the field is learned from L0 detector outcomes alone.
* ``hybrid``  — a residual fitted once against the L0 monocular-depth mean, then
                added either to that same L0 mean or to a target environment's
                recomputed mean.  The training mean must remain L0: subtracting a
                target-environment mean from L0 labels would silently refit a
                different residual for every target and would not be a frozen arm.

WHY THIS IS A SEPARATE PROCESS.  ``scripts/visibility_comparison/fit_belief_aware_gp``
does ``from common import ...`` and this package also has a ``common``.  One
interpreter would resolve that name to whichever directory reached ``sys.path``
first, silently handing the GP fitter the wrong module.  This process puts only the
fitter's directory on the path.  Same reason, same shape, as
``experiments/availability_paper/gp_refit.py``.

    python3 experiments/reconfiguration_holdout/gp_fields.py \
        --events-dir logs/studies/reconfiguration_holdout/work/events_L0 \
        --arm hybrid \
        --train-prior-dir logs/studies/reconfiguration_holdout/work/prior_L0 \
        --query-prior-dir logs/studies/reconfiguration_holdout/work/prior_L1 \
        --grid-npz logs/studies/reconfiguration_holdout/work/grid.npz \
        --out logs/studies/reconfiguration_holdout/fields/hybrid_L1.npz

Hyperparameters are frozen to the 2026-07-27 validation manifest's values, so these
fields differ from the published artifacts only in what they were trained on and
which prior they sit on.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Only the fitter's directory: this process must resolve `common` to theirs.
sys.path.insert(0, str(REPO / "scripts/visibility_comparison"))
sys.path.insert(0, str(REPO / "scripts/shared"))

import fit_belief_aware_gp as F  # noqa: E402

CAMERAS = ("external_camera", "external_camera_b", "external_camera_c", "external_camera_d")

HYPERPARAMETERS = dict(
    aggregate_resolution_m=0.3,
    max_bin_weight=20.0,
    gp_length_scale=1.2,
    gp_noise_var=0.05,
    pose_length_scale=0.35,
    min_certainty=0.05,
    spread_scale=1.0,
    min_prob=1e-4,
    beta=0.5,
)


class _Query:
    """What ``_predict_mode_at_events`` needs of an event set: points and covariances."""

    def __init__(self, points: np.ndarray) -> None:
        self.X = points
        self.cov = np.tile(np.eye(2) * 1e-6, (len(points), 1, 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-dir", required=True,
                    help="directory of <camera>_events.csv, the L0 training outcomes")
    ap.add_argument("--arm", required=True, choices=("gp", "hybrid"))
    ap.add_argument("--train-prior-dir", default="",
                    help="for hybrid: L0 <camera>_prior.npz maps used to define the residual")
    ap.add_argument("--query-prior-dir", default="",
                    help="for hybrid: target <camera>_prior.npz maps to which the frozen residual is added")
    ap.add_argument("--grid-npz", required=True,
                    help="npz carrying xs/ys: the one working grid every field lives on")
    ap.add_argument("--out", required=True, help="output .npz of per-camera fields")
    args = ap.parse_args(argv)

    fit_args = argparse.Namespace(**HYPERPARAMETERS)
    events_dir = Path(args.events_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grid = np.load(args.grid_npz)
    xs = np.asarray(grid["xs"], dtype=float)
    ys = np.asarray(grid["ys"], dtype=float)

    fields: dict[str, np.ndarray] = {}
    residual_fields: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        data = F._load_events(events_dir / f"{camera}_events.csv", target="hit",
                              min_prob=HYPERPARAMETERS["min_prob"])
        train_prior = None
        query_prior = None
        if args.arm == "hybrid":
            if not args.train_prior_dir or not args.query_prior_dir:
                raise SystemExit(
                    "hybrid requires both --train-prior-dir (L0) and "
                    "--query-prior-dir (target environment)")
            train_path = Path(args.train_prior_dir) / f"{camera}_prior.npz"
            query_path = Path(args.query_prior_dir) / f"{camera}_prior.npz"
            train_prior = F._load_prior_map(
                train_path, map_key="P_mean_map", min_prob=HYPERPARAMETERS["min_prob"])
            query_prior = F._load_prior_map(
                query_path, map_key="P_mean_map", min_prob=HYPERPARAMETERS["min_prob"])
            if train_prior is None or query_prior is None:
                raise SystemExit(f"could not load hybrid priors {train_path} and {query_path}")
            for role, prior in (("training", train_prior), ("query", query_prior)):
                if prior.xs.shape != xs.shape or prior.ys.shape != ys.shape:
                    raise SystemExit(
                        f"{camera}: {role} prior grid {prior.p_mean.shape} does not "
                        f"match the working grid ({len(ys)}, {len(xs)})")

        agg = F._aggregate_events(data,
                                  resolution_m=HYPERPARAMETERS["aggregate_resolution_m"],
                                  max_bin_weight=HYPERPARAMETERS["max_bin_weight"])
        XX, YY = np.meshgrid(xs, ys)
        query = _Query(np.column_stack([XX.ravel(), YY.ravel()]))
        if args.arm == "gp":
            pred = F._predict_mode_at_events(
                agg, query, mode="expected_kernel", prior=None, args=fit_args)
        else:
            # The canonical helper accepts one prior for both subtraction at the
            # training points and addition at query points.  Transfer needs two:
            # always subtract L0, then add the target environment.  Reuse the
            # canonical aggregation, noise weights, expected kernel and clipping.
            _x_fit, _y_fit, alpha, _summary = F._fit_inputs(
                agg,
                mode="expected_kernel",
                noise_var=float(fit_args.gp_noise_var),
                gp_length_scale=float(fit_args.gp_length_scale),
                pose_length_scale=float(fit_args.pose_length_scale),
                min_certainty=float(fit_args.min_certainty),
                spread_scale=float(fit_args.spread_scale),
            )
            train_latent = F._logit(agg.y) - F._prior_logit(train_prior, agg.X)
            residual, _sigma, _jitter = F._fit_predict_expected_kernel_latent(
                agg.X,
                train_latent,
                agg.cov,
                alpha,
                query.X,
                query_cov=query.cov,
                length_scale=float(fit_args.gp_length_scale),
            )
            residual_fields[camera] = np.asarray(residual, dtype=float).reshape(
                len(ys), len(xs))
            latent = F._prior_logit(query_prior, query.X) + residual
            pred = np.clip(
                1.0 / (1.0 + np.exp(-np.clip(latent, -60.0, 60.0))),
                HYPERPARAMETERS["min_prob"],
                1.0 - HYPERPARAMETERS["min_prob"],
            )
        field = np.clip(np.asarray(pred, dtype=float).reshape(len(ys), len(xs)),
                        1e-4, 1 - 1e-4)
        fields[camera] = field
        print(f"[{args.arm}] {camera}: {len(data.X)} events, "
              f"field mean {field.mean():.4f} min {field.min():.4f} max {field.max():.4f}")

    payload = {f"{c}__field": fields[c] for c in CAMERAS}
    if args.arm == "hybrid":
        payload.update({f"{c}__residual_latent": residual_fields[c] for c in CAMERAS})
    np.savez_compressed(
        out_path,
        xs=xs,
        ys=ys,
        residual_protocol=np.asarray(
            ["none" if args.arm == "gp" else "fit_against_L0_prior_then_freeze"]),
        train_prior_dir=np.asarray([args.train_prior_dir]),
        query_prior_dir=np.asarray([args.query_prior_dir]),
        **payload,
    )
    print(f"[{args.arm}] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
