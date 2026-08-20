#!/usr/bin/env python3
"""Fold-clean GP predictions for nominal-to-reconfigured transfer.

This process emits predictions, never a fitted deployment field.  For each camera
and outer spatial block it:

1. fits the GP on L0 events outside that block;
2. predicts both the held-out L0 block and the same block in every target
   environment;
3. creates nested out-of-sample predictions on the L0 training blocks so the
   two-parameter calibration link can also be fitted without leakage.

The hybrid residual is always defined relative to the L0 monocular prior.  At query
time that one frozen residual is added to the prior of the environment being scored.
This is intentionally more explicit than the canonical helper, which accepts only
one prior and therefore cannot express train-prior != query-prior transfer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FITTER_DIR = REPO / "scripts/visibility_comparison"

# Resolve the canonical fitter's own ``common`` module, not this study's module.
sys.path.insert(0, str(FITTER_DIR))
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


def block_ids(points: np.ndarray, x_edges: list[float], y_edges: list[float]) -> np.ndarray:
    """Apply the caller's frozen spatial-block edges."""

    pts = np.asarray(points, dtype=float)
    bx = np.clip(np.searchsorted(np.asarray(x_edges[1:-1]), pts[:, 0]),
                 0, len(x_edges) - 2)
    by = np.clip(np.searchsorted(np.asarray(y_edges[1:-1]), pts[:, 1]),
                 0, len(y_edges) - 2)
    return (bx * (len(y_edges) - 1) + by).astype(int)


def _load_prior(directory: Path, camera: str):
    path = directory / f"{camera}_prior.npz"
    prior = F._load_prior_map(path, map_key="P_mean_map",
                              min_prob=HYPERPARAMETERS["min_prob"])
    if prior is None:
        raise RuntimeError(f"could not load prior {path}")
    return prior


def _predict(
    train_data,
    queries: list[tuple[str, object, np.ndarray]],
    *,
    l0_prior,
    query_priors: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit each latent arm once and predict a collection of query subsets."""

    agg = F._aggregate_events(
        train_data,
        resolution_m=HYPERPARAMETERS["aggregate_resolution_m"],
        max_bin_weight=HYPERPARAMETERS["max_bin_weight"],
    )
    fit_args = argparse.Namespace(**HYPERPARAMETERS)
    _x_fit, _y_fit, alpha, _summary = F._fit_inputs(
        agg,
        mode="expected_kernel",
        noise_var=float(fit_args.gp_noise_var),
        gp_length_scale=float(fit_args.gp_length_scale),
        pose_length_scale=float(fit_args.pose_length_scale),
        min_certainty=float(fit_args.min_certainty),
        spread_scale=float(fit_args.spread_scale),
    )

    lengths = [int(np.sum(mask)) for _env, _data, mask in queries]
    query_x = np.concatenate([data.X[mask] for _env, data, mask in queries], axis=0)
    query_cov = np.concatenate([data.cov[mask] for _env, data, mask in queries], axis=0)

    gp_latent, _gp_sigma, _gp_jitter = F._fit_predict_expected_kernel_latent(
        agg.X,
        F._logit(agg.y),
        agg.cov,
        alpha,
        query_x,
        query_cov=query_cov,
        length_scale=HYPERPARAMETERS["gp_length_scale"],
    )
    hybrid_residual, _hy_sigma, _hy_jitter = F._fit_predict_expected_kernel_latent(
        agg.X,
        F._logit(agg.y) - F._prior_logit(l0_prior, agg.X),
        agg.cov,
        alpha,
        query_x,
        query_cov=query_cov,
        length_scale=HYPERPARAMETERS["gp_length_scale"],
    )

    gp_prob = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(gp_latent, -60.0, 60.0))),
        HYPERPARAMETERS["min_prob"], 1.0 - HYPERPARAMETERS["min_prob"],
    )
    hybrid_prob = np.empty_like(gp_prob)
    offset = 0
    for (environment, data, mask), length in zip(queries, lengths):
        sl = slice(offset, offset + length)
        query_prior = F._prior_logit(query_priors[environment], data.X[mask])
        latent = query_prior + hybrid_residual[sl]
        hybrid_prob[sl] = np.clip(
            1.0 / (1.0 + np.exp(-np.clip(latent, -60.0, 60.0))),
            HYPERPARAMETERS["min_prob"], 1.0 - HYPERPARAMETERS["min_prob"],
        )
        offset += length

    gp_out: dict[str, np.ndarray] = {}
    hybrid_out: dict[str, np.ndarray] = {}
    offset = 0
    for (environment, _data, _mask), length in zip(queries, lengths):
        sl = slice(offset, offset + length)
        gp_out[environment] = gp_prob[sl]
        hybrid_out[environment] = hybrid_prob[sl]
        offset += length
    return gp_out, hybrid_out


def validate_camera_rows(rows: list[dict[str, object]], *, camera: str,
                         event_counts: dict[str, int], n_blocks: int) -> None:
    """Fail closed if any query is missing, duplicated, or assigned to its fit fold."""
    camera_rows = [row for row in rows if row["camera"] == camera]
    test = [row for row in camera_rows if row["role"] == "test"]
    train = [row for row in camera_rows if row["role"] == "train_oos"]

    expected_test = sum(event_counts.values())
    expected_train = event_counts["L0"] * (n_blocks - 1)
    if len(test) != expected_test or len(train) != expected_train:
        raise RuntimeError(
            f"{camera}: prediction membership mismatch: test {len(test)}/{expected_test}, "
            f"train_oos {len(train)}/{expected_train}")

    test_keys = {(row["environment"], int(row["event_index"])) for row in test}
    if len(test_keys) != expected_test:
        raise RuntimeError(f"{camera}: duplicate or missing environment test event")
    train_keys = {(int(row["outer_fold"]), int(row["event_index"])) for row in train}
    if len(train_keys) != expected_train:
        raise RuntimeError(f"{camera}: duplicate or missing nested link event")
    if any(int(row["outer_fold"]) == int(row["block"]) for row in train):
        raise RuntimeError(f"{camera}: an outer test-block event entered link training")
    if any(int(row["outer_fold"]) != int(row["block"]) for row in test):
        raise RuntimeError(f"{camera}: a test event was emitted under the wrong outer fold")

    per_event: dict[int, int] = {}
    for row in train:
        event_index = int(row["event_index"])
        per_event[event_index] = per_event.get(event_index, 0) + 1
    if len(per_event) != event_counts["L0"] or set(per_event.values()) != {n_blocks - 1}:
        raise RuntimeError(f"{camera}: nested link coverage is not exactly n_blocks-1 per event")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l0-events-dir", required=True)
    ap.add_argument(
        "--environment", nargs=2, action="append", required=True,
        metavar=("KEY", "EVENTS_DIR"),
        help="target environment and its event directory; repeat as needed",
    )
    ap.add_argument(
        "--prior", nargs=2, action="append", required=True,
        metavar=("KEY", "PRIOR_DIR"),
        help="environment and its per-camera prior directory; must include L0",
    )
    ap.add_argument("--block-x-edges", nargs="+", type=float, required=True)
    ap.add_argument("--block-y-edges", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    event_dirs = {key: Path(path) for key, path in args.environment}
    prior_dirs = {key: Path(path) for key, path in args.prior}
    if "L0" in event_dirs:
        raise SystemExit("pass only changed environments via --environment; L0 has its own argument")
    needed_priors = {"L0", *event_dirs}
    if set(prior_dirs) != needed_priors:
        raise SystemExit(
            f"prior keys must be exactly {sorted(needed_priors)}, got {sorted(prior_dirs)}")

    n_blocks = ((len(args.block_x_edges) - 1) *
                (len(args.block_y_edges) - 1))
    rows: list[dict[str, object]] = []

    for camera in CAMERAS:
        camera_row_start = len(rows)
        l0 = F._load_events(
            Path(args.l0_events_dir) / f"{camera}_events.csv",
            target="hit", min_prob=HYPERPARAMETERS["min_prob"],
        )
        targets = {
            key: F._load_events(directory / f"{camera}_events.csv", target="hit",
                                min_prob=HYPERPARAMETERS["min_prob"])
            for key, directory in event_dirs.items()
        }
        all_data = {"L0": l0, **targets}
        folds = {
            key: block_ids(data.X, args.block_x_edges, args.block_y_edges)
            for key, data in all_data.items()
        }
        priors = {key: _load_prior(directory, camera)
                  for key, directory in prior_dirs.items()}

        def append_predictions(
            outer: int,
            role: str,
            train_mask: np.ndarray,
            query_specs: list[tuple[str, object, np.ndarray]],
        ) -> None:
            train = F._subset_events(l0, train_mask)
            gp, hybrid = _predict(
                train, query_specs, l0_prior=priors["L0"], query_priors=priors)
            for environment, data, mask in query_specs:
                indices = np.flatnonzero(mask)
                for k, event_index in enumerate(indices):
                    rows.append({
                        "environment": environment,
                        "camera": camera,
                        "outer_fold": outer,
                        "block": int(folds[environment][event_index]),
                        "role": role,
                        "event_index": int(event_index),
                        "m_x": f"{data.X[event_index, 0]:.6f}",
                        "m_y": f"{data.X[event_index, 1]:.6f}",
                        "hit": f"{data.y[event_index]:.0f}",
                        "n_fit": int(train.X.shape[0]),
                        "p_gp": f"{float(gp[environment][k]):.8f}",
                        "p_hybrid": f"{float(hybrid[environment][k]):.8f}",
                    })

        for outer in range(n_blocks):
            l0_test = folds["L0"] == outer
            l0_train = ~l0_test
            if not l0_test.any() or not l0_train.any():
                raise RuntimeError(f"{camera}: empty L0 outer fold {outer}")

            test_queries = [("L0", l0, l0_test)]
            for environment, data in targets.items():
                target_mask = folds[environment] == outer
                if not target_mask.any():
                    raise RuntimeError(f"{camera}: {environment} has empty block {outer}")
                test_queries.append((environment, data, target_mask))
            append_predictions(outer, "test", l0_train, test_queries)

            # Link-fitting rows are out of sample twice: neither the outer test
            # block nor the inner block being predicted participates in its GP fit.
            for inner in range(n_blocks):
                if inner == outer:
                    continue
                inner_mask = folds["L0"] == inner
                append_predictions(
                    outer,
                    "train_oos",
                    l0_train & ~inner_mask,
                    [("L0", l0, inner_mask)],
                )

            print(
                f"{camera} outer fold {outer}: test L0 + "
                f"{len(targets)} target environment(s), {n_blocks - 1} inner refits",
                flush=True,
            )

        validate_camera_rows(
            rows[camera_row_start:],
            camera=camera,
            event_counts={key: int(data.X.shape[0]) for key, data in all_data.items()},
            n_blocks=n_blocks,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "environment", "camera", "outer_fold", "block", "role", "event_index",
        "m_x", "m_y", "hit", "n_fit", "p_gp", "p_hybrid",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} fold-clean predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
