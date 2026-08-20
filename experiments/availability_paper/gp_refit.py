#!/usr/bin/env python3
"""Per-fold GP refit for E1 — emits held-out predictions, never a fitted field.

WHY THIS IS A SEPARATE PROCESS. ``scripts/visibility_comparison/fit_belief_aware_gp``
does ``from common import ...`` and this package also has a ``common``. Importing
both into one interpreter resolves that name to whichever directory reaches
sys.path first, so the GP fitter would silently get the wrong module. Running the
refit in its own process with only the fitter's directory on the path removes the
ambiguity entirely.

WHY IT EXISTS AT ALL. The GP fields cached under ``logs/.../spawn_grid_20260727/gp``
are fitted on **every** event, including the ones E1 holds out. Scoring them at
held-out points is training on the test set: doing exactly that put the GP at
Brier 0.021 / AUROC 0.991, against 0.157 for the same model under the repository's
own validated harness. This script refits from the training events of each fold.

Two arms are emitted per fold:

* ``gp``     — zero prior mean, so the field is learned from detector outcomes alone;
* ``hybrid`` — the geometric day-zero prior as the GP mean function, with the GP
               fitting only the residual.

Usage (invoked by e1_availability_calibration/run_experiment.py):

    python3 experiments/availability_paper/gp_refit.py \
        --out results/gp_fold_predictions.csv \
        --block-x-edges -11.7 -3.9 3.9 11.7 \
        --block-y-edges -9.0 -0.25 9.0
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

# Only the fitter's directory: this process must resolve `common` to theirs.
sys.path.insert(0, str(FITTER_DIR))
sys.path.insert(0, str(REPO / "scripts/shared"))

import fit_belief_aware_gp as F  # noqa: E402

EVENT_ROOT = REPO / "logs/studies/multicamera_commissioning_bigwarehouse/spawn_grid_20260727/events"
PRIOR_ROOT = (
    REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse"
    / "actual_commissioning_20260715/analysis/final_01/inputs"
)
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")

#: Hyperparameters frozen to the values in the 2026-07-27 validation manifest, so
#: the refit differs from the published artifact only in what it is trained on.
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


def _args_namespace() -> argparse.Namespace:
    return argparse.Namespace(**HYPERPARAMETERS)


def block_ids(pts: np.ndarray, x_edges: list[float], y_edges: list[float]) -> np.ndarray:
    """Apply block edges supplied by the caller.

    The block *rule* lives in ``experiments/availability_paper/common.py``; this
    function only applies the edges it is given, so the two processes cannot
    drift apart on anything but arithmetic. The caller verifies the per-fold
    counts agree.
    """

    bx = np.clip(np.searchsorted(np.asarray(x_edges[1:-1]), pts[:, 0]), 0, len(x_edges) - 2)
    by = np.clip(np.searchsorted(np.asarray(y_edges[1:-1]), pts[:, 1]), 0, len(y_edges) - 2)
    return (bx * (len(y_edges) - 1) + by).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block-x-edges", nargs="+", type=float, required=True)
    parser.add_argument("--block-y-edges", nargs="+", type=float, required=True)
    args = parser.parse_args()

    fit_args = _args_namespace()
    n_blocks = (len(args.block_x_edges) - 1) * (len(args.block_y_edges) - 1)
    rows: list[dict] = []

    for camera in CAMERAS:
        events_path = EVENT_ROOT / f"{camera}_events.csv"
        data = F._load_events(events_path, target="hit", min_prob=HYPERPARAMETERS["min_prob"])
        prior = F._load_prior_map(
            PRIOR_ROOT / f"{camera}_dayzero_prior.npz",
            map_key="P_mean_map",
            min_prob=HYPERPARAMETERS["min_prob"],
        )
        folds = block_ids(data.X, args.block_x_edges, args.block_y_edges)

        def predict_block(fit_mask: np.ndarray, target_block: int) -> None:
            """Fit on ``fit_mask`` events, predict on ``target_block``, append rows."""

            target_mask = folds == target_block
            if not target_mask.any() or not fit_mask.any():
                raise RuntimeError(f"{camera}: empty mask for block {target_block}")
            fit_data = F._subset_events(data, fit_mask)
            target = F._subset_events(data, target_mask)
            fit_agg = F._aggregate_events(
                fit_data,
                resolution_m=HYPERPARAMETERS["aggregate_resolution_m"],
                max_bin_weight=HYPERPARAMETERS["max_bin_weight"],
            )
            p_gp = F._predict_mode_at_events(
                fit_agg, target, mode="expected_kernel", prior=None, args=fit_args
            )
            p_hybrid = F._predict_mode_at_events(
                fit_agg, target, mode="expected_kernel", prior=prior, args=fit_args
            )
            target_index = np.flatnonzero(target_mask)
            for k in range(target.X.shape[0]):
                rows.append(
                    {
                        "camera": camera,
                        "outer_fold": outer,
                        "block": target_block,
                        "role": role,
                        "event_index": int(target_index[k]),
                        "m_x": f"{target.X[k, 0]:.6f}",
                        "m_y": f"{target.X[k, 1]:.6f}",
                        "hit": f"{target.y[k]:.0f}",
                        "n_fit": int(fit_data.X.shape[0]),
                        "p_gp": f"{float(p_gp[k]):.8f}",
                        "p_hybrid": f"{float(p_hybrid[k]):.8f}",
                    }
                )

        for outer in range(n_blocks):
            test_mask = folds == outer
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                raise RuntimeError(f"{camera}: empty fold {outer}")

            # Held-out predictions for the outer fold: fit on the other five blocks.
            role = "test"
            predict_block(train_mask, outer)

            # Out-of-sample predictions ON the training blocks, for fitting the
            # calibration link without letting it see the outer test block. Each
            # inner block is predicted from a model that saw neither it nor the
            # outer test block.
            role = "train_oos"
            for inner in range(n_blocks):
                if inner == outer:
                    continue
                predict_block(train_mask & (folds != inner), inner)

            print(f"{camera} outer fold {outer}: test {int(test_mask.sum())}, "
                  f"{n_blocks - 1} inner refits", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "camera",
        "outer_fold",
        "block",
        "role",
        "event_index",
        "m_x",
        "m_y",
        "hit",
        "n_fit",
        "p_gp",
        "p_hybrid",
    ]
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} held-out predictions)")


if __name__ == "__main__":
    main()
