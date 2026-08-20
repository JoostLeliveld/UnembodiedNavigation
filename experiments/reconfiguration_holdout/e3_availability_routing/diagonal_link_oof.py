#!/usr/bin/env python3
"""Create pooled out-of-fold GP/hybrid scores for E3 link calibration.

The deployment fields are fitted on every diagonal-heading L0 outcome.  Their
two-parameter probability links must not, however, be estimated from scores at
positions used to fit those same latent fields.  For each camera this helper fits
six spatially blocked models, predicts the held-out diagonal block, and emits one
out-of-fold score per diagonal event.  E3 pools the six held-out blocks when fitting
the final link used by the full-data deployment field.

Run this as a separate process.  It deliberately imports the canonical GP fitter
through ``gp_transfer_refit`` before any reconfiguration-study ``common`` module can
occupy that name in ``sys.modules``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

import gp_transfer_refit as G  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-dir", required=True)
    parser.add_argument("--l0-prior-dir", required=True)
    parser.add_argument("--block-x-edges", nargs="+", type=float, required=True)
    parser.add_argument("--block-y-edges", nargs="+", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    events_dir = Path(args.events_dir)
    prior_dir = Path(args.l0_prior_dir)
    n_blocks = ((len(args.block_x_edges) - 1) *
                (len(args.block_y_edges) - 1))
    if n_blocks < 2:
        raise SystemExit("at least two spatial blocks are required")

    rows: list[dict[str, object]] = []
    for camera in G.CAMERAS:
        data = G.F._load_events(
            events_dir / f"{camera}_events.csv",
            target="hit",
            min_prob=G.HYPERPARAMETERS["min_prob"],
        )
        folds = G.block_ids(data.X, args.block_x_edges, args.block_y_edges)
        prior = G._load_prior(prior_dir, camera)
        p_gp = np.full(data.X.shape[0], np.nan, dtype=float)
        p_hybrid = np.full(data.X.shape[0], np.nan, dtype=float)
        n_fit = np.zeros(data.X.shape[0], dtype=int)

        for fold in range(n_blocks):
            test = folds == fold
            train = ~test
            if not test.any() or not train.any():
                raise RuntimeError(f"{camera}: empty diagonal fold {fold}")
            fitted = G.F._subset_events(data, train)
            gp, hybrid = G._predict(
                fitted,
                [("L0", data, test)],
                l0_prior=prior,
                query_priors={"L0": prior},
            )
            p_gp[test] = gp["L0"]
            p_hybrid[test] = hybrid["L0"]
            n_fit[test] = int(train.sum())
            print(
                f"[e3 link OOF] {camera} fold {fold}: "
                f"fit {int(train.sum())}, predict {int(test.sum())}",
                flush=True,
            )

        if not (np.all(np.isfinite(p_gp)) and np.all(np.isfinite(p_hybrid))):
            raise RuntimeError(f"{camera}: incomplete out-of-fold predictions")
        for index in range(data.X.shape[0]):
            rows.append({
                "camera": camera,
                "event_index": index,
                "block": int(folds[index]),
                "m_x": f"{float(data.X[index, 0]):.6f}",
                "m_y": f"{float(data.X[index, 1]):.6f}",
                "hit": f"{float(data.y[index]):.0f}",
                "n_fit": int(n_fit[index]),
                "p_gp": f"{float(p_gp[index]):.8f}",
                "p_hybrid": f"{float(p_hybrid[index]):.8f}",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "camera", "event_index", "block", "m_x", "m_y", "hit", "n_fit",
        "p_gp", "p_hybrid",
    )
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[e3 link OOF] wrote {out} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

