#!/usr/bin/env python3
"""E2 — does the field notice the change, and does the residual on top hide it?

E1 measures how much skill each estimator loses.  It does not say why, and the two
recomputed arms lose very different amounts despite sharing a prior.  This experiment
asks the mechanical question directly: treat each estimator's own before-and-after
difference as a change detector, and score it against the change that actually
happened.

The reference is the fresh survey minus the stale survey -- a cell whose CAD ray-cast
visibility dropped is a cell that genuinely went dark.  That is evaluation-only
geometry, used here to report on the estimators and never as an input to them.

The comparison the paper turns on is between the monocular field and the hybrid.  They
share a prior, computed from the same frames; they differ only in that the hybrid
composes a Gaussian-process residual, fitted on nominal-warehouse outcomes, on top of
it.  If the residual pulls the composed field back toward the warehouse it was fitted
in, that shows up here as recall the prior had and the composition lost.

    python3 experiments/reconfiguration_holdout/e2_change_detection/run_experiment.py

Writes environment-keyed CSV and manifest files under
logs/studies/reconfiguration_holdout/e2_change_detection/.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402

RESULTS = C.OUT_ROOT / "e2_change_detection"
WORK = C.OUT_ROOT / "work/fields"

#: A cell counts as changed when its estimated availability moves by more than this.
#: 0.25 is a quarter of the full probability range: large enough that grid-resolution
#: jitter and depth noise do not register, small enough to catch a rack shadow.
CHANGE_THRESHOLD = 0.25

COLUMNS = ("camera", "arm", "label", "environment", "threshold",
           "truth_lost", "said_lost", "true_positive", "precision", "recall",
           "said_gained_spuriously", "cells")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--environment", default="L1")
    ap.add_argument("--threshold", type=float, default=CHANGE_THRESHOLD)
    args = ap.parse_args(argv)

    env = C.ENV_BY_KEY[args.environment]
    RESULTS.mkdir(parents=True, exist_ok=True)
    nominal = C.ENV_BY_KEY[C.DEVELOPMENT_ENV]

    arms: list[tuple[str, str, dict, dict]] = [
        ("mono_depth", "Monocular depth raycast",
         C.mono_depth_field(nominal.key), C.mono_depth_field(env.key)),
    ]
    hy0, hy1 = WORK / f"hybrid_{nominal.key}.npz", WORK / f"hybrid_{env.key}.npz"
    if hy0.is_file() and hy1.is_file():
        d0, d1 = np.load(hy0), np.load(hy1)
        arms.append(("hybrid", "Hybrid: GP residual on depth prior",
                     {c: d0[f"{c}__field"] for c in C.CAMERAS},
                     {c: d1[f"{c}__field"] for c in C.CAMERAS}))

    rows: list[dict] = []
    for camera in C.CAMERAS:
        cad0 = C.cad_field(nominal.world_name)[camera]
        cad1 = C.cad_field(env.world_name)[camera]
        truth_lost = (cad0 - cad1) > args.threshold
        for key, label, f0, f1 in arms:
            said_lost = (f0[camera] - f1[camera]) > args.threshold
            said_gain = (f1[camera] - f0[camera]) > args.threshold
            tp = int((said_lost & truth_lost).sum())
            rows.append({
                "camera": C.SHORT[camera], "arm": key, "label": label,
                "environment": env.key, "threshold": args.threshold,
                "truth_lost": int(truth_lost.sum()),
                "said_lost": int(said_lost.sum()),
                "true_positive": tp,
                "precision": tp / max(int(said_lost.sum()), 1),
                "recall": tp / max(int(truth_lost.sum()), 1),
                "said_gained_spuriously": int(said_gain.sum()),
                "cells": int(truth_lost.size),
            })

    result_path = RESULTS / f"e2_change_detection_{env.key}.csv"
    manifest_path = RESULTS / f"manifest_{env.key}.json"
    csv_paths = [result_path]
    if env.key == "L1":
        # Backward-compatible alias used by the existing figure script; later
        # environments never overwrite it.
        csv_paths.append(RESULTS / "e2_change_detection.csv")
    for csv_path in csv_paths:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
            w.writeheader()
            w.writerows(rows)

    print(f"Change detection against the fresh-minus-stale survey, threshold "
          f"{args.threshold}, environment {env.key}\n")
    print(f"{'camera':8s}{'arm':36s}{'truly lost':>11s}{'said lost':>11s}"
          f"{'precision':>11s}{'recall':>9s}{'spurious gains':>16s}")
    for r in rows:
        print(f"{r['camera']:8s}{r['label'][:35]:36s}{r['truth_lost']:11d}"
              f"{r['said_lost']:11d}{r['precision']:11.3f}{r['recall']:9.3f}"
              f"{r['said_gained_spuriously']:16d}")
    print()
    summary = {}
    for key, label, _f0, _f1 in arms:
        sub = [r for r in rows if r["arm"] == key]
        summary[key] = {
            "label": label,
            "precision_mean": float(np.mean([r["precision"] for r in sub])),
            "recall_mean": float(np.mean([r["recall"] for r in sub])),
            "truly_lost_total": int(sum(r["truth_lost"] for r in sub)),
            "spurious_gains_total": int(sum(r["said_gained_spuriously"] for r in sub)),
        }
        s = summary[key]
        print(f"{label:36s} precision {s['precision_mean']:.3f}  "
              f"recall {s['recall_mean']:.3f}  "
              f"{s['spurious_gains_total']} spurious 'gained' cells against "
              f"{s['truly_lost_total']} truly lost")

    if "mono_depth" in summary and "hybrid" in summary:
        lost = summary["mono_depth"]["recall_mean"] - summary["hybrid"]["recall_mean"]
        print(f"\nThe residual costs {lost:.3f} of the prior's recall: the composed field "
              f"notices {summary['hybrid']['recall_mean'] / max(summary['mono_depth']['recall_mean'], 1e-9):.0%} "
              f"of what its own prior noticed.")

    manifest_text = json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env.key, "nominal": nominal.key,
        "change_threshold": args.threshold,
        "reference": ("CAD raycast of the reconfigured world minus the nominal world; "
                      "evaluation-only geometry, never a model input"),
        "summary": summary,
    }, indent=2)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    if env.key == "L1":
        (RESULTS / "manifest.json").write_text(manifest_text, encoding="utf-8")
    print(f"\nwrote {result_path.relative_to(C.REPO)} and "
          f"{manifest_path.relative_to(C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
