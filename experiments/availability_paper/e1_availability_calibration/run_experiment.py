#!/usr/bin/env python3
"""E1 — held-out calibration of usable-observation probability.

Scores six availability estimators against the frozen four-camera spawn-grid
detector outcomes under leave-one-spatial-block-out cross-validation. The
calibration link is fitted inside the loop on training folds only.

Run:
    python3 experiments/availability_paper/e1_availability_calibration/run_experiment.py

Writes results/e1_folds.csv, results/e1_summary.csv, results/e1_paired.csv and
results/manifest.json. Reads no ground-truth pose.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402

RESULTS = C.OUT_ROOT / "e1_availability_calibration"
GP_PREDICTIONS = RESULTS / "gp_fold_predictions.csv"
#: Arms whose predictions come from the per-fold refit rather than a cached field.
REFIT_ARMS = {"gp": "p_gp", "hybrid": "p_hybrid"}

FOLD_COLUMNS = (
    "camera",
    "source",
    "variant",
    "fold",
    "n_train",
    "n_test",
    "link_a",
    "link_b",
    "brier",
    "logloss",
    "auroc",
    "ece",
    "pred_mean",
    "target_mean",
)
SUMMARY_COLUMNS = (
    "source",
    "label",
    "variant",
    "needs_surveyed_model",
    "n_units",
    "brier_mean",
    "brier_std",
    "logloss_mean",
    "logloss_std",
    "auroc_mean",
    "auroc_std",
    "ece_mean",
    "ece_std",
)
PAIRED_COLUMNS = (
    "source",
    "label",
    "reference",
    "metric",
    "n_units",
    "mean_difference",
    "std_difference",
    "ci95_low",
    "ci95_high",
    "units_source_better",
    "sign_test_p_two_sided",
)

#: Bootstrap resamples for the paired-difference interval. Fixed seed: the interval
#: in the paper must not move between runs.
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260817


def _paired_ci95(diff: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference.

    WHY AN INTERVAL AND NOT JUST A p-VALUE. "p = 0.15" is absence of evidence, which
    is not the same as evidence of equivalence. The interval says how large a
    difference the design could actually have detected, so a tie can be stated as a
    bound rather than as a failure to reject.
    """

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = diff.size
    if n < 2:
        return float("nan"), float("nan")
    means = diff[rng.integers(0, n, size=(BOOTSTRAP_N, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test; ties are dropped, as is conventional.

    Written out rather than pulled from scipy so the number in the paper does
    not depend on which scipy happens to be installed.
    """

    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1))
    return float(min(1.0, 2.0 * tail / (2.0**n)))


def load_refit_predictions(apparatus: C.Apparatus) -> dict:
    """Load per-fold GP predictions and verify they line up with the events.

    The refit runs in a separate process against the same event CSVs. If the two
    loaders ever disagreed about row order or row filtering, every GP number here
    would be silently attached to the wrong pose, so this asserts position
    agreement rather than trusting the index.
    """

    import csv

    if not GP_PREDICTIONS.is_file():
        raise RuntimeError(
            f"{GP_PREDICTIONS} not found. Run:\n"
            "  python3 experiments/availability_paper/gp_refit.py "
            f"--out {GP_PREDICTIONS} "
            f"--block-x-edges {' '.join(str(e) for e in C.BLOCK_X_EDGES)} "
            f"--block-y-edges {' '.join(str(e) for e in C.BLOCK_Y_EDGES)}"
        )

    table: dict = {}
    with open(GP_PREDICTIONS, newline="") as fh:
        for row in csv.DictReader(fh):
            camera = row["camera"]
            idx = int(row["event_index"])
            xy = apparatus.events[camera]["xy"][idx]
            if abs(float(row["m_x"]) - float(xy[0])) > 1e-5 or abs(float(row["m_y"]) - float(xy[1])) > 1e-5:
                raise RuntimeError(
                    f"{camera} event {idx}: refit position ({row['m_x']}, {row['m_y']}) does not "
                    f"match apparatus position {tuple(xy)}; the two event loaders disagree."
                )
            hit = float(apparatus.events[camera]["hit"][idx])
            if abs(float(row["hit"]) - hit) > 1e-9:
                raise RuntimeError(f"{camera} event {idx}: refit label {row['hit']} != {hit}")
            key = (camera, int(row["outer_fold"]), row["role"])
            entry = table.setdefault(key, {"index": [], "hit": [], "p_gp": [], "p_hybrid": []})
            entry["index"].append(idx)
            entry["hit"].append(hit)
            entry["p_gp"].append(float(row["p_gp"]))
            entry["p_hybrid"].append(float(row["p_hybrid"]))

    return {k: {kk: np.asarray(vv, dtype=float) for kk, vv in v.items()} for k, v in table.items()}


PREDICTION_COLUMNS = ("camera", "source", "fold", "m_x", "m_y", "hit", "p_linked")


def run(
    apparatus: C.Apparatus, refit: dict
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    fold_rows: list[dict] = []
    #: Every held-out prediction, so calibration curves and any later analysis do
    #: not have to re-derive them from the fold summaries.
    prediction_rows: list[dict] = []

    for camera in C.CAMERAS:
        ev = apparatus.events[camera]
        xy = np.asarray(ev["xy"], dtype=float)
        hits = np.asarray(ev["hit"], dtype=float)
        folds = C.block_ids(xy)

        for source in C.SOURCES:
            if source.key in REFIT_ARMS:
                scores = None  # supplied per fold by the refit
            elif source.key == "constant":
                scores = None
            else:
                scores = C.sample_field_at(
                    apparatus.field(source.key, camera), apparatus.xs, apparatus.ys, xy
                )

            for fold in range(C.N_BLOCKS):
                test = folds == fold
                train = ~test
                if not np.any(test) or not np.any(train):
                    raise RuntimeError(f"{camera}/{source.key}: empty fold {fold}")

                if source.key in REFIT_ARMS:
                    column = REFIT_ARMS[source.key]
                    test_entry = refit[(camera, fold, "test")]
                    train_entry = refit[(camera, fold, "train_oos")]
                    test_pred = test_entry[column]
                    a, b = C.fit_link(train_entry[column], train_entry["hit"])
                    n_train = int(train_entry["hit"].size)
                    variants = {
                        "linked": (C.apply_link(test_pred, a, b), a, b),
                        "raw": (np.clip(test_pred, 1e-4, 1 - 1e-4), float("nan"), float("nan")),
                    }
                    hits_test = test_entry["hit"]
                elif scores is None:
                    p_train_mean = float(np.mean(hits[train]))
                    n_train = int(train.sum())
                    variants = {
                        "linked": (np.full(int(test.sum()), p_train_mean), float("nan"), float("nan"))
                    }
                    hits_test = hits[test]
                else:
                    a, b = C.fit_link(scores[train], hits[train])
                    n_train = int(train.sum())
                    variants = {
                        "linked": (C.apply_link(scores[test], a, b), a, b),
                        "raw": (np.clip(scores[test], 1e-4, 1 - 1e-4), float("nan"), float("nan")),
                    }
                    hits_test = hits[test]

                if source.key in REFIT_ARMS:
                    test_xy = xy[refit[(camera, fold, "test")]["index"].astype(int)]
                else:
                    test_xy = xy[test]
                for k, p in enumerate(np.asarray(variants["linked"][0], dtype=float)):
                    prediction_rows.append(
                        {
                            "camera": camera,
                            "source": source.key,
                            "fold": fold,
                            "m_x": f"{test_xy[k, 0]:.4f}",
                            "m_y": f"{test_xy[k, 1]:.4f}",
                            "hit": f"{hits_test[k]:.0f}",
                            "p_linked": f"{p:.8f}",
                        }
                    )

                for variant, (pred, a, b) in variants.items():
                    scored = C.score_predictions(hits_test, pred)
                    fold_rows.append(
                        {
                            "camera": camera,
                            "source": source.key,
                            "variant": variant,
                            "fold": fold,
                            "n_train": n_train,
                            "n_test": int(hits_test.size),
                            "link_a": f"{a:.6f}" if np.isfinite(a) else "",
                            "link_b": f"{b:.6f}" if np.isfinite(b) else "",
                            **{k: f"{v:.8f}" if isinstance(v, float) else v for k, v in scored.items()},
                        }
                    )

    summary_rows = _summarise(fold_rows)
    paired_rows = _pair(fold_rows)
    return fold_rows, summary_rows, paired_rows, prediction_rows


def _units(fold_rows: list[dict], source: str, variant: str, metric: str) -> dict[tuple[str, int], float]:
    return {
        (r["camera"], int(r["fold"])): float(r[metric])
        for r in fold_rows
        if r["source"] == source and r["variant"] == variant and r[metric] != ""
    }


def _summarise(fold_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for source in C.SOURCES:
        for variant in ("linked", "raw"):
            rows = [r for r in fold_rows if r["source"] == source.key and r["variant"] == variant]
            if not rows:
                continue
            entry: dict = {
                "source": source.key,
                "label": source.label,
                "variant": variant,
                "needs_surveyed_model": str(source.needs_surveyed_model).lower(),
                "n_units": len(rows),
            }
            for metric in ("brier", "logloss", "auroc", "ece"):
                vals = np.asarray([float(r[metric]) for r in rows if r[metric] != ""], dtype=float)
                vals = vals[np.isfinite(vals)]
                entry[f"{metric}_mean"] = f"{np.mean(vals):.8f}" if vals.size else ""
                entry[f"{metric}_std"] = f"{np.std(vals, ddof=1):.8f}" if vals.size > 1 else ""
            out.append(entry)
    return out


def _pair(fold_rows: list[dict]) -> list[dict]:
    """Paired per-camera-fold contrasts against the two named references."""

    out: list[dict] = []
    for reference in ("cad_reference", "gp"):
        ref_by_metric = {m: _units(fold_rows, reference, "linked", m) for m in ("brier", "logloss", "auroc", "ece")}
        for source in C.SOURCES:
            if source.key == reference:
                continue
            for metric in ("brier", "logloss", "auroc", "ece"):
                src = _units(fold_rows, source.key, "linked", metric)
                ref = ref_by_metric[metric]
                keys = sorted(set(src) & set(ref))
                if not keys:
                    continue
                diff = np.asarray([src[k] - ref[k] for k in keys], dtype=float)
                diff = diff[np.isfinite(diff)]
                if diff.size == 0:
                    continue
                # Lower is better for brier/logloss/ece; higher is better for auroc.
                better = int(np.sum(diff > 0)) if metric == "auroc" else int(np.sum(diff < 0))
                worse = int(diff.size - better - np.sum(diff == 0))
                out.append(
                    {
                        "source": source.key,
                        "label": source.label,
                        "reference": reference,
                        "metric": metric,
                        "n_units": int(diff.size),
                        "mean_difference": f"{np.mean(diff):.8f}",
                        "std_difference": f"{np.std(diff, ddof=1):.8f}" if diff.size > 1 else "",
                        "ci95_low": f"{_paired_ci95(diff)[0]:.8f}",
                        "ci95_high": f"{_paired_ci95(diff)[1]:.8f}",
                        "units_source_better": better,
                        "sign_test_p_two_sided": f"{_sign_test_p(better, worse):.6f}",
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS))
    parser.add_argument(
        "--mono-depth-model",
        default=None,
        help="monocular depth model behind the mono_depth arm (default: the study's choice)",
    )
    args = parser.parse_args()
    out = Path(args.out)

    if args.mono_depth_model:
        C.set_mono_depth_model(args.mono_depth_model)
    apparatus = C.build_apparatus()
    block_counts = C.assert_blocks_match_capture(apparatus.events)

    refit = load_refit_predictions(apparatus)
    fold_rows, summary_rows, paired_rows, prediction_rows = run(apparatus, refit)

    C.write_csv(out / "e1_folds.csv", FOLD_COLUMNS, fold_rows)
    C.write_csv(out / "e1_summary.csv", SUMMARY_COLUMNS, summary_rows)
    C.write_csv(out / "e1_paired.csv", PAIRED_COLUMNS, paired_rows)
    C.write_csv(out / "e1_predictions.csv", PREDICTION_COLUMNS, prediction_rows)
    C.write_json(
        out / "manifest.json",
        {
            "experiment_id": "EXP-AVAIL-CAL",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": {
                "scheme": "leave_one_spatial_block_out",
                "blocks": C.N_BLOCKS,
                "block_x_edges": list(C.BLOCK_X_EDGES),
                "block_y_edges": list(C.BLOCK_Y_EDGES),
                "block_event_counts": block_counts,
            },
            "arms": {s.key: {"label": s.label, "needs_surveyed_model": s.needs_surveyed_model,
                             "operational_inputs": list(s.operational_inputs)} for s in C.SOURCES},
            "link": "P_D = sigmoid(a * logit(v) + b), fitted on training folds only",
            "mono_depth_model": C.MONO_DEPTH_MODEL,
            "refit_arms": {
                "arms": sorted(REFIT_ARMS),
                "why": (
                    "The cached GP fields are fitted on every event including the held-out "
                    "block. These arms are refitted from each fold's training events, and "
                    "their calibration link is fitted on inner out-of-sample predictions."
                ),
                "predictions": str(GP_PREDICTIONS.relative_to(C.REPO)),
                "predictions_sha256": C.sha256(GP_PREDICTIONS),
            },
            "prevalence_per_camera": apparatus.prevalence,
            "evaluation_only_inputs": [],
            "inputs_sha256": C.input_manifest(),
        },
    )

    print(f"wrote {out}/e1_folds.csv ({len(fold_rows)} rows)")
    print(f"wrote {out}/e1_summary.csv ({len(summary_rows)} rows)")
    print(f"wrote {out}/e1_paired.csv ({len(paired_rows)} rows)")

    print("\nPooled held-out means (linked variant), 24 camera-fold units per arm:")
    header = f"{'arm':<34}{'survey?':<9}{'Brier':>9}{'logloss':>10}{'AUROC':>8}{'ECE':>8}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        if row["variant"] != "linked":
            continue
        print(
            f"{row['label']:<34}{row['needs_surveyed_model']:<9}"
            f"{float(row['brier_mean']):>9.4f}{float(row['logloss_mean']):>10.4f}"
            f"{float(row['auroc_mean']):>8.4f}{float(row['ece_mean']):>8.4f}"
        )


if __name__ == "__main__":
    main()
