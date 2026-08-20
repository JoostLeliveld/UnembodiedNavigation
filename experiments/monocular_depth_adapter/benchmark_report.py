#!/usr/bin/env python3
"""Compare the depth models on several axes at once, and refuse to pick a winner.

    python3 experiments/monocular_depth_adapter/benchmark_report.py --run bs1_native_flip

A single depth RMSE would rank these models, and the ranking would not survive
contact with what they are actually for. So this report keeps the axes apart and
prints them side by side:

1. What it returns      convention, output size, fraction of usable pixels.
2. What it costs        median forward time per frame, peak GPU memory, weights.
3. Is it stable         flip-consistency spread and, per fixed camera, spread
                        across frames of a scene that never moved.
4. Do they agree        pairwise agreement between the metric models in metres,
                        and rank agreement (Spearman) for every pair including
                        the non-metric ones, which is the only comparison that
                        survives an unknown scale.
5. Does it know when
   it is wrong          rank correlation between a model's own confidence and
                        where the models disagree. A confidence head that
                        does not track disagreement is decoration.
6. What it thinks the
   camera is            models that estimate their own intrinsics, against the
                        calibration they were handed. A large gap is a direct
                        symptom of an out-of-distribution viewpoint.

No ground truth is read anywhere. There is none in the frozen set, and the point
of this stage is to establish that the plumbing and the cost are right, not to
grade accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import frozen_set as fs
import numpy as np

sys.path.insert(0, str(fs.REPO / "scripts" / "shared"))
from metrics import spearman as _spearman  # noqa: E402  THE shared one; never re-derive


def spearman(a, b) -> float:
    """The shared implementation returns ``(rho, n_finite)``; this study wants rho."""
    return float(_spearman(a, b)[0])

from monodepth import storage  # noqa: E402
from monodepth.conventions import align_affine  # noqa: E402
from monodepth.types import DepthConvention  # noqa: E402

OUT_ROOT = fs.REPO / "logs/studies/monocular_depth_adapter"

#: Rank correlations are computed on a strided pixel subsample. Full-frame
#: Spearman on 921 600 points per pair costs minutes and moves the third decimal.
RANK_STRIDE = 8


def _load_run(run_dir: Path) -> dict:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    runs = {}
    for model_name, entry in index["models"].items():
        model_dir = run_dir / entry["dir"]
        manifest = json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8"))
        runs[model_name] = {"manifest": manifest, "dir": model_dir}
    return {"index": index, "runs": runs}


def _cost_table(runs: dict) -> list[dict]:
    rows = []
    for model_name, blob in sorted(runs.items()):
        m = blob["manifest"]
        frames = m["per_frame"]
        forwards = np.array([f["timing"]["forward_s"] for f in frames])
        totals = np.array([f["timing"]["total_s"] for f in frames])
        rows.append({
            "model": model_name,
            "family": m["model"]["backend"],
            "convention": m["model"]["convention"],
            "params_m": m["model"]["parameter_count"] / 1e6,
            "batch_size": m["config"]["batch_size"],
            "median_forward_s": float(np.median(forwards)),
            "p95_forward_s": float(np.percentile(forwards, 95)),
            "median_uncertainty_s": float(np.median([f["timing"].get("uncertainty_s", 0.0)
                                                     for f in frames])),
            "median_total_s": float(np.median(totals)),
            "frames_per_s": float(1.0 / np.median(totals)) if np.median(totals) > 0 else float("nan"),
            "weights_mib": float(frames[0]["memory"]["weights_mib"]),
            "peak_gpu_mib": float(max(f["memory"]["gpu_peak_allocated_mib"] for f in frames)),
            "peak_host_rss_mib": float(max(f["memory"]["host_rss_mib"] for f in frames)),
            "load_s": float(m["load_seconds"]),
            "mean_valid_fraction": float(np.mean([f["valid_fraction"] for f in frames])),
            "oom_fallbacks": len(m["oom_events"]),
            "uses_intrinsics": bool(m["model"]["uses_intrinsics"]),
            "native_confidence": bool(m["model"]["provides_native_confidence"]),
        })
    return rows


def _stability_table(runs: dict) -> list[dict]:
    rows = []
    for model_name, blob in sorted(runs.items()):
        m = blob["manifest"]
        dev = [f for f in m["per_frame"] if f["role"] == "method_development"]
        kinds = {f["uncertainty_kind"] for f in dev}
        spreads = [f["uncertainty"].get("median") for f in dev
                   if f["uncertainty"].get("available") and f["uncertainty"].get("n_finite")]
        temporal = m.get("temporal_disagreement", {}).get("cameras", {})
        rows.append({
            "model": model_name,
            "convention": m["model"]["convention"],
            "uncertainty_kind": sorted(k for k in kinds if k) or ["none"],
            "median_uncertainty": float(np.median(spreads)) if spreads else float("nan"),
            "temporal_median_by_camera": {
                k: v["median_spread"] for k, v in sorted(temporal.items())
            },
        })
    return rows


def _agreement(runs: dict, role: str = "method_development") -> dict:
    """Pairwise agreement between models, on the frames they share."""
    preds: dict[str, dict[str, object]] = {}
    conventions: dict[str, DepthConvention] = {}
    for model_name, blob in sorted(runs.items()):
        loaded = {}
        for p in storage.load_dir(blob["dir"]):
            loaded[p.image_id] = p
            conventions[model_name] = p.convention
        wanted = {f["frame_id"] for f in blob["manifest"]["per_frame"] if f["role"] == role}
        preds[model_name] = {k: v for k, v in loaded.items() if k in wanted}

    frame_ids = sorted(set.intersection(*[set(v) for v in preds.values()])) if preds else []
    pairs = []
    for a, b in combinations(sorted(preds), 2):
        abs_diffs, rank_rhos, residuals, scales = [], [], [], []
        for fid in frame_ids:
            pa, pb = preds[a][fid], preds[b][fid]
            mask = pa.valid & pb.valid                      # type: ignore[union-attr]
            if not mask.any():
                continue
            da = pa.depth[mask]                              # type: ignore[union-attr]
            db = pb.depth[mask]                              # type: ignore[union-attr]
            if conventions[a].is_metric and conventions[b].is_metric:
                abs_diffs.append(float(np.median(np.abs(da - db))))
            sub_a, sub_b = da[::RANK_STRIDE], db[::RANK_STRIDE]
            if conventions[a].larger_is_nearer != conventions[b].larger_is_nearer:
                sub_b = -sub_b     # put both on a "larger = further" axis before ranking
            rank_rhos.append(float(spearman(sub_a, sub_b)))

            # Separates "they see a different scene" from "they see the same
            # scene at a different scale". One global scale+shift per frame is
            # the cheapest possible reconciliation; whatever survives it is
            # structural disagreement, which no anchoring downstream can fix.
            scale, shift = align_affine(pb.depth, pa.depth, mask)  # type: ignore[union-attr]
            residuals.append(float(np.median(np.abs(da - (scale * db + shift)))))
            scales.append(abs(scale))
        pairs.append({
            "model_a": a, "model_b": b,
            "convention_a": conventions[a].value, "convention_b": conventions[b].value,
            "n_frames": len(frame_ids),
            "median_abs_difference_m": float(np.median(abs_diffs)) if abs_diffs else None,
            "median_rank_agreement": float(np.median(rank_rhos)) if rank_rhos else None,
            "median_residual_after_scale_shift": float(np.median(residuals)) if residuals else None,
            "median_scale_ratio": float(np.median(scales)) if scales else None,
            "residual_unit": "m" if (conventions[a].is_metric and conventions[b].is_metric) else "unitless",
        })
    return {"role": role, "frame_ids": frame_ids, "pairs": pairs,
            "reading": ("a high rank agreement with a large raw gap and a small post-affine "
                        "residual means the models see the same geometry at different scales — "
                        "recoverable with an anchor. A large residual would mean they disagree "
                        "about the scene itself, which an anchor cannot fix.")}


def _confidence_tracks_disagreement(runs: dict, role: str = "method_development") -> list[dict]:
    """Does a model's own confidence go down where the models disagree?

    Disagreement per pixel is the spread of the metric models' predictions,
    which is the closest thing to an error signal available without truth. A
    useful confidence head should rank-correlate NEGATIVELY with it.
    """
    loaded: dict[str, dict] = {}
    conventions: dict[str, DepthConvention] = {}
    for model_name, blob in sorted(runs.items()):
        wanted = {f["frame_id"] for f in blob["manifest"]["per_frame"] if f["role"] == role}
        loaded[model_name] = {p.image_id: p for p in storage.load_dir(blob["dir"])
                              if p.image_id in wanted}
        if loaded[model_name]:
            conventions[model_name] = next(iter(loaded[model_name].values())).convention

    metric_models = [m for m, c in conventions.items() if c.is_metric]
    if len(metric_models) < 2:
        return []

    frame_ids = sorted(set.intersection(*[set(loaded[m]) for m in metric_models]))
    rows = []
    for model_name in sorted(loaded):
        rhos = []
        for fid in frame_ids:
            pred = loaded[model_name].get(fid)
            if pred is None or pred.native_confidence is None:
                continue
            stack, masks = [], []
            for other in metric_models:
                p = loaded[other][fid]
                stack.append(p.depth.astype(np.float64))
                masks.append(p.valid)
            disagreement = np.std(np.stack(stack), axis=0)
            mask = np.logical_and.reduce(masks) & pred.valid
            if mask.sum() < 100:
                continue
            rhos.append(float(spearman(pred.native_confidence[mask][::RANK_STRIDE],
                                       disagreement[mask][::RANK_STRIDE])))
        if rhos:
            rows.append({
                "model": model_name,
                "n_frames": len(rhos),
                "median_spearman_confidence_vs_disagreement": float(np.median(rhos)),
                "reference_models": metric_models,
                "reading": "negative = confidence drops where the models disagree, which is what you want",
            })
    return rows


def _self_estimated_camera(runs: dict) -> list[dict]:
    rows = []
    for model_name, blob in sorted(runs.items()):
        vals = [(f["extras"].get("supplied_fx_px"), f["extras"].get("model_self_estimated_fx_px"))
                for f in blob["manifest"]["per_frame"]
                if "model_self_estimated_fx_px" in f.get("extras", {})]
        if not vals:
            continue
        supplied = np.array([v[0] for v in vals], dtype=float)
        estimated = np.array([v[1] for v in vals], dtype=float)
        rows.append({
            "model": model_name,
            "n_frames": len(vals),
            "supplied_fx_px": float(np.median(supplied)),
            "self_estimated_fx_px": float(np.median(estimated)),
            "ratio": float(np.median(estimated / supplied)),
            "reading": ("the model's own guess at the focal length, against the calibration it "
                        "was given; far from 1.0 means this viewpoint is unlike its training data"),
        })
    return rows


def _fmt(value, spec=".3f", width=0) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        text = "-"
    else:
        text = format(value, spec)
    return text.rjust(width) if width else text


def print_report(report: dict) -> None:
    print(f"\nmonocular depth model benchmark — run {report['run_name']!r}, "
          f"frozen set {report['frozen_set']!r}")
    print(f"device: {report['environment'].get('device_name')} "
          f"({report['environment'].get('device_total_mib', float('nan')):.0f} MiB), "
          f"torch {report['environment'].get('torch')}")

    print("\n1-2. what it returns, and what it costs")
    head = (f"{'model':<28}{'convention':<17}{'par M':>7}{'fwd s':>8}{'+unc s':>8}"
            f"{'frame/s':>9}{'weights':>9}{'peak GPU':>10}{'valid':>8}{'K?':>4}{'conf?':>6}")
    print(head)
    print("-" * len(head))
    for row in report["cost"]:
        print(f"{row['model']:<28}{row['convention']:<17}{row['params_m']:>7.1f}"
              f"{row['median_forward_s']:>8.3f}{row['median_uncertainty_s']:>8.3f}"
              f"{row['frames_per_s']:>9.2f}"
              f"{row['weights_mib']:>8.0f}M{row['peak_gpu_mib']:>9.0f}M"
              f"{row['mean_valid_fraction']:>8.4f}"
              f"{'y' if row['uses_intrinsics'] else 'n':>4}"
              f"{'y' if row['native_confidence'] else 'n':>6}")
    print("  ('+unc s' is the extra forward pass the flip-consistency signal costs; "
          "frame/s includes it)")

    print("\n3. is it stable (spread in each model's OWN units — compare metric with metric only)")
    for row in report["stability"]:
        temporal = ", ".join(f"{k} {v:.3f}" for k, v in row["temporal_median_by_camera"].items())
        print(f"  {row['model']:<28} {'/'.join(row['uncertainty_kind']):<20} "
              f"median {_fmt(row['median_uncertainty'])}   per-camera over time: {temporal or '-'}")

    print("\n4. do they agree (development frames only)")
    for pair in report["agreement"]["pairs"]:
        print(f"  {pair['model_a']:<28} vs {pair['model_b']:<28} "
              f"raw gap {_fmt(pair['median_abs_difference_m']):>7} m   "
              f"rank agreement {_fmt(pair['median_rank_agreement']):>7}   "
              f"left over after one scale+shift "
              f"{_fmt(pair['median_residual_after_scale_shift']):>6} {pair['residual_unit']}"
              f"  (scale x{_fmt(pair['median_scale_ratio'], '.2f')})")
    print("  " + report["agreement"]["reading"])

    if report["confidence_vs_disagreement"]:
        print("\n5. does its confidence know where the models disagree "
              "(want negative; 0 means the confidence head carries no information here)")
        for row in report["confidence_vs_disagreement"]:
            print(f"  {row['model']:<28} spearman "
                  f"{row['median_spearman_confidence_vs_disagreement']:+.3f}  "
                  f"over {row['n_frames']} frames")

    if report["self_estimated_camera"]:
        print("\n6. what it thinks the camera is")
        for row in report["self_estimated_camera"]:
            print(f"  {row['model']:<28} supplied fx {row['supplied_fx_px']:.0f} px, "
                  f"self-estimated {row['self_estimated_fx_px']:.0f} px "
                  f"({row['ratio']:.2f}x)")

    print("\nNo winner is declared here. These axes disagree with each other by design, and "
          "\nnone of them is accuracy — the frozen set carries no depth labels. Picking an "
          "\noperational model needs the downstream task's own metric.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run subdirectory under logs/studies/monocular_depth_adapter")
    parser.add_argument("--json", action="store_true", help="also print the report as JSON")
    args = parser.parse_args()

    run_dir = OUT_ROOT / args.run
    if not (run_dir / "index.json").is_file():
        print(f"no run at {run_dir}; run run_inference.py first")
        return 1

    blob = _load_run(run_dir)
    runs = blob["runs"]
    any_manifest = next(iter(runs.values()))["manifest"]
    report = {
        "run_name": args.run,
        "frozen_set": blob["index"]["frozen_set"],
        "environment": any_manifest["environment"],
        "cost": _cost_table(runs),
        "stability": _stability_table(runs),
        "agreement": _agreement(runs),
        "confidence_vs_disagreement": _confidence_tracks_disagreement(runs),
        "self_estimated_camera": _self_estimated_camera(runs),
        "failures": blob["index"].get("failures", {}),
        "no_winner_note": (
            "Multi-axis by construction. No ranking is produced from a single depth "
            "error metric, and no ground truth was read."
        ),
    }
    out_path = run_dir / "benchmark.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n",
                        encoding="utf-8")
    print_report(report)
    if report["failures"]:
        print("\nmodels that could not run:")
        for name, why in report["failures"].items():
            print(f"  {name}: {why}")
    print(f"\nwrote {out_path.relative_to(fs.REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
