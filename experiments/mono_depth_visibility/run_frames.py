#!/usr/bin/env python3
"""Run ground anchoring + visibility inference over a scenario's frames.

Consumes what agents 1 and 2 emit and writes this method's output contract:

    agent 1: scenario manifest of frame records (calibration, timestamps)
    agent 2: one depth-prediction .npz per (frame, camera)
        -> logs/studies/mono_depth_visibility/<run>/<camera>_<frame>.npz + .json
        -> index.csv, one row per frame, for the event timeline

Oracle depth, simulator obstacle poses and oracle visibility grids are never
read: the loader whitelists the method-visible keys and records the rest as
withheld. Scoring against the oracle happens in a separate evaluator that is
allowed to open both.

Usage::

    python3 experiments/mono_depth_visibility/run_frames.py \\
        --manifest logs/studies/<scenario>/frames.json \\
        --predictions logs/studies/<scenario>/depth/<model> \\
        --out logs/studies/mono_depth_visibility/<run>

The manifest is JSON: either a list of frame records, or an object with a
``frames`` list plus optional ``drivable`` footprints, ``floor_plane`` and
``grid`` settings. A frame record follows agent 1's contract; this script uses
``scenario_id``, ``timestamp``, ``camera_id``, ``camera_intrinsics`` and
``camera_extrinsics``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ground_anchoring as ga  # noqa: E402


def _load_manifest(path: pathlib.Path) -> dict:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        payload = {"frames": payload}
    if "frames" not in payload:
        raise ga.ContractViolation(f"{path} has no 'frames' list")
    return payload


def _drivable_from(payload: dict) -> list[ga.Footprint]:
    regions = payload.get("drivable") or payload.get("drivable_footprints") or []
    return [
        ga.Footprint(
            xmin=float(r["xmin"]), xmax=float(r["xmax"]),
            ymin=float(r["ymin"]), ymax=float(r["ymax"]),
            name=str(r.get("name", "region")),
        )
        for r in regions
    ]


def _grid_from(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    g = payload.get("grid", {})
    res = float(g.get("resolution", 0.25))
    x0, x1 = float(g.get("xmin", -1.0)), float(g.get("xmax", 8.0))
    y0, y1 = float(g.get("ymin", -1.0)), float(g.get("ymax", 8.0))
    return np.arange(x0, x1 + res, res), np.arange(y0, y1 + res, res)


def _plane_from(payload: dict) -> ga.FloorPlane:
    p = payload.get("floor_plane")
    if not p:
        return ga.FloorPlane()
    return ga.FloorPlane(normal=np.asarray(p.get("normal", [0, 0, 1]), dtype=float),
                         offset=float(p.get("offset", 0.0)))


def _candidate_image_ids(record: dict) -> list[str]:
    """Every plausible spelling of this frame's id, most specific first.

    The two sides name frames independently: the scenario runner keys on
    ``camera_id`` + timestamp tag, the depth adapter on its own ``image_id``.
    Rather than legislate one, try the obvious combinations and require exactly
    one match.
    """
    cam = str(record.get("camera_id", "") or "")
    stem = pathlib.Path(str(record.get("rgb_path", "") or "")).stem
    explicit = str(record.get("frame_id", "") or "")
    ids = [explicit, f"{cam}_{stem}", f"{stem}_{cam}", stem]
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _prediction_path(
    index: dict[str, list[pathlib.Path]], record: dict, model: str | None
) -> tuple[pathlib.Path | None, str]:
    """Resolve one frame record to one prediction sidecar."""
    for image_id in _candidate_image_ids(record):
        hits = index.get(image_id, [])
        if model:
            hits = [h for h in hits if h.stem.endswith(f"__{model}")]
        if len(hits) == 1:
            return hits[0], ""
        if len(hits) > 1:
            names = ", ".join(sorted(h.name for h in hits))
            return None, f"image id {image_id!r} has {len(hits)} predictions ({names}); use --model"
    return None, f"no prediction for any of {_candidate_image_ids(record)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=pathlib.Path,
                    help="agent-1 scenario manifest (JSON)")
    ap.add_argument("--predictions", required=True, type=pathlib.Path,
                    help="directory of agent-2 depth predictions (.npz)")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="output directory")
    ap.add_argument("--strict", action="store_true", default=True,
                    help="raise on a depth-convention mismatch (default)")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="downgrade a convention mismatch to an all-unknown frame")
    ap.add_argument("--model", default=None,
                    help="depth-adapter model_name to use when a directory holds several")
    ap.add_argument(
        "--enhanced-anchors", action="store_true",
        help="erode floor masks, reject depth edges, and use model confidence/uncertainty",
    )
    ap.add_argument(
        "--temporal-anchoring", action="store_true",
        help="Bayesian-filter scale/shift independently per camera and model",
    )
    ap.add_argument("--limit", type=int, default=0, help="process at most N frames")
    args = ap.parse_args(argv)

    payload = _load_manifest(args.manifest)
    frames = payload["frames"][: args.limit or None]
    drivable = _drivable_from(payload)
    xs, ys = _grid_from(payload)
    plane = _plane_from(payload)
    cfg = ga.MethodConfig(
        anchors=ga.AnchorConfig(quality_filter=args.enhanced_anchors),
        fit=ga.FitConfig(strict_convention=args.strict),
    )
    temporal_filter = ga.TemporalGroundAnchorFilter() if args.temporal_anchoring else None

    args.out.mkdir(parents=True, exist_ok=True)
    index = ga.prediction_index(args.predictions)
    rows: list[dict] = []
    skipped: list[str] = []

    for record in frames:
        visible = ga.method_visible_record(record)
        pred_path, why = _prediction_path(index, record, args.model)
        if pred_path is None:
            skipped.append(f"{record.get('camera_id')} t={record.get('timestamp')}: {why}")
            continue
        pred = ga.load_prediction(pred_path)
        calib = ga.camera_from_record(record)

        t0 = time.perf_counter()
        result = ga.estimate_visibility(
            pred, calib, xs, ys,
            plane=plane, drivable=drivable, config=cfg,
            temporal_filter=temporal_filter,
            scenario_id=str(visible.get("scenario_id", "")),
            frame_id=str(record.get("frame_id", "") or pred.frame_id),
            timestamp=float(visible.get("timestamp", float("nan"))),
            extra_provenance={
                "prediction_path": str(pred_path),
                "withheld_record_keys": visible["_withheld"],
            },
        )
        elapsed = time.perf_counter() - t0
        ga.save_result(result, args.out)

        f = result.visibility
        rows.append({
            "scenario_id": result.scenario_id,
            "timestamp": result.timestamp,
            "camera_id": result.camera_id,
            "frame_id": result.frame_id,
            "status": result.status.value,
            "model_name": pred.model_name,
            "depth_convention": pred.convention.value,
            "scale": result.ground_fit.scale,
            "shift": result.ground_fit.shift,
            "n_anchor": result.ground_fit.n_anchor,
            "inlier_fraction": result.ground_fit.inlier_fraction,
            "residual_rms_m": result.ground_fit.residual_rms_m,
            "anchor_depth_span_m": result.ground_fit.anchor_depth_span_m,
            "n_shorter_than_floor": result.ground_fit.n_shorter_than_floor,
            "mean_p_visible": float(np.mean(f.p_visible)),
            "mean_p_occluded": float(np.mean(f.p_occluded)),
            "mean_p_unknown": float(np.mean(f.p_unknown)),
            "unobserved_in_fov_fraction": result.provenance.get(
                "unobserved_in_fov_fraction", float("nan")),
            "method_seconds": elapsed,
            "anchor_selection": "enhanced" if args.enhanced_anchors else "legacy",
            "temporal_anchor_mode": result.provenance.get("temporal_anchor", {}).get(
                "mode", "disabled"
            ),
        })
        print(f"{result.camera_id} {result.frame_id} t={result.timestamp:>8.3f} "
              f"{result.status.value:<28s} visible={np.mean(f.p_visible):.3f} "
              f"occluded={np.mean(f.p_occluded):.3f} unknown={np.mean(f.p_unknown):.3f}")

    if rows:
        index = args.out / "index.csv"
        with index.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} frames -> {index}")
        refused = [r for r in rows if r["status"] != "ok"]
        if refused:
            print(f"REFUSED {len(refused)}/{len(rows)} frames:")
            for r in refused:
                print(f"  {r['camera_id']} {r['frame_id']}: {r['status']}")
    else:
        print("no frames processed")
    for note in skipped:
        print(f"SKIPPED {note}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
