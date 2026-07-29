#!/usr/bin/env python3
"""CLI: export the usable-observation dataset + coverage/rate maps (P2).

Reuses the library in ``reliability.observation_exporter``. Method-dev corpus =
single-camera warehouse_aws honest_campaign_v1 (two-world rule).

Example:
    python3 scripts/reliability/export_observation_dataset.py \
        --gate-config config/usable_observation_gate_warehouse_aws.yaml \
        --campaign logs/visibility_comparison/honest_campaign_v1 \
        --output logs/studies/usable_observation/dataset_v1 \
        --holdout-routes route_apron_to_a3_mid
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# put src/* on path when run as a plain script (mirrors conftest for pytest)
_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in (_ROOT / "src").glob("*"):
    if (_pkg / _pkg.name).is_dir():
        sys.path.insert(0, str(_pkg))

from reliability.observation_exporter import ExporterConfig, export_observation_dataset  # noqa: E402
from reliability.observation_gates import UsableObservationGateConfig  # noqa: E402


def _binned_rate(ax, x, y, values, *, bins, extent, title, vmin=0.0, vmax=1.0, cmap="viridis"):
    import numpy as np

    sums, xe, ye = np.histogram2d(x, y, bins=bins, range=extent, weights=values)
    counts, _, _ = np.histogram2d(x, y, bins=bins, range=extent)
    with np.errstate(invalid="ignore"):
        rate = np.where(counts > 0, sums / counts, np.nan)
    im = ax.imshow(
        rate.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
        aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("belief x [m]  (BELIEF)")
    ax.set_ylabel("belief y [m]  (BELIEF)")
    return im


def make_plots(parquet_path: str, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    out = pathlib.Path(out_dir)
    written: list[str] = []

    x = df["state_x"].to_numpy()
    y = df["state_y"].to_numpy()
    pad = 1.0
    extent = [[x.min() - pad, x.max() + pad], [y.min() - pad, y.max() + pad]]
    bins = 28

    # 1. sample density
    fig, ax = plt.subplots(figsize=(6, 4.2))
    counts, xe, ye = np.histogram2d(x, y, bins=bins, range=extent)
    im = ax.imshow(np.log1p(counts.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   aspect="auto", cmap="magma")
    ax.set_title("sample density  log(1+count)  [BELIEF x,y]")
    ax.set_xlabel("belief x [m]"); ax.set_ylabel("belief y [m]")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    p = out / "map_sample_density.png"; fig.savefig(p, dpi=130); plt.close(fig); written.append(str(p))

    # 2-4. rate maps
    det = df["detection_label"].to_numpy().astype(float)
    qual = df["quality_label"].to_numpy().astype(float)
    use = df["usable_label"].to_numpy().astype(float)
    det_mask = df["detection_label"] == 1

    for name, xx, yy, vv, title in [
        ("map_p_det", x, y, det, "p_det  P(detection | s)"),
        ("map_p_qual", x[det_mask.to_numpy()], y[det_mask.to_numpy()], qual[det_mask.to_numpy()],
         "p_qual  P(quality | detection, s)"),
        ("map_p_use", x, y, use, "p_use  P(usable | s)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        im = _binned_rate(ax, xx, yy, vv, bins=bins, extent=extent, title=title)
        fig.colorbar(im, ax=ax); fig.tight_layout()
        p = out / f"{name}.png"; fig.savefig(p, dpi=130); plt.close(fig); written.append(str(p))

    # 5. failure histogram
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vc = df["failure_reason"].value_counts()
    ax.bar(vc.index, vc.values, color="#c05a3c")
    ax.set_title("failure_reason histogram"); ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    p = out / "failure_histogram.png"; fig.savefig(p, dpi=130); plt.close(fig); written.append(str(p))

    # 6. confidence distribution accepted vs rejected detections
    fig, ax = plt.subplots(figsize=(6, 3.8))
    conf = df["detector_confidence"]
    acc = df[(df["usable_label"] == 1)]["detector_confidence"].dropna()
    rej = df[(df["detection_label"] == 1) & (df["usable_label"] == 0)]["detector_confidence"].dropna()
    bins_c = np.linspace(0, 1, 41)
    ax.hist(acc, bins=bins_c, alpha=0.6, label=f"usable (n={len(acc)})", color="#2a7f62")
    ax.hist(rej, bins=bins_c, alpha=0.6, label=f"detected-but-unusable (n={len(rej)})", color="#c05a3c")
    ax.set_title("yolo_score_raw among detections (PIXEL)")
    ax.set_xlabel("detector_confidence"); ax.set_ylabel("count"); ax.legend()
    fig.tight_layout()
    p = out / "confidence_distribution.png"; fig.savefig(p, dpi=130); plt.close(fig); written.append(str(p))

    # 7. image-position of selected pixel, colored by usable
    fig, ax = plt.subplots(figsize=(6, 3.8))
    d = df[df["detection_label"] == 1]
    sc = ax.scatter(d["selected_pixel_u"], d["selected_pixel_v"], c=d["usable_label"],
                    cmap="RdYlGn", s=6, alpha=0.5, vmin=0, vmax=1)
    ax.set_title("selected pixel (bottom-centre), colour=usable  [PIXEL]")
    ax.set_xlabel("u [px]"); ax.set_ylabel("v [px]"); ax.invert_yaxis()
    fig.colorbar(sc, ax=ax); fig.tight_layout()
    p = out / "image_position_usable.png"; fig.savefig(p, dpi=130); plt.close(fig); written.append(str(p))

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", default="logs/visibility_comparison/honest_campaign_v1")
    ap.add_argument("--gate-config", default="config/usable_observation_gate_warehouse_aws.yaml")
    ap.add_argument("--output", default="logs/studies/usable_observation/dataset_v1")
    ap.add_argument("--holdout-routes", nargs="*", default=["route_apron_to_a3_mid"])
    ap.add_argument("--min-spacing-s", type=float, default=0.0)
    ap.add_argument("--detection-floor", type=float, default=0.05)
    ap.add_argument("--camera-id", default="external_camera_aws")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    gate = UsableObservationGateConfig.from_yaml(args.gate_config)
    exporter_cfg = ExporterConfig(
        camera_id=args.camera_id,
        detection_floor=args.detection_floor,
        min_spacing_s=args.min_spacing_s,
        holdout_routes=tuple(args.holdout_routes),
    )
    manifest = export_observation_dataset(
        args.campaign, gate, exporter_cfg, args.output, write_csv=not args.no_csv
    )

    plots: list[str] = []
    if not args.no_plots:
        plots = make_plots(manifest["artifacts"]["parquet"], args.output)
    manifest["artifacts"]["plots"] = plots
    with open(pathlib.Path(args.output) / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(json.dumps({
        "rows": manifest["row_count"],
        "runs": manifest["runs"],
        "routes": manifest["routes"],
        "class_balance": manifest["class_balance"],
        "failure_reason_counts": manifest["failure_reason_counts"],
        "gt_firewall_passed": manifest["gt_firewall_audit"]["passed"],
        "gate_config_hash": manifest["gate_config_hash"],
        "output": args.output,
        "plots": len(plots),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
