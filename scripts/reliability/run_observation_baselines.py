#!/usr/bin/env python3
"""CLI: run P3 observability baselines with leave-one-route-out eval on dataset_v1.

    python3 scripts/reliability/run_observation_baselines.py \
        --dataset logs/studies/usable_observation/dataset_v1/observations.parquet \
        --output  logs/studies/usable_observation/baselines_v1
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in (_ROOT / "src").glob("*"):
    if (_pkg / _pkg.name).is_dir():
        sys.path.insert(0, str(_pkg))

import numpy as np  # noqa: E402

from reliability.observation_baselines import (  # noqa: E402
    AWS_CAM_POS,
    DistanceLogistic,
    FovRangeLogistic,
    GlobalConstant,
    GridFrequency,
    aws_camera_model,
    bootstrap_ci_by_run,
    leave_one_route_out,
)

FACTORIES = {
    "B0_constant": GlobalConstant,
    "B1_distance_logistic": DistanceLogistic,
    "B2_fov_range_logistic": FovRangeLogistic,
    "B3_grid_frequency": GridFrequency,
}


def _projection_sanity(df) -> dict[str, float]:
    """Median pixel error between the calibrated camera model and the logged selected
    pixel, on usable detections — validates B2's calibration is meaningful (not GT)."""
    cam = aws_camera_model()
    d = df[(df["usable_label"] == 1) & df["selected_pixel_u"].notna()]
    errs = []
    for _, r in d.iterrows():
        u, v, _vis = cam.world_to_pixel(float(r["state_x"]), float(r["state_y"]), 0.0)
        errs.append(math.hypot(u - float(r["selected_pixel_u"]), v - float(r["selected_pixel_v"])))
    errs = np.array(errs) if errs else np.array([np.nan])
    return {"n": int(len(d)), "median_px": float(np.nanmedian(errs)), "p90_px": float(np.nanpercentile(errs, 90))}


def _reliability_fig(oof_y, oof_p, path, title, bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = ~np.isnan(oof_p)
    y, p = oof_y[valid], oof_p[valid]
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.scatter(xs, ys, s=[max(12, n / 20) for n in ns], color="#2a6f97")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical rate")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="logs/studies/usable_observation/dataset_v1/observations.parquet")
    ap.add_argument("--output", default="logs/studies/usable_observation/baselines_v1")
    ap.add_argument("--targets", nargs="*", default=["detection_label", "usable_label"])
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "dataset": args.dataset,
        "camera_pos": list(AWS_CAM_POS),
        "n_rows": int(len(df)),
        "routes": sorted(df["route_id"].unique().tolist()),
        "projection_sanity_px": _projection_sanity(df),
        "targets": {},
    }

    for target in args.targets:
        prevalence = float(df[target].mean())
        tgt = {"prevalence": prevalence, "baselines": {}}
        for name, factory in FACTORIES.items():
            ev = leave_one_route_out(df, factory, target)
            ci = bootstrap_ci_by_run(df, ev["oof_y"], ev["oof_p"], metric="brier", n_boot=args.n_boot)
            entry = {"pooled": ev["pooled"], "per_route": ev["per_route"], "brier_ci95_by_run": ci}
            if name == "B3_grid_frequency":
                model = factory().fit(df[["state_x", "state_y"]].to_numpy(), df[target].to_numpy().astype(float))
                entry["sparse_fraction"] = model.sparse_fraction(df[["state_x", "state_y"]].to_numpy())
            tgt["baselines"][name] = entry
            # reliability fig for each baseline on the primary target
            if target == "detection_label":
                _reliability_fig(ev["oof_y"], ev["oof_p"], out / f"reliability_{name}.png",
                                 f"{name}  p_det (LORO)")
        results["targets"][target] = tgt

    with open(out / "baseline_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    # markdown results table
    lines = ["# Observability baselines — leave-one-route-out\n",
             f"dataset: `{args.dataset}`  ·  rows: {len(df)}  ·  "
             f"projection sanity (B2 calib vs logged pixel): "
             f"median {results['projection_sanity_px']['median_px']:.1f}px\n"]
    for target, tgt in results["targets"].items():
        lines.append(f"\n## {target}  (prevalence {tgt['prevalence']:.3f})\n")
        lines.append("| baseline | Brier | NLL | ECE | AUROC | AUPRC | Brier 95% CI (by run) |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, e in tgt["baselines"].items():
            pl = e["pooled"]; ci = e["brier_ci95_by_run"]
            lines.append(
                f"| {name} | {pl['brier']:.4f} | {pl['nll']:.4f} | {pl['ece']:.4f} | "
                f"{pl['auroc']:.3f} | {pl['auprc']:.3f} | [{ci['lo95']:.4f}, {ci['hi95']:.4f}] |"
            )
    (out / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(out),
        "projection_sanity_px": results["projection_sanity_px"],
        "detection_label": {n: round(e["pooled"]["brier"], 4)
                            for n, e in results["targets"]["detection_label"]["baselines"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
