#!/usr/bin/env python3
"""CLI: P4 learned observability GP vs baselines (leave-one-route-out), Gate 4.

    python3 scripts/reliability/run_observation_gp.py \
        --dataset   logs/studies/usable_observation/dataset_v1/observations.parquet \
        --baselines logs/studies/usable_observation/baselines_v1/baseline_results.json \
        --output    logs/studies/usable_observation/gp_v1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in (_ROOT / "src").glob("*"):
    if (_pkg / _pkg.name).is_dir():
        sys.path.insert(0, str(_pkg))

import numpy as np  # noqa: E402

from reliability.observation_baselines import (  # noqa: E402
    bootstrap_ci_by_run,
    leave_one_route_out,
    _metrics,
)
from reliability.observation_gp import ObservabilityGP, TwoStageGP  # noqa: E402

HARD_ROUTE = "route_west_to_a1_upper"


def _two_stage_loro(df, resolution_m, length_scale, noise_var, pseudocount):
    routes = sorted(df["route_id"].unique())
    oof_y = np.full(len(df), np.nan)
    oof_p = np.full(len(df), np.nan)
    per_route = {}
    idx = {r: np.where(df["route_id"].to_numpy() == r)[0] for r in routes}
    for held in routes:
        tr = df["route_id"].to_numpy() != held
        te = idx[held]
        model = TwoStageGP(resolution_m=resolution_m, length_scale=length_scale,
                           noise_var=noise_var, pseudocount=pseudocount).fit(
            df.loc[tr, ["state_x", "state_y"]].to_numpy(),
            df.loc[tr, "detection_label"].to_numpy().astype(float),
            df.loc[tr, "quality_label"].to_numpy().astype(float),
        )
        xy_te = df.iloc[te][["state_x", "state_y"]].to_numpy()
        p_te = model.predict_proba(xy_te)
        y_te = df.iloc[te]["usable_label"].to_numpy().astype(float)
        oof_y[te] = y_te
        oof_p[te] = p_te
        per_route[held] = _metrics(y_te, p_te)
    valid = ~np.isnan(oof_p)
    return {"pooled": _metrics(oof_y[valid], oof_p[valid]), "per_route": per_route,
            "oof_y": oof_y, "oof_p": oof_p}


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
    ax.scatter(xs, ys, s=[max(12, n / 20) for n in ns], color="#7b3f9e")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical rate")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _p_use_map(df, path, length_scale, resolution_m, noise_var, pseudocount):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = df[["state_x", "state_y"]].to_numpy()
    model = ObservabilityGP(length_scale=length_scale, resolution_m=resolution_m,
                            noise_var=noise_var, pseudocount=pseudocount).fit(
        xy, df["usable_label"].to_numpy().astype(float))
    xg = np.linspace(xy[:, 0].min() - 0.5, xy[:, 0].max() + 0.5, 90)
    yg = np.linspace(xy[:, 1].min() - 0.5, xy[:, 1].max() + 0.5, 60)
    XX, YY = np.meshgrid(xg, yg)
    P = model.predict_proba(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.pcolormesh(XX, YY, P, cmap="viridis", vmin=0, vmax=1, shading="auto")
    ax.scatter(xy[:, 0], xy[:, 1], s=1, c="white", alpha=0.06)
    ax.set_title("learned p_use(s)  GP (fit on all runs)  [BELIEF x,y]")
    ax.set_xlabel("belief x [m]"); ax.set_ylabel("belief y [m]")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="logs/studies/usable_observation/dataset_v1/observations.parquet")
    ap.add_argument("--baselines", default="logs/studies/usable_observation/baselines_v1/baseline_results.json")
    ap.add_argument("--output", default="logs/studies/usable_observation/gp_v1")
    ap.add_argument("--length-scales", nargs="*", type=float, default=[0.75, 1.0, 1.5])
    ap.add_argument("--resolution-m", type=float, default=0.5)
    ap.add_argument("--noise-var", type=float, default=0.05)
    ap.add_argument("--pseudocount", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    baselines = json.load(open(args.baselines)) if pathlib.Path(args.baselines).exists() else {}

    kw = dict(resolution_m=args.resolution_m, noise_var=args.noise_var, pseudocount=args.pseudocount)
    results: dict = {"dataset": args.dataset, "length_scales": args.length_scales, "hard_route": HARD_ROUTE,
                     "direct_p_det": {}, "direct_p_use": {}, "two_stage_p_use": {}}

    # length-scale sensitivity on the primary target
    for ls in args.length_scales:
        ev = leave_one_route_out(df, lambda ls=ls: ObservabilityGP(length_scale=ls, **kw), "detection_label")
        ci = bootstrap_ci_by_run(df, ev["oof_y"], ev["oof_p"], metric="brier", n_boot=args.n_boot)
        results["direct_p_det"][f"ls={ls}"] = {
            "pooled": ev["pooled"], "hard_route": ev["per_route"][HARD_ROUTE],
            "per_route": ev["per_route"], "brier_ci95_by_run": ci,
        }
        if ls == 1.0:
            _reliability_fig(ev["oof_y"], ev["oof_p"], out / "reliability_GP_p_det.png", "GP p_det (LORO, ls=1.0)")

    # direct p_use and two-stage product at ls=1.0
    ls = 1.0
    ev_use = leave_one_route_out(df, lambda: ObservabilityGP(length_scale=ls, **kw), "usable_label")
    results["direct_p_use"] = {"pooled": ev_use["pooled"], "hard_route": ev_use["per_route"][HARD_ROUTE],
                               "per_route": ev_use["per_route"]}
    ev_ts = _two_stage_loro(df, args.resolution_m, ls, args.noise_var, args.pseudocount)
    results["two_stage_p_use"] = {"pooled": ev_ts["pooled"], "hard_route": ev_ts["per_route"][HARD_ROUTE],
                                  "per_route": ev_ts["per_route"]}
    _reliability_fig(ev_ts["oof_y"], ev_ts["oof_p"], out / "reliability_GP_two_stage_p_use.png",
                     "two-stage p_det*p_qual (LORO)")
    _p_use_map(df, out / "map_gp_p_use.png", ls, args.resolution_m, args.noise_var, args.pseudocount)

    # Gate 4 comparison on the hard route (generalization), p_det
    best_ls = min(results["direct_p_det"], key=lambda k: results["direct_p_det"][k]["hard_route"]["brier"])
    gp_hard = results["direct_p_det"][best_ls]["hard_route"]["brier"]
    gp_pooled = results["direct_p_det"][best_ls]["pooled"]["brier"]
    base_hard = {}
    if baselines:
        b = baselines["targets"]["detection_label"]["baselines"]
        base_hard = {n: e["per_route"][HARD_ROUTE]["brier"] for n, e in b.items()}
    best_base_name = min(base_hard, key=base_hard.get) if base_hard else None
    best_base_hard = base_hard.get(best_base_name) if best_base_name else float("nan")
    smooth_base_hard = min(base_hard.get("B1_distance_logistic", np.inf),
                           base_hard.get("B2_fov_range_logistic", np.inf)) if base_hard else float("nan")
    results["gate4"] = {
        "metric": "Brier on held-out hard route (generalization)",
        "gp_best_ls": best_ls, "gp_hard_route_brier": gp_hard, "gp_pooled_brier": gp_pooled,
        "baseline_hard_route_brier": base_hard,
        "best_smooth_baseline_hard_route_brier": smooth_base_hard,
        "gp_beats_smooth_baselines_on_hard_route": bool(gp_hard < smooth_base_hard) if base_hard else None,
    }

    with open(out / "gp_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    # markdown
    L = ["# Learned observability GP vs baselines — leave-one-route-out (P4)\n",
         f"dataset `{args.dataset}` · rows {len(df)} · canonical GP reused from fit_belief_aware_gp.py\n",
         "\n## p_det — length-scale sensitivity (Brier)\n",
         "| model | pooled | **hard route** | Brier 95% CI (by run) |", "|---|---|---|---|"]
    for k, e in results["direct_p_det"].items():
        ci = e["brier_ci95_by_run"]
        L.append(f"| GP p_det {k} | {e['pooled']['brier']:.4f} | **{e['hard_route']['brier']:.4f}** | "
                 f"[{ci['lo95']:.4f}, {ci['hi95']:.4f}] |")
    if base_hard:
        L.append("\n## Gate 4 — generalization to the hard route (p_det Brier)\n")
        L.append("| model | hard-route Brier |")
        L.append("|---|---|")
        for n, v in sorted(base_hard.items()):
            L.append(f"| {n} | {v:.4f} |")
        L.append(f"| **GP ({best_ls})** | **{gp_hard:.4f}** |")
    L.append("\n## p_use — direct vs two-stage product (ls=1.0, Brier)\n")
    L.append("| model | pooled | hard route |")
    L.append("|---|---|---|")
    L.append(f"| GP direct p_use | {results['direct_p_use']['pooled']['brier']:.4f} | "
             f"{results['direct_p_use']['hard_route']['brier']:.4f} |")
    L.append(f"| GP two-stage p_det*p_qual | {results['two_stage_p_use']['pooled']['brier']:.4f} | "
             f"{results['two_stage_p_use']['hard_route']['brier']:.4f} |")
    (out / "gp_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({"output": str(out), "gate4": results["gate4"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
