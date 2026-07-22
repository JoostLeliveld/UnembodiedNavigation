#!/usr/bin/env python3
"""Paper-2 containment go/no-go harness — real capture -> E4 subset sweep + Δ_fault.

Wires the validated real-data bridge (`load_commissioning_run`) to the existing
offline replay sweeps (`replay_sweeps`) so the Paper-2 headline experiments R3/R4
fill in from ONE real handover capture, no synthetic data:

  R3 (E4, nominal)   — every camera subset through the same fusion mode;
                       fusion_gain_p95 = best-single-camera p95 − full-set p95.
  R4 (Δ_fault)       — inject a calibration drift into one camera and compare the
                       full system's error to simply DROPPING that camera:
                         Δ_fault(c, bias) = p95(system with c drifted)
                                          − p95(healthy subset without c).
                       Δ_fault ≤ 0  ⇒ the fusion CONTAINS the fault (keeping-but-
                       down-weighting the bad camera is no worse than dropping it);
                       Δ_fault ≫ 0 ⇒ the bad camera pollutes the estimate.

All conditions replay IDENTICAL detections through ONE fusion pipeline (§21 gate
discipline): any difference is attributable to the transform, never a retuned
filter. Ground truth arrives only as EvaluationFrames (firewall enforced by the
bridge). Cross-run paired CIs are a later step (many runs → campaign_statistics);
a single capture gives point estimates only — stated honestly in the output.

Usage:
  python3 run_containment_pilot.py <capture_run_dir> [--fusion-mode SEQUENTIAL_FUSION]
      [--bias-m 0.2 0.5 1.0] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

import load_commissioning_run as bridge  # noqa: E402
import replay_sweeps as rs  # noqa: E402
from reliability.replay import ReplayConfig, ReplayMode, run_replay  # noqa: E402

DEFAULT_BIAS_LEVELS_M = (0.2, 0.5, 1.0)
DEFAULT_DIRECTION = (1.0, 0.0)


def _p95(frames, evaluation_frames, cfg) -> float:
    result = run_replay(frames, cfg, evaluation_frames=evaluation_frames)
    return result.metrics.p95_error_m if result.metrics is not None else math.nan


def delta_fault_for_camera(
    frames, evaluation_frames, camera_id, cfg, *, bias_levels_m, direction=DEFAULT_DIRECTION
):
    """Δ_fault(c, bias) = p95(system with c drifted) − p95(healthy subset without c)."""
    norm = math.hypot(direction[0], direction[1]) or 1.0
    ux, uy = direction[0] / norm, direction[1] / norm
    p95_dropped = _p95(rs.drop_camera_permanent(frames, camera_id), evaluation_frames, cfg)
    rows = []
    for b in bias_levels_m:
        biased = rs.bias_camera_position(frames, camera_id, (b * ux, b * uy))
        p95_bad = _p95(biased, evaluation_frames, cfg)
        rows.append({
            "bias_m": float(b),
            "p95_with_bad_m": p95_bad,
            "p95_dropped_m": p95_dropped,
            "delta_fault_m": (p95_bad - p95_dropped) if (math.isfinite(p95_bad) and math.isfinite(p95_dropped)) else math.nan,
        })
    return rows


def run_pilot(run_dir, *, fusion_mode=ReplayMode.SEQUENTIAL_FUSION,
              bias_levels_m=DEFAULT_BIAS_LEVELS_M, direction=DEFAULT_DIRECTION,
              projection_calibration=None, world_sdf=None):
    loaded = bridge.load_run(
        run_dir, projection_calibration=projection_calibration, world_sdf=world_sdf
    )
    cfg = ReplayConfig(mode=fusion_mode, nis_gate=9.21)
    cams = tuple(sorted({o.camera_id for fr in loaded.frames for o in fr.observations}))

    # R3 — nominal subset sweep (fusion gain vs best single)
    subset = rs.run_camera_subset_sweep(loaded.frames, loaded.evaluation_frames, config=cfg)
    p95_healthy_full = _p95(loaded.frames, loaded.evaluation_frames, cfg)

    # R4 — Δ_fault per camera across the bias severity axis
    delta = {c: delta_fault_for_camera(loaded.frames, loaded.evaluation_frames, c, cfg,
                                       bias_levels_m=bias_levels_m, direction=direction)
             for c in cams}

    return {
        "run_dir": str(run_dir),
        "fusion_mode": fusion_mode.name,
        "cameras": list(cams),
        "observation_counts": loaded.observation_counts,
        "n_frames": len(loaded.frames),
        "n_eval_frames": len(loaded.evaluation_frames),
        "p95_healthy_full_m": p95_healthy_full,
        "subset": {
            "best_single_camera": subset.best_single_camera,
            "best_single_p95_m": subset.best_single_p95,
            "full_set": subset.full_set,
            "full_set_p95_m": subset.full_set_p95,
            "fusion_gain_p95_m": subset.fusion_gain_p95,
            "per_subset_p95_m": {"|".join(s): m.p95_error_m for s, m in subset.per_subset.items()},
        },
        "delta_fault": delta,
    }


def _fmt(x):
    return "n/a" if (x is None or not math.isfinite(x)) else f"{x:.3f}"


def write_results(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub = res["subset"]
    subset_rows = "\n".join(
        f"| {s} | {_fmt(p)} |" for s, p in sorted(sub["per_subset_p95_m"].items(), key=lambda kv: (len(kv[0]), kv[0]))
    )
    dfault_rows = []
    for c in res["cameras"]:
        for r in res["delta_fault"][c]:
            dfault_rows.append(f"| {c} | {r['bias_m']:.2f} | {_fmt(r['p95_with_bad_m'])} | "
                               f"{_fmt(r['p95_dropped_m'])} | **{_fmt(r['delta_fault_m'])}** |")
    md = f"""# Paper-2 containment pilot — E4 subset sweep + Δ_fault

**Provenance:** real capture `{res['run_dir']}` via `load_commissioning_run`
(firewall enforced). Fusion mode: `{res['fusion_mode']}` @ NIS gate 9.21.
Cameras: {res['cameras']}; frames {res['n_frames']}, eval frames {res['n_eval_frames']};
observations {res['observation_counts']}.

> **Status:** point estimates from a SINGLE capture — plumbing + directional only,
> not paper evidence. Cross-run paired CIs need many captures
> (`reliability.campaign_statistics`). If this run used the v1 (OOD) detector it is
> detector-limited; the evidence run needs the v2 handover capture.

## R3 — nominal localization: fusion gain vs best single (E4)
- best single camera = **{sub['best_single_camera']}** (p95 {_fmt(sub['best_single_p95_m'])} m)
- full set {sub['full_set']} p95 = {_fmt(sub['full_set_p95_m'])} m
- **fusion_gain_p95 = {_fmt(sub['fusion_gain_p95_m'])} m** (best-single − full-set; >0 ⇒ fusion helps)
- healthy full-set p95 (reference) = {_fmt(res['p95_healthy_full_m'])} m

| camera subset | held-out p95 err (m) |
|---|---|
{subset_rows}

## R4 — Δ_fault: does the fusion CONTAIN an injected calibration drift?
`Δ_fault = p95(system WITH camera drifted) − p95(healthy subset WITHOUT it)`.
**Δ_fault ≤ 0 ⇒ contained** (keeping-but-down-weighting the bad camera ≤ dropping it);
**Δ_fault ≫ 0 ⇒ the bad camera pollutes the estimate** (containment fails for this mode).

| drifted camera | bias (m) | p95 with bad (m) | p95 dropped (m) | Δ_fault (m) |
|---|---|---|---|---|
{chr(10).join(dfault_rows)}

*Reading Δ_fault vs bias: Δ_fault→0 at LARGE bias means the NIS gate rejected the
gross fault outright (gating already contains it); a Δ_fault PEAK at MODERATE bias is
the gate-evading regime — the drift passes the NIS gate yet pollutes the estimate. That
peak is exactly what the bias-EWMA health monitor (WP5) is for.*

*The containment claim is a mode comparison: naive `SEQUENTIAL_FUSION` (M5) shows the
moderate-bias peak (the drifted camera pollutes); the health-aware `HEALTH_AWARE_FUSION`
(B6) mode drives it toward ≤0 by detecting the drift (innovation/bias EWMA → DEGRADED) and
inflating/rejecting the bad camera. Run both — `--fusion-mode SEQUENTIAL_FUSION` and
`--fusion-mode HEALTH_AWARE_FUSION` — and compare Δ_fault. (B6 needs no extra config; the
selection modes M6/M7/M8 additionally need quality providers.)*

*Generated by experiments/multicamera_fusion_extension/tools/run_containment_pilot.py.*
"""
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(res, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o)),
        encoding="utf-8")
    return out / "RESULTS.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--fusion-mode", default="SEQUENTIAL_FUSION",
                    help="e.g. SEQUENTIAL_FUSION (naive M5) or HEALTH_AWARE_FUSION (B6); run both to compare containment")
    ap.add_argument("--bias-m", nargs="+", type=float, default=list(DEFAULT_BIAS_LEVELS_M))
    ap.add_argument("--projection-calibration", default=None,
                    help="v2 projection_calibration.json: re-project obs from recorded pixels offline "
                         "(the recorder writes pred_world UNCORRECTED, so gate-quality runs need this)")
    ap.add_argument("--world-sdf", default=None,
                    help="world SDF for building camera models (required with --projection-calibration)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        mode = ReplayMode[args.fusion_mode]
    except KeyError:
        ap.error(f"unknown fusion mode {args.fusion_mode!r}; choices: {[m.name for m in ReplayMode]}")
    res = run_pilot(args.run_dir, fusion_mode=mode, bias_levels_m=tuple(args.bias_m),
                    projection_calibration=args.projection_calibration, world_sdf=args.world_sdf)
    out = args.out or (ROOT / "logs" / "studies" / "multicamera_fusion_extension" / "containment_pilot")
    path = write_results(res, out)
    print("fusion_gain_p95 =", _fmt(res["subset"]["fusion_gain_p95_m"]), "m")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
