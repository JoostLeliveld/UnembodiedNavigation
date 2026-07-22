#!/usr/bin/env python3
"""Paper-2 containment go/no-go harness — real capture -> E4 subset sweep + Δ_fault.

Wires the validated real-data bridge (`load_commissioning_run`) to the existing
offline replay sweeps (`replay_sweeps`) so the Paper-2 headline experiments R3/R4
fill in from ONE real handover capture, no synthetic data:

  R3 (E4, nominal)   — every camera subset through the same fusion mode;
                       fusion_gain_p95 = best-single-camera p95 − full-set p95.
  R4 (Δ_fault)       — inject a fault into one camera and compare the full system's
                       error to simply DROPPING that camera:
                         Δ_fault(c, s) = p95(system with c faulted at severity s)
                                       − p95(healthy subset without c).
                       Δ_fault ≤ 0  ⇒ the fusion CONTAINS the fault (keeping-but-
                       down-weighting the bad camera is no worse than dropping it);
                       Δ_fault ≫ 0 ⇒ the bad camera pollutes the estimate.
  Detection (E5/E6)  — when the fusion mode logs health (HEALTH_AWARE_FUSION / B6),
                       the same pass scores detection delay / isolation / false-alarm
                       from the health-state timeline (evaluate_fault_detection).

Two fault models (pre-reg §2):
  * ``position``    — constant world-position bias (`bias_camera_position`); coarse,
                      needs no camera model. Severity axis = metres.
  * ``calibration`` — the faithful model: perturb the camera's yaw and re-project
                      (`perturb_camera_calibration`), so the world bias is range- and
                      viewing-angle-dependent. Severity axis = degrees. Needs a
                      per-camera `PinholeGroundCamera` (built from the world SDF).

All conditions replay IDENTICAL detections through ONE fusion pipeline (§21 gate
discipline): any difference is attributable to the transform, never a retuned
filter. Ground truth arrives only as EvaluationFrames (firewall enforced by the
bridge). A single capture gives point estimates only — stated honestly.

Usage:
  python3 run_containment_pilot.py <capture_run_dir> [--fusion-mode HEALTH_AWARE_FUSION]
      [--fault-model position|calibration] [--bias-m ...] [--yaw-deg ...]
      [--world-sdf world.sdf] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))
sys.path.insert(0, str(ROOT / "experiments" / "multicamera_fusion_extension" / "tools"))

import load_commissioning_run as bridge  # noqa: E402
import replay_sweeps as rs  # noqa: E402
from reliability.calibration_perturbation import (  # noqa: E402
    CalibrationPerturbation,
    PinholeGroundCamera,
    perturb_camera_calibration,
)
from reliability.replay import ReplayConfig, ReplayMode, run_replay  # noqa: E402
from experiment_evaluators import (  # noqa: E402
    evaluate_fault_detection,
    summarize_fault_detection,
)

DEFAULT_BIAS_LEVELS_M = (0.2, 0.5, 1.0)
DEFAULT_YAW_LEVELS_DEG = (0.5, 1.0, 2.0, 5.0)
DEFAULT_DIRECTION = (1.0, 0.0)


def _p95(frames, evaluation_frames, cfg) -> float:
    result = run_replay(frames, cfg, evaluation_frames=evaluation_frames)
    return result.metrics.p95_error_m if result.metrics is not None else math.nan


# --------------------------------------------------------------------------- #
# Fault models: fault_fn(frames, camera_id, severity) -> frames.
# --------------------------------------------------------------------------- #
def position_fault_fn(direction=DEFAULT_DIRECTION):
    norm = math.hypot(direction[0], direction[1]) or 1.0
    ux, uy = direction[0] / norm, direction[1] / norm

    def _fn(frames, camera_id, severity):
        return rs.bias_camera_position(frames, camera_id, (severity * ux, severity * uy))

    return _fn


def calibration_fault_fn(camera_models):
    """Faithful fault: perturb the camera's yaw (deg) and re-project each observation."""

    def _fn(frames, camera_id, severity):
        camera = camera_models.get(camera_id)
        if camera is None:
            return frames  # no calibration for this camera -> cannot perturb
        return perturb_camera_calibration(
            frames, camera_id, camera, CalibrationPerturbation(yaw_deg=float(severity))
        )

    return _fn


def delta_fault_for_camera(
    frames, evaluation_frames, camera_id, cfg, *, fault_fn, severities, severity_key="bias_m"
):
    """Δ_fault(c, s) = p95(system with c faulted) − p95(healthy subset without c)."""
    p95_dropped = _p95(rs.drop_camera_permanent(frames, camera_id), evaluation_frames, cfg)
    rows = []
    for s in severities:
        faulted = fault_fn(frames, camera_id, s)
        p95_bad = _p95(faulted, evaluation_frames, cfg)
        rows.append({
            severity_key: float(s),
            "p95_with_bad_m": p95_bad,
            "p95_dropped_m": p95_dropped,
            "delta_fault_m": (p95_bad - p95_dropped) if (math.isfinite(p95_bad) and math.isfinite(p95_dropped)) else math.nan,
        })
    return rows


def _health_timeline(result):
    return [(step.timestamp_s, step.health_state_by_camera) for step in result.steps]


def detection_for_camera(
    frames, evaluation_frames, camera_id, cfg, *, fault_fn, severities, severity_key="bias_m"
):
    """Score detection (delay/isolation/false-alarm) per severity for a B6 run.

    The fault is persistent across the whole capture, so onset = the first frame
    time and the detection delay is time-to-DEGRADED from the run start.
    """
    onset = min((fr.timestamp_s for fr in frames), default=0.0)
    results = []
    for s in severities:
        faulted = fault_fn(frames, camera_id, s)
        replay = run_replay(faulted, cfg, evaluation_frames=evaluation_frames)
        det = evaluate_fault_detection(
            f"{camera_id}@{s:g}",
            _health_timeline(replay),
            faulted_camera=camera_id,
            onset_s=onset,
        )
        results.append((float(s), det))
    return results


def build_camera_models_from_world(world_sdf, camera_ids):
    """Build a PinholeGroundCamera per camera id from SDF `<include>` poses.

    Mirrors ``reliability.projection.camera_model_from_world`` (cam_pos + ground
    look-at from the six-value pose) but returns the ROS-free pinhole model the
    calibration perturbation uses. Matches a camera id to an include by exact
    name/model-name, else by substring; raises listing the includes if a camera
    id cannot be matched (the id<->include mapping is capture-specific).
    """
    root = ET.parse(Path(world_sdf)).getroot()
    includes = {}
    for include in root.findall(".//include"):
        name = (include.findtext("name") or "").strip()
        uri = (include.findtext("uri") or "").strip()
        model_name = uri.removeprefix("model://").split("/", 1)[0]
        pose = include.findtext("pose")
        if pose:
            for key in {name, model_name}:
                if key:
                    includes[key] = pose

    models = {}
    for cam in camera_ids:
        pose_text = includes.get(cam)
        if pose_text is None:
            matches = [key for key in includes if cam in key or key in cam]
            pose_text = includes[matches[0]] if matches else None
        if pose_text is None:
            raise RuntimeError(
                f"could not match camera {cam!r} to an SDF include; available: {sorted(includes)}"
            )
        x, y, z, _roll, pitch, yaw = (float(v) for v in pose_text.split())
        forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
        if forward[2] >= -1.0e-6:
            raise RuntimeError(f"camera {cam!r} does not point down toward the ground")
        scale = -z / forward[2]
        look_at = (x + scale * forward[0], y + scale * forward[1], 0.0)
        models[cam] = PinholeGroundCamera.looking_at(
            (x, y, z), look_at, fov_h_deg=90.0, width=1280, height=720
        )
    return models


def run_pilot(run_dir, *, fusion_mode=ReplayMode.SEQUENTIAL_FUSION,
              fault_model="position", bias_levels_m=DEFAULT_BIAS_LEVELS_M,
              yaw_levels_deg=DEFAULT_YAW_LEVELS_DEG, direction=DEFAULT_DIRECTION,
              camera_models=None, projection_calibration=None, world_sdf=None):
    loaded = bridge.load_run(
        run_dir, projection_calibration=projection_calibration, world_sdf=world_sdf
    )
    cfg = ReplayConfig(mode=fusion_mode, nis_gate=9.21)
    cams = tuple(sorted({o.camera_id for fr in loaded.frames for o in fr.observations}))

    if fault_model == "calibration":
        if camera_models is None:
            if world_sdf is None:
                raise ValueError("fault_model='calibration' needs camera_models or --world-sdf")
            camera_models = build_camera_models_from_world(world_sdf, cams)
        fault_fn = calibration_fault_fn(camera_models)
        severities, severity_key = tuple(yaw_levels_deg), "yaw_deg"
    else:
        fault_fn = position_fault_fn(direction)
        severities, severity_key = tuple(bias_levels_m), "bias_m"

    # R3 — nominal subset sweep (fusion gain vs best single)
    subset = rs.run_camera_subset_sweep(loaded.frames, loaded.evaluation_frames, config=cfg)
    p95_healthy_full = _p95(loaded.frames, loaded.evaluation_frames, cfg)

    # R4 — Δ_fault per camera across the severity axis
    delta = {c: delta_fault_for_camera(loaded.frames, loaded.evaluation_frames, c, cfg,
                                       fault_fn=fault_fn, severities=severities, severity_key=severity_key)
             for c in cams}

    out = {
        "run_dir": str(run_dir),
        "fusion_mode": fusion_mode.name,
        "fault_model": fault_model,
        "severity_key": severity_key,
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

    # Detection metrics (E5/E6) — only meaningful when the mode logs health (B6).
    if fusion_mode == ReplayMode.HEALTH_AWARE_FUSION:
        episodes = []
        detection = {}
        for c in cams:
            per_cam = detection_for_camera(
                loaded.frames, loaded.evaluation_frames, c, cfg,
                fault_fn=fault_fn, severities=severities, severity_key=severity_key)
            detection[c] = [{"severity": s, **det.as_row()} for s, det in per_cam]
            episodes.extend(det for _s, det in per_cam)
        # One nominal (no-fault) episode captures spurious DEGRADED = the false-alarm rate.
        nominal = run_replay(loaded.frames, cfg, evaluation_frames=loaded.evaluation_frames)
        episodes.append(evaluate_fault_detection("nominal", _health_timeline(nominal)))
        summary = summarize_fault_detection(episodes)
        out["detection"] = detection
        out["detection_summary"] = summary.as_row()

    return out


def _fmt(x):
    return "n/a" if (x is None or not isinstance(x, (int, float)) or not math.isfinite(x)) else f"{x:.3f}"


def _detection_section(res) -> str:
    if "detection" not in res:
        return (
            "\n## Detection (E5/E6)\n"
            f"Not scored: fusion mode `{res['fusion_mode']}` does not log health state. "
            "Re-run with `--fusion-mode HEALTH_AWARE_FUSION` to score detection.\n"
        )
    key = res["severity_key"]
    rows = []
    for c in res["cameras"]:
        for r in res["detection"][c]:
            rows.append(
                f"| {c} | {r['severity']:g} | {r['detected']} | {_fmt(r['detection_delay_s'])} | "
                f"{_fmt(r['escalation_delay_s'])} | {r['isolated']} | {r['false_alarm']} |"
            )
    s = res["detection_summary"]

    def _p(est):
        return "n/a" if est is None else f"{est['point']:.2f} [{est['low']:.2f},{est['high']:.2f}]"

    return f"""
## Detection (E5/E6) — health monitor vs the injected fault
Persistent fault from the run start; delay = time-to-DEGRADED. Isolation is scored
only with ≥3 cameras (else `None`). GROUND TRUTH IS NOT USED by the monitor; it is
used here only to know which camera was faulted.

| drifted camera | {key} | detected | detection delay (s) | escalation delay (s) | isolated | false alarm |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

**Summary:** detection rate {_p(s['detection_rate'])} · isolation-TPR {_p(s['isolation_tpr'])} ·
false-isolation {_p(s['false_isolation_rate'])} · nominal FAR {_p(s['nominal_far'])} ·
median detection delay {_fmt(s['median_detection_delay_s'])} s.
**Critical-failure stop rule (§5) passed: {s['stop_rule_passed']}** (isolation-TPR must
exceed false-isolation AND nominal FAR within gate).
"""


def write_results(res, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub = res["subset"]
    key = res["severity_key"]
    subset_rows = "\n".join(
        f"| {s} | {_fmt(p)} |" for s, p in sorted(sub["per_subset_p95_m"].items(), key=lambda kv: (len(kv[0]), kv[0]))
    )
    dfault_rows = []
    for c in res["cameras"]:
        for r in res["delta_fault"][c]:
            dfault_rows.append(f"| {c} | {r[key]:g} | {_fmt(r['p95_with_bad_m'])} | "
                               f"{_fmt(r['p95_dropped_m'])} | **{_fmt(r['delta_fault_m'])}** |")
    md = f"""# Paper-2 containment pilot — E4 subset sweep + Δ_fault

**Provenance:** real capture `{res['run_dir']}` via `load_commissioning_run`
(firewall enforced). Fusion mode: `{res['fusion_mode']}` @ NIS gate 9.21; fault model:
`{res['fault_model']}`. Cameras: {res['cameras']}; frames {res['n_frames']}, eval frames
{res['n_eval_frames']}; observations {res['observation_counts']}.

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

## R4 — Δ_fault: does the fusion CONTAIN an injected fault ({res['fault_model']} model)?
`Δ_fault = p95(system WITH camera faulted) − p95(healthy subset WITHOUT it)`.
**Δ_fault ≤ 0 ⇒ contained**; **Δ_fault ≫ 0 ⇒ the bad camera pollutes the estimate**.

| drifted camera | {key} | p95 with bad (m) | p95 dropped (m) | Δ_fault (m) |
|---|---|---|---|---|
{chr(10).join(dfault_rows)}

*Reading Δ_fault vs severity: Δ_fault→0 at LARGE severity means the NIS gate rejected the
gross fault outright; a Δ_fault PEAK at MODERATE severity is the gate-evading regime — the
drift passes the NIS gate yet pollutes the estimate. That peak is exactly what the bias-EWMA
health monitor (WP5) is for, and the `calibration` fault model makes it range-dependent.*

*The containment claim is a mode comparison: naive `SEQUENTIAL_FUSION` (M5) shows the
moderate-severity peak; `HEALTH_AWARE_FUSION` (B6) drives it toward ≤0 by detecting the drift
and inflating/rejecting the bad camera. Run both and compare Δ_fault + the detection table.*
{_detection_section(res)}
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
                    help="e.g. SEQUENTIAL_FUSION (naive M5) or HEALTH_AWARE_FUSION (B6); "
                         "B6 also emits the detection table")
    ap.add_argument("--fault-model", choices=("position", "calibration"), default="position",
                    help="position = constant world bias (metres); calibration = faithful yaw "
                         "drift re-projected through the camera (degrees; needs --world-sdf)")
    ap.add_argument("--bias-m", nargs="+", type=float, default=list(DEFAULT_BIAS_LEVELS_M))
    ap.add_argument("--yaw-deg", nargs="+", type=float, default=list(DEFAULT_YAW_LEVELS_DEG))
    ap.add_argument("--projection-calibration", default=None,
                    help="v2 projection_calibration.json: re-project obs from recorded pixels offline")
    ap.add_argument("--world-sdf", default=None,
                    help="world SDF for building camera models (required for --fault-model calibration)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        mode = ReplayMode[args.fusion_mode]
    except KeyError:
        ap.error(f"unknown fusion mode {args.fusion_mode!r}; choices: {[m.name for m in ReplayMode]}")
    res = run_pilot(args.run_dir, fusion_mode=mode, fault_model=args.fault_model,
                    bias_levels_m=tuple(args.bias_m), yaw_levels_deg=tuple(args.yaw_deg),
                    projection_calibration=args.projection_calibration, world_sdf=args.world_sdf)
    out = args.out or (ROOT / "logs" / "studies" / "multicamera_fusion_extension" / "containment_pilot")
    path = write_results(res, out)
    print("fusion_gain_p95 =", _fmt(res["subset"]["fusion_gain_p95_m"]), "m")
    if "detection_summary" in res:
        print("detection stop-rule passed =", res["detection_summary"]["stop_rule_passed"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
