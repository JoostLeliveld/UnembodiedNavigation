#!/usr/bin/env python3
"""Create an honest pilot-study showcase from evaluation-only camera probes.

The input probes are produced by ``probe_multicamera_accuracy.py``.  This tool
reconstructs manager inputs from operational fields only: camera ID, detector
score, selected pixel, camera calibration, and diagnostic timestamp.  The
probe's truth-derived homography errors are read only after selection to report
evaluation metrics (selection regret and map error).

It writes ``pilot_summary.json``, ``PILOT_RESULTS.md``, and two figures.  The
result is intentionally called a *static commissioning pilot*, not a completed
D0-D5 navigation campaign.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
for relative in ("src/reliability", "src/unav_common"):
    source = str(REPO / relative)
    if source not in sys.path:
        sys.path.insert(0, source)

from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.contracts import CameraQuality
from reliability.fusion import MapObservation

RECORDER_PATH = REPO / "experiments/multicamera_commissioning_bigwarehouse/tools/record_operational_logs.py"
STUDY_CONFIG_PATH = REPO / "experiments/multicamera_commissioning_bigwarehouse/config/study.yaml"
PROBE_PATH = REPO / "scripts/reliability/probe_multicamera_accuracy.py"
MAX_OVERLAP_TIME_DELTA_S = 0.05
MAX_CROSS_CAMERA_DISAGREEMENT_M = 0.30
MIN_OVERLAP_PAIRS = 30
MAX_OVERLAP_OUTLIER_RATE = 0.10


def _load_recorder_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bigwarehouse_recorder_for_showcase", RECORDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load camera calibration helper {RECORDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_probe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("results"), list) or not isinstance(payload.get("poses"), list):
        raise ValueError(f"{path} is not a camera-probe report")
    return payload


def _artifact_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": resolved.stat().st_size,
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _detected(row: dict[str, Any]) -> bool:
    return bool(row.get("detected")) and str(row.get("status")) == "ok"


def _group_probe_frames(report: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Keep the probe's pose order; a frame contains one row per camera."""

    results = [dict(row) for row in report["results"]]
    cameras = sorted({str(row["camera_id"]) for row in results})
    if len(cameras) != 2:
        raise ValueError("The pilot showcase expects exactly two cameras")
    if len(results) % len(cameras):
        raise ValueError("Probe result count is not a whole number of camera frames")
    frames = []
    for start in range(0, len(results), len(cameras)):
        group = results[start : start + len(cameras)]
        if {str(row["camera_id"]) for row in group} != set(cameras):
            raise ValueError("Each probe frame must contain one result from each camera")
        frames.append(group)
    return frames


def _map_observation(row: dict[str, Any], cameras: dict[str, Any]) -> MapObservation | None:
    """Build a manager-visible map observation; no truth field is accessed."""

    if not _detected(row):
        return None
    camera_id = str(row["camera_id"])
    u = _finite(row.get("selected_u_px"))
    v = _finite(row.get("selected_v_px"))
    score = _finite(row.get("score"))
    stamp = _finite(row.get("diagnostic_stamp_s"))
    if u is None or v is None or score is None or stamp is None:
        return None
    xy = cameras[camera_id].pixel_to_world_at_z(u, v, 0.05)
    if xy is None:
        return None
    return MapObservation(
        camera_id=camera_id,
        timestamp_s=stamp,
        xy_m=xy,
        covariance_m2=((0.08**2, 0.0), (0.0, 0.08**2)),
        quality=CameraQuality(
            camera_id=camera_id,
            p_available=max(0.0, min(1.0, score)),
            conditional_cov_uv=((2.5**2, 0.0), (0.0, 2.5**2)),
            association_confidence=1.0,
            source_model="pilot_detector_score",
        ),
        source="pilot_operational_projection",
    )


def _selected_error(row: dict[str, Any] | None) -> float | None:
    if row is None or not _detected(row):
        return None
    return _finite(row.get("homography_xy_error_m"))


def _summary_stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    clean = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if not clean.size:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "n": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p90": float(np.quantile(clean, 0.90)),
        "max": float(np.max(clean)),
    }


def _switch_count(selected_ids: list[str | None]) -> int:
    ids = [item for item in selected_ids if item is not None]
    return sum(current != previous for previous, current in zip(ids, ids[1:]))


def _camera_metrics(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report["results"]:
        per_camera[str(row["camera_id"])].append(dict(row))
    output: dict[str, dict[str, Any]] = {}
    for camera_id, rows in sorted(per_camera.items()):
        detected = [row for row in rows if _detected(row)]
        output[camera_id] = {
            "samples": len(rows),
            "detections": len(detected),
            "detection_rate": len(detected) / max(len(rows), 1),
            "homography_xy_error_m": _summary_stats(
                _finite(row.get("homography_xy_error_m")) for row in detected
            ),
            "selected_pixel_error_px": _summary_stats(
                _finite(row.get("selected_pixel_error_px")) for row in detected
            ),
            "frame_age_s": _summary_stats(
                _finite(row.get("frame_age_at_publish_s")) for row in detected
            ),
        }
    return output


def _overlap_metrics(report: dict[str, Any], cameras: dict[str, Any]) -> dict[str, Any]:
    pair_distances: list[float] = []
    pair_time_deltas: list[float] = []
    geometric_pair_count = 0
    detected_pair_count = 0
    synchronized_pair_count = 0
    for frame in _group_probe_frames(report):
        geometric = []
        points = []
        for row in frame:
            expected_u = _finite(row.get("expected_u_px"))
            expected_v = _finite(row.get("expected_v_px"))
            geometric.append(
                expected_u is not None and expected_v is not None and 0.0 <= expected_u <= 1280.0 and 0.0 <= expected_v <= 720.0
            )
            obs = _map_observation(row, cameras)
            if obs is not None:
                points.append(obs.xy_m)
        if all(geometric):
            geometric_pair_count += 1
        if len(points) == 2:
            detected_pair_count += 1
            pair_distances.append(math.dist(points[0], points[1]))
            stamps = [_finite(row.get("diagnostic_stamp_s")) for row in frame]
            if all(stamp is not None for stamp in stamps):
                delta = abs(float(stamps[0]) - float(stamps[1]))
                pair_time_deltas.append(delta)
                if delta <= MAX_OVERLAP_TIME_DELTA_S:
                    synchronized_pair_count += 1
    reliable = [
        distance
        for distance, delta in zip(pair_distances, pair_time_deltas)
        if distance <= MAX_CROSS_CAMERA_DISAGREEMENT_M
        and delta <= MAX_OVERLAP_TIME_DELTA_S
    ]
    outlier_rate = (
        (len(pair_distances) - len(reliable)) / len(pair_distances)
        if pair_distances
        else None
    )
    max_time_delta = max(pair_time_deltas, default=None)
    gate_checks = {
        "enough_overlap_pairs": len(pair_distances) >= MIN_OVERLAP_PAIRS,
        "time_delta_within_limit": (
            max_time_delta is not None and max_time_delta <= MAX_OVERLAP_TIME_DELTA_S
        ),
        "outlier_rate_within_limit": (
            outlier_rate is not None and outlier_rate <= MAX_OVERLAP_OUTLIER_RATE
        ),
    }
    return {
        "frames": len(_group_probe_frames(report)),
        "geometric_overlap_frames": geometric_pair_count,
        "detected_overlap_frames": detected_pair_count,
        "near_synchronous_detected_overlap_frames_at_0_05s": synchronized_pair_count,
        "reliable_overlap_frames_at_0_30m": len(reliable),
        "map_disagreement_m": _summary_stats(pair_distances),
        "time_delta_s": _summary_stats(pair_time_deltas),
        "outlier_rate_at_0_30m": outlier_rate,
        "commissioning_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "thresholds": {
                "max_overlap_time_delta_s": MAX_OVERLAP_TIME_DELTA_S,
                "max_cross_camera_disagreement_m": MAX_CROSS_CAMERA_DISAGREEMENT_M,
                "min_overlap_pairs": MIN_OVERLAP_PAIRS,
                "max_overlap_outlier_rate": MAX_OVERLAP_OUTLIER_RATE,
            },
        },
    }


def _run_handover(report: dict[str, Any], cameras: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = CameraManagerConfig(
        min_spatial_trust=0.45,
        candidate_score_margin=0.08,
        required_consecutive_better_frames=3,
        max_cross_camera_disagreement_m=0.30,
        require_consistency_when_source_available=True,
    )
    manager = CameraManager(cfg)
    manager_selected: list[str | None] = []
    score_selected: list[str | None] = []
    manager_errors: list[float | None] = []
    score_errors: list[float | None] = []
    oracle_errors: list[float | None] = []
    trace: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(_group_probe_frames(report)):
        observations = tuple(obs for row in frame if (obs := _map_observation(row, cameras)) is not None)
        timestamp = max((_finite(row.get("diagnostic_stamp_s")) or float(frame_index)) for row in frame)
        decision = manager.select(timestamp_s=timestamp, observations=observations)
        manager_id = decision.selected_camera_id or None
        score_observation = max(
            observations,
            key=lambda item: item.quality.p_available * item.quality.association_confidence,
            default=None,
        )
        score_id = None if score_observation is None else score_observation.camera_id
        by_camera = {str(row["camera_id"]): row for row in frame}
        valid_errors = [_selected_error(row) for row in frame]
        valid_errors = [value for value in valid_errors if value is not None]
        manager_error = _selected_error(by_camera.get(manager_id)) if manager_id else None
        score_error = _selected_error(by_camera.get(score_id)) if score_id else None
        oracle_error = min(valid_errors) if valid_errors else None
        manager_selected.append(manager_id)
        score_selected.append(score_id)
        manager_errors.append(manager_error)
        score_errors.append(score_error)
        oracle_errors.append(oracle_error)
        trace.append(
            {
                "frame": frame_index,
                "pose_label": str(frame[0].get("pose_label", f"frame_{frame_index}")),
                "timestamp_s": timestamp,
                "camera_scores": {
                    str(row["camera_id"]): _finite(row.get("score")) if _detected(row) else None
                    for row in frame
                },
                "manager_selected_camera": manager_id,
                "score_only_selected_camera": score_id,
                "oracle_camera": (
                    min(
                        (row for row in frame if _selected_error(row) is not None),
                        key=lambda row: _selected_error(row) or math.inf,
                        default=None,
                    ) or {}
                ).get("camera_id"),
                "manager_decision": decision.to_dict(),
                "manager_error_m": manager_error,
                "score_only_error_m": score_error,
                "oracle_error_m": oracle_error,
            }
        )

    def regret(selected: list[float | None]) -> list[float | None]:
        return [
            None if error is None or oracle is None else max(error - oracle, 0.0)
            for error, oracle in zip(selected, oracle_errors)
        ]

    metrics = {
        "frames": len(trace),
        "manager_switches": _switch_count(manager_selected),
        "score_only_switches": _switch_count(score_selected),
        "manager_map_error_m": _summary_stats(manager_errors),
        "score_only_map_error_m": _summary_stats(score_errors),
        "manager_selection_regret_m": _summary_stats(regret(manager_errors)),
        "score_only_selection_regret_m": _summary_stats(regret(score_errors)),
        "manager_selected_camera_ids": manager_selected,
        "score_only_selected_camera_ids": score_selected,
        "manager_no_selection_frames": sum(item is None for item in manager_selected),
    }
    return metrics, trace


def _plot_static(report: dict[str, Any], output: Path) -> None:
    rows = [dict(row) for row in report["results"]]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    colors = {"camera_A": "#2563eb", "camera_B": "#d97706"}
    for camera_id in sorted({str(row["camera_id"]) for row in rows}):
        subset = [row for row in rows if str(row["camera_id"]) == camera_id]
        y = [float(row["commanded_y_m"]) for row in subset]
        detection = [1.0 if _detected(row) else 0.0 for row in subset]
        errors = [_selected_error(row) for row in subset]
        axes[0].scatter(y, detection, color=colors[camera_id], s=42, label=camera_id)
        axes[1].scatter(
            [float(row["commanded_y_m"]) for row, error in zip(subset, errors) if error is not None],
            [float(error) for error in errors if error is not None],
            color=colors[camera_id],
            s=42,
            label=camera_id,
        )
    axes[0].set(title="Static commissioning detections", xlabel="Warehouse y [m]", ylabel="Detection valid")
    axes[0].set_yticks([0.0, 1.0], ["no", "yes"])
    axes[1].set(title="Evaluation-only map error", xlabel="Warehouse y [m]", ylabel="Projection error [m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_handover(trace: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 5.6), sharex=True, constrained_layout=True)
    frame = np.asarray([entry["frame"] for entry in trace], dtype=int)
    for camera_id, color in (("camera_A", "#2563eb"), ("camera_B", "#d97706")):
        values = [entry["camera_scores"].get(camera_id) for entry in trace]
        axes[0].plot(frame, values, marker="o", color=color, label=camera_id)
    axes[0].axhline(0.45, color="#64748b", linestyle="--", linewidth=1, label="trust gate")
    axes[0].set(ylabel="Operational detector score", title="Repeated-pose handover sequence")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    mapping = {None: -0.2, "camera_A": 0.0, "camera_B": 1.0}
    axes[1].step(frame, [mapping[entry["score_only_selected_camera"]] for entry in trace], where="mid", label="score-only", color="#94a3b8")
    axes[1].step(frame, [mapping[entry["manager_selected_camera"]] for entry in trace], where="mid", label="M8 hysteretic", color="#059669", linewidth=2.3)
    axes[1].set(
        xlabel="Probe frame (south → overlap → central → north)",
        ylabel="Selected camera",
        yticks=[-0.2, 0.0, 1.0],
        yticklabels=["none (safe)", "camera_A", "camera_B"],
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_overlap(report: dict[str, Any], cameras: dict[str, Any], output: Path) -> None:
    labels: list[str] = []
    disagreements: list[float] = []
    time_deltas: list[float] = []
    for frame in _group_probe_frames(report):
        observations = [
            obs for row in frame if (obs := _map_observation(row, cameras)) is not None
        ]
        stamps = [_finite(row.get("diagnostic_stamp_s")) for row in frame]
        if len(observations) != 2 or any(stamp is None for stamp in stamps):
            continue
        labels.append(str(frame[0].get("pose_label", f"pair_{len(labels)}")))
        disagreements.append(math.dist(observations[0].xy_m, observations[1].xy_m))
        time_deltas.append(abs(float(stamps[0]) - float(stamps[1])))

    x = np.arange(len(labels), dtype=int)
    colors = [
        "#dc2626" if value > MAX_CROSS_CAMERA_DISAGREEMENT_M else "#059669"
        for value in disagreements
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10.8, 6.2), sharex=True, constrained_layout=True)
    axes[0].bar(x, disagreements, color=colors)
    axes[0].axhline(
        MAX_CROSS_CAMERA_DISAGREEMENT_M,
        color="#64748b",
        linestyle="--",
        label="0.30 m gate",
    )
    axes[0].set(title="Empirical overlap commissioning", ylabel="A–B disagreement [m]")
    axes[0].legend(frameon=False)
    axes[1].scatter(x, time_deltas, color="#2563eb", s=48)
    axes[1].axhline(
        MAX_OVERLAP_TIME_DELTA_S,
        color="#64748b",
        linestyle="--",
        label="0.05 s gate",
    )
    axes[1].set(
        ylabel="Capture-time delta [s]",
        xticks=x,
        xticklabels=labels,
        ylim=(-0.005, 0.055),
    )
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _markdown(summary: dict[str, Any]) -> str:
    static = summary["static_probe"]["cameras"]
    overlap = summary["static_probe"]["overlap"]
    handover = summary["handover_sequence"]
    gate = overlap["commissioning_gate"]
    gate_checks = gate["checks"]
    thresholds = gate["thresholds"]
    outlier_rate = overlap["outlier_rate_at_0_30m"]
    lines = [
        "# Big-Warehouse Two-Camera Commissioning — Pilot Results",
        "",
        "This is a static, simulation-only commissioning pilot. It is not the full repeated D0–D5 campaign.",
        "The camera manager received only detector score, selected pixel, calibration geometry, timestamp, and its configured gates. Ground truth was used only after selection for the metrics below.",
        "",
        "## Per-camera static audit",
        "",
        "| Camera | Detection rate | Median projection error [m] | p90 projection error [m] |",
        "| --- | ---: | ---: | ---: |",
    ]
    for camera_id, values in sorted(static.items()):
        error = values["homography_xy_error_m"]
        lines.append(
            f"| {camera_id} | {100.0 * values['detection_rate']:.1f}% | {error['median']:.3f} | {error['p90']:.3f} |"
        )
    lines.extend(["", "![Static commissioning audit](figures/static_commissioning_audit.png)"])
    disagreement = overlap["map_disagreement_m"]
    lines.extend(
        [
            "",
            "## Empirical overlap",
            "",
            f"- Geometric overlap: {overlap['geometric_overlap_frames']}/{overlap['frames']} sampled frames.",
            f"- Detected overlap: {overlap['detected_overlap_frames']}/{overlap['frames']} frames.",
            f"- Near-synchronous overlap (≤0.05 s): {overlap['near_synchronous_detected_overlap_frames_at_0_05s']}/{overlap['detected_overlap_frames']} detected pairs.",
            f"- Reliable overlap (≤0.05 s and ≤0.30 m): {overlap['reliable_overlap_frames_at_0_30m']}/{overlap['detected_overlap_frames']} pairs; p90 pair disagreement: {disagreement['p90']:.3f} m.",
            f"- D2 evidence gate: **{'PASS' if gate['passed'] else 'FAIL'}**. Sample count {overlap['detected_overlap_frames']}/{thresholds['min_overlap_pairs']}; spatial outlier rate {100.0 * outlier_rate:.1f}%/{100.0 * thresholds['max_overlap_outlier_rate']:.1f}% maximum; maximum time delta {overlap['time_delta_s']['max']:.3f}/{thresholds['max_overlap_time_delta_s']:.3f} s.",
            "",
            "![Overlap commissioning gate](figures/overlap_commissioning_gate.png)",
            "",
            "## Hysteretic handover",
            "",
            f"- Score-only selection switched {handover['score_only_switches']} times; M8 switched {handover['manager_switches']} time(s) and emitted no selected camera in {handover['manager_no_selection_frames']} frame(s).",
            f"- Mean selection regret: score-only {handover['score_only_selection_regret_m']['mean']:.3f} m; M8 {handover['manager_selection_regret_m']['mean']:.3f} m.",
            f"- Mean selected projection error: score-only {handover['score_only_map_error_m']['mean']:.3f} m; M8 {handover['manager_map_error_m']['mean']:.3f} m.",
            "",
            "![Hysteretic handover](figures/hysteretic_handover.png)",
            "",
            (
                "The corrected synchronized probe passes the timing check, but this pilot does **not** pass D2: "
                f"overlap sample count {'passes' if gate_checks['enough_overlap_pairs'] else 'fails'} and spatial outlier rate "
                f"{'passes' if gate_checks['outlier_rate_within_limit'] else 'fails'}. M8 demonstrates a stability trade-off "
                "(fewer switches) rather than an accuracy improvement, so active planner handover remains disabled. "
                "Held-out navigation performance, GP calibration, and failure robustness remain open in `TODO.md`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-probe", type=Path, required=True)
    parser.add_argument("--handover-probe", type=Path, required=True)
    parser.add_argument("--world-sdf", type=Path, default=REPO.parent / "_archive/src/sim/gazebo_worlds/worlds/warehouse_big_2cam.world.sdf"  # HISTORICAL pilot world (archived))
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=REPO / "logs/perception_models/warehouse_yolo_detector_v1/model.pt",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    recorder = _load_recorder_module()
    cameras = {
        "camera_A": recorder.camera_model_from_world(args.world_sdf, include_name="external_camera"),
        "camera_B": recorder.camera_model_from_world(args.world_sdf, include_name="external_camera_2"),
    }
    static = _read_probe(args.static_probe)
    handover = _read_probe(args.handover_probe)
    handover_metrics, trace = _run_handover(handover, cameras)
    summary = {
        "study_id": "multicamera_commissioning_bigwarehouse_v1",
        "evidence_level": "static_simulation_pilot",
        "static_probe": {
            "source": str(args.static_probe.resolve()),
            "cameras": _camera_metrics(static),
            "overlap": _overlap_metrics(static, cameras),
        },
        "handover_sequence": {
            "source": str(args.handover_probe.resolve()),
            **handover_metrics,
        },
        "handover_trace": trace,
        "truth_boundary": "Ground truth from the evaluation-only probe was not passed to CameraManager.",
    }
    out_dir = args.out_dir.resolve()
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "PILOT_RESULTS.md").write_text(_markdown(summary), encoding="utf-8")
    _plot_static(static, figures / "static_commissioning_audit.png")
    _plot_overlap(static, cameras, figures / "overlap_commissioning_gate.png")
    _plot_handover(trace, figures / "hysteretic_handover.png")
    manifest = {
        "study_id": summary["study_id"],
        "evidence_level": summary["evidence_level"],
        "inputs": {
            "world_sdf": _artifact_entry(args.world_sdf),
            "detector_model": _artifact_entry(args.detector_model),
            "study_config": _artifact_entry(STUDY_CONFIG_PATH),
            "static_probe": _artifact_entry(args.static_probe),
            "handover_probe": _artifact_entry(args.handover_probe),
            "probe_implementation": _artifact_entry(PROBE_PATH),
            "showcase_implementation": _artifact_entry(Path(__file__)),
        },
        "contracts": {
            "camera_ids": ["camera_A", "camera_B"],
            "robot_contact_plane_z_m": 0.05,
            "max_overlap_time_delta_s": MAX_OVERLAP_TIME_DELTA_S,
            "max_cross_camera_disagreement_m": MAX_CROSS_CAMERA_DISAGREEMENT_M,
            "min_overlap_pairs": MIN_OVERLAP_PAIRS,
            "max_overlap_outlier_rate": MAX_OVERLAP_OUTLIER_RATE,
            "ground_truth_available_to_manager": False,
        },
        "outputs": {
            "pilot_summary": _artifact_entry(out_dir / "pilot_summary.json"),
            "pilot_results": _artifact_entry(out_dir / "PILOT_RESULTS.md"),
            "static_figure": _artifact_entry(figures / "static_commissioning_audit.png"),
            "overlap_figure": _artifact_entry(figures / "overlap_commissioning_gate.png"),
            "handover_figure": _artifact_entry(figures / "hysteretic_handover.png"),
        },
    }
    (out_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
