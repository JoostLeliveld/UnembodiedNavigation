#!/usr/bin/env python3
"""Run one deterministic, manifest-backed D3 replay case.

The command applies only pre-declared replay perturbations to operational replay
frames. Evaluation frames are never changed.  Each output therefore records the
same task/condition/seed for every baseline and carries hashes for the source
export and all four frozen camera-specific GP posteriors.

The injected ``low_light_proxy`` is deliberately an observation-thinning proxy,
not a claim about physical illumination.  It belongs in a robustness table, not
in a real-world lighting claim.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO / "experiments" / "multicamera_commissioning_bigwarehouse"
PROTOCOL = STUDY_DIR / "config" / "paper_protocol.yaml"
ANALYSIS = STUDY_DIR / "config" / "paper_analysis_plan.yaml"
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
sys.path.insert(0, str(REPO / "src" / "reliability"))

from reliability.benchmark import run_replay_benchmark  # noqa: E402
from reliability.cli import _load_evaluation_frames, _load_replay_frames  # noqa: E402
from reliability.fusion import MapObservation  # noqa: E402
from reliability.providers import GridMapReliabilityProvider  # noqa: E402
from reliability.replay import ReplayFrame  # noqa: E402


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected YAML mapping at {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _method_hashes(analysis: Mapping[str, Any]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for relative in analysis["method_freeze"]["source_files"]:
        path = (STUDY_DIR / str(relative)).resolve()
        hashes[str(relative)] = _sha256(path) if path.is_file() else None
    return hashes


def _parse_assignments(values: Sequence[str], *, flag: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in str(value):
            raise RuntimeError(f"{flag} must use camera_id=path, got {value!r}")
        camera, text_path = (part.strip() for part in str(value).split("=", 1))
        path = Path(text_path).expanduser().resolve()
        if camera not in CAMERAS or not path.is_file() or camera in out:
            raise RuntimeError(f"Invalid {flag} entry {value!r}")
        out[camera] = path
    if set(out) != set(CAMERAS):
        raise RuntimeError(f"{flag} must provide exactly {', '.join(CAMERAS)}")
    return out


def _stable_uniform(*parts: object) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _modify_observation(obs: MapObservation, *, timestamp_offset_s: float = 0.0, availability_scale: float = 1.0) -> MapObservation:
    quality = replace(
        obs.quality,
        p_available=max(0.0, min(1.0, float(obs.quality.p_available) * availability_scale)),
        association_confidence=max(0.0, min(1.0, float(obs.quality.association_confidence) * availability_scale)),
        source_model=f"{obs.quality.source_model}:d3_injected",
    )
    return MapObservation(
        camera_id=obs.camera_id,
        timestamp_s=float(obs.timestamp_s) - float(timestamp_offset_s),
        xy_m=obs.xy_m,
        covariance_m2=obs.covariance_m2,
        quality=quality,
        source=f"{obs.source or 'map_observation'}:d3_injected",
    )


def apply_condition(frames: Sequence[ReplayFrame], *, condition: str, seed: int) -> tuple[tuple[ReplayFrame, ...], dict[str, Any]]:
    """Apply the frozen D3 fault model without inspecting evaluation truth."""

    if condition not in {"nominal", "low_light_proxy", "camera_latency", "camera_dropout", "odometry_stress", "combined_shift"}:
        raise RuntimeError(f"Unknown D3 condition {condition!r}")
    target_camera = CAMERAS[int(seed) % len(CAMERAS)]
    apply_light = condition in {"low_light_proxy", "combined_shift"}
    apply_latency = condition in {"camera_latency", "combined_shift"}
    apply_dropout = condition in {"camera_dropout", "combined_shift"}
    apply_odom = condition in {"odometry_stress", "combined_shift"}
    rng = random.Random(20260716 + int(seed))
    previous_odom: tuple[float, float] | None = None
    odom_bias = [0.0, 0.0]
    output: list[ReplayFrame] = []
    for index, frame in enumerate(frames):
        observations: list[MapObservation] = []
        for obs in frame.observations:
            if apply_light and _stable_uniform(condition, seed, index, obs.camera_id, "light") < 0.25:
                continue
            if apply_dropout and obs.camera_id == target_camera and _stable_uniform(condition, seed, index, obs.camera_id, "dropout") < 0.70:
                continue
            observations.append(
                _modify_observation(
                    obs,
                    timestamp_offset_s=0.10 if apply_latency and obs.camera_id == target_camera else 0.0,
                    availability_scale=0.70 if apply_light else 1.0,
                )
            )
        odom = frame.odometry_xy_m
        if apply_odom:
            if previous_odom is None:
                previous_odom = odom
            distance = math.hypot(odom[0] - previous_odom[0], odom[1] - previous_odom[1])
            sigma = 0.010 + 0.030 * math.sqrt(max(distance, 0.0))
            odom_bias[0] += rng.gauss(0.0, sigma)
            odom_bias[1] += rng.gauss(0.0, sigma)
            odom = (odom[0] + odom_bias[0], odom[1] + odom_bias[1])
            previous_odom = frame.odometry_xy_m
        output.append(ReplayFrame(timestamp_s=frame.timestamp_s, odometry_xy_m=odom, observations=tuple(observations)))
    return tuple(output), {
        "type": "deterministic_replay_fault_injection",
        "condition": condition,
        "seed": int(seed),
        "affected_camera": target_camera if (apply_latency or apply_dropout) else "",
        "low_light_proxy": {"observation_drop_probability": 0.25, "quality_scale": 0.70} if apply_light else None,
        "camera_latency": {"delay_s": 0.10} if apply_latency else None,
        "camera_dropout": {"observation_drop_probability": 0.70} if apply_dropout else None,
        "odometry_stress": {"random_walk_base_sigma_m": 0.010, "distance_sigma_m": 0.030} if apply_odom else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--camera-gp", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    protocol = _yaml(PROTOCOL)
    analysis = _yaml(ANALYSIS)
    task_ids = {str(task["id"]) for task in protocol["navigation_tasks"]}
    if args.task_id not in task_ids:
        raise RuntimeError(f"Unknown task ID {args.task_id!r}")
    if args.condition not in set(protocol["randomization"]["conditions"]):
        raise RuntimeError(f"Condition {args.condition!r} is absent from the frozen protocol")
    if int(args.seed) not in {int(value) for value in protocol["randomization"]["seed_values"]}:
        raise RuntimeError(f"Seed {args.seed} is absent from the frozen protocol")
    out_dir = args.out_dir.expanduser().resolve()
    allowed_root = (REPO / "logs" / "studies" / "multicamera_commissioning_bigwarehouse").resolve()
    if out_dir != allowed_root and allowed_root not in out_dir.parents:
        raise RuntimeError(f"--out-dir must stay under {allowed_root}")
    export_dir = args.export_dir.expanduser().resolve()
    replay_path = export_dir / "operational" / "replay_frames.jsonl"
    evaluation_path = export_dir / "evaluation_only" / "evaluation_frames.jsonl"
    if not replay_path.is_file() or not evaluation_path.is_file():
        raise RuntimeError("export directory must contain operational replay and evaluation-only truth frames")
    gp_paths = _parse_assignments(args.camera_gp, flag="--camera-gp")
    frames = _load_replay_frames(replay_path)
    evaluation = _load_evaluation_frames(evaluation_path)
    if not frames or not evaluation:
        raise RuntimeError("D3 replay requires non-empty operational frames and evaluation-only truth frames")
    perturbed, perturbation = apply_condition(frames, condition=args.condition, seed=args.seed)
    providers = {camera: GridMapReliabilityProvider.from_npz(path, camera_id=camera) for camera, path in gp_paths.items()}
    summary = run_replay_benchmark(perturbed, evaluation_frames=evaluation, include_multicamera=True, quality_providers=providers).summary()
    summary["frames"] = len(perturbed)
    summary["evaluation_frames"] = len(evaluation)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "analysis_plan_id": analysis["analysis_plan_id"],
        "task_id": args.task_id,
        "condition": args.condition,
        "seed": int(args.seed),
        "source_export": str(export_dir),
        "source_export_sha256": {"replay_frames": _sha256(replay_path), "evaluation_frames": _sha256(evaluation_path)},
        "camera_gp_sha256": {camera: _sha256(path) for camera, path in gp_paths.items()},
        "source_method_sha256": _method_hashes(analysis),
        "policy_ids": list(protocol["offline_replay"]["policies"]),
        "perturbation": perturbation,
        "evaluation_truth_modified": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "frames": len(perturbed), "evaluation_frames": len(evaluation)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
