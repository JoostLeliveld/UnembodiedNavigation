"""Command-line tools for reliability export and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from reliability.contracts import CameraQuality
from reliability.benchmark import run_replay_benchmark
from reliability.export import export_multicamera_run_files, export_run_directory
from reliability.fusion import MapObservation
from reliability.overlap import validate_camera_overlap
from reliability.providers import GridMapReliabilityProvider
from reliability.replay import EvaluationFrame, ReplayConfig, ReplayFrame, required_replay_configs, run_replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reliability_tools")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export-run", help="Split a run directory into operational/evaluation JSONL records")
    export_parser.add_argument("--run-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--run-id", default=None)
    export_parser.add_argument("--task-id", default="")
    export_parser.add_argument("--seed", type=int, default=0)
    export_parser.add_argument("--config-hash", default="")

    multi_export_parser = sub.add_parser("export-multicamera", help="Split one perception CSV per camera into replay records")
    multi_export_parser.add_argument("--camera-csv", action="append", required=True, help="camera_id=path/to/perception.csv")
    multi_export_parser.add_argument("--experiment-csv", default="")
    multi_export_parser.add_argument("--output-dir", required=True)
    multi_export_parser.add_argument("--run-id", default="")
    multi_export_parser.add_argument("--task-id", default="")
    multi_export_parser.add_argument("--seed", type=int, default=0)
    multi_export_parser.add_argument("--config-hash", default="")
    multi_export_parser.add_argument("--frame-time-round-digits", type=int, default=3)

    replay_parser = sub.add_parser("replay", help="Run required replay configs from an exported split directory")
    replay_parser.add_argument("--export-dir", required=True)
    replay_parser.add_argument("--summary-out", default="")
    replay_parser.add_argument("--gp-artifact", default="")
    replay_parser.add_argument("--camera-id", default="camera_A")

    benchmark_parser = sub.add_parser("benchmark", help="Run replay benchmark suite from an exported split directory")
    benchmark_parser.add_argument("--export-dir", required=True)
    benchmark_parser.add_argument("--summary-out", default="")
    benchmark_parser.add_argument("--gp-artifact", default="")
    benchmark_parser.add_argument("--camera-id", default="camera_A")
    benchmark_mode = benchmark_parser.add_mutually_exclusive_group()
    benchmark_mode.add_argument("--include-multicamera", action="store_true")
    benchmark_mode.add_argument("--single-camera-only", action="store_true")

    overlap_parser = sub.add_parser("validate-overlap", help="Validate camera A/B map-estimate agreement in overlap frames")
    overlap_parser.add_argument("--export-dir", required=True)
    overlap_parser.add_argument("--camera-a", default="camera_A")
    overlap_parser.add_argument("--camera-b", default="camera_B")
    overlap_parser.add_argument("--max-time-delta-s", type=float, default=0.05)
    overlap_parser.add_argument("--max-disagreement-m", type=float, default=0.30)
    overlap_parser.add_argument("--summary-out", default="")

    args = parser.parse_args(argv)
    if args.command == "export-run":
        export = export_run_directory(
            args.run_dir,
            args.output_dir,
            run_id=args.run_id,
            task_id=args.task_id,
            seed=args.seed,
            config_hash=args.config_hash,
        )
        print(
            json.dumps(
                {
                    "operational_samples": len(export.operational_samples),
                    "camera_observations": len(export.camera_observations),
                    "replay_frames": len(export.replay_frames),
                    "evaluation_samples": len(export.evaluation_samples),
                    "evaluation_frames": len(export.evaluation_frames),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "export-multicamera":
        camera_csvs = _parse_camera_csv_args(args.camera_csv)
        export = export_multicamera_run_files(
            camera_csvs=camera_csvs,
            experiment_csv=args.experiment_csv or None,
            output_dir=args.output_dir,
            run_id=args.run_id,
            task_id=args.task_id,
            seed=args.seed,
            config_hash=args.config_hash,
            frame_time_round_digits=args.frame_time_round_digits,
        )
        print(
            json.dumps(
                {
                    "cameras": sorted(camera_csvs.keys()),
                    "operational_samples": len(export.operational_samples),
                    "camera_observations": len(export.camera_observations),
                    "replay_frames": len(export.replay_frames),
                    "evaluation_samples": len(export.evaluation_samples),
                    "evaluation_frames": len(export.evaluation_frames),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "replay":
        summary = replay_export_dir(
            args.export_dir,
            gp_artifact=args.gp_artifact,
            camera_id=args.camera_id,
        )
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.summary_out:
            Path(args.summary_out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.command == "validate-overlap":
        summary = validate_overlap_export_dir(
            args.export_dir,
            camera_a_id=args.camera_a,
            camera_b_id=args.camera_b,
            max_time_delta_s=args.max_time_delta_s,
            max_allowed_disagreement_m=args.max_disagreement_m,
        )
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.summary_out:
            Path(args.summary_out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.command == "benchmark":
        include_multicamera = None
        if args.include_multicamera:
            include_multicamera = True
        if args.single_camera_only:
            include_multicamera = False
        summary = benchmark_export_dir(
            args.export_dir,
            include_multicamera=include_multicamera,
            gp_artifact=args.gp_artifact,
            camera_id=args.camera_id,
        )
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.summary_out:
            Path(args.summary_out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    parser.error(f"Unsupported command {args.command!r}")
    return 2


def replay_export_dir(
    export_dir: str | Path,
    *,
    gp_artifact: str | Path = "",
    camera_id: str = "camera_A",
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    frames = _load_replay_frames(export_dir / "operational" / "replay_frames.jsonl")
    evaluation = _load_evaluation_frames(export_dir / "evaluation_only" / "evaluation_frames.jsonl")
    providers = _quality_providers(gp_artifact=gp_artifact, camera_id=camera_id)
    summary: dict[str, Any] = {
        "frames": len(frames),
        "evaluation_frames": len(evaluation),
        "results": {},
    }
    for config in required_replay_configs(quality_providers=providers):
        result = run_replay(frames, config, evaluation_frames=evaluation)
        metrics = result.metrics
        summary["results"][config.mode.value] = {
            "steps": len(result.steps),
            "rmse_m": None if metrics is None else _none_nan(metrics.rmse_m),
            "max_error_m": None if metrics is None else _none_nan(metrics.max_error_m),
            "final_error_m": None if metrics is None else _none_nan(metrics.final_error_m),
            "mean_nis": None if metrics is None else _none_nan(metrics.mean_nis),
            "mean_nees": None if metrics is None else _none_nan(metrics.mean_nees),
            "update_acceptance_rate": None if metrics is None else _none_nan(metrics.update_acceptance_rate),
            "divergence_count": None if metrics is None else metrics.divergence_count,
        }
    return summary


def benchmark_export_dir(
    export_dir: str | Path,
    *,
    include_multicamera: bool | None = None,
    gp_artifact: str | Path = "",
    camera_id: str = "camera_A",
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    frames = _load_replay_frames(export_dir / "operational" / "replay_frames.jsonl")
    evaluation = _load_evaluation_frames(export_dir / "evaluation_only" / "evaluation_frames.jsonl")
    providers = _quality_providers(gp_artifact=gp_artifact, camera_id=camera_id)
    suite = run_replay_benchmark(
        frames,
        evaluation_frames=evaluation,
        include_multicamera=include_multicamera,
        quality_providers=providers,
    )
    summary = suite.summary()
    summary["frames"] = len(frames)
    summary["evaluation_frames"] = len(evaluation)
    return summary


def validate_overlap_export_dir(
    export_dir: str | Path,
    *,
    camera_a_id: str = "camera_A",
    camera_b_id: str = "camera_B",
    max_time_delta_s: float = 0.05,
    max_allowed_disagreement_m: float = 0.30,
) -> dict[str, Any]:
    export_dir = Path(export_dir)
    frames = _load_replay_frames(export_dir / "operational" / "replay_frames.jsonl")
    summary = validate_camera_overlap(
        frames,
        camera_a_id=camera_a_id,
        camera_b_id=camera_b_id,
        max_time_delta_s=max_time_delta_s,
        max_allowed_disagreement_m=max_allowed_disagreement_m,
    )
    return summary.to_dict()


def _quality_providers(
    *,
    gp_artifact: str | Path = "",
    camera_id: str = "camera_A",
) -> dict[str, GridMapReliabilityProvider]:
    if not str(gp_artifact or "").strip():
        return {}
    provider = GridMapReliabilityProvider.from_npz(
        gp_artifact,
        camera_id=camera_id,
        out_of_bounds_policy="clamp",
    )
    return {camera_id: provider}


def _parse_camera_csv_args(values: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        text = str(value)
        if "=" in text:
            camera_id, path = text.split("=", 1)
        elif ":" in text:
            camera_id, path = text.split(":", 1)
        else:
            raise SystemExit(f"--camera-csv must be camera_id=path, got {text!r}")
        camera_id = camera_id.strip()
        path = path.strip()
        if not camera_id or not path:
            raise SystemExit(f"--camera-csv must be camera_id=path, got {text!r}")
        out[camera_id] = Path(path)
    if len(out) < 2:
        raise SystemExit("export-multicamera requires at least two --camera-csv entries")
    return out


def _load_replay_frames(path: Path) -> tuple[ReplayFrame, ...]:
    if not path.is_file():
        return tuple()
    frames = []
    for payload in _read_jsonl(path):
        observations = []
        for obs_payload in payload.get("observations", []):
            quality_payload = obs_payload.get("quality", {})
            quality = CameraQuality.from_dict(quality_payload)
            observations.append(
                MapObservation(
                    camera_id=obs_payload["camera_id"],
                    timestamp_s=obs_payload["timestamp_s"],
                    xy_m=obs_payload["xy_m"],
                    covariance_m2=obs_payload["covariance_m2"],
                    quality=quality,
                    source=obs_payload.get("source", ""),
                )
            )
        frames.append(
            ReplayFrame(
                timestamp_s=payload["timestamp_s"],
                odometry_xy_m=payload["odometry_xy_m"],
                observations=tuple(observations),
            )
        )
    return tuple(frames)


def _load_evaluation_frames(path: Path) -> tuple[EvaluationFrame, ...]:
    if not path.is_file():
        return tuple()
    return tuple(
        EvaluationFrame(
            timestamp_s=payload["timestamp_s"],
            truth_xy_m=payload["truth_xy_m"],
        )
        for payload in _read_jsonl(path)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _none_nan(value: float) -> float | None:
    return None if value != value else float(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
