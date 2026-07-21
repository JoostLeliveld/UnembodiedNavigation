from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.cli import benchmark_export_dir, main, replay_export_dir, validate_overlap_export_dir  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_cli_export_and_replay_round_trip(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    export_dir = tmp_path / "export"
    _write_csv(
        run_dir / "perception.csv",
        [
            {
                "diag_stamp": "0.0",
                "detected": "1",
                "true_x": "0.0",
                "true_y": "0.0",
                "obs_u": "10",
                "obs_v": "20",
                "pixel_pose_available": "1",
                "pixel_pose_u": "10",
                "pixel_pose_v": "20",
                "pixel_pose_age_s": "0.01",
                "pixel_pose_fresh": "1",
                "pred_world_x_calibrated": "0.0",
                "pred_world_y_calibrated": "0.0",
                "yolo_score_raw": "0.9",
                "yolo_score_selected": "0.9",
                "yolo_detected_after_threshold": "1",
            },
            {
                "diag_stamp": "1.0",
                "detected": "1",
                "true_x": "1.0",
                "true_y": "0.0",
                "obs_u": "11",
                "obs_v": "20",
                "pixel_pose_available": "1",
                "pixel_pose_u": "11",
                "pixel_pose_v": "20",
                "pixel_pose_age_s": "0.01",
                "pixel_pose_fresh": "1",
                "pred_world_x_calibrated": "1.0",
                "pred_world_y_calibrated": "0.0",
                "yolo_score_raw": "0.9",
                "yolo_score_selected": "0.9",
                "yolo_detected_after_threshold": "1",
            },
        ],
    )
    _write_csv(
        run_dir / "experiment.csv",
        [
            {"stamp": "0.0", "odom_noisy_x": "0.0", "odom_noisy_y": "0.0"},
            {"stamp": "1.0", "odom_noisy_x": "1.1", "odom_noisy_y": "0.0"},
        ],
    )

    assert main(["export-run", "--run-dir", str(run_dir), "--output-dir", str(export_dir), "--run-id", "run"]) == 0
    export_stdout = json.loads(capsys.readouterr().out)
    assert export_stdout["camera_observations"] == 2
    assert export_stdout["evaluation_frames"] == 2

    assert main(["replay", "--export-dir", str(export_dir)]) == 0
    replay_stdout = json.loads(capsys.readouterr().out)
    assert replay_stdout["frames"] == 2
    assert replay_stdout["results"]["R0_odom_only"]["steps"] == 2
    assert replay_stdout["results"]["R1_single_fixed_R"]["rmse_m"] is not None

    direct_summary = replay_export_dir(export_dir)
    assert direct_summary["frames"] == 2

    assert main(["benchmark", "--export-dir", str(export_dir), "--include-multicamera"]) == 0
    benchmark_stdout = json.loads(capsys.readouterr().out)
    assert benchmark_stdout["frames"] == 2
    assert "M5_sequential_fusion" in benchmark_stdout["results"]

    direct_benchmark = benchmark_export_dir(export_dir, include_multicamera=False)
    assert "M5_sequential_fusion" not in direct_benchmark["results"]

    np = pytest.importorskip("numpy")
    artifact = tmp_path / "gp.npz"
    np.savez(
        artifact,
        xs=np.asarray([0.0, 1.0]),
        ys=np.asarray([0.0, 1.0]),
        P_conservative_plan_map=np.asarray([[0.8, 0.8], [0.8, 0.8]]),
    )
    gp_summary = benchmark_export_dir(export_dir, include_multicamera=False, gp_artifact=artifact)
    assert gp_summary["results"]["R4_current_GP_R"]["steps"] == 2


def test_cli_multicamera_export_benchmark_and_overlap(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "multi_run"
    export_dir = tmp_path / "multi_export"
    base_rows = [
        {
            "diag_stamp": "0.0",
            "detected": "1",
            "true_x": "0.0",
            "true_y": "0.0",
            "obs_u": "10",
            "obs_v": "20",
            "pixel_pose_available": "1",
            "pixel_pose_u": "10",
            "pixel_pose_v": "20",
            "pixel_pose_age_s": "0.01",
            "pixel_pose_fresh": "1",
            "pred_world_x_calibrated": "0.0",
            "pred_world_y_calibrated": "0.0",
            "yolo_score_raw": "0.9",
            "yolo_score_selected": "0.9",
            "yolo_detected_after_threshold": "1",
        },
        {
            "diag_stamp": "1.0",
            "detected": "1",
            "true_x": "1.0",
            "true_y": "0.0",
            "obs_u": "11",
            "obs_v": "20",
            "pixel_pose_available": "1",
            "pixel_pose_u": "11",
            "pixel_pose_v": "20",
            "pixel_pose_age_s": "0.01",
            "pixel_pose_fresh": "1",
            "pred_world_x_calibrated": "1.0",
            "pred_world_y_calibrated": "0.0",
            "yolo_score_raw": "0.9",
            "yolo_score_selected": "0.9",
            "yolo_detected_after_threshold": "1",
        },
    ]
    b_rows = [dict(row) for row in base_rows]
    b_rows[0]["diag_stamp"] = "0.0004"
    b_rows[0]["pred_world_x_calibrated"] = "0.02"
    b_rows[1]["diag_stamp"] = "1.0004"
    b_rows[1]["pred_world_x_calibrated"] = "1.02"

    _write_csv(run_dir / "camera_A_perception.csv", base_rows)
    _write_csv(run_dir / "camera_B_perception.csv", b_rows)
    _write_csv(
        run_dir / "experiment.csv",
        [
            {"stamp": "0.0", "odom_noisy_x": "0.0", "odom_noisy_y": "0.0"},
            {"stamp": "1.0", "odom_noisy_x": "1.1", "odom_noisy_y": "0.0"},
        ],
    )

    assert main(
        [
            "export-multicamera",
            "--camera-csv",
            f"camera_A={run_dir / 'camera_A_perception.csv'}",
            "--camera-csv",
            f"camera_B={run_dir / 'camera_B_perception.csv'}",
            "--experiment-csv",
            str(run_dir / "experiment.csv"),
            "--output-dir",
            str(export_dir),
            "--run-id",
            "multi",
        ]
    ) == 0
    export_stdout = json.loads(capsys.readouterr().out)
    assert export_stdout["cameras"] == ["camera_A", "camera_B"]
    assert export_stdout["replay_frames"] == 2

    assert main(["benchmark", "--export-dir", str(export_dir)]) == 0
    benchmark_stdout = json.loads(capsys.readouterr().out)
    assert "M5_sequential_fusion" in benchmark_stdout["results"]

    assert main(["validate-overlap", "--export-dir", str(export_dir), "--max-disagreement-m", "0.05"]) == 0
    overlap_stdout = json.loads(capsys.readouterr().out)
    assert overlap_stdout["pass_gate"] is True
    assert overlap_stdout["pair_count"] == 2
    assert overlap_stdout["trust"]["pass_gate"] is True
    assert overlap_stdout["trust"]["bias_norm_m"] == pytest.approx(0.02)

    direct_overlap = validate_overlap_export_dir(export_dir, max_allowed_disagreement_m=0.05)
    assert direct_overlap["pair_count"] == 2
    assert direct_overlap["trust"]["pair_count"] == 2
