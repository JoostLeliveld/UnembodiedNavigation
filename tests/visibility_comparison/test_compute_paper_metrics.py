import importlib.util
from pathlib import Path


def _load_metrics_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "visibility_comparison" / "compute_paper_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_paper_metrics", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collision_valid_run_false_counts_as_collision_not_invalid():
    metrics = _load_metrics_module()

    result = metrics._classify_outcome({
        "valid_run": False,
        "completion_reason": "collision",
        "collision_any": True,
        "max_wall_penetration_m": 0.02,
    })

    assert result["is_collision"] is True
    assert result["is_penetration"] is True
    assert result["is_invalid"] is False


def test_non_outcome_valid_run_false_counts_as_invalid():
    metrics = _load_metrics_module()

    result = metrics._classify_outcome({
        "valid_run": False,
        "completion_reason": "",
        "collision_any": False,
        "max_wall_penetration_m": 0.0,
        "max_obstacle_penetration_m": 0.0,
    })

    assert result["is_collision"] is False
    assert result["is_penetration"] is False
    assert result["is_invalid"] is True


def test_goal_reached_stable_counts_as_clean_success():
    metrics = _load_metrics_module()

    result = metrics._classify_outcome({
        "valid_run": True,
        "completion_reason": "goal_reached_stable",
        "collision_any": False,
        "max_wall_penetration_m": 0.0,
        "max_obstacle_penetration_m": 0.0,
    })

    assert result["is_clean_success"] is True
    assert result["is_invalid"] is False


def test_stuck_counts_as_explicit_failure_not_invalid():
    metrics = _load_metrics_module()

    result = metrics._classify_outcome({
        "valid_run": True,
        "completion_reason": "stuck",
        "collision_any": False,
        "max_wall_penetration_m": 0.0,
        "max_obstacle_penetration_m": 0.0,
    })

    assert result["is_stuck"] is True
    assert result["is_invalid"] is False
    assert result["is_clean_success"] is False


def test_yolo_detection_rate_uses_capture_stamp_for_runtime_filter(tmp_path):
    metrics = _load_metrics_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "experiment.csv").write_text(
        "stamp,cmd_v,cmd_w,odom_map_available,odom_map_x,odom_map_y,planner_belief_available,planner_belief_x,planner_belief_y,planner_cov_x,planner_cov_xy,planner_cov_y\n"
        "9.9,0.0,0.0,1,0,0,1,0,0,0.01,0,0.01\n"
        "10.0,0.2,0.0,1,0,0,1,0,0,0.01,0,0.01\n",
        encoding="utf-8",
    )
    (run_dir / "perception.csv").write_text(
        "diag_stamp,log_stamp,yolo_detected_after_threshold,pixel_pose_fresh,state_pos_error\n"
        "9.9,10.5,1,1,0.1\n"
        "10.1,10.2,0,1,0.2\n",
        encoding="utf-8",
    )

    result = metrics._compute_run_metrics(
        run_dir,
        {
            "completion_reason": "goal_reached",
            "goal_reached": True,
            "path_length_m": 1.0,
            "minimum_goal_distance": 0.0,
            "final_goal_distance": 0.0,
            "elapsed_after_first_cmd_s": 1.0,
            "mean_solve_time_ms": 0.0,
        },
        None,
        None,
    )

    assert result["yolo_detection_rate"] == 0.0
