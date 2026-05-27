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
