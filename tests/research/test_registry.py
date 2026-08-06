from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "research" / "validate_registry.py"
SPEC = importlib.util.spec_from_file_location("validate_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_canonical_registry_is_valid() -> None:
    data = MODULE.load_registry()
    assert MODULE.validate(data, ROOT) == []


def test_evaluation_truth_leak_is_rejected() -> None:
    data = MODULE.load_registry()
    experiment = data["experiments"][0]
    experiment["operational_interface_inputs"].append(experiment["evaluation_only_inputs"][0])
    errors = MODULE.validate(data, ROOT)
    assert any("evaluation-only inputs leak" in error for error in errors)


def test_retired_dependency_is_rejected_for_active_focus() -> None:
    data = MODULE.load_registry()
    focus_id = data["current_focus"]["research_experiment_id"]
    focus = next(row for row in data["experiments"] if row["experiment_id"] == focus_id)
    focus["status"] = "ACTIVE"
    focus["dependencies"].append("ASSET-LEGACY-UIGP")
    errors = MODULE.validate(data, ROOT)
    assert any("depends on retired code" in error for error in errors)


def test_blocked_experiment_can_remain_the_current_focus() -> None:
    data = MODULE.load_registry()
    focus_id = data["current_focus"]["research_experiment_id"]
    focus = next(row for row in data["experiments"] if row["experiment_id"] == focus_id)
    focus["status"] = "BLOCKED"
    assert MODULE.validate(data, ROOT) == []


def test_planned_experiment_cannot_be_the_current_focus() -> None:
    data = MODULE.load_registry()
    focus_id = data["current_focus"]["research_experiment_id"]
    focus = next(row for row in data["experiments"] if row["experiment_id"] == focus_id)
    focus["status"] = "PLANNED"
    errors = MODULE.validate(data, ROOT)
    assert any("current research focus must be ACTIVE or BLOCKED" in error for error in errors)
