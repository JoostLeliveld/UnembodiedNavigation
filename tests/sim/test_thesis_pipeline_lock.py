from pathlib import Path
import hashlib
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_current_thesis_pipeline_locked_prefix_is_reproducible():
    result = subprocess.run(
        [sys.executable, "pipeline/verify_pipeline_lock.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stage09_v5_uses_one_availability_independent_feasible_set():
    protocol_path = (
        ROOT / "pipeline/stage09_route_selection_protocol_v5.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert "availability_support_gate" not in protocol
    assert "No q threshold" in protocol["availability_role"]
    assert "one common feasible subset" in protocol["selection_rule"]

    selector_path = ROOT / "pipeline/select_stage09_routes_v5.py"
    selector = selector_path.read_bytes()
    assert hashlib.sha256(selector).hexdigest() == protocol["selector_sha256"]
    source = selector.decode("utf-8")
    assert "CommissionedAvailabilityModel" not in source
    assert "minimum_single_camera_probability" not in source
    assert "feasibility_by_arm[arm] != first_feasibility" in source
