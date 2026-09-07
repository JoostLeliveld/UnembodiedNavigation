"""Reproduce audit 07 against its source snapshot without changing the workspace.

Another workstream repaired batching during final QA. The original source is
preserved by the shared checkpoint below. Extract only the needed paths into a
temporary directory, expose the registered logs for reading, and compare all
three probe outputs to the retained evidence. No worktree or ROS graph is used.
"""
from pathlib import Path
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = "3c4ddeae4a7427bf374514dad6a1d65dc728a91c"
PATHS = ["src", "tests", "conftest.py", "pyproject.toml", "config", "schemas",
         "experiments/estimator_consistency/module_review_20260906/probe.py",
         "experiments/fusion_on_fixed_routes/aligned.py",
         "experiments/icra_commissioning/network_navigation_runtime_pilot.yaml",
         "docs/localization_metrics_registry.json"]
SUITES = ["tests/reliability/test_fusion_v2.py",
          "tests/reliability/test_fusion_arms.py",
          "tests/reliability/test_dynamic_occlusion_fusion.py",
          "tests/reliability/test_camera_manager_node.py",
          "tests/reliability/test_bias_floor.py",
          "tests/planning/test_planner_node_per_camera_correction.py",
          "tests/planning/test_runtime_transactions.py",
          "tests/perception/test_camera_acquisition_audit_04.py"]
PROBES = ["07_fusion_probe", "07_fusion_node_probe", "07_registered_event_probe"]
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           PYTHONDONTWRITEBYTECODE="1")
summary = {"snapshot": SNAPSHOT, "scope": "Original audit source; not later batching repairs"}
archive = subprocess.check_output(["git", "archive", SNAPSHOT, *PATHS], cwd=ROOT)
with tempfile.TemporaryDirectory(prefix="audit07-snapshot-") as temp:
    checkout = Path(temp)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(checkout, filter="data")
    (checkout / "logs").symlink_to(ROOT / "logs", target_is_directory=True)
    target = checkout / "docs/module_audits"
    target.mkdir(parents=True, exist_ok=True)
    for name in PROBES:
        shutil.copy2(Path(__file__).with_name(name + ".py"), target / (name + ".py"))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", *SUITES],
                         cwd=checkout, env=env, text=True, capture_output=True)
    summary["regressions"] = {"returncode": run.returncode, "output": run.stdout + run.stderr}
    if run.returncode:
        print(json.dumps(summary, indent=2))
        raise SystemExit(run.returncode)
    for name in PROBES:
        output = subprocess.check_output([sys.executable, str(target / (name + ".py"))],
                                         cwd=checkout, env=env, text=True)
        got = json.loads(output)
        expected = json.loads(Path(__file__).with_name(name + "_results.json").read_text())
        assert got == expected, f"{name} differs from saved audit evidence"
        summary[name] = {"assertions_passed": True, "matches_saved_results": True}
print(json.dumps(summary, indent=2))
