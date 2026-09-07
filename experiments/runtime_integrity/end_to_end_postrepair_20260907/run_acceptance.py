"""Run source-bound successor acceptance only after explicit handoff preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from preflight import check

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def source_snapshot() -> dict[str, str]:
    paths = subprocess.check_output(["rg", "--files", "src", "tests", "experiments", "docs/module_audits"], cwd=ROOT, text=True).splitlines()
    allowed = {".py", ".yaml", ".json", ".sdf", ".xacro", ".xml", ".toml", ".md"}
    result = {}
    for name in sorted(set(paths)):
        path = ROOT / name
        if path.is_file() and path.suffix in allowed:
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def main(stage: str) -> int:
    gate = check(stage)
    if not gate["ready"]:
        print(json.dumps(gate, indent=2))
        return 2

    config = json.loads((HERE / "acceptance_commands.json").read_text())
    groups = config["desired_invariant_groups"]
    if stage == "physical":
        groups = [{"name": "physical_integration", "command": config["physical_integration"]["command"]}]
    elif stage == "final":
        groups = [{"name": "recorded_cross_chain", "command": config["recorded_cross_chain"]["command"]}]

    before = source_snapshot()
    record = {
        "stage": stage,
        "started_wall_unix": time.time(),
        "ready_record": gate["ready_record"],
        "sources_before": before,
        "groups": [],
    }
    environment = dict(os.environ)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    result_code = 0
    for group in groups:
        output_path = HERE / f"{group['name']}.txt"
        with output_path.open("w") as output:
            completed = subprocess.run(group["command"], cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
        record["groups"].append({"name": group["name"], "command": group["command"], "returncode": completed.returncode, "output": output_path.name})
        if completed.returncode:
            result_code = completed.returncode
            break

    after = source_snapshot()
    record["sources_after"] = after
    record["changed_during_execution"] = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if record["changed_during_execution"]:
        result_code = result_code or 3
    record["finished_wall_unix"] = time.time()
    record["returncode"] = result_code
    (HERE / f"execution_{stage}.json").write_text(json.dumps(record, indent=2) + "\n")
    return result_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("deterministic", "physical", "final"), default="deterministic")
    args = parser.parse_args()
    raise SystemExit(main(args.stage))
