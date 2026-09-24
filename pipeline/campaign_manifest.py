#!/usr/bin/env python3
"""Lock every campaign input by path and SHA-256 before the first campaign run.

Writes logs/thesis/campaign/manifest.json: the code commit (a dirty tree is refused), the
dataset lock, the gate, the detector, every fit artifact, the world, the tasks, the three
per-seed configs and the 30 solved routes. Re-running it after the campaign has started
compares instead of overwriting, so any drift is reported.

    python3 pipeline/campaign_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "logs/thesis"
OUT = R / "campaign/manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs() -> dict[str, str]:
    files = [REPO / p for p in (
        "pipeline/dataset_lock.json", "config/sensor_gate.yaml", "pipeline/tasks.yaml",
        "pipeline/execution_template.yaml", "pipeline/route_planning_template.yaml",
        "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf",
        "src/experiments/config/world_profiles.yaml",
        "logs/thesis/detector/training/imgsz960/upper_finetune/weights/best.pt",
        "logs/thesis/final_audit_protocol.json")]
    for sub in ("runtime_r012", "planning_precision", "covariance", "correction"):
        files += sorted(p for p in (R / "fits" / sub).rglob("*") if p.is_file())
    files += sorted((R / "campaign_configs").glob("*.yaml"))
    files += sorted((R / "routes").glob("*/*.route.json"))
    return {str(p.relative_to(REPO)): sha(p) for p in files}


def main() -> int:
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print(f"refusing: tracked changes in the working tree\n{dirty}", file=sys.stderr)
        return 1
    manifest = {"schema": "thesis_campaign_manifest.v1", "commit": git("rev-parse", "HEAD"),
                "inputs": inputs()}
    n_routes = sum(k.endswith(".route.json") for k in manifest["inputs"])
    if n_routes != 30:
        print(f"refusing: expected 30 solved routes, found {n_routes}", file=sys.stderr)
        return 1
    if OUT.exists():
        old = json.loads(OUT.read_text())
        drift = sorted(k for k in set(old["inputs"]) | set(manifest["inputs"])
                       if old["inputs"].get(k) != manifest["inputs"].get(k))
        drift += ["commit"] if old["commit"] != manifest["commit"] else []
        print(json.dumps({"manifest": str(OUT), "drift": drift}, indent=1))
        return 1 if drift else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"manifest": str(OUT), "files": len(manifest["inputs"]), "commit": manifest["commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
