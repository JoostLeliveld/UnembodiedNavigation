#!/usr/bin/env python3
"""Rebuild the current Track-A thesis plot suite and record its provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PAPER_FIGURES = ROOT.parent / "papers" / "Thesis" / "figures"
EXPORT = ROOT / "logs/thesis_current/paper_exports"
ROUTE_RUNS = (
    ROOT / "logs/thesis_final_pipeline_v1/stage09_navigation/planned_routes/thesis09_lane08E_to_lane12W",
    ROOT / "logs/thesis_final_pipeline_v1/stage09_navigation/planned_routes/thesis09_blind_corridor_west_to_east",
)
OUTPUTS = (
    "data_collection_roles.pdf",
    "planner_facing_field.pdf",
    "planning_information_fields.pdf",
    "covariance_diagnostics.pdf",
    "fusion_final_audit.pdf",
    "navigation_detailed_routes.pdf",
    "navigation_campaign_routes.pdf",
    "navigation_single_run.pdf",
    "fusion_trust_weighting.pdf",
)


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run(str(HERE / "make_data_and_information_figures.py"))
    run(str(HERE / "make_thesis_evaluation_figures.py"))
    run(str(HERE / "make_thesis_fusion_figure.py"))
    run(str(HERE / "make_single_run_figure.py"))
    run(str(HERE / "make_trust_weighting_figure.py"))
    if all((path / "manifest.json").is_file() and (path / ".complete").is_file()
           for path in ROUTE_RUNS):
        run(str(HERE / "make_thesis_navigation_figures.py"),
            *(str(path) for path in ROUTE_RUNS))

    missing = [name for name in OUTPUTS if not (PAPER_FIGURES / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing thesis figures: {missing}")
    manifest = {
        "schema": "thesis_track_a_figure_suite.v1",
        "status": "frozen_models_and_planned_routes_pending_gazebo_execution",
        "method": "canonical_R0_R1_R2_direct_information",
        "figures": {
            name: {"path": str(PAPER_FIGURES / name),
                   "sha256": digest(PAPER_FIGURES / name)}
            for name in OUTPUTS
        },
        "navigation_reports": [
            {"path": str(path / "manifest.json"),
             "sha256": digest(path / "manifest.json")} for path in ROUTE_RUNS
        ],
    }
    EXPORT.mkdir(parents=True, exist_ok=True)
    target = EXPORT / "figure_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
