#!/usr/bin/env python3
"""Derive the v8 route-planning and execution configs from the stage-09/10 ones.

Only artifact identities change: the runtime R0/R1/R2 models, the matched planning
precision and the navigation detector are repointed to the v8 uniform campaign and their
hashes re-derived. Every navigation, planner and follower setting is inherited unchanged.
The execution template is written once per seed so the campaign can run seed by seed
(amendment 2026-09-23).

    python3 pipeline/campaign_configs.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "pipeline"
V8 = REPO / "logs/thesis_final_pipeline_v1/recapture_v8_uniform"
OLD_RUNTIME = REPO / "logs/thesis_final_pipeline_v1/final_bayesian/runtime_r012"
OLD_PLANNING = REPO / "logs/thesis_final_pipeline_v1/planning_precision"
OLD_DETECTOR = "logs/thesis_final_pipeline_v1/stage05_detector_training/imgsz960/upper_finetune/weights/best.pt"
SEEDS = (91500, 91501, 91502)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


GATE = "config/sensor_gate.yaml"


def repoint(cfg: dict, detector: str) -> dict:
    cfg["yolo_model"] = detector
    cfg["manager_sensor_gate_config_path"] = GATE
    for condition in cfg["conditions"].values():
        runtime = Path(condition["manager_visibility_sensor_model_path"])
        if runtime.parent != OLD_RUNTIME:
            raise RuntimeError(f"unexpected runtime model {runtime}")
        new_runtime = V8 / "runtime_r012" / runtime.name
        condition["manager_visibility_sensor_model_path"] = str(new_runtime)
        condition["manager_visibility_sensor_model_expected_sha256"] = sha(new_runtime)
        if "camera_network_artifact_path" in condition:
            planning = Path(condition["camera_network_artifact_path"])
            if planning.parent != OLD_PLANNING:
                raise RuntimeError(f"unexpected planning artifact {planning}")
            new_planning = V8 / "planning_precision" / planning.name
            condition["camera_network_artifact_path"] = str(new_planning)
            condition["camera_network_expected_sha256"] = sha(new_planning)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default=OLD_DETECTOR,
                        help="repo-relative path of the frozen navigation detector (v5 by default)")
    args = parser.parse_args()
    if not (REPO / args.detector).is_file():
        raise FileNotFoundError(args.detector)
    out_dir = V8 / "campaign_configs"
    out_dir.mkdir(exist_ok=True)

    planning_cfg = yaml.safe_load((HERE / "route_planning_template.yaml").read_text())
    if planning_cfg["yolo_model"] != OLD_DETECTOR:
        raise RuntimeError("route-planning template has an unexpected detector")
    repoint(planning_cfg, args.detector)
    route_path = out_dir / "route_planning_campaign.yaml"
    route_path.write_text(yaml.safe_dump(planning_cfg, sort_keys=False))

    execution = yaml.safe_load((HERE / "execution_template.yaml").read_text())
    repoint(execution, args.detector)
    # Collisions end a run through the physical /world_contacts channel only. The stage-10
    # template still asks for termination on ground-truth geometry, which the experiment
    # logger now refuses because it leaks ground truth into the experiment.
    execution["terminate_on_geom_collision"] = False
    # The runtime gate must be the one the covariance was fitted under (amendment
    # 2026-09-23: gate v3, no edge or size check).
    for task in execution["tasks"].values():
        task.pop("preselected_routes", None)
    for seed in SEEDS:
        per_seed = yaml.safe_load(yaml.safe_dump(execution))
        per_seed["study_title"] = f"v8 five-task camera-dropout campaign, seed {seed}"
        for task in per_seed["tasks"].values():
            task["seeds"] = [seed]
        (out_dir / f"execution_template_seed{seed}.yaml").write_text(
            yaml.safe_dump(per_seed, sort_keys=False))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
