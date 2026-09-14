#!/usr/bin/env python3
"""Build the one-seed, three-arm thesis-story navigation sanity pilot."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments/thesis_pipeline_lock/stage09_navigation_frozen_routes_v9.yaml"
MEASUREMENT = REPO / (
    "logs/thesis_final_pipeline_v1/stage07_correction_covariance/"
    "run_mlp_v1/measurement_model.json"
)
HULL_MEASUREMENT = REPO / (
    "logs/thesis_final_pipeline_v1/stage07_correction_covariance/"
    "run_hull_R_v1/measurement_model.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    source_task = config["tasks"]["thesis09_west_to_east_north"]
    route_agnostic = dict(source_task["preselected_routes"]["P0"])
    route_available = dict(source_task["preselected_routes"]["P1"])

    config["study_title"] = "Thesis commissioning-story one-seed sanity pilot V1"
    config["study_comparison"] = (
        "Diagnostic, non-inferential three-arm pilot at 1 m/s. HULL and MLP execute "
        "the identical availability-agnostic route with the same learned R; FULL keeps "
        "the MLP and R but executes the previously commissioned-availability route. "
        "Detector, gate, independent fusion, estimator, controller, noise and seed are fixed."
    )
    config["conditions"] = {
        "P0": {
            "label": "HULL_no_NN_matching_learned_R",
            "manager_observation_model": "hull_stage07_r",
            "manager_stage07_measurement_model_path": str(
                HULL_MEASUREMENT.relative_to(REPO)
            ),
            "manager_stage07_measurement_model_expected_sha256": sha256(
                HULL_MEASUREMENT
            ),
        },
        "P1": {
            "label": "MLP_same_route_same_R",
            "manager_observation_model": "hull_mlp",
        },
        "P2": {
            "label": "FULL_MLP_R_availability_route",
            "manager_observation_model": "hull_mlp",
        },
    }
    config["tasks"] = {
        "thesis09_west_to_east_north": {
            "conditions": ["P0", "P1", "P2"],
            "seeds": [900],
            "preselected_routes": {
                "P0": dict(route_agnostic),
                "P1": dict(route_agnostic),
                "P2": dict(route_available),
            },
        }
    }
    config["manager_observation_model"] = "hull_mlp"
    config["manager_covariance_profile"] = "stage07_r2c"
    config["manager_stage07_measurement_model_path"] = str(
        MEASUREMENT.relative_to(REPO)
    )
    config["manager_stage07_measurement_model_expected_sha256"] = sha256(MEASUREMENT)
    config["manager_fusion_rule"] = "independent"
    config["v_max"] = 1.0
    config["max_predict_speed_mps"] = 1.0
    config["local_controller_type"] = "ff_fb"
    config["horizon"] = 20
    config["dt"] = 0.25
    config["local_horizon"] = 20
    config["local_plan_rate"] = 4.0
    config["cmd_publish_rate"] = 10.0
    config["waypoint_spacing_m"] = 0.2
    config["waypoint_arrival_radius_m"] = 0.1
    config["run_timeout_after_first_cmd_s"] = 420
    config["headless"] = True
    config["use_rviz"] = False
    config["ros_domain_id_base"] = 221
    config["cleanup_mode"] = "isolated"
    config["cleanup_sim_stragglers"] = False
    output.write_text(
        yaml.safe_dump(config, sort_keys=False, width=1000), encoding="utf-8"
    )
    print(output)
    print(f"measurement_sha256={sha256(MEASUREMENT)}")
    print(f"hull_measurement_sha256={sha256(HULL_MEASUREMENT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
