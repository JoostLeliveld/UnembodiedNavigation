#!/usr/bin/env python3
"""Development-only canonical one-task planning comparison for M0/M1/M2.

Every camera opportunity is retained. An admitted opportunity contributes
``inv(Rk_run)`` and a miss/refusal contributes zero. Geometry, candidate routes,
dynamics, process noise, objective, prior and feasibility are identical across arms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO / 'src/planning'), str(REPO / 'src/reliability'),
    str(REPO / 'src/unav_common'), str(REPO / 'src/experiments'),
]

from experiments.core.world_profiles import (  # noqa: E402
    serialize_collision_geometry_from_world,
    serialize_driveable_geometry_from_profile,
)
from planning.planners.base_planner import UnicyclePlannerBase  # noqa: E402
from reliability.planning_information import (  # noqa: E402
    fit_constant_planning_information, fit_planning_information_field,
)
from reliability.projection import camera_model_from_world  # noqa: E402
from unav_common.lane_graph_routes import (  # noqa: E402
    generate_route_seeds, repair_route_seeds_for_footprint,
)

# The covariance module's CLI imports its capture loader at module import time.
# This comparison reuses only its pure R0/R1/R2 fitting functions and supplies
# opportunities directly, so install an explicit non-callable shim rather than
# reviving the superseded campaign-analysis module.
_capture_shim = types.ModuleType('analyze_commissioned_model')
_capture_shim.load_capture = lambda *_args, **_kwargs: (_ for _ in ()).throw(
    RuntimeError('capture loader is not part of the R0/R1/R2 planning comparison'))
sys.modules.setdefault('analyze_commissioned_model', _capture_shim)
import run_current_residual_covariance as covariance  # noqa: E402


CAMERAS = tuple(f'camera_{letter}' for letter in 'ABCDE')
METHODS = ('M0_global', 'M1_per_camera', 'M2_spatial')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bool_field(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true'}


def load_development_population(campaign_root: Path, prediction_root: Path):
    print('loading development opportunities and corrected residuals', flush=True)
    profiles = yaml.safe_load(
        (REPO / 'src/experiments/config/world_profiles.yaml').read_text()
    )['worlds']['warehouse_v2.world.sdf']
    world = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
    camera_xy = {
        camera_id: camera_model_from_world(world, include_name=include).cam_pos[:2]
        for camera_id, include in zip(
            profiles['camera_ids'], profiles['camera_model_includes'], strict=True)
    }
    prediction_path = prediction_root / 'candidate_predictions.npz'
    with np.load(prediction_path, allow_pickle=False) as archive:
        predicted = {key: np.asarray(archive[key]) for key in archive.files}
    selected = predicted['partition'].astype(str) == 'development'
    drive = predicted['drive_id'].astype(str)[selected]
    camera_id = predicted['camera_id'].astype(str)[selected]
    stamp = predicted['capture_stamp_ns'].astype(np.int64)[selected]
    frame = predicted['source_frame_id'].astype(str)[selected]
    identities = list(zip(drive, camera_id, stamp, frame, strict=True))
    opportunities = []
    opportunity_by_identity = {}
    for drive_id in sorted(set(drive.tolist())):
        table = campaign_root / drive_id / 'tables/camera_opportunities.csv'
        if not table.is_file():
            recovered = sorted(campaign_root.parent.glob(
                f'*/{drive_id}/tables/camera_opportunities.csv'))
            hashes = {digest(path) for path in recovered}
            if not recovered or len(hashes) != 1:
                raise RuntimeError(
                    f'{drive_id}: recovery opportunity tables are missing or differ')
            table = recovered[0]
        with table.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        opportunities.extend(rows)
        for row in rows:
            key = (row['drive_id'], row['camera_id'], int(row['capture_stamp_ns']),
                   row['source_frame_id'])
            opportunity_by_identity[key] = row
    matched = [opportunity_by_identity[key] for key in identities]
    position = np.asarray([
        [float(row['reference_x']), float(row['reference_y'])] for row in matched
    ])
    basis = []
    for row in matched:
        raw = np.asarray([float(row['raw_projected_x']), float(row['raw_projected_y'])])
        ray = raw - np.asarray(camera_xy[row['camera_id']], dtype=float)
        ray /= np.linalg.norm(ray)
        basis.append(np.stack([ray, np.asarray([-ray[1], ray[0]])], axis=1))
    basis = np.stack(basis)
    residual_ray_raw_basis = (
        predicted['candidate_prediction_ray_m'][selected]
        - predicted['target_ray_m'][selected]
    )
    residual_world = np.einsum('nij,nj->ni', basis, residual_ray_raw_basis)
    correction = predicted['candidate_prediction_ray_m'][selected]
    raw = np.asarray([
        [float(row['raw_projected_x']), float(row['raw_projected_y'])]
        for row in matched
    ])
    corrected = raw + np.einsum('nij,nj->ni', basis, correction)
    covariance_basis = []
    for camera_name, point in zip(camera_id, corrected, strict=True):
        along = point - camera_xy[camera_name]
        along /= np.linalg.norm(along)
        covariance_basis.append(np.column_stack((along, [-along[1], along[0]])))
    covariance_basis = np.stack(covariance_basis)
    residual_ray = np.einsum('nji,nj->ni', covariance_basis, residual_world)
    corrected_by_identity = dict(zip(identities, corrected, strict=True))

    # Query a covariance model for EVERY admitted opportunity in its held-out
    # drive. The residual bundle need only fit the model; it is not allowed to
    # define which admitted opportunities survive into the planning target.
    covariance_by_identity = {method: {} for method in METHODS}
    for held_drive in sorted(set(drive.tolist())):
        print(f'fitting held-drive covariance: {held_drive}', flush=True)
        train = drive != held_drive
        r0, r1 = covariance.fit_constants(
            residual_ray[train], drive[train], camera_id[train])
        query_rows = [
            row for row in opportunities
            if row['drive_id'] == held_drive
            and bool_field(row['sensor_gate_admitted'])
            and (row['drive_id'], row['camera_id'], int(row['capture_stamp_ns']),
                 row['source_frame_id']) in corrected_by_identity
        ]
        query_position = np.asarray([
            [float(row['reference_x']), float(row['reference_y'])] for row in query_rows
        ])
        query_camera = np.asarray([row['camera_id'] for row in query_rows])
        predicted_ray = {}
        predicted_ray['M0_global'], predicted_ray['M1_per_camera'] = (
            covariance.predict_constants(query_camera, r0, r1))
        predicted_ray['M2_spatial'] = covariance.predict_spatial(
            position[train], residual_ray[train], drive[train], camera_id[train],
            query_position, query_camera, r1)
        for method in METHODS:
            predicted_world = []
            for row, covariance_ray in zip(query_rows, predicted_ray[method], strict=True):
                key = (row['drive_id'], row['camera_id'], int(row['capture_stamp_ns']),
                       row['source_frame_id'])
                point = corrected_by_identity[key]
                along = point - camera_xy[row['camera_id']]
                along /= np.linalg.norm(along)
                query_basis = np.column_stack((along, [-along[1], along[0]]))
                predicted_world.append(query_basis @ covariance_ray @ query_basis.T)
            covariance_by_identity[method].update({
                (row['drive_id'], row['camera_id'], int(row['capture_stamp_ns']),
                 row['source_frame_id']): value
                for row, value in zip(query_rows, predicted_world, strict=True)
            })
    return opportunities, covariance_by_identity, prediction_path


def export_fields(opportunities, covariance_by_identity, output: Path, source_hashes):
    positions = np.asarray([
        [float(row['reference_x']), float(row['reference_y'])] for row in opportunities
    ])
    camera_ids = np.asarray([row['camera_id'] for row in opportunities])
    def usable(row):
        key = (row['drive_id'], row['camera_id'], int(row['capture_stamp_ns']),
               row['source_frame_id'])
        return bool_field(row['sensor_gate_admitted']) and key in covariance_by_identity['M0_global']

    admitted = np.asarray([usable(row) for row in opportunities])
    headings = np.asarray([float(row['reference_yaw']) for row in opportunities])
    xs = np.arange(-11.25, 11.25 + 1e-9, 0.50)
    ys = np.arange(-9.25, 9.25 + 1e-9, 0.50)
    paths = {}
    for method in METHODS:
        print(f'fitting direct expected-information field: {method}', flush=True)
        runtime_covariance = np.full((len(opportunities), 2, 2), np.nan)
        for index, row in enumerate(opportunities):
            if not admitted[index]:
                continue
            key = (row['drive_id'], row['camera_id'], int(row['capture_stamp_ns']),
                   row['source_frame_id'])
            runtime_covariance[index] = covariance_by_identity[method][key]
        shape = (len(CAMERAS), len(ys), len(xs), 2, 2)
        support_shape = shape[:3]
        if method == 'M0_global':
            constant, _per_camera = fit_constant_planning_information(
                positions, camera_ids, admitted, runtime_covariance,
                camera_ids=CAMERAS, headings_rad=headings)
            expected_information = np.broadcast_to(constant, shape).copy()
            raw_expected_information = expected_information.copy()
            opportunity_support = np.full(
                support_shape, float(len(np.unique(positions, axis=0))))
        elif method == 'M1_per_camera':
            _global, per_camera = fit_constant_planning_information(
                positions, camera_ids, admitted, runtime_covariance,
                camera_ids=CAMERAS, headings_rad=headings)
            constants = np.stack([per_camera[camera_id] for camera_id in CAMERAS])
            expected_information = np.broadcast_to(
                constants[:, None, None, :, :], shape).copy()
            raw_expected_information = expected_information.copy()
            opportunity_support = np.broadcast_to(np.asarray([
                len(np.unique(positions[camera_ids == camera_id], axis=0))
                for camera_id in CAMERAS
            ])[:, None, None], support_shape).astype(float).copy()
        else:
            field = fit_planning_information_field(
                positions, camera_ids, admitted, runtime_covariance,
                camera_ids=CAMERAS, xs=xs, ys=ys,
                length_scale_m=1.0, support_radius_m=2.0, support_tau=5.0,
                headings_rad=headings,
            )
            expected_information = field.expected_information
            raw_expected_information = field.raw_expected_information
            opportunity_support = field.opportunity_support
        path = output / f'{method}.npz'
        metadata = {
            'schema': 'camera_network.thesis_stage09.v3',
            'status': 'development_only_not_final_evidence',
            'reference': 'robot_ground_reference_xy', 'frame': 'map_bev',
            'covariance_units': 'm2', 'information_units': 'm-2',
            'planning_target': 'admitted_runtime_precision_else_zero',
            'camera_network_model': method,
            'aggregation_resolution': {
                'M0_global': 'one_global_constant_shared_by_cameras',
                'M1_per_camera': 'one_constant_per_camera',
                'M2_spatial': 'one_spatial_field_per_camera',
            }[method],
            'fit_parameters': (
                {'aggregation': 'position_balanced_after_heading_pooling'}
                if method == 'M0_global' else
                {'aggregation': 'per_camera_position_balanced_after_heading_pooling'}
                if method == 'M1_per_camera' else
                {'grid_step_m': 0.5, 'length_scale_m': 1.0,
                 'support_radius_m': 2.0, 'support_tau': 5.0}
            ),
            'source_hashes': source_hashes,
        }
        np.savez_compressed(
            path, xs=xs, ys=ys,
            camera_ids=np.asarray(CAMERAS),
            expected_information_m2_inv=expected_information,
            opportunity_support=opportunity_support,
            raw_expected_information_m2_inv=raw_expected_information,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        paths[method] = path
    return paths


def planner(path: Path, collision_json: str, boundary_json: str,
            active_cameras: tuple[str, ...]):
    return UnicyclePlannerBase(
        horizon=75, dt=1.0, v_min=0.0, v_max=1.0, w_min=-1.0, w_max=1.0,
        control_weight=0.0, process_noise_xy=0.02, process_noise_theta=0.08,
        obs_noise_uv=2.5, goal_sigma_uv=2.0, risk_weight_obs=1.0,
        ambiguity_weight=1.0, optimizer_maxiter=60, optimizer_gtol=1e-5,
        optimizer_warm_start=False, optimizer_maxfun=500, optimizer_ftol=1e-6,
        optimizer_terminal_goal_tolerance_m=0.35, seed=900,
        camera_params={'cam_pos': [-6.0, -10.0, 5.0], 'look_at': [0.0, 0.0, 0.0],
                       'img_width': 1280, 'img_height': 720, 'fov_h_rad': 1.2},
        use_visibility_model=True, camera_network_artifact_path=str(path),
        camera_network_camera_ids=','.join(CAMERAS),
        camera_network_active_camera_ids=','.join(active_cameras),
        camera_network_objective='metric_expected_belief', network_goal_std_m=0.10,
        network_goal_std_start_m=5.0, camera_network_updates_per_step=1,
        kouw_et1_ambiguity=True, terminal_risk_only=False,
        goal_tightening_power=0.9, observation_risk_scale=1.0,
        collision_geometry_json=collision_json, driveable_geometry_json=boundary_json,
        use_nogo_cost=True, nogo_mode='keep_in', nogo_weight=40.0,
        nogo_safe_distance=0.0, nogo_logbarrier_eps=0.05,
        nogo_warning_band=0.05, nogo_near_weight=50.0,
        use_belief_nogo_cost=True, nogo_belief_kappa=1.0,
        robot_collision_radius_m=math.hypot(0.4, 0.275),
        robot_length_m=0.80, robot_width_m=0.55,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--task', default='thesis09_blind_corridor_west_to_east')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    staging = output.with_name(output.name + '.incomplete')
    if output.exists() or staging.exists():
        raise FileExistsError(output if output.exists() else staging)
    staging.mkdir(parents=True)
    print(f'created staging directory: {staging}', flush=True)
    opportunities, covariance_by_identity, prediction_path = load_development_population(
        args.campaign_root.resolve(), args.prediction_root.resolve())
    source_hashes = {
        'development_candidate_predictions.npz': digest(prediction_path),
        'run_one_task_r012_planning.py': digest(Path(__file__)),
        'lane_graph_routes.py': digest(
            REPO / 'src/unav_common/unav_common/lane_graph_routes.py'),
    }
    fields = export_fields(opportunities, covariance_by_identity, staging, source_hashes)
    print('evaluating common route candidates', flush=True)

    profiles_path = REPO / 'src/experiments/config/world_profiles.yaml'
    profile = yaml.safe_load(profiles_path.read_text())['worlds']['warehouse_v2.world.sdf']
    world = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
    collision_json = serialize_collision_geometry_from_world(
        str(world), tuple(profile['collision_model_names']),
        tuple(profile['collision_include_names']), profile)
    boundary_json = serialize_driveable_geometry_from_profile(profile)
    tasks = yaml.safe_load(
        (REPO / 'experiments/thesis_pipeline_lock/stage09_navigation_tasks.yaml').read_text()
    )['tasks'][world.name]
    task = next(item for item in tasks if item['name'] == args.task)
    start = np.asarray([task['start'][key] for key in ('x', 'y', 'yaw')], dtype=float)
    goal = np.asarray([task['goal'][key] for key in ('x', 'y')], dtype=float)
    seeds = generate_route_seeds(collision_json, start[:2], goal)
    seeds = repair_route_seeds_for_footprint(
        seeds, collision_json, boundary_json, start,
        robot_length_m=0.80, robot_width_m=0.55,
        target_clearance_m=0.02,
    )
    prior = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])
    results = {}
    network_states = {
        'intact': CAMERAS,
        'removal_camera_D': tuple(camera for camera in CAMERAS if camera != 'camera_D'),
    }
    for method, path in fields.items():
        results[method] = {}
        for network_state, active_cameras in network_states.items():
            print(f'evaluating planner arm: {method}/{network_state}', flush=True)
            model = planner(path, collision_json, boundary_json, active_cameras)
            rows = []
            for seed in seeds:
                controls = model._controls_for_waypoints(start, seed['waypoints'])
                result = model.evaluate_rollout_controls(start, prior, goal, controls)
                rows.append({
                    'candidate': seed['name'],
                    **{key: result[key] for key in (
                        'total_cost', 'risk_cost', 'ambiguity_cost', 'control_cost',
                        'obstacle_cost', 'terminal_goal_distance_pred',
                        'min_predicted_obstacle_distance_m', 'rollout_valid',
                        'invalid_reason', 'expected_information_trace_mean',
                    ) if key in result},
                    'waypoints': seed['waypoints'],
                })
            feasible = [row for row in rows if row['rollout_valid']
                        and row['terminal_goal_distance_pred'] <= 0.35]
            if not feasible:
                results[method][network_state] = {
                    'status': 'no_feasible_candidate', 'candidates': rows}
                continue
            winner = min(feasible, key=lambda row: (row['total_cost'], row['candidate']))
            results[method][network_state] = {
                'status': 'selected', 'winner': winner, 'candidates': rows}
    report = {
        'schema': 'one_task_m012_direct_information_planning.v1',
        'status': 'development_only_not_final_evidence',
        'task': task, 'opportunities': len(opportunities),
        'admitted_opportunities': sum(bool_field(r['sensor_gate_admitted']) for r in opportunities),
        'candidate_names': [seed['name'] for seed in seeds],
        'invariants': {
            'same_opportunities': True, 'same_geometry': True,
            'same_candidates': True, 'same_dynamics_and_objective': True,
            'only_covariance_method_or_active_camera_set_changes': True,
            'removed_camera': 'camera_D',
        },
        'field_sha256': {method: digest(path) for method, path in fields.items()},
        'results': results, 'source_hashes': source_hashes,
    }
    (staging / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    os.replace(staging, output)
    print(json.dumps({
        method: {state: value.get('winner', {'status': value['status']})
                 for state, value in states.items()}
        for method, states in results.items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
