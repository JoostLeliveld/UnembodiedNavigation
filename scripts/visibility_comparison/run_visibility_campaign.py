#!/usr/bin/env python3
"""Run a visibility-comparison campaign from a locked config file.

Usage:
    python run_visibility_campaign.py --config scripts/visibility_comparison/warehouse_visibility_campaign.yaml [--dry-run] [--resume]

Each run result is written immediately to campaign_log.json so the campaign
can be interrupted and resumed with --resume (already-completed runs are skipped).
Completion reasons are exactly: goal_reached, timeout_after_first_cmd, collision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / 'logs' / 'visibility_comparison'
UNAV_COMMON_SRC = REPO_ROOT / 'src' / 'unav_common'
if str(UNAV_COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(UNAV_COMMON_SRC))

from unav_common.preselected_route import (  # noqa: E402
    canonicalize_polyline_json,
    route_sha256,
    validate_preselected_route,
)

# Map condition ID to planner name (must match ALLOWED_PLANNERS in launch file).
CONDITION_PLANNER = {
    'C0': 'geometric_shortest_path',
    'C1': 'constant_R_efe',
    'C2': 'visibility_aware_efe',
    # Same visibility-aware planner and field as C2; the condition-level
    # use_hit_miss_mixture override selects the Bernoulli observation model.
    'C3': 'visibility_aware_efe',
    # Prospective closed-loop successor: these are route arms, not new planner
    # objectives. Both execute through the same geometry/local-tracker path.
    'gp': 'geometric_shortest_path',
    'mono_depth': 'geometric_shortest_path',
    # The six-arm fusion comparison. Not planner variants: every arm drives ONE frozen
    # preselected route through the same geometry/local-tracker path, so the only thing that
    # differs is how the camera network's readings become one measurement (F1-F4) and what a
    # detector's box is taken to mean (O1, O2). See
    # experiments/fusion_on_fixed_routes/README.md.
    'F1': 'geometric_shortest_path',
    'F2': 'geometric_shortest_path',
    'F3': 'geometric_shortest_path',
    'F4': 'geometric_shortest_path',
    'O1': 'geometric_shortest_path',
    'O2': 'geometric_shortest_path',
    # Commissioning arms: same frozen route and the same fusion rule, differing only in
    # what the commissioning artifact is allowed to say about the cameras.
    'K0': 'geometric_shortest_path',
    'K1': 'geometric_shortest_path',
    'K2': 'geometric_shortest_path',
    'K3': 'geometric_shortest_path',
    # Heading arms: the same frozen route and fusion rule, differing only in whether the
    # camera update is allowed to move the heading through the position-heading covariance.
    'H0': 'geometric_shortest_path',
    'H1': 'geometric_shortest_path',
}

PRESELECTED_ROUTE_KEYS = (
    'preselected_route_json',
    'preselected_route_sha256',
    'preselected_route_source_path',
    'preselected_route_source_sha256',
    'preselected_route_clearance_m',
    'preselected_route_endpoint_tolerance_m',
    'preselected_route_sample_step_m',
)


def _load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg, path)
    return cfg


def _validate_config(cfg: dict, path: Path) -> None:
    for key in ('world', 'launch_file', 'conditions', 'tasks',
                'yolo_model', 'horizon', 'dt', 'goal_success_radius',
                'run_timeout_after_first_cmd_s'):
        if key not in cfg:
            raise RuntimeError(f"Campaign config {path} is missing required key: '{key}'")
    for key in ('yolo_model',):
        if str(cfg[key]).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} has unfilled placeholder for '{key}': {cfg[key]!r}\n"
                f"Fill in all [FILL] entries before running."
            )
    for key in ('observation_risk_scale', 'ambiguity_term_scale'):
        if key in cfg and str(cfg[key]).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} has unfilled placeholder for '{key}': {cfg[key]!r}\n"
                f"Verify the lambda mapping against the planner source and fill it in."
            )
    for condition_id in cfg['conditions']:
        if condition_id not in CONDITION_PLANNER:
            raise RuntimeError(
                f"Campaign config {path} uses unsupported active condition '{condition_id}'. "
                f"Allowed conditions are: {', '.join(CONDITION_PLANNER)}"
            )
        condition_cfg = cfg['conditions'].get(condition_id, {}) or {}
        if not isinstance(condition_cfg, dict):
            raise RuntimeError(
                f"Condition '{condition_id}' in {path} must be a mapping"
            )
        declared_planner = condition_cfg.get('planner')
        expected_planner = CONDITION_PLANNER[condition_id]
        if declared_planner is not None and str(declared_planner) != expected_planner:
            raise RuntimeError(
                f"Condition '{condition_id}' declares planner {declared_planner!r}, "
                f"but the active runner contract requires {expected_planner!r}"
            )
    for task_name, task_cfg in cfg['tasks'].items():
        for condition_id in task_cfg.get('conditions', []):
            if condition_id not in CONDITION_PLANNER:
                raise RuntimeError(
                    f"Task '{task_name}' in {path} uses unsupported active condition "
                    f"'{condition_id}'. Allowed conditions are: {', '.join(CONDITION_PLANNER)}"
                )

    active_cells = [
        (task_name, condition_id)
        for task_name, task_cfg in cfg['tasks'].items()
        for condition_id in task_cfg.get('conditions', [])
    ]
    needs_gp = any(
        CONDITION_PLANNER[condition_id] == 'visibility_aware_efe'
        and str(_effective_value(cfg, task_name, condition_id, 'global_planner_mode') or 'efe')
        != 'preselected_route'
        for task_name, condition_id in active_cells
    )
    if needs_gp:
        if 'gp_artifact' not in cfg or str(cfg.get('gp_artifact', '')).startswith('[FILL'):
            raise RuntimeError(
                f"Campaign config {path} requires a filled gp_artifact for its "
                "visibility-aware global-planner condition"
            )

    if any(
        str(_effective_value(cfg, task_name, condition_id, 'global_planner_mode') or '')
        == 'preselected_route'
        for task_name, condition_id in active_cells
    ):
        _validate_preselected_campaign_routes(cfg, path)


def _terminate_process_group(pgid: int, *, grace_s: float = 3.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(grace_s)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


_OWN_NODE_PATTERNS = [
    'yolo_robot_detector_node',
    'pixel_to_bev_state_node',
    'goal_mission_node',
    'goal_marker_node',
    'experiment_logger',
    'install/planning/lib/planning/efe_agent',
    'wait_for_odom',
    'wait_for_clock',
    'reset_world',
    # Sim noise nodes MUST be reaped: a leftover encoder_noise_node from a prior
    # run publishes a second (pi-offset) heading on /odom_noisy, corrupting the
    # planner's heading. This caused C1's spurious start-divergence (2026-06-03).
    'encoder_noise_node',
    'actuation_noise_node',
]


def _reap_own_node_stragglers() -> None:
    for pattern in _OWN_NODE_PATTERNS:
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _reap_sim_stragglers(world: str) -> None:
    """Stop stale Gazebo servers for this campaign world.

    Gazebo Transport is outside ROS_DOMAIN_ID. A leftover server with the same
    world name can publish duplicate clocks/sensors into a fresh ROS run.
    """
    world_name = str(world or '').strip()
    if not world_name:
        return
    for pattern in (
        f'ign gazebo.*{world_name}',
        f'ruby /usr/bin/ign gazebo.*{world_name}',
        f'gz sim.*{world_name}',
    ):
        try:
            subprocess.run(['pkill', '-f', pattern], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass


# Heavy stragglers that were NOT being reaped and accumulate CPU across a campaign:
# the ros_gz bridges, robot_state_publisher, rviz, and the bare Gazebo server/client
# (which often re-parents away from the launch process group and so survives killpg).
_EXTRA_STRAGGLER_PATTERNS = [
    'ros_gz_bridge', 'parameter_bridge', 'ros_gz_image', 'image_bridge',
    'robot_state_publisher', 'rviz', 'static_transform_publisher',
    'ros2 launch experiments', 'spawn_entity',
]
_SIM_STRAGGLER_PATTERNS = [
    'ign gazebo', 'gz sim', 'gzserver', 'gzclient', 'ruby /usr/bin/ign gazebo',
]


def _all_straggler_patterns() -> list:
    return list(_OWN_NODE_PATTERNS) + _EXTRA_STRAGGLER_PATTERNS + _SIM_STRAGGLER_PATTERNS


def _patterns_alive(patterns) -> list:
    alive = []
    for p in patterns:
        try:
            r = subprocess.run(['pgrep', '-f', p], capture_output=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                alive.append(p)
        except (subprocess.TimeoutExpired, OSError):
            pass
    return alive


def _force_fresh(*, settle_s: float = 1.5, max_wait_s: float = 25.0) -> None:
    """Guarantee a clean slate before/after a run: SIGTERM all known run processes,
    then SIGKILL and POLL until none remain (or timeout). This prevents Gazebo/bridge
    stragglers from accumulating and slowly starving the next run's global solve."""
    patterns = _all_straggler_patterns()
    for p in patterns:
        try:
            subprocess.run(['pkill', '-TERM', '-f', p], timeout=2, capture_output=True, check=False)
        except (subprocess.TimeoutExpired, OSError):
            pass
    time.sleep(settle_s)
    deadline = time.time() + max_wait_s
    while True:
        for p in patterns:
            try:
                subprocess.run(['pkill', '-KILL', '-f', p], timeout=2, capture_output=True, check=False)
            except (subprocess.TimeoutExpired, OSError):
                pass
        time.sleep(0.5)
        alive = _patterns_alive(patterns)
        if not alive:
            return
        if time.time() > deadline:
            print(f'  WARN: stragglers still alive after force-clean: {alive}')
            return


def _run_key(task: str, condition: str, seed: int) -> str:
    return f'{task}__{condition}__seed{seed}'


def _load_run_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}
    try:
        return json.loads(log_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_run_log(log_path: Path, log: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')


def _resolve_repo_path(path_str: str, *, strict: bool = False) -> Path:
    """Resolve config paths, treating relative paths as repo-root relative."""
    expanded = os.path.expandvars(str(path_str))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=strict)


def _resolve_for_compare(path_str: str) -> Path:
    return _resolve_repo_path(path_str, strict=False)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')


def _route_overrides(cfg: dict, task_name: str, condition_id: str) -> dict:
    task_cfg = cfg.get('tasks', {}).get(task_name, {}) or {}
    routes = task_cfg.get('preselected_routes', {}) or {}
    if not isinstance(routes, dict):
        raise RuntimeError(
            f"Task '{task_name}' preselected_routes must be a mapping by condition"
        )
    route_cfg = routes.get(condition_id, {}) or {}
    if not isinstance(route_cfg, dict):
        raise RuntimeError(
            f"Task '{task_name}' route '{condition_id}' must be a mapping"
        )
    return route_cfg


def _effective_value(cfg: dict, task_name: str, condition_id: str, key: str):
    """Resolve route > condition > task > campaign overrides deterministically."""

    task_cfg = cfg.get('tasks', {}).get(task_name, {}) or {}
    condition_cfg = cfg.get('conditions', {}).get(condition_id, {}) or {}
    route_cfg = _route_overrides(cfg, task_name, condition_id)
    if key in route_cfg:
        return route_cfg[key]
    if key in condition_cfg:
        return condition_cfg[key]
    if key in task_cfg:
        return task_cfg[key]
    return cfg.get(key)


def _task_spec_from_yaml(cfg: dict, task_name: str, config_path: Path) -> dict:
    tasks_path = _resolve_repo_path(
        cfg.get('tasks_yaml', 'src/experiments/config/tasks.yaml'), strict=True
    )
    payload = yaml.safe_load(tasks_path.read_text(encoding='utf-8')) or {}
    world_tasks = (payload.get('tasks') or {}).get(cfg['world'])
    if not isinstance(world_tasks, list):
        raise RuntimeError(
            f"Campaign {config_path}: no tasks list for world {cfg['world']!r} "
            f"in {tasks_path}"
        )
    for task in world_tasks:
        if isinstance(task, dict) and task.get('name') == task_name:
            return task
    raise RuntimeError(
        f"Campaign {config_path}: task {task_name!r} is not registered for "
        f"world {cfg['world']!r} in {tasks_path}"
    )


def _profile_driveable_geometry(cfg: dict, config_path: Path) -> str:
    profiles_path = _resolve_repo_path(
        cfg.get('world_profiles', 'src/experiments/config/world_profiles.yaml'),
        strict=True,
    )
    payload = yaml.safe_load(profiles_path.read_text(encoding='utf-8')) or {}
    profile = (payload.get('worlds') or {}).get(cfg['world'])
    if not isinstance(profile, dict):
        raise RuntimeError(
            f"Campaign {config_path}: world {cfg['world']!r} is not present in "
            f"{profiles_path}"
        )
    prisms = []
    for region in profile.get('known_2d_regions', []) or []:
        if str(region.get('type', '')).strip().lower() != 'traversable':
            continue
        try:
            prisms.append({
                'name': str(region.get('name', 'lane')),
                'xmin': float(region['xmin']),
                'xmax': float(region['xmax']),
                'ymin': float(region['ymin']),
                'ymax': float(region['ymax']),
                'zmin': 0.0,
                'zmax': 0.1,
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Campaign {config_path}: malformed traversable region {region!r}"
            ) from exc
    if not prisms:
        raise RuntimeError(
            f"Campaign {config_path}: world profile has no traversable driveable regions"
        )
    return json.dumps({'prisms': prisms, 'model_name': 'driveable_region'})


def _validate_preselected_campaign_routes(cfg: dict, config_path: Path) -> None:
    """Make ``--dry-run`` exercise the same route identity/geometry gate as launch."""

    profile_driveable = None
    for task_name, task_cfg in cfg['tasks'].items():
        for condition_id in task_cfg.get('conditions', []):
            mode = str(
                _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or ''
            ).strip().lower()
            route_cfg = _route_overrides(cfg, task_name, condition_id)
            if mode != 'preselected_route':
                if route_cfg:
                    raise RuntimeError(
                        f"Task {task_name!r} condition {condition_id!r} supplies a "
                        "preselected route but global_planner_mode is not preselected_route"
                    )
                continue
            if not _as_bool(
                _effective_value(cfg, task_name, condition_id, 'use_hierarchical')
            ):
                raise RuntimeError(
                    f"Task {task_name!r} condition {condition_id!r}: "
                    "preselected_route requires use_hierarchical=true"
                )
            if _as_bool(
                _effective_value(cfg, task_name, condition_id, 'use_truth_localization')
            ):
                raise RuntimeError(
                    f"Task {task_name!r} condition {condition_id!r}: control from "
                    "ground truth is forbidden by the closed-loop contract"
                )
            identity_keys = PRESELECTED_ROUTE_KEYS[:4]
            missing = [key for key in identity_keys if not str(route_cfg.get(key, '') or '')]
            if missing:
                raise RuntimeError(
                    f"Task {task_name!r} condition {condition_id!r} is missing "
                    f"route-specific fields: {', '.join(missing)}"
                )

            task = _task_spec_from_yaml(cfg, task_name, config_path)
            if task.get('waypoints'):
                raise RuntimeError(
                    f"Task {task_name!r} is a waypoint tour; preselected_route accepts "
                    "one start-to-goal polyline"
                )
            try:
                start = (float(task['start']['x']), float(task['start']['y']))
                goal = (float(task['goal']['x']), float(task['goal']['y']))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Task {task_name!r} has malformed start/goal coordinates"
                ) from exc

            driveable = _effective_value(
                cfg, task_name, condition_id, 'driveable_geometry_json'
            )
            if not str(driveable or '').strip():
                if profile_driveable is None:
                    profile_driveable = _profile_driveable_geometry(cfg, config_path)
                driveable = profile_driveable

            endpoint_tolerance = float(
                _effective_value(
                    cfg,
                    task_name,
                    condition_id,
                    'preselected_route_endpoint_tolerance_m',
                )
                or 0.25
            )
            sample_step = float(
                _effective_value(
                    cfg, task_name, condition_id, 'preselected_route_sample_step_m'
                )
                or 0.04
            )
            if not 0.0 <= endpoint_tolerance <= 0.25:
                raise RuntimeError(
                    "preselected_route_endpoint_tolerance_m must be within [0, 0.25]"
                )
            if not 0.0 < sample_step <= 0.04:
                raise RuntimeError(
                    "preselected_route_sample_step_m must be within (0, 0.04]"
                )
            source_path = _resolve_repo_path(
                route_cfg['preselected_route_source_path'], strict=True
            )
            try:
                validate_preselected_route(
                    route_cfg['preselected_route_json'],
                    route_cfg['preselected_route_sha256'],
                    start_xy=start,
                    goal_xy=goal,
                    driveable_geometry_json=str(driveable),
                    declared_clearance_m=float(
                        _effective_value(
                            cfg,
                            task_name,
                            condition_id,
                            'preselected_route_clearance_m',
                        )
                        or 0.25
                    ),
                    source_path=source_path,
                    expected_source_sha256=route_cfg[
                        'preselected_route_source_sha256'
                    ],
                    endpoint_tolerance_m=endpoint_tolerance,
                    sample_step_m=sample_step,
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Campaign {config_path}: route gate failed for task "
                    f"{task_name!r}, condition {condition_id!r}: {exc}"
                ) from exc


def _float_close(a, b, *, tol: float = 1e-8) -> bool:
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= tol


def _load_run_manifest(run_dir: Path) -> dict:
    p = run_dir / 'run_manifest.json'
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}


def _verify_preselected_run_artifacts(
    run_dir: Path,
    cfg: dict,
    task_name: str,
    condition_id: str,
    *,
    manifest: dict | None = None,
) -> tuple[bool, str]:
    """Require the executed route bytes and no-global-solve provenance on disk."""

    mode = str(
        _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or ''
    ).strip().lower()
    if mode != 'preselected_route':
        return True, ''
    route_cfg = _route_overrides(cfg, task_name, condition_id)
    try:
        points, canonical = canonicalize_polyline_json(
            route_cfg['preselected_route_json']
        )
    except (KeyError, ValueError) as exc:
        return False, f'invalid configured preselected route: {exc}'
    expected_hash = str(route_cfg.get('preselected_route_sha256', '') or '')
    if route_sha256(canonical) != expected_hash:
        return False, 'configured canonical preselected-route hash no longer matches'
    expected_source = str(
        _resolve_repo_path(route_cfg.get('preselected_route_source_path', ''), strict=False)
    )
    expected_source_hash = str(
        route_cfg.get('preselected_route_source_sha256', '') or ''
    )

    run_manifest = manifest if manifest is not None else _load_run_manifest(run_dir)
    strict_manifest_values = {
        'global_planner_mode': 'preselected_route',
        'preselected_route_json': canonical,
        'preselected_route_sha256': expected_hash,
        'preselected_route_source_path': expected_source,
        'preselected_route_source_sha256': expected_source_hash,
    }
    for key, expected in strict_manifest_values.items():
        actual = str(run_manifest.get(key, '') or '')
        if key == 'preselected_route_source_path':
            if _resolve_for_compare(actual) != _resolve_for_compare(expected):
                return False, f'run manifest {key} mismatch'
        elif actual != expected:
            return False, f'run manifest {key} mismatch'

    route_path = run_dir / 'preselected_route.json'
    try:
        route_bytes = route_path.read_bytes()
        route_text = route_bytes.decode('utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        return False, f'missing/malformed exact preselected route artifact: {exc}'
    if route_text != canonical:
        return False, 'preselected_route.json bytes differ from canonical campaign route'
    if route_sha256(route_text) != expected_hash:
        return False, 'preselected_route.json has the wrong route SHA-256'

    provenance_path = run_dir / 'preselected_route_provenance.json'
    meta_path = run_dir / 'global_plan_meta.json'
    try:
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f'missing/malformed preselected route provenance: {exc}'
    if provenance.get('validation_status') != 'passed':
        return False, 'preselected route provenance does not record a passed gate'
    if provenance.get('route_sha256') != expected_hash:
        return False, 'preselected route provenance hash mismatch'
    if provenance.get('route_points') != [[x, y] for x, y in points]:
        return False, 'preselected route provenance coordinates mismatch'
    if provenance.get('source_sha256') != expected_source_hash:
        return False, 'preselected route source provenance hash mismatch'
    if _resolve_for_compare(provenance.get('source_path', '')) != _resolve_for_compare(
        expected_source
    ):
        return False, 'preselected route source provenance path mismatch'
    if meta.get('global_planner_mode') != 'preselected_route':
        return False, 'global plan metadata does not identify preselected_route mode'
    if meta.get('global_solve_invoked') is not False:
        return False, 'global plan metadata does not prove the global solve was skipped'
    if meta.get('route_sha256') != expected_hash:
        return False, 'global plan metadata route hash mismatch'
    return True, ''


def _existing_entry_matches_config(entry: dict, cfg: dict) -> tuple[bool, str]:
    condition_id = str(entry.get('condition', ''))
    run_dir_str = str(entry.get('run_dir', '') or '')
    if not run_dir_str:
        return False, 'missing run_dir'
    manifest = _load_run_manifest(Path(run_dir_str))
    if not manifest:
        return False, f'missing run_manifest.json in {run_dir_str}'

    task_name = str(entry.get('task', '') or '')

    def expected_value(key: str):
        return _effective_value(cfg, task_name, condition_id, key)

    expected_yolo_model = str(_resolve_repo_path(cfg['yolo_model'], strict=False))
    actual_yolo_model = str(manifest.get('yolo_model', '') or '')
    if _resolve_for_compare(actual_yolo_model) != _resolve_for_compare(expected_yolo_model):
        return False, f'yolo_model mismatch: run used {actual_yolo_model or "<missing>"}, config expects {expected_yolo_model}'

    numeric_keys = (
        'horizon', 'dt', 'goal_success_radius', 'goal_success_hold_s',
        'optimizer_terminal_goal_tolerance_m',
        'run_timeout_after_first_cmd_s', 'r_visible_uv', 'r_miss_uv',
        'process_noise_xy', 'process_noise_theta', 'risk_weight_obs',
        'ambiguity_weight', 'observation_risk_scale', 'ambiguity_term_scale',
        'control_weight', 'v_max', 'discount_gamma', 'yolo_conf_threshold',
        'yolo_iou_threshold', 'odom_heading_timeout_s', 'optimizer_maxiter',
        'optimizer_maxfun', 'optimizer_ftol', 'optimizer_gtol',
        'goal_prior_u_std_start', 'goal_prior_v_std_start',
        'goal_prior_u_std_final', 'goal_prior_v_std_final',
        'goal_tightening_power', 'nogo_weight', 'nogo_safe_distance',
        'nogo_logbarrier_eps',
        'nogo_belief_kappa',
        'pixel_correction_nis_threshold',
        'robot_collision_radius_m',
        'global_horizon', 'global_dt', 'local_horizon', 'local_plan_rate',
        'local_optimizer_maxiter', 'local_nogo_weight',
        'local_nogo_safe_distance',
        'local_goal_prior_u_std_start', 'local_goal_prior_v_std_start',
        'local_goal_prior_u_std_final', 'local_goal_prior_v_std_final',
        'waypoint_spacing_m', 'waypoint_arrival_radius_m',
        'local_replan_min_remaining_s', 'cmd_publish_rate',
        'encoder_noise_linear_slip_mean',
        'encoder_noise_linear_slip_std',
        'encoder_noise_angular_slip_mean',
        'encoder_noise_angular_slip_std',
        'encoder_noise_linear_additive_std',
        'encoder_noise_angular_additive_std',
        'encoder_noise_correlation_alpha',
        'preselected_route_clearance_m',
        'preselected_route_endpoint_tolerance_m',
        'preselected_route_sample_step_m',
    )
    for key in numeric_keys:
        expected = expected_value(key)
        if expected is not None and key in manifest and not _float_close(manifest.get(key), expected):
            return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")}, config expects {expected}'

    bool_keys = (
        'use_nogo_cost',
        'use_belief_nogo_cost',
        'use_command_noise',
        'use_encoder_noise',
        'use_odom_for_predict',
        'optimizer_multistart',
        'optimizer_multistart_include_direct',
        'use_hierarchical',
        'global_use_ambiguity',
        'local_use_ambiguity',
        'local_use_obs_risk',
        'global_optimizer_multistart',
        'local_optimizer_multistart',
        'local_use_visibility_model',
        'local_use_belief_nogo_cost',
        'local_replan_on_waypoint_change',
        'latency_compensate_plan_handoff',
        'use_truth_localization',
        'use_hit_miss_mixture',
    )
    for key in bool_keys:
        expected = expected_value(key)
        if expected is not None and key in manifest and bool(manifest.get(key)) != bool(expected):
            return False, f'{key} mismatch: run used {manifest.get(key)}, config expects {expected}'

    string_keys = (
        'nogo_mode',
        'nogo_penalty_type',
        'yolo_device',
        'optimizer_initial_routes_json',
        'optimizer_route_seed_mode',
        'local_nogo_penalty_type',
        'heading_update_mode',
        'state_correction_mode',
        'local_controller_type',
        'global_planner_mode',
    )
    for key in string_keys:
        expected = expected_value(key)
        if expected is not None and key in manifest and str(manifest.get(key, '')) != str(expected):
            return False, f'{key} mismatch: run used {manifest.get(key, "<missing>")!r}, config expects {expected!r}'

    route_matches, route_reason = _verify_preselected_run_artifacts(
        Path(run_dir_str), cfg, task_name, condition_id, manifest=manifest
    )
    if not route_matches:
        return False, route_reason

    # Only an actually solved visibility-aware planner consumes the GP; C1
    # (constant_R_efe) and C0 (geometric_shortest_path) are camera-model-free.
    if (
        CONDITION_PLANNER.get(condition_id) == 'visibility_aware_efe'
        and str(expected_value('global_planner_mode') or 'efe') != 'preselected_route'
    ):
        actual = str(manifest.get('visibility_artifact_path', '') or '')
        expected = str(_resolve_repo_path(cfg['gp_artifact'], strict=False))
        if _resolve_for_compare(actual) != _resolve_for_compare(expected):
            return (
                False,
                f'visibility_artifact_path mismatch: run used {actual or "<missing>"}, config expects {expected}',
            )
    return True, ''


def _build_run_matrix(cfg: dict) -> list[tuple[str, str, int]]:
    """Return list of (task_name, condition_id, seed) in deterministic order."""
    runs = []
    for task_name, task_cfg in cfg['tasks'].items():
        for condition_id in task_cfg['conditions']:
            for seed in task_cfg['seeds']:
                runs.append((task_name, condition_id, seed))
    return runs


def _ros_domain_for_run(cfg: dict, run_idx: int) -> str | None:
    """Return the per-run ROS domain when campaign isolation is enabled."""
    if 'ros_domain_id_base' not in cfg:
        return None
    return str(int(cfg['ros_domain_id_base']) + int(run_idx))


def _build_launch_cmd(cfg: dict, task_name: str, condition_id: str, seed: int, log_dir: Path) -> list[str]:
    planner = CONDITION_PLANNER[condition_id]
    global_mode = str(
        _effective_value(cfg, task_name, condition_id, 'global_planner_mode') or 'efe'
    ).strip().lower()
    gp_artifact = None
    if planner == 'visibility_aware_efe' and global_mode != 'preselected_route':
        gp_artifact = str(_resolve_repo_path(cfg['gp_artifact'], strict=True))
    yolo_model = str(_resolve_repo_path(cfg['yolo_model'], strict=True))
    odom_topic = str(cfg.get('odom_topic', '/odom_noisy'))
    if not bool(cfg.get('use_encoder_noise', True)) and odom_topic == '/odom_noisy':
        odom_topic = '/odom'

    cmd = [
        'ros2', 'launch', 'experiments', str(cfg['launch_file']),
        f'world:={cfg["world"]}',
        f'task:={task_name}',
        f'planner:={planner}',
        f'seed:={seed}',
        f'log_dir:={log_dir}',
        f'perception_backend:={cfg.get("perception_backend", "yolo")}',
        f'horizon:={cfg["horizon"]}',
        f'dt:={cfg["dt"]}',
        f'goal_success_radius:={cfg["goal_success_radius"]}',
        f'goal_success_hold_s:={cfg.get("goal_success_hold_s", 2.0)}',
        f'run_timeout_after_first_cmd_s:={cfg["run_timeout_after_first_cmd_s"]}',
        f'auto_stop_on_goal:=true',
        f'headless:={str(cfg.get("headless", False)).lower()}',
        f'use_rviz:={str(cfg.get("use_rviz", False)).lower()}',
        f'reset_world:={str(cfg.get("reset_world", False)).lower()}',
        f'r_visible_uv:={cfg.get("r_visible_uv", 2.5)}',
        f'r_miss_uv:={cfg.get("r_miss_uv", 120.0)}',
        f'discount_gamma:={cfg.get("discount_gamma", 0.98)}',
        f'v_max:={cfg.get("v_max", 0.22)}',
        f'max_predict_speed_mps:={cfg.get("max_predict_speed_mps", 0.0)}',
        f'state_correction_mode:={cfg.get("state_correction_mode", "fused")}',
        f'state_max_correction_jump_m:={cfg.get("state_max_correction_jump_m", 0.0)}',
        f'process_noise_xy:={cfg.get("process_noise_xy", 0.01)}',
        f'process_noise_theta:={cfg.get("process_noise_theta", 0.02)}',
        f'control_weight:={cfg.get("control_weight", 0.0)}',
        f'optimizer_maxiter:={cfg.get("optimizer_maxiter", 80)}',
        f'optimizer_maxfun:={cfg.get("optimizer_maxfun", 500)}',
        f'optimizer_multistart:={str(cfg.get("optimizer_multistart", False)).lower()}',
        f'optimizer_multistart_include_direct:={str(cfg.get("optimizer_multistart_include_direct", True)).lower()}',
        f'optimizer_terminal_goal_tolerance_m:={cfg.get("optimizer_terminal_goal_tolerance_m", 0.0)}',
        f'use_command_noise:={str(cfg.get("use_command_noise", True)).lower()}',
        f'use_encoder_noise:={str(cfg.get("use_encoder_noise", True)).lower()}',
        f'use_odom_for_predict:={str(cfg.get("use_odom_for_predict", True)).lower()}',
        f'odom_topic:={odom_topic}',
        f'command_noise_linear_slip_mean:={cfg.get("command_noise_linear_slip_mean", 0.03)}',
        f'command_noise_linear_slip_std:={cfg.get("command_noise_linear_slip_std", 0.06)}',
        f'command_noise_angular_slip_std:={cfg.get("command_noise_angular_slip_std", 0.04)}',
        f'command_noise_linear_additive_std:={cfg.get("command_noise_linear_additive_std", 0.008)}',
        f'command_noise_angular_additive_std:={cfg.get("command_noise_angular_additive_std", 0.035)}',
        f'command_noise_correlation_alpha:={cfg.get("command_noise_correlation_alpha", 0.85)}',
        f'encoder_noise_linear_slip_mean:={cfg.get("encoder_noise_linear_slip_mean", 0.02)}',
        f'encoder_noise_linear_slip_std:={cfg.get("encoder_noise_linear_slip_std", 0.05)}',
        f'encoder_noise_angular_slip_mean:={cfg.get("encoder_noise_angular_slip_mean", 0.0)}',
        f'encoder_noise_angular_slip_std:={cfg.get("encoder_noise_angular_slip_std", 0.03)}',
        f'encoder_noise_linear_additive_std:={cfg.get("encoder_noise_linear_additive_std", 0.004)}',
        f'encoder_noise_angular_additive_std:={cfg.get("encoder_noise_angular_additive_std", 0.020)}',
        f'encoder_noise_correlation_alpha:={cfg.get("encoder_noise_correlation_alpha", 0.80)}',
        f'odom_heading_timeout_s:={cfg.get("odom_heading_timeout_s", 0.75)}',
        f'yolo_model:={yolo_model}',
        f'yolo_device:={cfg.get("yolo_device", "")}',
        f'yolo_imgsz:={cfg.get("yolo_imgsz", 640)}',
        f'yolo_conf_threshold:={cfg.get("yolo_conf_threshold", 0.25)}',
        f'yolo_iou_threshold:={cfg.get("yolo_iou_threshold", 0.45)}',
        f'yolo_target_class:={cfg.get("yolo_target_class", "robot")}',
        f'yolo_class_id:={cfg.get("yolo_class_id", -1)}',
        f'yolo_use_masks:={str(cfg.get("yolo_use_masks", True)).lower()}',
        f'yolo_min_mask_area_px:={cfg.get("yolo_min_mask_area_px", 12.0)}',
        f'yolo_mask_bottom_band_px:={cfg.get("yolo_mask_bottom_band_px", 3.0)}',
        f'yolo_min_bbox_area_px:={cfg.get("yolo_min_bbox_area_px", 0.0)}',
        f'yolo_debug_frame_dir:={cfg.get("yolo_debug_frame_dir", "")}',
        f'yolo_use_torchscript:={str(cfg.get("yolo_use_torchscript", False)).lower()}',
        f'yolo_warmup_iters:={cfg.get("yolo_warmup_iters", 3)}',
        f'yolo_inference_in_callback:={str(cfg.get("yolo_inference_in_callback", True)).lower()}',
    ]
    if global_mode == 'preselected_route':
        cmd.append(f'comparison_method_id:=closed_loop_{condition_id}')

    # Planner-specific args: pass GP artifact only for the visibility-aware
    # planner (C2). C1 (constant_R_efe) and C0 (geometric_shortest_path) are
    # camera-model-free and must not receive it.
    if gp_artifact is not None:
        cmd.append(f'visibility_artifact_path:={gp_artifact}')

    for key in (
        'observation_risk_scale', 'ambiguity_term_scale',
        'risk_weight_obs', 'ambiguity_weight',
        'belief_publish_rate',
        'heading_update_mode',
        'use_pixel_correction', 'pixel_topic', 'command_noise_output_topic',
        'pixel_timeout_s', 'skip_stale_pixel_correction',
        'bev_y_calibration_offset_m', 'bev_affine_calibration', 'pixel_max_correction_jump_m',
        'pixel_correction_nis_threshold', 'use_truth_localization',
        'debug_runtime',
        'optimizer_ftol', 'optimizer_gtol', 'optimizer_warm_start',
        'optimizer_initial_routes_json',
        'optimizer_terminal_goal_tolerance_m',
        'optimizer_route_seed_mode',
        'driveable_geometry_json',
        'use_hierarchical', 'global_horizon', 'global_dt', 'local_horizon',
        'local_plan_rate', 'local_optimizer_maxiter',
        'global_use_ambiguity', 'local_use_ambiguity', 'local_use_obs_risk',
        'global_optimizer_multistart', 'local_optimizer_multistart',
        'local_use_visibility_model', 'local_use_belief_nogo_cost',
        'local_nogo_penalty_type', 'local_nogo_weight',
        'local_nogo_safe_distance',
        'local_goal_prior_u_std_start', 'local_goal_prior_v_std_start',
        'local_goal_prior_u_std_final', 'local_goal_prior_v_std_final',
        'waypoint_spacing_m', 'waypoint_arrival_radius_m',
        'local_replan_min_remaining_s', 'local_replan_on_waypoint_change',
        'latency_compensate_plan_handoff',
        'local_controller_type',
        'cmd_publish_rate',
        'goal_prior_u_std_start', 'goal_prior_v_std_start',
        'goal_prior_u_std_final', 'goal_prior_v_std_final',
        'goal_tightening_power',
        'use_nogo_cost', 'nogo_mode', 'nogo_penalty_type', 'nogo_weight',
        'nogo_safe_distance',
        'nogo_logbarrier_eps', 'nogo_warning_band', 'nogo_near_weight',
        'use_belief_nogo_cost',
        'nogo_belief_kappa',
        'use_hit_miss_mixture',
        'robot_collision_radius_m',
        'terminate_on_geom_collision',
        'global_planner_mode',
        *PRESELECTED_ROUTE_KEYS,
        'bridge_camera_b', 'bridge_camera_c', 'bridge_camera_d',
        'multicam_belief', 'manager_gp_artifact_template',
        'manager_min_spatial_trust',
        'manager_decision_rate_hz',
        'manager_fusion_disagreement_gate_m',
        'manager_fusion_max_timestamp_spread_s',
        'manager_covariance_profile',
        'manager_commissioned_calibration_path', 'manager_commissioned_sigma_px',
        'manager_commissioned_per_camera_sigma',
        'manager_fusion_common_mode_std_m',
        'manager_fusion_rule', 'manager_observation_model', 'manager_fixed_offset_m',
        'manager_correction_timestamp_compensation', 'manager_admission_gate',
        'manager_correction_residual_interval_s',
        'manager_correction_propagation_drift_std',
        'manager_max_measurement_age_s', 'manager_age_decay_s',
        'manager_min_association_confidence',
        'manager_required_consecutive_better_frames',
        'manager_max_cross_camera_disagreement_m',
        'manager_require_consistency_when_source_available',
        'manager_camera_ids', 'manager_fusion_mode', 'manager_require_gp_artifacts',
        'state_correction_ekf',
        'wait_for_belief_before_first_goal', 'initial_belief_max_sigma_m',
        'multicam_scheduled', 'scheduled_coverage_artifact',
        'scheduled_report_std_m', 'scheduled_rate_hz',
        'stuck_window_s', 'stuck_max_displacement_m',
        'stuck_max_goal_improvement_m', 'stuck_cmd_fraction_min',
        'stuck_idle_cmd_fraction_max',
    ):
        val = _effective_value(cfg, task_name, condition_id, key)
        if key == 'preselected_route_json' and val is not None:
            _points, val = canonicalize_polyline_json(str(val))
        elif key == 'preselected_route_source_path' and val is not None:
            val = str(_resolve_repo_path(str(val), strict=True))
        if val is not None and not str(val).startswith('[FILL'):
            cmd.append(f'{key}:={val}')

    # Drop empty-valued launch args (e.g. yolo_device when unset): ros2 launch
    # rejects a bare 'name:=', and omitting it lets the launch file use its own
    # default (the behaviour earlier runs relied on).
    cmd = [arg for arg in cmd if not arg.endswith(':=')]
    return cmd


def _read_run_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / 'run_summary.json'
    if not summary_path.is_file():
        return None
    try:
        return json.loads(summary_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def _find_latest_run_dir(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None
    candidates = sorted(log_dir.iterdir(), reverse=True)
    for d in candidates:
        if d.is_dir() and (d / 'run_summary.json').is_file():
            return d
    return None


def _find_latest_experiment_dir(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None
    candidates = sorted((d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('experiment_')), reverse=True)
    return candidates[0] if candidates else None


def _command_activity(run_dir: Path | None) -> tuple[int, bool]:
    """Return (experiment rows, whether any nonzero command has been logged)."""
    if run_dir is None:
        return 0, False
    path = run_dir / 'experiment.csv'
    if not path.is_file():
        return 0, False
    command_keys = ('cmd_v', 'cmd_w', 'cmd_raw_v', 'cmd_raw_w', 'exec_cmd_v', 'exec_cmd_w')
    rows = 0
    try:
        with path.open('r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows += 1
                for key in command_keys:
                    try:
                        value = float(row.get(key, 'nan'))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value) and abs(value) > 1e-6:
                        return rows, True
    except OSError:
        return rows, False
    return rows, False


def main() -> int:
    parser = argparse.ArgumentParser(description='Run a locked visibility-comparison campaign.')
    parser.add_argument('--config', default='scripts/visibility_comparison/warehouse_visibility_campaign.yaml',
                        help='Path to the locked campaign config YAML.')
    parser.add_argument('--log-root', default=str(LOGS_ROOT / 'warehouse_visibility_campaign_v1'),
                        help='Root directory for all run logs.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be run without executing.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip runs already marked completed in campaign_log.json.')
    parser.add_argument('--run-timeout', type=float, default=420.0,
                        help='Wall-clock timeout per run including simulator startup (seconds). '
                             'Sized so the one-shot global solve (median ~146s, contended tail to ~220s wall) '
                             'completes AND leaves ~150-200s for path execution.')
    parser.add_argument('--first-cmd-timeout', type=float, default=270.0,
                        help='Kill a live run when experiment.csv has rows but no nonzero command for this many seconds; <=0 disables. '
                             'Must exceed the worst-case global-solve wall time (~220s under contention) or slow solves are '
                             'guillotined mid-optimization with no command (the dominant past failure mode).')
    parser.add_argument('--cleanup-delay', type=float, default=8.0,
                        help='Sleep between runs for process cleanup (seconds).')
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config, strict=False)
    if not config_path.is_file():
        print(f'ERROR: config file not found: {config_path}', file=sys.stderr)
        return 1

    cfg = _load_config(config_path)
    log_root = Path(args.log_root).expanduser().resolve()
    campaign_log_path = log_root / 'campaign_log.json'
    ros_log_dir = Path(os.environ.get('ROS_LOG_DIR') or (log_root / '_ros_logs')).expanduser().resolve()
    if not args.dry_run:
        log_root.mkdir(parents=True, exist_ok=True)
        ros_log_dir.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    child_env['ROS_LOG_DIR'] = str(ros_log_dir)

    run_matrix = _build_run_matrix(cfg)
    if 'ros_domain_id_base' in cfg:
        domain_base = int(cfg['ros_domain_id_base'])
        domain_max = domain_base + max(len(run_matrix) - 1, 0)
        if domain_base < 0 or domain_max > 232:
            raise RuntimeError(
                f'ros_domain_id_base={domain_base} with {len(run_matrix)} runs would use '
                f'ROS_DOMAIN_ID up to {domain_max}; expected range is 0..232.'
            )
    existing_log = _load_run_log(campaign_log_path) if args.resume else {}

    print(f'Campaign: {len(run_matrix)} runs total')
    print(f'Config: {config_path}')
    print(f'Log root: {log_root}')
    print(f'Campaign log: {campaign_log_path}')
    print(f'ROS log dir: {ros_log_dir}')
    if cfg.get('cleanup_sim_stragglers', False):
        print(f'Gazebo cleanup: enabled for {cfg["world"]}')
    if args.dry_run:
        print('DRY RUN — no processes will be started.\n')

    if not args.dry_run:
        _force_fresh()

    campaign_log = dict(existing_log)

    for run_idx, (task_name, condition_id, seed) in enumerate(run_matrix):
        key = _run_key(task_name, condition_id, seed)
        label = f'[{run_idx + 1}/{len(run_matrix)}] task={task_name} condition={condition_id} seed={seed}'

        if args.resume and key in campaign_log and campaign_log[key].get('outcome') not in (None, 'infra_invalid'):
            matches, reason = _existing_entry_matches_config(campaign_log[key], cfg)
            if not matches:
                raise RuntimeError(
                    f'Cannot resume campaign with stale run entry for {label}: {reason}. '
                    'Start a fresh log root or rerun the campaign without --resume.'
                )
            print(f'  SKIP (already done): {label}')
            continue

        run_log_dir = log_root / task_name / condition_id / f'seed{seed}'
        run_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = _build_launch_cmd(cfg, task_name, condition_id, seed, run_log_dir)
        ros_domain_id = _ros_domain_for_run(cfg, run_idx)
        print(f'\n{label}')
        if ros_domain_id is not None:
            print(f'  ROS_DOMAIN_ID: {ros_domain_id}')
        print('  CMD:', ' '.join(str(p) for p in cmd))

        if args.dry_run:
            continue

        if run_idx > 0:
            _force_fresh()

        run_entry: dict = {
            'task': task_name,
            'condition': condition_id,
            'seed': seed,
            'planner': CONDITION_PLANNER[condition_id],
            'global_planner_mode': str(
                _effective_value(
                    cfg, task_name, condition_id, 'global_planner_mode'
                ) or 'efe'
            ),
            'preselected_route_sha256': _effective_value(
                cfg, task_name, condition_id, 'preselected_route_sha256'
            ),
            'run_log_dir': str(run_log_dir),
            'started_at': datetime.now().isoformat(),
            'outcome': None,
            'completion_reason': None,
            'goal_reached': None,
            'crashed': None,
            'path_length_m': None,
            'mean_belief_error_gt_m': None,
            'elapsed_after_first_cmd_s': None,
            'minimum_goal_distance': None,
            'ros_domain_id': ros_domain_id,
            'first_cmd_timeout_s': args.first_cmd_timeout,
        }
        campaign_log[key] = run_entry
        _save_run_log(campaign_log_path, campaign_log)

        run_env = dict(child_env)
        if ros_domain_id is not None:
            run_env['ROS_DOMAIN_ID'] = ros_domain_id

        process = subprocess.Popen(cmd, start_new_session=True, env=run_env)
        pgid = os.getpgid(process.pid)

        # Optional: stream-record the external camera for this run (opt-in).
        recorder_pgid = None
        if os.environ.get('CAMPAIGN_RECORD_CAMERA'):
            cam_out = run_log_dir / 'camera_frames'
            cam_out.mkdir(parents=True, exist_ok=True)
            rec_proc = subprocess.Popen(
                ['python3', str(REPO_ROOT / 'scripts/paper_figures/record_camera_stream.py'),
                 '--out-dir', str(cam_out)],
                start_new_session=True, env=run_env)
            try:
                recorder_pgid = os.getpgid(rec_proc.pid)
            except Exception:
                recorder_pgid = None

        timed_out = False
        no_first_cmd_timeout = False
        started_at = time.time()
        first_cmd_watch_started_at = None
        try:
            while True:
                if process.poll() is not None:
                    break
                elapsed_wall = time.time() - started_at
                if elapsed_wall >= args.run_timeout:
                    timed_out = True
                    print(f'  Wall-clock timeout after {args.run_timeout:.0f}s — killing.')
                    break
                live_run_dir = _find_latest_experiment_dir(run_log_dir)
                rows, has_command = _command_activity(live_run_dir)
                if has_command:
                    first_cmd_watch_started_at = None
                elif rows > 0 and args.first_cmd_timeout > 0:
                    if first_cmd_watch_started_at is None:
                        first_cmd_watch_started_at = time.time()
                    elif (time.time() - first_cmd_watch_started_at) >= args.first_cmd_timeout:
                        no_first_cmd_timeout = True
                        print(
                            f'  INFRA INVALID: no nonzero command after '
                            f'{args.first_cmd_timeout:.0f}s of logged experiment rows — killing.'
                        )
                        break
                time.sleep(2.0)
        finally:
            if recorder_pgid is not None:
                _terminate_process_group(recorder_pgid)
            _terminate_process_group(pgid)
            _force_fresh()

        # Read run summary written by experiment_logger
        run_dir = _find_latest_run_dir(run_log_dir)
        summary = _read_run_summary(run_dir) if run_dir else None
        route_artifact_ok = True
        route_artifact_reason = ''
        if run_dir is not None:
            route_artifact_ok, route_artifact_reason = _verify_preselected_run_artifacts(
                run_dir, cfg, task_name, condition_id
            )

        if no_first_cmd_timeout:
            outcome = 'infra_invalid'
            completion_reason = 'no_first_cmd_timeout'
            print(f'  INFRA INVALID: live run never produced a nonzero command.')
        elif summary is None:
            outcome = 'infra_invalid'
            completion_reason = 'no_summary'
            print(f'  INFRA INVALID: no run_summary.json found in {run_log_dir}')
        elif not route_artifact_ok:
            outcome = 'infra_invalid'
            completion_reason = 'wrong_route_artifact'
            print(f'  INFRA INVALID: {route_artifact_reason}')
        elif not summary.get('completed', False) and timed_out:
            outcome = 'infra_invalid'
            completion_reason = 'wall_clock_timeout'
            print(f'  INFRA INVALID: wall-clock timeout, logger did not complete.')
        else:
            completion_reason = str(summary.get('completion_reason', ''))
            crashed = bool(summary.get('crashed', False))
            if crashed:
                outcome = 'collision'
            elif completion_reason in ('goal_reached', 'goal_reached_stable'):
                outcome = 'goal_reached'
            elif completion_reason == 'timeout_after_first_cmd':
                outcome = 'timeout'
            else:
                outcome = completion_reason or 'completed'

        run_entry.update({
            'finished_at': datetime.now().isoformat(),
            'outcome': outcome,
            'completion_reason': completion_reason,
            'goal_reached': outcome == 'goal_reached',
            'crashed': bool(summary.get('crashed', False)) if summary else None,
            'path_length_m': summary.get('path_length_m') if summary else None,
            'mean_belief_error_gt_m': (summary.get('mean_belief_error_gt_after_first_cmd_m',
                                                    summary.get('mean_belief_error_gt_m')) if summary else None),
            'elapsed_after_first_cmd_s': summary.get('elapsed_after_first_cmd_s') if summary else None,
            'minimum_goal_distance': summary.get('minimum_goal_distance') if summary else None,
            'run_dir': str(run_dir) if run_dir else None,
            'route_artifact_verified': route_artifact_ok if run_dir else False,
            'route_artifact_verification_reason': route_artifact_reason,
        })
        campaign_log[key] = run_entry
        _save_run_log(campaign_log_path, campaign_log)

        goal_str = 'YES' if outcome == 'goal_reached' else 'no'
        print(f'  -> outcome={outcome}, goal={goal_str}, reason={completion_reason}')

    if args.dry_run:
        print('\n=== Dry run complete ===')
    else:
        print('\n=== Campaign complete ===')
        total = len(run_matrix)
        completed = sum(1 for e in campaign_log.values() if e.get('outcome') not in (None, 'infra_invalid'))
        goals = sum(1 for e in campaign_log.values() if e.get('outcome') == 'goal_reached')
        infra = sum(1 for e in campaign_log.values() if e.get('outcome') == 'infra_invalid')
        print(f'  {completed}/{total} runs completed, {goals} goal_reached, {infra} infra_invalid')
        print(f'  Full log: {campaign_log_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
