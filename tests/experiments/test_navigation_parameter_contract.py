"""Reject impossible parameters before launching a campaign."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from unav_common.navigation_parameters import validate_navigation_parameters
from experiments.core.visibility_launch_common import PAPER_LAUNCH_DEFAULTS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('values', [
    {'v_max': 0}, {'dt': -1}, {'horizon': 2.5},
    {'local_plan_rate': float('nan')}, {'robot_collision_radius_m': float('inf')},
    {'v_max': True}, {'dt': None}, {'process_noise_xy': -.01},
    {'discount_gamma': 1.1}, {'waypoint_arrival_radius_m': 0},
    {'nogo_logbarrier_eps': 0}, {'nogo_warning_band': 0},
    {'nogo_weight': -40}, {'nogo_near_weight': float('nan')},
    {'optimizer_maxiter': 1.5}, {'optimizer_ftol': -1},
    {'nogo_mode': 'keep_ni'}, {'control_weight': -1},
])
def test_invalid_parameters(values):
    with pytest.raises(ValueError):
        validate_navigation_parameters(values)


def test_documented_zero_sentinels_and_defaults():
    validate_navigation_parameters(PAPER_LAUNCH_DEFAULTS)
    validate_navigation_parameters({'global_dt': 0, 'process_noise_xy': 0,
                                    'pixel_correction_nis_threshold': 0})


@pytest.mark.parametrize('overrides', [
    {'weight': -1}, {'safe_distance': float('nan')}, {'warning_band': 0},
    {'logbarrier_eps': 0}, {'near_weight': -1},
])
def test_direct_nogo_model_does_not_silently_repair_invalid_tuning(overrides):
    from planning.core.nogo_cost import NogoCostConfig, NogoZoneCostModel
    with pytest.raises(ValueError):
        NogoZoneCostModel(NogoCostConfig(**overrides))


def test_default_disc_encloses_spawned_amr_body():
    import re
    text = (ROOT / 'src/sim/robot_description/urdf/warehouse_amr.urdf.xacro').read_text()
    dimensions = [float(re.search(r'name="' + key + r'" value="([\d.]+)"', text)[1])
                  for key in ('body_l', 'body_w')]
    radius = float(PAPER_LAUNCH_DEFAULTS['robot_collision_radius_m'])
    assert radius + 1e-12 >= sum((d / 2) ** 2 for d in dimensions) ** .5
    assert float(PAPER_LAUNCH_DEFAULTS['robot_length_m']) == dimensions[0]
    assert float(PAPER_LAUNCH_DEFAULTS['robot_width_m']) == dimensions[1]


def test_campaign_checks_effective_condition_overrides():
    spec = importlib.util.spec_from_file_location(
        'parameter_contract_campaign', ROOT / 'scripts/visibility_comparison/run_visibility_campaign.py')
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    path = ROOT / 'experiments/icra_commissioning/network_navigation_speed_candidate.yaml'
    cfg = yaml.safe_load(path.read_text())
    campaign._validate_config(cfg, path)
    cfg['conditions']['P0']['v_max'] = -1
    with pytest.raises(ValueError, match='fusion_network_traverse/P0: v_max'):
        campaign._validate_config(cfg, path)


def test_speed_candidate_changes_no_other_navigation_tuning():
    folder = ROOT / 'experiments/icra_commissioning'
    baseline = yaml.safe_load((folder / 'network_navigation_runtime_pilot.yaml').read_text())
    candidate = yaml.safe_load((folder / 'network_navigation_speed_candidate.yaml').read_text())
    for key in ('v_max', 'ros_domain_id_base', 'study_title', 'study_comparison'):
        baseline.pop(key, None)
        candidate.pop(key, None)
    assert candidate == baseline
