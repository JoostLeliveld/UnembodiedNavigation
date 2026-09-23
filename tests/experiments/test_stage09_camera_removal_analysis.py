from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'pipeline/analyze_campaign.py'


def load_analysis_module():
    spec = importlib.util.spec_from_file_location('stage09_navigation_analysis', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def protocol():
    return {
        'design': {
            'tasks': ['route_a'],
            'conditions': ['intact', 'stale_removal', 'updated_removal'],
            'matched_seeds': [10, 11],
            'expected_cells': 6,
            'primary_route_choice_tasks': ['route_a'],
            'null_route_choice_task': 'route_a',
        },
        'comparisons': {
            'primary_pair': {
                'baseline': 'stale_removal',
                'treatment': 'updated_removal',
            },
        },
    }


def test_camera_removal_cells_and_primary_pair_are_explicit():
    analysis = load_analysis_module()
    value = protocol()
    assert analysis.condition_ids(value) == [
        'intact', 'stale_removal', 'updated_removal']
    assert analysis.primary_pair(value) == ('stale_removal', 'updated_removal')
    assert analysis.expected_cells(value) == [
        ('route_a', 'intact', 10),
        ('route_a', 'intact', 11),
        ('route_a', 'stale_removal', 10),
        ('route_a', 'stale_removal', 11),
        ('route_a', 'updated_removal', 10),
        ('route_a', 'updated_removal', 11),
    ]


def test_primary_pair_excludes_intact_reference_from_difference():
    analysis = load_analysis_module()
    rows = []
    values = {
        'intact': (0.10, 0.12),
        'stale_removal': (0.30, 0.40),
        'updated_removal': (0.20, 0.25),
    }
    for condition, rmses in values.items():
        for seed, rmse in zip((10, 11), rmses, strict=True):
            rows.append({
                'task': 'route_a', 'condition': condition, 'seed': seed,
                'campaign_outcome': 'goal_reached',
                'strict_success': True, 'collision': False,
                'duration_sim_s': 1.0, 'path_length_m': 1.0,
                'belief_error_m_rmse': rmse, 'belief_error_m_p95': rmse,
                'longest_correction_gap_s': 0.1,
                'accepted_fraction_of_fresh': 1.0,
                'fused_error_m_p95': rmse,
            })
    pairs = analysis.paired_results(rows, protocol())
    assert [
        row['difference_updated_removal_minus_stale_removal_belief_error_m_rmse']
        for row in pairs
    ] == pytest.approx([-0.10, -0.15])
    assert all('intact_belief_error_m_rmse' not in row for row in pairs)
