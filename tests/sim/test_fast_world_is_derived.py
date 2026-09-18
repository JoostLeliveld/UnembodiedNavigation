"""The fast development world must stay a derivative of the canonical world.

`warehouse_v2_low_cpu.world.sdf` was retired because it claimed to share the
canonical geometry and did not: it was missing `bin_office`. Its replacement can
drift the same way, so the property is asserted here rather than documented.

The only permitted difference is the contact sensor rate, which the physics
contact report reads and no rendered pixel does.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / 'src' / 'sim' / 'gazebo_worlds' / 'worlds'
CANONICAL = WORLDS / 'warehouse_v2.world.sdf'
FAST = WORLDS / 'warehouse_v2_fast.world.sdf'

sys.path.insert(0, str(ROOT / 'experiments' / 'warehouse_v2_sketches'))


def _core(text: str) -> str:
    """Comment-free body with the contact rate normalised back to canonical."""
    stripped = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    return stripped.replace('<update_rate>20</update_rate>',
                            '<update_rate>60</update_rate>')


def test_fast_world_differs_from_canonical_only_in_contact_rate():
    canonical = CANONICAL.read_text(encoding='utf-8')
    fast = FAST.read_text(encoding='utf-8')
    assert _core(fast) == _core(canonical), (
        'the fast world has diverged from the canonical geometry; regenerate it '
        'with experiments/warehouse_v2_sketches/derive_fast_world.py'
    )


def test_fast_world_actually_lowers_the_contact_rate():
    fast = FAST.read_text(encoding='utf-8')
    assert '<update_rate>60</update_rate>' not in fast
    assert fast.count('<update_rate>20</update_rate>') > 0


def test_derivation_is_reproducible():
    """Running the deriver again must reproduce the committed file exactly."""
    derive_fast_world = pytest.importorskip('derive_fast_world')
    expected = derive_fast_world.derive(CANONICAL.read_text(encoding='utf-8'))
    assert expected == FAST.read_text(encoding='utf-8'), (
        'the committed fast world is not what derive_fast_world.py produces'
    )


@pytest.mark.parametrize('probe, why', [
    ('0.8378', 'camera C pitch 48 deg, changed 2026-09-18'),
    ('obs_Cm_p3', 'the fourth Cm stack, which make_world.py does not emit'),
])
def test_fast_world_carries_the_canonical_edits(probe, why):
    """Guards the two specific ways make_world.py --fast would diverge."""
    assert probe in FAST.read_text(encoding='utf-8'), f'fast world is missing {why}'


def test_fast_world_has_no_leftover_three_tier_barrels():
    fast = FAST.read_text(encoding='utf-8')
    assert not re.search(r'drum_Cm_p\d+_[12]_', fast), (
        'the fast world still has de-stacked Cm barrel tiers'
    )
