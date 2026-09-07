import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORLDS = ROOT / 'src' / 'sim' / 'gazebo_worlds' / 'worlds'


def _world_text(name):
    return (WORLDS / name).read_text(encoding='utf-8')


def test_low_cpu_world_only_changes_documented_rates():
    reference = _world_text('warehouse_v2.world.sdf')
    low_cpu = _world_text('warehouse_v2_low_cpu.world.sdf')

    assert '<max_step_size>0.005</max_step_size>' in low_cpu
    assert '<real_time_update_rate>200</real_time_update_rate>' in low_cpu
    assert low_cpu.count('<update_rate>20</update_rate>') == 46

    normalized = re.sub(
        r'  <!-- LOW-CPU DERIVATIVE:.*?\n       ',
        '  <!-- ',
        low_cpu,
        count=1,
    )
    normalized = normalized.replace(
        '<max_step_size>0.005</max_step_size>',
        '<max_step_size>0.001</max_step_size>',
    ).replace(
        '<real_time_update_rate>200</real_time_update_rate>',
        '<real_time_update_rate>1000</real_time_update_rate>',
    ).replace('<update_rate>20</update_rate>', '<update_rate>60</update_rate>')
    assert normalized == reference
