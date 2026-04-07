from experiments.core.thesis_launch import make_thesis_launch_description


def generate_launch_description():
    return make_thesis_launch_description(
        default_planner='efe1',
        allowed_planners=('efe1', 'geometric_baseline'),
        planner_description='Primary thesis comparison: efe1 | geometric_baseline',
    )
