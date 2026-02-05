"""EFE-UT planner (Unscented Transform approximation)."""

from planning.planners.base_planner import UnicyclePlannerBase


class EfeUtPlanner(UnicyclePlannerBase):
    APPROX_METHOD = 'UT'
    USE_OBS_RISK = True
    USE_AMBIGUITY = True
