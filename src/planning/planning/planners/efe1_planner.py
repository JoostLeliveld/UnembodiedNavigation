"""EFE1 planner (ET1 approximation)."""

from planning.planners.base_planner import UnicyclePlannerBase


class Efe1Planner(UnicyclePlannerBase):
    APPROX_METHOD = 'ET1'
    USE_OBS_RISK = True
    USE_AMBIGUITY = True
