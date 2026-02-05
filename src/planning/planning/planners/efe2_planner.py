"""EFE2 planner (ET2 approximation)."""

from planning.planners.base_planner import UnicyclePlannerBase


class Efe2Planner(UnicyclePlannerBase):
    APPROX_METHOD = 'ET2'
    USE_OBS_RISK = True
    USE_AMBIGUITY = True
