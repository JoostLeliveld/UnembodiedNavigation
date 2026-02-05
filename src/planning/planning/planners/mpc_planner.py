"""MPC-style planner (state risk + control cost only)."""

from planning.planners.base_planner import UnicyclePlannerBase


class MpcPlanner(UnicyclePlannerBase):
    APPROX_METHOD = 'ET2'
    USE_OBS_RISK = False
    USE_AMBIGUITY = False
