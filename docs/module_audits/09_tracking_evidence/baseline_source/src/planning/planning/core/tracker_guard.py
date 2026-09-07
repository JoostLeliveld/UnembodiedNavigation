"""Checked local action selection; no estimator, map, or clearance relaxation."""
from dataclasses import dataclass

import numpy as np

from planning.core.dynamics import unicycle_step


@dataclass(frozen=True)
class GuardedControls:
    controls: np.ndarray
    safe_steps: int
    reason: str
    rotation_recovery: bool = False


def checked_tracker_controls(controls, state, target, *, dt, w_min, w_max,
                             safety_check, allow_rotation_recovery=False):
    """Try a bounded stationary turn only after the proposed first step is refused.

    Keep the same waypoint and the same gate. A blocked translation while already
    aligned remains blocked; this is not a route search or a collision exemption.
    The gate must model the robot's footprint for rotation as well as translation.
    The active platform uses a conservative circumscribed collision disc.
    """
    controls = np.asarray(controls, dtype=float)
    n_safe, reason = safety_check(controls, state)
    original = GuardedControls(controls, n_safe, reason)
    if n_safe > 0 or not allow_rotation_recovery:
        return original
    if not str(reason).startswith(('driveable_clearance_violation_step_0:',
                                   'collision_geometry_violation_step_0:')):
        return original
    pose = np.asarray(state, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    if (pose.shape != (3,) or target.shape != (2,) or controls.ndim != 2
            or controls.shape[1] != 2 or not np.isfinite(pose).all()
            or not np.isfinite(target).all()):
        return original
    turn = np.zeros_like(controls)
    for i in range(len(turn)):
        desired = np.arctan2(target[1]-pose[1], target[0]-pose[0])
        delta = np.arctan2(np.sin(desired-pose[2]), np.cos(desired-pose[2]))
        turn[i, 1] = np.clip(2.*delta, w_min, w_max)
        pose = unicycle_step(pose, turn[i], dt)
    if not len(turn) or abs(turn[0, 1]) < 1e-3:
        return original
    turn_safe, turn_reason = safety_check(turn, state)
    if turn_safe <= 0:
        return original
    return GuardedControls(turn, turn_safe, 'rotation_recovery_after:'+reason, True)
