"""Camera absence must not discard available motion, especially a blind turn."""
import math

import numpy as np
import pytest

from planning.core.dynamics import unicycle_step
from planning.core.motion_history import covers_interval
from test_planner_node_state_correction import make_state_node
from test_planner_node_correction_wiring import stamp


@pytest.mark.parametrize('entries,start,end,expected', [
    ([(0., .2, 0.), (1., .2, 0.), (2., .2, 0.)], .1, 2.5, True),
    ([(1., .2, 0.), (2., .2, 0.)], .1, 2.5, False),
    ([(0., .2, 0.), (2., .2, 0.)], .1, 2.5, False),
    ([(0., .2, 0.), (1., .2, 0.)], .1, 3., False),
    ([(0., .2, 0.), (1., math.nan, 0.)], .1, 1.2, False),
    ([(0., .2, 0.), (1., .2, 0.), (.5, .2, 0.)], .1, 1.2, False),
    ([(0., .2, 0.)], 2., 2.1, False),
    ([], .1, 2., False),
])
def test_motion_coverage(entries, start, end, expected):
    assert covers_interval(entries, start, end, 1.5) is expected


class MotionModel:
    @staticmethod
    def predict(m, P, u, dt):
        # Independent exact unicycle mean oracle, with nonzero process growth.
        return unicycle_step(m, u, dt), P + np.diag([.01, .01, .02])**2 * dt


def test_returning_camera_cannot_erase_a_recorded_blind_turn():
    node = make_state_node(belief_stamp_s=0., now_s=10.1)
    node.heading_update_mode = 'coupled'
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    # Drive east, turn north, then drive north. Every input sample is present;
    # the camera alone is absent for ten seconds.
    node._odom_log = [(float(t), .2 if t < 2 or t >= 4 else 0.,
                       math.pi/4 if 2 <= t < 4 else 0.)
                      for t in np.arange(0., 10.01, .1)]
    expected, covariance = node._predict_belief_to_now(
        node.belief_m.copy(), node.belief_S.copy(), node.last_cmd, 10., stamp(10.))
    node._advance_belief_over_outage(stamp(10.), 10.)
    np.testing.assert_allclose(node.belief_m, expected, atol=1e-10)
    np.testing.assert_allclose(node.belief_S, covariance, atol=1e-10)
    assert node.belief_m[2] == pytest.approx(math.pi/2)
    assert node._stamp_to_float(node.belief_stamp) == pytest.approx(10.)


def test_unsupported_motion_is_not_treated_as_a_complete_replay():
    node = make_state_node(belief_stamp_s=0., now_s=10.1)
    node.use_odom_for_predict = True
    node.planner = MotionModel()
    node._odom_log = [(9., .2, 0.), (10., .2, 0.)]
    before = node.belief_S.copy()
    node._advance_belief_over_outage(stamp(10.), 10.)
    # This is the existing conservative fallback, not the complete-history case.
    assert node.belief_m[0] <= .3
    assert node.belief_S[0, 0] > before[0, 0] + 1.
