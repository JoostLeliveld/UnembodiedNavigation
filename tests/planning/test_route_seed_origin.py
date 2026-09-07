"""Map translation must not remove a corridor from the route seed family."""
import json

import numpy as np
import pytest

from unav_common.lane_graph_routes import generate_route_seeds


def box(name, xmin, xmax, ymin, ymax):
    return dict(name=name, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=0., zmax=1.)


@pytest.mark.parametrize('origin_y', [0., -1., 1.])
def test_straight_corridor_has_a_route_at_every_map_origin(origin_y):
    geometry = json.dumps({'prisms': [box('corridor', -2., 2., origin_y-.5, origin_y+.5)]})
    seeds = generate_route_seeds(geometry, (-1., origin_y), (1., origin_y))
    assert len(seeds) == 1
    assert np.allclose(np.asarray(seeds[0]['waypoints']) - [0., origin_y], [[1., 0.]])


def test_zero_centred_corridor_is_between_existing_negative_and_positive_seeds():
    prisms = [box(f'cross_{y}', -2., 2., y-.4, y+.4) for y in (-2., 0., 2.)]
    prisms += [box('left_aisle', -1.4, -.6, -2.4, 2.4),
               box('right_aisle', .6, 1.4, -2.4, 2.4)]
    seeds = generate_route_seeds(json.dumps({'prisms': prisms}), (-1., -2.), (1., 2.))
    assert [s['name'] for s in seeds] == ['below_main_aisle', 'centre_cross_aisle', 'above_connector']
    assert seeds[1]['waypoints'] == [[-1., 0.], [1., 0.], [1., 2.]]
    assert seeds[0]['waypoints'] == [[1., -2.], [1., 2.]]
    assert seeds[2]['waypoints'] == [[-1., 2.], [1., 2.]]
