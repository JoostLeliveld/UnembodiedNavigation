"""A* planner core (no ROS)."""

from dataclasses import dataclass
import numpy as np

from planning.core import search_based_path_planning


@dataclass
class AStarResult:
    path_world: np.ndarray
    path_grid: np.ndarray


class AStarPlannerCore:
    def __init__(self, max_cost=100.0, diagonal_connectivity=True):
        self.max_cost = float(max_cost)
        self.diagonal_connectivity = bool(diagonal_connectivity)

    def plan(self, costmap_matrix, origin, resolution, start_xy, goal_xy):
        costmap_matrix = np.array(costmap_matrix, dtype=float)
        # Mark obstacles as -1
        costmap_matrix[costmap_matrix >= self.max_cost] = -1

        start_cell = search_based_path_planning.world_to_grid(
            start_xy, origin=origin, resolution=resolution
        )[0]
        goal_cell = search_based_path_planning.world_to_grid(
            goal_xy, origin=origin, resolution=resolution
        )[0]

        path_grid = search_based_path_planning.shortest_path_networkx(
            costmap_matrix, start_cell, goal_cell, diagonal_connectivity=self.diagonal_connectivity
        )
        path_world = search_based_path_planning.grid_to_world(path_grid, origin, resolution)

        return AStarResult(path_world=path_world, path_grid=path_grid)
