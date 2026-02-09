import numpy as np

# Compatibility shim for old NetworkX with NumPy >= 2.0
# (NetworkX <= 2.x uses deprecated np.int alias)
if not hasattr(np, 'int'):
    np.int = int

import networkx as nx


def world_to_grid(xy, origin, resolution):
    # Converts world coordinates in meters to grid indices
    xy = np.reshape(xy, (-1, 2))
    origin = np.reshape(origin, (-1, 2))

    if xy.size == 0:
        ij = xy
    else:
        ij = np.floor((xy - origin) / resolution)
    ij = ij[:, [1, 0]]  # Convert from (columns,rows) to (rows, columns) order
    return np.int64(ij)


def grid_to_world(ij, origin, resolution):
    # Converts grid indices to world coordinates in meters

    ij = np.reshape(ij, (-1, 2))
    ij = ij[:, [1, 0]]  # Convert from (rows,columns) to (columns,rows) order
    origin = np.reshape(origin, (-1, 2))

    if ij.size == 0:
        xy = np.zeros((0, 2))
    else:
        xy = ij * resolution + (origin + 0.5 * resolution)

    return xy


def costmap_connectivity(costmap_matrix, connectivity_threshold=0.0, diagonal_connectivity=False, edge_weight='mean'):

    if edge_weight.lower() == 'mean':
        edge_weight_func = lambda a, b: (a + b) / 2.0
    elif edge_weight.lower() == 'sum':
        edge_weight_func = lambda a, b: (a + b)
    elif edge_weight.lower() == 'min':
        edge_weight_func = lambda a, b: np.minimum(a, b)
    elif edge_weight.lower() == 'max':
        edge_weight_func = lambda a, b: np.maximum(a, b)
    else:
        raise ValueError('Unknown edge weight type')

    costmap_matrix = np.array(costmap_matrix)

    costmap_connectivity = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
    if diagonal_connectivity:
        costmap_connectivity.extend([(-1, -1), (-1, +1), (+1, -1), (+1, +1)])

    number_of_rows = costmap_matrix.shape[0]
    number_of_cols = costmap_matrix.shape[1]
    edge_list = []
    weight_list = []

    for delta_row, delta_col in costmap_connectivity:

        start_row_indices, start_col_indices = np.meshgrid(
            np.arange(max(0, -delta_row), min(number_of_rows, number_of_rows - delta_row)),
            np.arange(max(0, -delta_col), min(number_of_cols, number_of_cols - delta_col))
        )
        end_row_indices, end_col_indices = np.meshgrid(
            np.arange(max(0, delta_row), min(number_of_rows, number_of_rows + delta_row)),
            np.arange(max(0, delta_col), min(number_of_cols, number_of_cols + delta_col))
        )

        start_flat_indices = np.ravel_multi_index(
            (start_row_indices.flatten(), start_col_indices.flatten()),
            dims=(number_of_rows, number_of_cols),
        )
        end_flat_indices = np.ravel_multi_index(
            (end_row_indices.flatten(), end_col_indices.flatten()),
            dims=(number_of_rows, number_of_cols),
        )
        start_cost = costmap_matrix[start_row_indices, start_col_indices].flatten()
        end_cost = costmap_matrix[end_row_indices, end_col_indices].flatten()
        if delta_row * delta_col == 0.0:
            weights = edge_weight_func(start_cost, end_cost)
        else:
            weights = edge_weight_func(start_cost, end_cost) * np.sqrt(2)

        # Treat costs >= 0 as traversable (free or low-cost cells). Obstacles are < 0.
        valid_edges = np.logical_and(start_cost >= 0, end_cost >= 0)
        new_edges = list(zip(start_flat_indices[valid_edges], end_flat_indices[valid_edges]))
        new_weights = weights[valid_edges]

        edge_list.extend(new_edges)
        weight_list.extend(new_weights)

    return edge_list, weight_list


def costmap_graph_networkx(costmap_matrix, diagonal_connectivity=False):

    edge_list, weight_list = costmap_connectivity(costmap_matrix=costmap_matrix, diagonal_connectivity=diagonal_connectivity)

    weighted_edges = [(x[0], x[1], y) for (x, y) in zip(edge_list, weight_list)]
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)

    return graph


def shortest_path_networkx(costmap_matrix, source_cell, target_cell, diagonal_connectivity=False):

    graph = costmap_graph_networkx(costmap_matrix, diagonal_connectivity=diagonal_connectivity)
    source_node = np.ravel_multi_index(source_cell, dims=np.shape(costmap_matrix))
    target_node = np.ravel_multi_index(target_cell, dims=np.shape(costmap_matrix))

    if source_node not in graph or target_node not in graph:
        return np.zeros((0, 2))

    try:
        path = nx.dijkstra_path(graph, source_node, target_node)
        path = np.unravel_index(path, costmap_matrix.shape)
        path = np.column_stack((path[0], path[1]))
    except nx.exception.NetworkXNoPath:
        path = np.zeros((0, 2))

    return path
