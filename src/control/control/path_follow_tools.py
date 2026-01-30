import numpy as np

def line_distance_to_point(start, end, point):
    
    start = np.asarray(start)
    if start.ndim < 2:
        start = start.reshape((1,-1))
    end = np.asarray(end)
    if end.ndim < 2: 
        end = end.reshape(1,-1)
    point = np.asarray(point)
    if point.ndim < 2:
        point = point.reshape((1, -1))
    
    start = start[:, np.newaxis,:]
    end = end[:, np.newaxis,:]
    point = point[np.newaxis,:,:]
    distance_start_end = np.sum(np.power(start-end, 2), axis=2)
    distance_point_end = np.sum((point-end)*(start-end), axis=2)
    alpha = distance_point_end/distance_start_end
    alpha = np.minimum(np.maximum(alpha, 0.0), 1.0)
    distance = np.sum(np.power(alpha[:,:, np.newaxis]*start + (1.0-alpha[:,:, np.newaxis])*end - point, 2), axis=2)
    distance = np.sqrt(distance)
    
    return distance

def line_intersect_sphere(start, end, center, radius):

    start = np.asarray(start)
    if start.ndim < 2:
        start = start.reshape((1,-1))
    end = np.asarray(end)
    if end.ndim < 2: 
        end = end.reshape(1,-1)
    
    center = np.asarray(center).reshape((-1, start.shape[1]))

    start = start - center
    end = end - center

    distance = line_distance_to_point(start, end, 0*center)
    is_valid = distance.squeeze() <= radius 
    start = start[is_valid,:]
    end = end[is_valid,:]

    start_end_diff = start - end

    a = np.sum(np.power(start_end_diff, 2), axis=1)
    b = 2*np.sum(start_end_diff*end, axis=1)
    c = np.sum(np.power(end, 2), axis=1) - np.power(radius, 2)
    
    discriminant = np.power(b,2) - 4*a*c
    valid = discriminant>=0
    start = start[valid,:]
    end = end[valid,:]
    a = a[valid]
    b = b[valid]
    c = c[valid]
    discriminant = discriminant[valid] 

    weight_min = (-b-np.sqrt(discriminant))/(2*a)
    weight_min = np.maximum(np.minimum(weight_min[:, np.newaxis],1),0)
    weight_max = (-b+np.sqrt(discriminant))/(2*a)
    weight_max = np.maximum(np.minimum(weight_max[:,np.newaxis],1),0)
 
    xline_start = weight_max*start + (1-weight_max)*end
    xline_end = weight_min*start + (1-weight_min)*end

    xline_start = xline_start + center
    xline_end = xline_end + center

    return xline_start, xline_end    

def line_intersect_halfplane(start, end, base, normal):

    start = np.asarray(start)
    if start.ndim < 2:
        start = start.reshape((1,-1))
    end = np.asarray(end)
    if end.ndim < 2: 
        end = end.reshape(1,-1)
    
    xline_start = start
    xline_end = end

    distance_start = np.sum((xline_start - base)*normal, axis=1)
    distance_end = np.sum((xline_end - base) * normal, axis=1)
    
    is_valid_edge = np.logical_or(distance_start >= 0, distance_end >= 0)  
    xline_start = xline_start[is_valid_edge,:]
    xline_end = xline_end[is_valid_edge,:]
    distance_start = distance_start[is_valid_edge]
    distance_end = distance_end[is_valid_edge]

    start_weight = - distance_end/(distance_start-distance_end)
    start_weight = np.maximum(np.minimum(start_weight[:, np.newaxis], 1.0), 0.0)
    edge_interection = start_weight*xline_start + (1-start_weight)*xline_end
    xline_start[distance_start<0] = edge_interection[distance_start<0]
    xline_end[distance_end<0] = edge_interection[distance_end<0]

    return xline_start, xline_end

def path_goal_sphere(path, center, radius):
    
    path = np.asarray(path)
    if path.ndim < 2:
        path = path.reshape((1,-1))
    edge_start = path[:-1] 
    edge_end = path[1:]

    xline_start, xline_end = line_intersect_sphere(edge_start, edge_end, center, radius)
    
    if xline_end.shape[0] > 0:
        goal = xline_end[-1]
    else:
        goal = None

    return goal

def path_goal_circular_corridor(path, center, radius):
    return path_goal_sphere(path, center, radius)

def path_goal_support_corridor(path, center, boundary):

    path = np.asarray(path)
    if path.ndim < 2:
        path = path.reshape((1,-1))

    center = np.asarray(center).reshape((-1, path.shape[1]))
    boundary = np.asarray(boundary).reshape((-1, path.shape[1]))

    xline_start = path[:-1]
    xline_end = path[1:]
    for point in boundary:
        xline_start, xline_end = line_intersect_halfplane(xline_start, xline_end, point, center-point)

    if xline_end.shape[0] > 0:
        goal = xline_end[-1]
    else:
        goal = None

    return goal
    



        




    pass

def path_length(path):
    pass

def path_linspace(path):
    pass

def path_value(path, param):
    pass

def path_point(path, param):
    return path_value(path, param)
