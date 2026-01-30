import numpy as np
import scipy.ndimage
import skimage
import skimage.morphology 

def distance_transform(binary_occupancy_matrix):
    
    distance_matrix = scipy.ndimage.distance_transform_edt(1-binary_occupancy_matrix)
    
    return distance_matrix

def dilate(binary_occupancy_matrix, radius=0.0):
    
    # Create a disk-shaped structuring element with the desired radius
    dilation_disk = skimage.morphology.disk(radius)
    # Apply binary dilation using the structuring element
    dilated_occupancy_matrix = skimage.morphology.binary_dilation(binary_occupancy_matrix, dilation_disk)

    return dilated_occupancy_matrix

def inverse_distance_costmap_exponential_decay(binary_occupancy_matrix, safety_margin=0.0, decay_rate=1.0, min_cost=0.0, max_cost=1.0):

    distance_matrix = distance_transform(binary_occupancy_matrix)
    distance_matrix = distance_matrix - safety_margin
    distance_matrix = np.maximum(distance_matrix, 0.0)

    cost_matrix = max_cost * np.exp(-decay_rate*distance_matrix)
    cost_matrix = np.maximum(cost_matrix, min_cost) 

    return cost_matrix

def inverse_distance_costmap_linear_decay(binary_occupancy_matrix, safety_margin=0.0, decay_rate=1.0, min_cost=0.0, max_cost=1.0):

    distance_matrix = distance_transform(binary_occupancy_matrix)
    distance_matrix = distance_matrix - safety_margin
    distance_matrix = np.maximum(distance_matrix, 0.0)

    cost_matrix = max_cost*(1.0 - decay_rate*distance_matrix)
    cost_matrix = np.maximum(cost_matrix, min_cost) 

    return cost_matrix

def inverse_distance_costmap_polynomial_decay(binary_occupancy_matrix, safety_margin=0.0, decay_rate=1.0, min_cost=0.0, max_cost=1.0):

    distance_matrix = distance_transform(binary_occupancy_matrix)
    distance_matrix = distance_matrix - safety_margin
    distance_matrix = np.maximum(distance_matrix, 0.0)

    cost_matrix = max_cost*(np.power(1.0 + distance_matrix, -decay_rate))
    cost_matrix = np.maximum(cost_matrix, min_cost) 

    return cost_matrix
