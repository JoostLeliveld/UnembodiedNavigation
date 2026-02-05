import numpy as np


def load_camera_params(node):
    cam_pos = np.array(node.get_parameter('cam_pos').value, dtype=float)
    look_at = np.array(node.get_parameter('look_at').value, dtype=float)
    img_width = int(node.get_parameter('img_width').value)
    img_height = int(node.get_parameter('img_height').value)
    fov_h_rad = float(node.get_parameter('fov_h_rad').value)
    return cam_pos, look_at, img_width, img_height, fov_h_rad
