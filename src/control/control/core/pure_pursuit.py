import math
import numpy as np

from control.core import path_follow


class PurePursuitParams:
    def __init__(
        self,
        lookahead_distance=0.5,
        max_speed=0.22,
        kp_angular=2.0,
        max_angular=1.0,
        turn_in_place_threshold=0.7,
        slowdown_distance=0.5,
        goal_tolerance=0.1,
    ):
        self.lookahead_distance = float(lookahead_distance)
        self.max_speed = float(max_speed)
        self.kp_angular = float(kp_angular)
        self.max_angular = float(max_angular)
        self.turn_in_place_threshold = float(turn_in_place_threshold)
        self.slowdown_distance = float(slowdown_distance)
        self.goal_tolerance = float(goal_tolerance)


def compute_control(pose_x, pose_y, pose_yaw, path_points, params: PurePursuitParams):
    if path_points is None or len(path_points) == 0:
        return None, None, True, None

    robot_pos = np.array([pose_x, pose_y], dtype=float)
    path = np.asarray(path_points, dtype=float)

    dist_to_end = np.linalg.norm(robot_pos - path[-1])
    if dist_to_end < params.goal_tolerance:
        return 0.0, 0.0, True, path[-1]

    lookahead_point = path_follow.path_goal_sphere(path, robot_pos, params.lookahead_distance)
    if lookahead_point is None:
        lookahead_point = path[-1]

    target_x, target_y = lookahead_point
    dx = target_x - pose_x
    dy = target_y - pose_y

    desired_yaw = math.atan2(dy, dx)
    yaw_error = desired_yaw - pose_yaw

    while yaw_error > math.pi:
        yaw_error -= 2 * math.pi
    while yaw_error < -math.pi:
        yaw_error += 2 * math.pi

    angular_vel = params.kp_angular * yaw_error
    angular_vel = max(-params.max_angular, min(params.max_angular, angular_vel))

    if abs(yaw_error) > params.turn_in_place_threshold:
        linear_vel = 0.0
    else:
        heading_scale = max(0.1, math.cos(yaw_error))
        linear_vel = params.max_speed * heading_scale
        if dist_to_end < params.slowdown_distance:
            linear_vel *= max(0.1, dist_to_end / params.slowdown_distance)

    return float(linear_vel), float(angular_vel), False, lookahead_point
