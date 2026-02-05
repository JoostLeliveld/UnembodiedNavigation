import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import yaml
from ament_index_python.packages import get_package_share_directory


VALID_PLANNERS = {"astar", "efe1", "efe2"}


def load_world_profiles(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"world_profiles not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    worlds = data.get("worlds")
    if not isinstance(worlds, dict) or not worlds:
        raise RuntimeError("world_profiles.yaml must contain a non-empty 'worlds' mapping")
    return worlds


def get_worlds_dir() -> str:
    sim_share = get_package_share_directory("sim")
    return os.path.join(sim_share, "gazebo_worlds", "worlds")


def resolve_world_path(world_file: str) -> str:
    return os.path.join(get_worlds_dir(), world_file)


def _parse_included_models(world_path: str) -> List[str]:
    try:
        tree = ET.parse(world_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse world file '{world_path}': {exc}")
    root = tree.getroot()
    world_name = None
    for node in root.iter():
        if node.tag.endswith("world") and "name" in node.attrib:
            world_name = node.attrib.get("name")
            break
    models: List[str] = []

    for include in root.iter():
        if not include.tag.endswith("include"):
            continue
        uri_node = None
        for child in list(include):
            if child.tag.endswith("uri"):
                uri_node = child
                break
        if uri_node is None or not uri_node.text:
            continue
        uri = uri_node.text.strip()
        if not uri.startswith("model://"):
            continue
        model_path = uri[len("model://"):]
        model_name = model_path.split("/")[0]
        if model_name:
            models.append(model_name)

    return models, world_name


def _gz_resource_paths() -> List[str]:
    sim_share = get_package_share_directory("sim")
    sim_share_parent = os.path.dirname(sim_share)
    return [
        sim_share_parent,
        os.path.join(sim_share, "models"),
        os.path.join(sim_share, "gazebo_worlds", "models"),
        os.path.join(sim_share, "gazebo_worlds"),
        os.path.join(sim_share, "robot_description"),
        sim_share,
    ]


def _model_exists(model_name: str) -> bool:
    for base in _gz_resource_paths():
        if os.path.isdir(os.path.join(base, model_name)):
            return True
    return False


def _ensure_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected '{name}' to be a mapping")
    return value


def _ensure_number(value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"Expected '{name}' to be a number")


def validate_profile(world_file: str, profile: Dict[str, Any], world_path: str) -> None:
    _ensure_mapping(profile, world_file)
    if "world_name" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'world_name'")
    if "spawn" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'spawn'")
    if "planner_default" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'planner_default'")
    if "camera" not in profile:
        raise RuntimeError(f"Profile for '{world_file}' missing 'camera'")

    spawn = _ensure_mapping(profile["spawn"], "spawn")
    for key in ("x", "y", "z", "yaw"):
        _ensure_number(spawn.get(key), f"spawn.{key}")

    planner = profile.get("planner_default")
    if planner not in VALID_PLANNERS:
        raise RuntimeError(
            f"Invalid planner_default '{planner}' for '{world_file}'. "
            f"Valid: {', '.join(sorted(VALID_PLANNERS))}"
        )

    camera = _ensure_mapping(profile["camera"], "camera")
    for key in ("cam_pos", "look_at", "img_width", "img_height", "fov_h_rad"):
        if key not in camera:
            raise RuntimeError(f"Camera config missing '{key}' in '{world_file}'")
    if not isinstance(camera["cam_pos"], list) or len(camera["cam_pos"]) != 3:
        raise RuntimeError("camera.cam_pos must be a list of 3 numbers")
    if not isinstance(camera["look_at"], list) or len(camera["look_at"]) != 3:
        raise RuntimeError("camera.look_at must be a list of 3 numbers")

    if not os.path.isfile(world_path):
        raise RuntimeError(f"World file not found: {world_path}")

    included_models, parsed_world_name = _parse_included_models(world_path)
    missing = [model for model in included_models if not _model_exists(model)]
    if missing:
        raise RuntimeError(
            f"World '{world_file}' references missing models: {', '.join(sorted(missing))}"
        )

    if "external_camera" not in included_models:
        raise RuntimeError(
            f"World '{world_file}' does not include 'external_camera' but camera profile is set"
        )
    if parsed_world_name and parsed_world_name != profile["world_name"]:
        raise RuntimeError(
            f"World '{world_file}' name '{parsed_world_name}' does not match profile "
            f"world_name '{profile['world_name']}'"
        )


def load_profile(path: str, world_file: str) -> Dict[str, Any]:
    profiles = load_world_profiles(path)
    if world_file not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        raise RuntimeError(
            f"No profile for world '{world_file}'. Available: {known or 'none'}"
        )
    profile = profiles[world_file]
    world_path = resolve_world_path(world_file)
    validate_profile(world_file, profile, world_path)
    return profile


def compute_camera_quaternion(cam_pos: List[float], look_at: List[float]) -> List[float]:
    dx = look_at[0] - cam_pos[0]
    dy = look_at[1] - cam_pos[1]
    dz = look_at[2] - cam_pos[2]
    yaw = math.atan2(dy, dx)
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.atan2(dz, horiz)
    roll = 0.0

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]
