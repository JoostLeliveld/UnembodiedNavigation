"""Shared occlusion geometry helpers for warehouse-style visibility models."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np


_EMPTY_HEIGHT = -1.0e6


@dataclass(frozen=True)
class AxisAlignedPrism:
    """Axis-aligned vertical prism used for fast occlusion queries."""

    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def center_xy(self) -> np.ndarray:
        return np.array([
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
        ], dtype=float)

    @property
    def size_xy(self) -> np.ndarray:
        return np.array([
            max(self.xmax - self.xmin, 1e-6),
            max(self.ymax - self.ymin, 1e-6),
        ], dtype=float)

    def contains_xy(self, x: float, y: float) -> bool:
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def signed_distance_xy(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        dx = np.maximum(np.maximum(self.xmin - pts[:, 0], 0.0), pts[:, 0] - self.xmax)
        dy = np.maximum(np.maximum(self.ymin - pts[:, 1], 0.0), pts[:, 1] - self.ymax)
        outside = np.hypot(dx, dy)

        inside_x = np.minimum(pts[:, 0] - self.xmin, self.xmax - pts[:, 0])
        inside_y = np.minimum(pts[:, 1] - self.ymin, self.ymax - pts[:, 1])
        inside_depth = np.minimum(inside_x, inside_y)
        inside = (dx <= 0.0) & (dy <= 0.0)
        signed = outside
        signed[inside] = -inside_depth[inside]
        return signed

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'xmin': float(self.xmin),
            'xmax': float(self.xmax),
            'ymin': float(self.ymin),
            'ymax': float(self.ymax),
            'zmin': float(self.zmin),
            'zmax': float(self.zmax),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AxisAlignedPrism':
        return cls(
            name=str(data.get('name', 'prism')),
            xmin=float(data['xmin']),
            xmax=float(data['xmax']),
            ymin=float(data['ymin']),
            ymax=float(data['ymax']),
            zmin=float(data['zmin']),
            zmax=float(data['zmax']),
        )


@dataclass(frozen=True)
class OcclusionScene:
    prisms: tuple[AxisAlignedPrism, ...]
    source_world: str = ''
    model_name: str = 'warehouse_rack_occluders'

    def to_dict(self) -> dict:
        return {
            'source_world': self.source_world,
            'model_name': self.model_name,
            'prisms': [prism.to_dict() for prism in self.prisms],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(',', ':'))

    @classmethod
    def from_dict(cls, data: dict) -> 'OcclusionScene':
        prisms = tuple(AxisAlignedPrism.from_dict(item) for item in data.get('prisms', []))
        return cls(
            prisms=prisms,
            source_world=str(data.get('source_world', '')),
            model_name=str(data.get('model_name', 'warehouse_rack_occluders')),
        )

    @classmethod
    def from_json(cls, text: str | None) -> 'OcclusionScene':
        if text is None:
            return cls(prisms=())
        payload = str(text).strip()
        if not payload:
            return cls(prisms=())
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class _Pose6D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


def _tag_matches(node: ET.Element, name: str) -> bool:
    return node.tag == name or node.tag.endswith(name)


def _find_child(node: ET.Element, name: str) -> ET.Element | None:
    for child in list(node):
        if _tag_matches(child, name):
            return child
    return None


def _parse_pose_node(node: ET.Element | None) -> _Pose6D:
    if node is None or not (node.text and node.text.strip()):
        return _Pose6D()
    parts = [p for p in node.text.replace(',', ' ').split() if p]
    if len(parts) != 6:
        raise RuntimeError(f'Expected pose with 6 values, got {len(parts)}')
    vals = [float(part) for part in parts]
    return _Pose6D(*vals)


def _compose_pose(base: _Pose6D, local: _Pose6D) -> _Pose6D:
    cy = math.cos(base.yaw)
    sy = math.sin(base.yaw)
    x = base.x + cy * local.x - sy * local.y
    y = base.y + sy * local.x + cy * local.y
    z = base.z + local.z
    return _Pose6D(
        x=x,
        y=y,
        z=z,
        roll=base.roll + local.roll,
        pitch=base.pitch + local.pitch,
        yaw=base.yaw + local.yaw,
    )


def _parse_box_prism(name: str, pose: _Pose6D, size_text: str) -> AxisAlignedPrism:
    sx, sy, sz = [float(part) for part in size_text.replace(',', ' ').split() if part]
    cy = abs(math.cos(pose.yaw))
    syaw = abs(math.sin(pose.yaw))
    hx = 0.5 * (cy * sx + syaw * sy)
    hy = 0.5 * (syaw * sx + cy * sy)
    hz = 0.5 * sz
    return AxisAlignedPrism(
        name=name,
        xmin=pose.x - hx,
        xmax=pose.x + hx,
        ymin=pose.y - hy,
        ymax=pose.y + hy,
        zmin=pose.z - hz,
        zmax=pose.z + hz,
    )


def _parse_cylinder_prism(name: str, pose: _Pose6D, radius_text: str, length_text: str) -> AxisAlignedPrism:
    radius = float(radius_text)
    length = float(length_text)
    hz = 0.5 * length
    return AxisAlignedPrism(
        name=name,
        xmin=pose.x - radius,
        xmax=pose.x + radius,
        ymin=pose.y - radius,
        ymax=pose.y + radius,
        zmin=pose.z - hz,
        zmax=pose.z + hz,
    )


def _geometry_prisms_from_node(
    *,
    model_name: str,
    link_name: str,
    geometry_node: ET.Element | None,
    pose: _Pose6D,
    element_name: str,
) -> list[AxisAlignedPrism]:
    if geometry_node is None:
        return []

    prisms: list[AxisAlignedPrism] = []
    box = _find_child(geometry_node, 'box')
    if box is not None:
        size_node = _find_child(box, 'size')
        if size_node is not None and size_node.text:
            prisms.append(
                _parse_box_prism(
                    f'{model_name}/{link_name}:{element_name}',
                    pose,
                    size_node.text.strip(),
                )
            )
        return prisms

    cylinder = _find_child(geometry_node, 'cylinder')
    if cylinder is not None:
        radius_node = _find_child(cylinder, 'radius')
        length_node = _find_child(cylinder, 'length')
        if radius_node is not None and radius_node.text and length_node is not None and length_node.text:
            prisms.append(
                _parse_cylinder_prism(
                    f'{model_name}/{link_name}:{element_name}',
                    pose,
                    radius_node.text.strip(),
                    length_node.text.strip(),
                )
            )
    return prisms


def _normalize_names(names: str | Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        value = str(names).strip()
        return (value,) if value else ()
    out = []
    for item in names:
        value = str(item).strip()
        if value:
            out.append(value)
    return tuple(out)


def _normalize_geometry_tags(geometry_tags: Iterable[str] | None) -> tuple[str, ...]:
    if geometry_tags is None:
        return ('visual',)
    out = []
    for item in geometry_tags:
        value = str(item).strip().lower()
        if value:
            out.append(value)
    return tuple(out) if out else ('visual',)


def parse_occlusion_scene_from_world(
    world_path: str,
    *,
    model_name: str | Sequence[str] = 'warehouse_rack_occluders',
    geometry_tags: Iterable[str] | None = None,
) -> OcclusionScene:
    if not os.path.isfile(world_path):
        raise RuntimeError(f'World file not found: {world_path}')
    try:
        tree = ET.parse(world_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Failed to parse world file '{world_path}': {exc}") from exc

    root = tree.getroot()
    model_names = _normalize_names(model_name)
    geometry_kinds = _normalize_geometry_tags(geometry_tags)
    matched_models: list[ET.Element] = []
    for node in root.iter():
        if not _tag_matches(node, 'model'):
            continue
        node_name = str(node.attrib.get('name', '')).strip()
        if node_name in model_names:
            matched_models.append(node)

    if not matched_models:
        return OcclusionScene(
            prisms=(),
            source_world=world_path,
            model_name=','.join(model_names) if model_names else '',
        )

    prisms: list[AxisAlignedPrism] = []
    for model_node in matched_models:
        node_model_name = str(model_node.attrib.get('name', 'model')).strip() or 'model'
        model_pose = _parse_pose_node(_find_child(model_node, 'pose'))
        for link in list(model_node):
            if not _tag_matches(link, 'link'):
                continue
            link_name = str(link.attrib.get('name', 'link'))
            link_pose = _compose_pose(model_pose, _parse_pose_node(_find_child(link, 'pose')))
            for element in list(link):
                if element.tag.split('}')[-1].lower() not in geometry_kinds:
                    continue
                element_kind = element.tag.split('}')[-1].lower()
                element_name = str(element.attrib.get('name', element_kind))
                element_pose = _compose_pose(link_pose, _parse_pose_node(_find_child(element, 'pose')))
                prisms.extend(
                    _geometry_prisms_from_node(
                        model_name=node_model_name,
                        link_name=link_name,
                        geometry_node=_find_child(element, 'geometry'),
                        pose=element_pose,
                        element_name=element_name,
                    )
                )
    return OcclusionScene(
        prisms=tuple(prisms),
        source_world=world_path,
        model_name=','.join(model_names) if model_names else '',
    )


def parse_collision_scene_from_world(
    world_path: str,
    *,
    model_names: Sequence[str] = ('warehouse_walls', 'warehouse_rack_occluders'),
) -> OcclusionScene:
    return parse_occlusion_scene_from_world(
        world_path,
        model_name=model_names,
        geometry_tags=('collision',),
    )


def scene_from_json(text: str | None) -> OcclusionScene:
    return OcclusionScene.from_json(text)


def scene_to_json(scene: OcclusionScene) -> str:
    return scene.to_json()


def top_heights_for_xy(prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism], xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    heights = np.full(pts.shape[0], _EMPTY_HEIGHT, dtype=float)
    for prism in prisms:
        mask = (
            (pts[:, 0] >= prism.xmin)
            & (pts[:, 0] <= prism.xmax)
            & (pts[:, 1] >= prism.ymin)
            & (pts[:, 1] <= prism.ymax)
        )
        if np.any(mask):
            heights[mask] = np.maximum(heights[mask], prism.zmax)
    return heights


def top_height_at_xy(prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism], x: float, y: float) -> float:
    return float(top_heights_for_xy(prisms, np.array([[x, y]], dtype=float))[0])


def signed_distance_to_union_xy(prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism], xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    if not prisms:
        return np.full(pts.shape[0], np.inf, dtype=float)
    signed = np.full(pts.shape[0], np.inf, dtype=float)
    for prism in prisms:
        signed = np.minimum(signed, prism.signed_distance_xy(pts))
    return signed


def segment_intersects_prism(start_xyz: np.ndarray, end_xyz: np.ndarray, prism: AxisAlignedPrism, *, eps: float = 1e-9) -> bool:
    p0 = np.asarray(start_xyz, dtype=float)
    p1 = np.asarray(end_xyz, dtype=float)
    direction = p1 - p0
    bounds_min = np.array([prism.xmin, prism.ymin, prism.zmin], dtype=float)
    bounds_max = np.array([prism.xmax, prism.ymax, prism.zmax], dtype=float)

    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) <= eps:
            if p0[axis] < bounds_min[axis] or p0[axis] > bounds_max[axis]:
                return False
            continue
        inv = 1.0 / direction[axis]
        t0 = (bounds_min[axis] - p0[axis]) * inv
        t1 = (bounds_max[axis] - p0[axis]) * inv
        if t0 > t1:
            t0, t1 = t1, t0
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_max < t_min:
            return False
    return (t_max >= max(t_min, 0.0)) and (t_min <= 1.0)


def segment_occluded(
    prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism],
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
) -> bool:
    return any(segment_intersects_prism(start_xyz, end_xyz, prism) for prism in prisms)
