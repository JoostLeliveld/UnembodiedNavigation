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


import functools

@functools.lru_cache(maxsize=16)
def _get_union_boundary_segments(prisms: tuple[AxisAlignedPrism, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    if not prisms:
        return []
    
    prisms_list = list(prisms)

    def covered_outside(px, py, ignore_idx):
        """True if the point (just outside one prism's edge) lies inside ANY other prism.
        Used to drop INTERNAL union seams: an edge of prism P is part of the true union
        boundary only if the space immediately on its OUTWARD side is free. This correctly
        dissolves both overlapping prisms AND exactly-abutting prisms (whose shared edge the
        old midpoint-interior test missed, leaving phantom internal boundary segments)."""
        for idx2, q in enumerate(prisms_list):
            if idx2 == ignore_idx:
                continue
            if (q.xmin - 1e-6 <= px <= q.xmax + 1e-6) and (q.ymin - 1e-6 <= py <= q.ymax + 1e-6):
                return True
        return False

    probe = 1e-3  # outward offset (smaller than any lane width, larger than float noise)
    boundary_segments = []
    for idx, p in enumerate(prisms_list):
        # each edge carries its OUTWARD normal (pointing away from this prism's interior)
        edges = [
            (np.array([p.xmin, p.ymin]), np.array([p.xmin, p.ymax]), 'vertical', np.array([-1.0, 0.0])),
            (np.array([p.xmax, p.ymin]), np.array([p.xmax, p.ymax]), 'vertical', np.array([1.0, 0.0])),
            (np.array([p.xmin, p.ymin]), np.array([p.xmax, p.ymin]), 'horizontal', np.array([0.0, -1.0])),
            (np.array([p.xmin, p.ymax]), np.array([p.xmax, p.ymax]), 'horizontal', np.array([0.0, 1.0])),
        ]
        for p1, p2, orient, outward in edges:
            splits = [0.0, 1.0]
            if orient == 'vertical':
                x_val = p1[0]
                y_start, y_end = p1[1], p2[1]
                for o_idx, op in enumerate(prisms_list):
                    if o_idx == idx:
                        continue
                    if op.xmin - 1e-5 < x_val < op.xmax + 1e-5:
                        y_min_clip = max(y_start, op.ymin)
                        y_max_clip = min(y_end, op.ymax)
                        if y_min_clip < y_max_clip:
                            splits.append((y_min_clip - y_start) / (y_end - y_start))
                            splits.append((y_max_clip - y_start) / (y_end - y_start))
            else:
                y_val = p1[1]
                x_start, x_end = p1[0], p2[0]
                for o_idx, op in enumerate(prisms_list):
                    if o_idx == idx:
                        continue
                    if op.ymin - 1e-5 < y_val < op.ymax + 1e-5:
                        x_min_clip = max(x_start, op.xmin)
                        x_max_clip = min(x_end, op.xmax)
                        if x_min_clip < x_max_clip:
                            splits.append((x_min_clip - x_start) / (x_end - x_start))
                            splits.append((x_max_clip - x_start) / (x_end - x_start))

            splits = sorted(list(set([float(np.clip(s, 0.0, 1.0)) for s in splits])))
            for i in range(len(splits) - 1):
                s1, s2 = splits[i], splits[i+1]
                if s2 - s1 < 1e-5:
                    continue
                smid = 0.5 * (s1 + s2)
                pt_mid = p1 + smid * (p2 - p1)
                # keep only TRUE boundary: the outward side of this sub-edge is free space
                # (not covered by an overlapping OR abutting prism)
                if not covered_outside(pt_mid[0] + probe * outward[0],
                                       pt_mid[1] + probe * outward[1], idx):
                    boundary_segments.append((
                        p1 + s1 * (p2 - p1),
                        p1 + s2 * (p2 - p1)
                    ))
    return boundary_segments


def signed_distance_to_union_xy(
    prisms: tuple[AxisAlignedPrism, ...] | list[AxisAlignedPrism],
    xy: np.ndarray,
    *,
    keep_in: bool = True,
) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)
    is_1d = (pts.ndim == 1)
    if is_1d:
        pts = pts.reshape(1, 2)
    if not prisms:
        return np.full(pts.shape[0], np.inf, dtype=float)
    
    if not keep_in:
        signed = np.full(pts.shape[0], np.inf, dtype=float)
        for prism in prisms:
            signed = np.minimum(signed, prism.signed_distance_xy(pts))
        return signed
        
    segs = _get_union_boundary_segments(tuple(prisms))
    if not segs:
        return np.full(pts.shape[0], np.inf, dtype=float)
        
    n_pts = pts.shape[0]
    min_dists = np.full(n_pts, np.inf, dtype=float)
    
    for p1, p2 in segs:
        v = p2 - p1
        v_len_sq = np.sum(v**2)
        if v_len_sq < 1e-9:
            d = np.linalg.norm(pts - p1, axis=1)
        else:
            w = pts - p1
            t = np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0)
            closest = p1 + t[:, np.newaxis] * v
            d = np.linalg.norm(pts - closest, axis=1)
        min_dists = np.minimum(min_dists, d)
        
    is_inside = np.zeros(n_pts, dtype=bool)
    for p in prisms:
        dx = np.maximum(np.maximum(p.xmin - pts[:, 0], 0.0), pts[:, 0] - p.xmax)
        dy = np.maximum(np.maximum(p.ymin - pts[:, 1], 0.0), pts[:, 1] - p.ymax)
        is_inside |= (dx <= 0.0) & (dy <= 0.0)
        
    signed = np.where(is_inside, -min_dists, min_dists)
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
