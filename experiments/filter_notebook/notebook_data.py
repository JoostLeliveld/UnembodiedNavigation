#!/usr/bin/env python3
"""Load the notebook demonstration capture: frames, detections, odometry, truth.

Why this exists rather than ``rcond_common.load_operational_capture``:

* That loader back-projects pixels through ``legacy_projection.project_pixel_to_world``
  with the ``projection_calibration_v2`` corrections. Those corrections were
  **deleted from the runtime on 2026-08-07** -- raw inverse perspective mapping beat
  every fitted variant (66.6 mm vs 68.2 mm for v2), so the deployed path is now the
  parameter-free ``camera.pixel_to_world(u, v)``. A notebook that explains the
  current system must use the current mapping.
* It also keys captures off a frozen ``CAPTURES`` dict of campaign evidence. This
  capture is a demonstration (see ``CAPTURE_ROLE.md``) and does not belong there.

Everything below the EVALUATION ONLY line reads ground truth. No filter, and no
quantity a filter consumes, may be built from it -- it exists to score.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

import numpy as np

_HERE = Path(__file__).resolve()
REPO = next(p for p in _HERE.parents if (p / "src").is_dir() and (p / "logs").is_dir())

import sys

for _path in ("src/reliability", "src/unav_common", "src/state"):
    sys.path.insert(0, str(REPO / _path))

from reliability.projection import camera_model_from_world  # noqa: E402

STUDY_ROOT = REPO / "logs/studies/filter_notebook"
TRUTH_TOL_S = 0.05
ASSOC_TOL_S = 0.15


@dataclass(frozen=True)
class World:
    """Which world a capture was recorded in, and what it has in it.

    There are two, and they are not interchangeable. Method development belongs in the
    single-camera AWS warehouse (`research/06_world_camera_design.md`); the frozen
    four-camera world evaluates methods that are already settled. The filter notebooks
    are method development, so they run in the first -- but the four-camera captures
    still exist and still load, so both have to be describable.
    """

    key: str
    world_sdf_name: str
    cameras: tuple[str, ...]
    model_includes: dict
    image_topics: dict
    commissioned_file: str
    detector_model: str
    description: str

    @property
    def world_sdf(self):
        return REPO / "src/sim/gazebo_worlds/worlds" / self.world_sdf_name


AWS_SINGLE = World(
    key="warehouse_aws",
    world_sdf_name="warehouse_aws.world.sdf",
    cameras=("camera_A",),
    model_includes={"camera_A": "external_camera"},
    image_topics={"camera_A": "/external_camera/image_raw"},
    commissioned_file="commissioned_observation_noise_aws.json",
    # Captured and trained IN this world, 2026-06-17, with an occlusion gate on the
    # labels. imgsz 960 must match training or box placement degrades.
    detector_model="warehouse_yolo_detector_v1",
    description="AWS warehouse, one ceiling camera on the south wall",
)

FULL_4CAM = World(
    key="warehouse_full_4cam",
    world_sdf_name="warehouse_full_4cam.world.sdf",
    cameras=("camera_A", "camera_B", "camera_C", "camera_D"),
    model_includes={
        "camera_A": "external_camera",
        "camera_B": "external_camera_b",
        "camera_C": "external_camera_c",
        "camera_D": "external_camera_d",
    },
    image_topics={
        "camera_A": "/external_camera/image_raw",
        "camera_B": "/external_camera_b/image_raw",
        "camera_C": "/external_camera_c/image_raw",
        "camera_D": "/external_camera_d/image_raw",
    },
    commissioned_file="commissioned_observation_noise.json",
    detector_model="warehouse_yolo_detector_4cam_v3_960",
    description="frozen flagship world, four cameras",
)

WORLDS = {w.key: w for w in (AWS_SINGLE, FULL_4CAM)}

# The world every loader below currently speaks for. `load_capture` sets it from the
# capture's own manifest, so a notebook picks it up by loading its data and never has to
# be told. It is module state rather than an argument threaded through forty call sites
# because a notebook works in exactly one world for its whole life.
ACTIVE = FULL_4CAM
WORLD_SDF = ACTIVE.world_sdf
CAMERAS = ACTIVE.cameras
MODEL_INCLUDES = ACTIVE.model_includes
IMAGE_TOPICS = ACTIVE.image_topics


def use_world(world) -> World:
    """Point every loader at one of the two worlds. Returns the world now active."""
    global ACTIVE, WORLD_SDF, CAMERAS, MODEL_INCLUDES, IMAGE_TOPICS
    ACTIVE = WORLDS[world] if isinstance(world, str) else world
    WORLD_SDF = ACTIVE.world_sdf
    CAMERAS = ACTIVE.cameras
    MODEL_INCLUDES = ACTIVE.model_includes
    IMAGE_TOPICS = ACTIVE.image_topics
    return ACTIVE


def world_of(run_tag=None) -> World:
    """Which world a capture was recorded in, from the manifest it carries.

    Captures recorded before the manifest existed are all four-camera ones, so that is
    what an absent manifest means.
    """
    try:
        root = capture_root(run_tag)
    except FileNotFoundError:
        return FULL_4CAM
    path = root / "raw" / "capture_manifest.json"
    if not path.is_file():
        return FULL_4CAM
    try:
        return WORLDS[json.loads(path.read_text(encoding="utf-8"))["world_name"]]
    except (json.JSONDecodeError, KeyError):
        return FULL_4CAM


def detector_of(run_tag=None):
    """The detector weights a capture was recorded with, from its own manifest.

    A figure that re-runs YOLO on a recorded frame has to run the SAME weights the
    pipeline ran, or the boxes it draws are not the boxes that became the observations.
    The two worlds have different detectors: the AWS world's was captured and trained in
    that world with an occlusion gate on the labels; the flagship world's was trained on
    its four viewpoints.
    """
    default = REPO / "logs/perception_models" / ACTIVE.detector_model / "model.pt"
    if run_tag is None:
        return default
    try:
        path = capture_root(run_tag) / "raw" / "capture_manifest.json"
        return Path(json.loads(path.read_text(encoding="utf-8"))["detector_model"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return default


def camera_models(world=None) -> dict:
    """Every camera in the world, posed from the world file itself. No fitted parameters."""
    spec = ACTIVE if world is None else (WORLDS[world] if isinstance(world, str) else world)
    return {
        cam: camera_model_from_world(spec.world_sdf, include_name=spec.model_includes[cam])
        for cam in spec.cameras
    }


# Captures used to COMMISSION the observation noise. Deliberately disjoint from the
# notebook capture: a covariance fitted on the same run it is then evaluated on tells
# you nothing about whether it generalises. All three predate the notebook capture and
# were recorded on the same world with the same camera poses.
COMMISSIONING_CAPTURES = {
    "smoke1_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke_20260716",
    "smoke2_20260716": REPO
    / "logs/studies/multicamera_commissioning_bigwarehouse/gt_validation_smoke2_20260716",
    "fusion_handover_20260721": REPO
    / "logs/studies/multicamera_fusion_extension/fusion_handover_real_20260721/data",
}


def capture_root(run_tag: str | None = None) -> Path:
    """The capture directory: an explicit path, a named run, or the newest recorded."""
    if isinstance(run_tag, Path):
        if not run_tag.is_dir():
            raise FileNotFoundError(f"no capture at {run_tag}")
        return run_tag
    if run_tag and run_tag in COMMISSIONING_CAPTURES:
        return COMMISSIONING_CAPTURES[run_tag]
    if run_tag:
        root = STUDY_ROOT / run_tag
        if not root.is_dir():
            raise FileNotFoundError(f"no capture at {root}")
        return root
    candidates = sorted(
        p for p in STUDY_ROOT.glob("*")
        if (p / "raw" / "experiment.csv").is_file()
        and (p.name.startswith("notebook") or p.name.startswith("aws_"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"no notebook capture under {STUDY_ROOT}; run capture_aws_notebook_dataset.sh"
        )
    return candidates[-1]


@dataclass(frozen=True)
class Detection:
    """One detection, as the runtime sees it: a pixel and where it lands on the floor."""

    camera: str
    stamp: float
    u: float
    v: float
    world: tuple[float, float]
    range_m: float


@dataclass(frozen=True)
class Capture:
    name: str
    root: Path
    stamps: np.ndarray            # odometry stamps, simulated seconds
    odom: np.ndarray              # (N, 2) warehouse metres
    odom_cov: np.ndarray          # (N, 2, 2)
    detections: dict[str, list[Detection]]
    world: "World" = None         # which world, and therefore which cameras exist

    @property
    def cameras(self) -> tuple:
        return (self.world or ACTIVE).cameras

    @property
    def n_steps(self) -> int:
        return int(self.stamps.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.stamps[-1] - self.stamps[0]) if self.n_steps > 1 else 0.0

    @property
    def n_detections(self) -> int:
        return sum(len(v) for v in self.detections.values())

    def frames(self, camera: str) -> list[tuple[float, Path]]:
        """(stamp, png path) for every recorded frame of one camera."""
        index = self.root / "views" / camera / "index.csv"
        if not index.is_file():
            return []
        out = []
        with index.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    stamp = float(row["stamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                path = index.parent / row["file"]
                if path.is_file():
                    out.append((stamp, path))
        return sorted(out)

    def frame_at(self, camera: str, stamp: float, tol_s: float = 0.5):
        """The recorded frame nearest a stamp, or None if none is close enough."""
        frames = self.frames(camera)
        if not frames:
            return None
        stamps = np.asarray([f[0] for f in frames])
        idx = int(np.argmin(np.abs(stamps - stamp)))
        if abs(float(stamps[idx]) - stamp) > tol_s:
            return None
        return frames[idx]


ROBOT_MESHES = REPO / "src/sim/robot_description/meshes/turtlebot3_description/meshes"
ROBOT_URDF = REPO / "src/sim/robot_description/urdf/turtlebot3_burger.urdf.xacro"


def _load_stl(path: Path, scale: float = 0.001) -> np.ndarray:
    """Vertices of a binary STL, in metres."""
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    vertices = np.empty((count * 3, 3), dtype=float)
    for i in range(count):
        record = raw[84 + 50 * i:84 + 50 * (i + 1)]
        for j in range(3):
            vertices[3 * i + j] = struct.unpack("<3f", record[12 + 12 * j:24 + 12 * j])
    return vertices * scale


def robot_point_cloud() -> np.ndarray:
    """The robot's actual surface, in its own frame, assembled from the URDF meshes.

    The placements are the ones in `turtlebot3_burger.urdf.xacro`: base_link sits 10 mm
    above base_footprint, the base mesh is offset -32 mm in x inside it, the tires hang at
    y = +-80 mm, and the lidar sits on top. base_footprint is the frame the ground-truth
    stream reports, so this cloud is expressed in exactly the frame the truth is in.

    There is nothing fitted here. It is what Gazebo renders, which is what the detector
    sees.
    """
    def rot_x(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    parts = [_load_stl(ROBOT_MESHES / "bases/burger_base.stl") + np.array([-0.032, 0.0, 0.010])]
    for side, y in (("left", 0.08), ("right", -0.08)):
        tire = _load_stl(ROBOT_MESHES / f"wheels/{side}_tire.stl")
        tire = tire @ rot_x(1.57).T @ rot_x(-1.57).T      # visual rpy, then joint rpy
        parts.append(tire + np.array([0.0, y, 0.023 + 0.010]))
    parts.append(_load_stl(ROBOT_MESHES / "sensors/lds.stl")
                 + np.array([-0.032 - 0.045, 0.0, 0.172 + 0.010]))
    return np.vstack(parts)


def route_window(run_tag: str | None = None) -> tuple[float, float] | None:
    """(start, end) simulated seconds of the driven route, from the driver's record.

    The recorders keep running after the route finishes, so the robot sits parked in
    one camera's view for as long as it takes to shut down. Left in, that stationary
    tail dominates that camera's statistics -- in the first full capture it tripled
    camera B's detection count. Everything downstream should be trimmed to this.
    """
    path = capture_root(run_tag) / "raw" / "route_completion.json"
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        start = float(record["route_started_sim_time_s"])
        end = float(record["route_completed_sim_time_s"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return (start, end) if end > start else None


def _float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def load_capture(run_tag: str | None = None, models=None) -> Capture:
    """Odometry and detections. Truth is not read here, by construction.

    Loading a capture also switches every loader in this module to the world that
    capture was recorded in, so a notebook never has to declare which world it is in --
    it says which drive it is looking at and the rest follows.
    """
    root = capture_root(run_tag)
    world = use_world(world_of(run_tag))
    models = models if models is not None else camera_models()

    stamps, odom, cov = [], [], []
    with (root / "raw" / "experiment.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = _float(row, "stamp")
            x, y = _float(row, "odom_noisy_x"), _float(row, "odom_noisy_y")
            if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
                continue
            stamps.append(t)
            odom.append((x, y))
            cxx = _float(row, "odom_noisy_cov_xx")
            cxy = _float(row, "odom_noisy_cov_xy")
            cyy = _float(row, "odom_noisy_cov_yy")
            cov.append(((cxx, cxy), (cxy, cyy)))

    order = np.argsort(np.asarray(stamps, dtype=float))
    stamps_arr = np.asarray(stamps, dtype=float)[order]
    odom_arr = np.asarray(odom, dtype=float)[order]
    cov_arr = np.asarray(cov, dtype=float)[order]

    detections: dict[str, list[Detection]] = {}
    for cam in world.cameras:
        src = root / "raw" / f"{cam}_perception.csv"
        found: list[Detection] = []
        if src.is_file():
            model = models[cam]
            cam_x, cam_y = float(model.cam_pos[0]), float(model.cam_pos[1])
            with src.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("detected") != "1":
                        continue
                    u, v = _float(row, "obs_u"), _float(row, "obs_v")
                    t = _float(row, "diag_stamp")
                    if not all(math.isfinite(q) for q in (u, v, t)):
                        continue
                    # The deployed mapping: one homography, zero fitted parameters.
                    point = model.pixel_to_world(u, v)
                    if point is None or not all(math.isfinite(c) for c in point):
                        continue
                    found.append(
                        Detection(
                            camera=cam, stamp=t, u=u, v=v,
                            world=(float(point[0]), float(point[1])),
                            range_m=float(math.hypot(point[0] - cam_x, point[1] - cam_y)),
                        )
                    )
        detections[cam] = found

    return Capture(
        name=root.name, root=root, stamps=stamps_arr, odom=odom_arr,
        odom_cov=cov_arr, detections=detections, world=world,
    )


def load_messages(run_tag: str | None = None) -> dict[str, list[tuple[float, bool]]]:
    """(stamp, detected) for EVERY observation message, misses included.

    ``load_capture`` keeps only the detections, because that is all a filter consumes.
    But the misses are most of the traffic -- the detector returns nothing for between
    45% and 83% of the frames it is handed -- and any honest account of how the
    perception stage behaves has to show them.
    """
    root = capture_root(run_tag)
    out: dict[str, list[tuple[float, bool]]] = {}
    for camera in world_of(run_tag).cameras:
        src = root / "raw" / f"{camera}_perception.csv"
        rows: list[tuple[float, bool]] = []
        if src.is_file():
            with src.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    stamp = _float(row, "diag_stamp")
                    if math.isfinite(stamp):
                        rows.append((stamp, row.get("detected") == "1"))
        out[camera] = sorted(rows)
    return out


def associate(capture: Capture, stamp: float, tol_s: float = ASSOC_TOL_S):
    """Nearest odometry index for a stamp, or None if no step is within ``tol_s``."""
    idx = int(np.argmin(np.abs(capture.stamps - stamp)))
    if abs(float(capture.stamps[idx]) - stamp) > tol_s:
        return None
    return idx


# --------------------------------------------------------------------------- #
# EVALUATION ONLY below this line
# --------------------------------------------------------------------------- #


def load_truth(run_tag: str | None = None):
    """(stamps, xy, yaw) from the ground-truth stream. Scoring only."""
    root = capture_root(run_tag)
    src = root / "evaluation_only" / "ground_truth.csv"
    stamps, xy, yaw = [], [], []
    with src.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = _float(row, "stamp")
            x, y = _float(row, "gt_x"), _float(row, "gt_y")
            th = _float(row, "gt_yaw")
            if not all(math.isfinite(q) for q in (t, x, y)):
                continue
            stamps.append(t)
            xy.append((x, y))
            yaw.append(th)
    order = np.argsort(np.asarray(stamps, dtype=float))
    return (np.asarray(stamps, dtype=float)[order],
            np.asarray(xy, dtype=float)[order],
            np.asarray(yaw, dtype=float)[order])


def truth_at(table, stamp: float, tol_s: float = TRUTH_TOL_S):
    """Nearest truth pose to a stamp, or None outside ``tol_s``."""
    stamps, xy, yaw = table
    if stamps.size == 0:
        return None
    idx = int(np.argmin(np.abs(stamps - stamp)))
    if abs(float(stamps[idx]) - stamp) > tol_s:
        return None
    return float(xy[idx, 0]), float(xy[idx, 1]), float(yaw[idx])


if __name__ == "__main__":
    cap = load_capture()
    print(f"capture   {cap.name}")
    print(f"odometry  {cap.n_steps} steps over {cap.duration_s:.1f} s simulated")
    for cam in CAMERAS:
        n = len(cap.detections[cam])
        hz = n / cap.duration_s if cap.duration_s else 0.0
        print(f"  {cam}  {n:5d} detections ({hz:5.2f} Hz sim), "
              f"{len(cap.frames(cam)):4d} frames recorded")
    stamps, xy, _ = load_truth()
    print(f"truth     {stamps.size} poses (evaluation only)")
