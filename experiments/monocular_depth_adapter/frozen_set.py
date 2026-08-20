"""Define and load the frozen image set the depth models are measured on.

This module is the boundary between the repo and the adapter. It is the only
place that knows about capture directories, world profiles, and the camera
model; the ``monodepth`` package below it knows only about pixels and a 3x3 K.

"Frozen" means the manifest pins each frame by SHA-256. If a capture directory is
ever re-rendered, verification fails loudly instead of quietly changing what a
benchmark number refers to.

Two roles of frame, and they are not interchangeable:

``method_development``
    Frames from ``warehouse_aws``, the world the repo reserves for building
    methods. Every accuracy, uncertainty, and model-comparison statement comes
    from these.

``batch_plumbing_only``
    Frames from all four cameras of ``warehouse_full_4cam``. That world is
    reserved for evaluating frozen methods, so these frames are used ONLY to
    show that batch inference runs across four cameras and to record what it
    costs. No model ranking, accuracy claim, or tuning decision may be drawn
    from them.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
for _pkg in ("src/unav_common", "src/experiments"):
    _p = str(REPO / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from monodepth.determinism import file_sha256              # noqa: E402
from monodepth.types import CameraIntrinsics, DepthRequest  # noqa: E402

MANIFEST_DIR = _HERE / "frozen_sets"
DEFAULT_SET = "monodepth_frozen_v1"

#: Capture directories the frozen set draws from. Both are existing real-Gazebo
#: renders; nothing here generates imagery.
AWS_CAPTURE = REPO / "logs/visibility_comparison/warehouse_visibility_capture_v1"
FOURCAM_CAPTURE = REPO / "logs/visibility_comparison/commissioning_grid_20260807"

#: Suffix in the 4-camera capture's filenames -> camera letter. Camera A has no
#: suffix because it is the capture's primary ``external_camera`` frame.
FOURCAM_SUFFIX_TO_CAMERA = {
    "": "A",
    "_external_camera_b": "B",
    "_external_camera_c": "C",
    "_external_camera_d": "D",
}


@dataclass(frozen=True)
class FrozenFrame:
    """One pinned image plus the calibration that produced it."""

    frame_id: str
    image_path: str          # repo-relative
    sha256: str
    width: int
    height: int
    camera_id: str
    camera_frame: str
    world: str
    role: str
    intrinsics: dict
    camera_pose_xyzrpy: list
    source_capture: str
    source_sample_id: str

    def absolute_path(self) -> Path:
        return REPO / self.image_path

    def camera_intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(**self.intrinsics)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------- intrinsics
def _intrinsics_from_capture(img_width: int, img_height: int, fov_h_rad: float,
                             cam_pos, look_at) -> CameraIntrinsics:
    """Derive K through the repo's canonical camera, not a local formula.

    ``ObliqueCameraModel`` is the single definition of what these cameras' K is;
    re-deriving ``f = (W/2)/tan(fov/2)`` here would be a second copy that can
    drift from it.
    """
    from unav_common.camera_model import ObliqueCameraModel

    cam = ObliqueCameraModel(cam_pos=cam_pos, look_at=look_at, img_width=img_width,
                             img_height=img_height, fov_h_rad=fov_h_rad)
    return CameraIntrinsics.from_matrix(cam.K, img_width, img_height)


def _look_at(pose_xyzrpy: Sequence[float]) -> list:
    from experiments.core.world_profiles import compute_look_at_from_pose

    return list(compute_look_at_from_pose(list(pose_xyzrpy[:3]), pose_xyzrpy[3],
                                          pose_xyzrpy[4], pose_xyzrpy[5]))


# ------------------------------------------------------------------- construction
def _capture_metadata(capture_dir: Path) -> dict:
    """Camera metadata, from whichever manifest the capture happens to carry."""
    for name in ("manifest.json", "capture_manifest.json"):
        path = capture_dir / name
        if not path.is_file():
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        meta = blob.get("capture_metadata", blob)
        if "img_width" in meta:
            return meta
    raise FileNotFoundError(f"no capture metadata with intrinsics under {capture_dir}")


def _read_samples(capture_dir: Path) -> list[dict]:
    with open(capture_dir / "samples.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pick_evenly(rows: Sequence[dict], count: int) -> list[dict]:
    """Deterministic even spread across the capture order — no RNG anywhere."""
    if count >= len(rows):
        return list(rows)
    idx = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[i] for i in dict.fromkeys(idx.tolist())]


def build_aws_frames(count: int = 12) -> list[FrozenFrame]:
    """Method-development frames: one wall camera in ``warehouse_aws``."""
    meta = _capture_metadata(AWS_CAPTURE)
    intr = _intrinsics_from_capture(meta["img_width"], meta["img_height"],
                                    meta["fov_h_rad"], meta["camera_pos"], meta["look_at"])
    rows = _pick_evenly(_read_samples(AWS_CAPTURE), count)

    frames = []
    for row in rows:
        path = AWS_CAPTURE / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(FrozenFrame(
            frame_id=f"aws_camA_s{int(row['sample_id']):05d}",
            image_path=str(path.relative_to(REPO)),
            sha256=file_sha256(path),
            width=int(meta["img_width"]), height=int(meta["img_height"]),
            camera_id="A", camera_frame=meta.get("camera_frame", "external_camera"),
            world=meta["world"], role="method_development",
            intrinsics=intr.as_dict(),
            camera_pose_xyzrpy=[float(v) for v in meta["camera_pose"]],
            source_capture=AWS_CAPTURE.name, source_sample_id=str(row["sample_id"]),
        ))
    return frames


def build_fourcam_frames(count_per_camera: int = 3) -> list[FrozenFrame]:
    """Plumbing-only frames: all four cameras of ``warehouse_full_4cam``.

    Same robot pose across the four cameras, so a batch is four genuinely
    different viewpoints rather than four copies of one.
    """
    meta = _capture_metadata(FOURCAM_CAPTURE)
    mounts = {"": list(meta["camera_pose"])}
    for frame_name, pose in meta.get("extra_camera_mounts", {}).items():
        mounts[f"_{frame_name}"] = list(pose)

    intrinsics_by_suffix = {}
    for suffix, pose in mounts.items():
        intrinsics_by_suffix[suffix] = _intrinsics_from_capture(
            meta["img_width"], meta["img_height"], meta["fov_h_rad"],
            pose[:3], _look_at(pose))

    # samples.csv holds one row per (pose, camera); keep only the primary rows so
    # a pose is picked once and then read out across all four cameras.
    primary = [r for r in _read_samples(FOURCAM_CAPTURE) if r["camera_frame"] == "external_camera"]
    rows = _pick_evenly(primary, count_per_camera)
    frames = []
    for row in rows:
        stem = Path(row["image_path"]).stem
        for suffix, camera_id in FOURCAM_SUFFIX_TO_CAMERA.items():
            path = FOURCAM_CAPTURE / "images" / f"{stem}{suffix}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            pose = mounts[suffix]
            frames.append(FrozenFrame(
                frame_id=f"full4cam_cam{camera_id}_s{int(row['sample_id']):05d}",
                image_path=str(path.relative_to(REPO)),
                sha256=file_sha256(path),
                width=int(meta["img_width"]), height=int(meta["img_height"]),
                camera_id=camera_id,
                camera_frame="external_camera" if suffix == "" else suffix.lstrip("_"),
                world=meta["world"], role="batch_plumbing_only",
                intrinsics=intrinsics_by_suffix[suffix].as_dict(),
                camera_pose_xyzrpy=[float(v) for v in pose],
                source_capture=FOURCAM_CAPTURE.name, source_sample_id=str(row["sample_id"]),
            ))
    return frames


def build_manifest(name: str = DEFAULT_SET, *, aws_count: int = 12,
                   fourcam_per_camera: int = 3) -> dict:
    frames = build_aws_frames(aws_count) + build_fourcam_frames(fourcam_per_camera)
    return {
        "name": name,
        "schema_version": 1,
        "purpose": (
            "Frozen image set for monocular depth adapter acceptance tests and "
            "model benchmarking. Frames pinned by SHA-256."
        ),
        "role_contract": {
            "method_development": (
                "warehouse_aws frames. All accuracy, uncertainty and model-comparison "
                "statements come from these."
            ),
            "batch_plumbing_only": (
                "warehouse_full_4cam frames. Used ONLY to show four-camera batch "
                "inference runs and what it costs. No model ranking or accuracy claim."
            ),
        },
        "sources": {
            "warehouse_aws": str(AWS_CAPTURE.relative_to(REPO)),
            "warehouse_full_4cam": str(FOURCAM_CAPTURE.relative_to(REPO)),
        },
        "frames": [f.as_dict() for f in frames],
    }


# ------------------------------------------------------------------------- loading
def manifest_path(name: str = DEFAULT_SET) -> Path:
    return MANIFEST_DIR / f"{name}.json"


def load_manifest(name: str = DEFAULT_SET) -> dict:
    path = manifest_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"frozen set {name!r} not found at {path}. "
            "Build it with: python3 experiments/monocular_depth_adapter/build_frozen_set.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(name: str = DEFAULT_SET, *, role: str | None = None) -> list[FrozenFrame]:
    frames = [FrozenFrame(**f) for f in load_manifest(name)["frames"]]
    if role is not None:
        frames = [f for f in frames if f.role == role]
    return frames


def verify(name: str = DEFAULT_SET) -> list[str]:
    """Re-hash every frame. Returns a list of problems; empty means the set is intact."""
    problems = []
    for frame in load_frames(name):
        path = frame.absolute_path()
        if not path.is_file():
            problems.append(f"missing: {frame.image_path}")
            continue
        actual = file_sha256(path)
        if actual != frame.sha256:
            problems.append(f"changed: {frame.image_path} ({frame.sha256[:12]} -> {actual[:12]})")
    return problems


def load_image(frame: FrozenFrame) -> np.ndarray:
    """Read one frame as (H, W, 3) uint8 RGB."""
    from PIL import Image

    with Image.open(frame.absolute_path()) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if array.shape[:2] != (frame.height, frame.width):
        raise ValueError(
            f"{frame.image_path} is {array.shape[1]}x{array.shape[0]}, "
            f"manifest says {frame.width}x{frame.height}"
        )
    return array


def to_requests(frames: Iterable[FrozenFrame]) -> list[DepthRequest]:
    """Turn frozen frames into adapter inputs, carrying the pinned hash along."""
    return [
        DepthRequest(
            image_id=f.frame_id,
            image=load_image(f),
            intrinsics=f.camera_intrinsics(),
            source_path=f.image_path,
            image_sha256=f.sha256,
        )
        for f in frames
    ]


__all__ = [
    "FrozenFrame", "DEFAULT_SET", "MANIFEST_DIR", "REPO",
    "build_manifest", "manifest_path", "load_manifest", "load_frames",
    "verify", "load_image", "to_requests",
]
