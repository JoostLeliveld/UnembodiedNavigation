"""ROS-independent synchronization for the fixed four-camera detector.

The runtime deliberately does not approximate-time synchronize by reusing the
last image from a slow camera.  A batch is emitted only when every camera has
contributed one strictly newer image.  Pending images are latest-only and are
discarded when their wall age or inter-camera stamp skew exceeds the configured
bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Sequence


CAMERA_ORDER = ("camera_A", "camera_B", "camera_C", "camera_D")
# A small ROS/Gazebo scheduling tolerance is allowed, but a materially future
# image stamp is an integrity fault. Clamping it to zero would otherwise turn a
# reversed or mismatched simulation clock into a plausible fresh observation.
MAX_FUTURE_IMAGE_STAMP_S = 0.005


class BatchContractError(ValueError):
    """Raised when a frame or inference result violates the batch contract."""


@dataclass(frozen=True)
class PendingFrame:
    """One camera frame and the operational times recorded at receipt."""

    camera_id: str
    stamp_ns: int
    receive_stamp_s: float
    receive_wall_s: float
    payload: Any


@dataclass(frozen=True)
class BatchDecision:
    """Outcome of offering one frame to :class:`FourCameraBatcher`."""

    status: str
    batch: tuple[PendingFrame, ...] | None = None
    dropped_camera_ids: tuple[str, ...] = ()


def stamp_parts_to_ns(sec: Any, nanosec: Any) -> int:
    """Validate ROS-style stamp parts and return an exact integer nanosecond key."""

    if isinstance(sec, bool) or isinstance(nanosec, bool):
        raise BatchContractError("stamp fields must be integers, not booleans")
    try:
        sec_i = int(sec)
        nanosec_i = int(nanosec)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchContractError("stamp fields must be integers") from exc
    if sec_i != sec or nanosec_i != nanosec:
        raise BatchContractError("stamp fields must be exact integers")
    if sec_i < 0 or not 0 <= nanosec_i < 1_000_000_000:
        raise BatchContractError("stamp is outside the ROS time domain")
    return sec_i * 1_000_000_000 + nanosec_i


def frame_age_at_publish_s(
    *,
    publish_stamp_s: float,
    image_stamp_s: float,
    future_tolerance_s: float = MAX_FUTURE_IMAGE_STAMP_S,
) -> float:
    """Return a non-negative frame age or reject a materially future image.

    A tiny negative value can arise because the simulator advances the image
    and ROS clocks on neighbouring callbacks. A larger negative age is not a
    freshness success: it signals a clock/order contract failure and must tear
    down the batch detector before it emits a deceptive availability row.
    """

    try:
        publish = float(publish_stamp_s)
        image = float(image_stamp_s)
        tolerance = float(future_tolerance_s)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchContractError(
            "publish/image stamps and future tolerance must be numeric"
        ) from exc
    if not all(math.isfinite(value) for value in (publish, image, tolerance)):
        raise BatchContractError("publish/image stamps and future tolerance must be finite")
    if tolerance < 0.0:
        raise BatchContractError("future image-stamp tolerance must be non-negative")
    age = publish - image
    if age < -tolerance:
        raise BatchContractError(
            "image stamp is materially in the future at publish time: "
            f"publish={publish:.9f}, image={image:.9f}, tolerance={tolerance:.9f}"
        )
    return max(age, 0.0)


def validate_batch_results(results: Any, expected_size: int = 4) -> tuple[Any, ...]:
    """Return a trustworthy ordered result tuple or fail the complete batch.

    Ultralytics returns one result per source image.  A missing entry would
    silently shift camera identity if it were accepted, so length, null entries,
    and the minimum result interface are checked before any result is published.
    Empty ``boxes`` collections are valid non-detections; a missing ``boxes``
    attribute is malformed.
    """

    if isinstance(results, (str, bytes)) or results is None:
        raise BatchContractError("batch inference returned no result sequence")
    try:
        ordered = tuple(results)
    except TypeError as exc:
        raise BatchContractError("batch inference result is not iterable") from exc
    if len(ordered) != int(expected_size):
        raise BatchContractError(
            f"batch inference returned {len(ordered)} results; expected {expected_size}"
        )
    for index, result in enumerate(ordered):
        if result is None or not hasattr(result, "boxes") or result.boxes is None:
            raise BatchContractError(f"batch result {index} is malformed")
    return ordered


class FourCameraBatcher:
    """Thread-safe, latest-only synchronizer with no frame reuse."""

    def __init__(
        self,
        *,
        camera_order: Sequence[str] = CAMERA_ORDER,
        max_stamp_skew_s: float = 0.10,
        max_pending_wall_s: float = 0.50,
    ) -> None:
        order = tuple(str(camera_id) for camera_id in camera_order)
        if len(order) != 4 or len(set(order)) != 4:
            raise BatchContractError("camera_order must contain exactly four unique cameras")
        if not math.isfinite(max_stamp_skew_s) or float(max_stamp_skew_s) < 0.0:
            raise BatchContractError("max_stamp_skew_s must be finite and non-negative")
        if not math.isfinite(max_pending_wall_s) or float(max_pending_wall_s) <= 0.0:
            raise BatchContractError("max_pending_wall_s must be finite and positive")
        self.camera_order = order
        self.max_stamp_skew_ns = int(round(float(max_stamp_skew_s) * 1.0e9))
        self.max_pending_wall_s = float(max_pending_wall_s)
        self._pending: dict[str, PendingFrame] = {}
        self._last_seen_stamp_ns = {camera_id: -1 for camera_id in order}
        self._last_batched_stamp_ns = {camera_id: -1 for camera_id in order}
        self._lock = threading.Lock()

    @property
    def pending_camera_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(camera_id for camera_id in self.camera_order if camera_id in self._pending)

    @property
    def last_batched_stamp_ns(self) -> dict[str, int]:
        with self._lock:
            return dict(self._last_batched_stamp_ns)

    def _expire_locked(self, now_wall_s: float) -> tuple[str, ...]:
        dropped = tuple(
            camera_id
            for camera_id in self.camera_order
            if camera_id in self._pending
            and now_wall_s - self._pending[camera_id].receive_wall_s > self.max_pending_wall_s
        )
        for camera_id in dropped:
            self._pending.pop(camera_id, None)
        return dropped

    def expire(self, now_wall_s: float) -> tuple[str, ...]:
        """Discard pending frames that can no longer form a fresh batch."""

        now_wall_s = float(now_wall_s)
        if not math.isfinite(now_wall_s):
            raise BatchContractError("now_wall_s must be finite")
        with self._lock:
            return self._expire_locked(now_wall_s)

    def offer(self, frame: PendingFrame) -> BatchDecision:
        """Offer a frame and possibly return a deterministic A--D batch."""

        if frame.camera_id not in self._last_seen_stamp_ns:
            raise BatchContractError(f"unknown camera_id: {frame.camera_id!r}")
        if isinstance(frame.stamp_ns, bool) or not isinstance(frame.stamp_ns, int):
            raise BatchContractError("stamp_ns must be an integer")
        if frame.stamp_ns < 0:
            raise BatchContractError("stamp_ns must be non-negative")
        if not math.isfinite(float(frame.receive_stamp_s)):
            raise BatchContractError("receive_stamp_s must be finite")
        if not math.isfinite(float(frame.receive_wall_s)):
            raise BatchContractError("receive_wall_s must be finite")

        with self._lock:
            expired = self._expire_locked(float(frame.receive_wall_s))
            last_seen = self._last_seen_stamp_ns[frame.camera_id]
            if frame.stamp_ns == last_seen:
                return BatchDecision("duplicate", dropped_camera_ids=expired)
            if frame.stamp_ns < last_seen:
                return BatchDecision("out_of_order", dropped_camera_ids=expired)

            replaced = frame.camera_id in self._pending
            self._last_seen_stamp_ns[frame.camera_id] = frame.stamp_ns
            self._pending[frame.camera_id] = frame
            if len(self._pending) != len(self.camera_order):
                return BatchDecision(
                    "accepted_replaced" if replaced else "accepted_waiting",
                    dropped_camera_ids=expired,
                )

            stamps = {camera_id: self._pending[camera_id].stamp_ns for camera_id in self.camera_order}
            newest_stamp = max(stamps.values())
            oldest_stamp = min(stamps.values())
            if newest_stamp - oldest_stamp > self.max_stamp_skew_ns:
                # Drop every frame outside the allowable window relative to the
                # newest image.  Retaining the newer pending frames avoids
                # needlessly lowering output rate while still forbidding stale
                # reuse.
                cutoff = newest_stamp - self.max_stamp_skew_ns
                skew_dropped = tuple(
                    camera_id
                    for camera_id in self.camera_order
                    if stamps[camera_id] < cutoff
                )
                if not skew_dropped:  # defensive against integer roundoff
                    skew_dropped = tuple(
                        camera_id
                        for camera_id in self.camera_order
                        if stamps[camera_id] == oldest_stamp
                    )
                for camera_id in skew_dropped:
                    self._pending.pop(camera_id, None)
                return BatchDecision(
                    "stamp_skew",
                    dropped_camera_ids=tuple(dict.fromkeys((*expired, *skew_dropped))),
                )

            batch = tuple(self._pending.pop(camera_id) for camera_id in self.camera_order)
            for item in batch:
                if item.stamp_ns <= self._last_batched_stamp_ns[item.camera_id]:
                    # This should be unreachable because last-seen stamps are
                    # strict, but keep the final no-reuse invariant fail-closed.
                    raise BatchContractError(
                        f"attempted to reuse {item.camera_id} stamp {item.stamp_ns}"
                    )
                self._last_batched_stamp_ns[item.camera_id] = item.stamp_ns
            return BatchDecision("batch_ready", batch=batch, dropped_camera_ids=expired)
