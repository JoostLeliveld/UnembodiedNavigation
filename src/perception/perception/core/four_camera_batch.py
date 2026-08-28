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


# warehouse_v2 has FIVE wall cameras and the fusion study needs them in one
# inference cycle, so the batch is A--E as of 2026-08-21. The module keeps its
# historical file name; the batch size is len(CAMERA_ORDER) everywhere and is not
# written as a literal anywhere. Measured on this GPU before the change was made:
# one batch of five 1280x720 frames at imgsz 960 takes 99 ms, against the 200 ms
# the 5 Hz cameras allow, so five cameras still fit in the cadence.
CAMERA_ORDER = ("camera_A", "camera_B", "camera_C", "camera_D", "camera_E")
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


def validate_batch_results(
    results: Any, expected_size: int = len(CAMERA_ORDER)
) -> tuple[Any, ...]:
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
        expected = len(CAMERA_ORDER)
        if len(order) != expected or len(set(order)) != expected:
            raise BatchContractError(
                f"camera_order must contain exactly {expected} unique cameras: "
                f"{', '.join(CAMERA_ORDER)}"
            )
        if not math.isfinite(max_stamp_skew_s) or float(max_stamp_skew_s) < 0.0:
            raise BatchContractError("max_stamp_skew_s must be finite and non-negative")
        if not math.isfinite(max_pending_wall_s) or float(max_pending_wall_s) <= 0.0:
            raise BatchContractError("max_pending_wall_s must be finite and positive")
        self.camera_order = order
        self.max_stamp_skew_ns = int(round(float(max_stamp_skew_s) * 1.0e9))
        self.max_pending_wall_s = float(max_pending_wall_s)
        self._pending: dict[str, PendingFrame] = {}
        #: stamp bucket -> the frames of that round, at most one per camera
        self._buckets: dict[int, dict[str, PendingFrame]] = {}
        self._last_seen_stamp_ns = {camera_id: -1 for camera_id in order}
        self._last_batched_stamp_ns = {camera_id: -1 for camera_id in order}
        self._lock = threading.Lock()

    @property
    def pending_camera_ids(self) -> tuple[str, ...]:
        with self._lock:
            waiting = {c for bucket in self._buckets.values() for c in bucket}
            return tuple(camera_id for camera_id in self.camera_order if camera_id in waiting)

    @property
    def bucket_report(self) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Which cameras each open round is still waiting on.

        A round that never completes is otherwise silent: the batcher returns
        "accepted_waiting" and the node publishes nothing, with no warning to
        say which camera never arrived.
        """
        with self._lock:
            return tuple(
                (key, tuple(c for c in self.camera_order if c in bucket))
                for key, bucket in sorted(self._buckets.items())
            )

    @property
    def last_batched_stamp_ns(self) -> dict[str, int]:
        with self._lock:
            return dict(self._last_batched_stamp_ns)

    def _bucket_key_locked(self, stamp_ns: int) -> int:
        """The round this stamp belongs to: an existing one within tolerance, or its own."""
        for key in self._buckets:
            if abs(stamp_ns - key) <= self.max_stamp_skew_ns:
                return key
        return int(stamp_ns)

    def _expire_locked(self, now_wall_s: float) -> tuple[str, ...]:
        dropped: list[str] = []
        for key in list(self._buckets):
            bucket = self._buckets[key]
            for camera_id in [
                cam for cam, frame in bucket.items()
                if now_wall_s - frame.receive_wall_s > self.max_pending_wall_s
            ]:
                bucket.pop(camera_id, None)
                dropped.append(camera_id)
            if not bucket:
                self._buckets.pop(key, None)
        # report in camera order, once each, however many rounds they came from
        return tuple(c for c in self.camera_order if c in set(dropped))

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

            # Group by STAMP, not by camera.
            #
            # This used to be one slot per camera holding that camera's newest frame,
            # which is correct only while the consumer keeps up with the cameras. It
            # does not here: one inference cycle over five 1280x720 images outlasts
            # the 200 ms camera period, so a newer frame for one camera overwrote its
            # slot mid-cycle and the set that finally completed mixed one camera's
            # round N+1 with another's round N. Their stamps then differed by a whole
            # period and the batch was rejected for skew -- in a 55 s live run, every
            # single batch, so the node published nothing at all.
            #
            # The cameras were never the problem: measured on the same rig, all five
            # deliver 5.05 Hz on the simulation clock and 101 of 102 rounds carry
            # byte-identical stamps. Keying the pending set by stamp makes a round
            # complete or not on its own merits, and it cannot be broken by the
            # arrival of the next one. `max_stamp_skew_ns` becomes the grouping
            # tolerance rather than a rejection test.
            self._last_seen_stamp_ns[frame.camera_id] = frame.stamp_ns
            key = self._bucket_key_locked(frame.stamp_ns)
            replaced = frame.camera_id in self._buckets.setdefault(key, {})
            self._buckets[key][frame.camera_id] = frame
            if len(self._buckets[key]) != len(self.camera_order):
                return BatchDecision(
                    "accepted_replaced" if replaced else "accepted_waiting",
                    dropped_camera_ids=expired,
                )
            self._pending = self._buckets[key]

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
            # a completed round makes every earlier round unreachable: discard them
            # instead of leaving them to expire and be reported as drops later
            for stale in [k for k in self._buckets if k <= key]:
                self._buckets.pop(stale, None)
            for item in batch:
                if item.stamp_ns <= self._last_batched_stamp_ns[item.camera_id]:
                    # This should be unreachable because last-seen stamps are
                    # strict, but keep the final no-reuse invariant fail-closed.
                    raise BatchContractError(
                        f"attempted to reuse {item.camera_id} stamp {item.stamp_ns}"
                    )
                self._last_batched_stamp_ns[item.camera_id] = item.stamp_ns
            return BatchDecision("batch_ready", batch=batch, dropped_camera_ids=expired)
