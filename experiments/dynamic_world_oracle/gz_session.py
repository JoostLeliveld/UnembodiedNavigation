#!/usr/bin/env python3
"""A Gazebo server you drive one step at a time.

Reproducibility is the whole point of this module.  A normal ``gz sim -r`` run
advances on wall-clock, so two runs of the same scenario see the obstacle spawn
at different simulation times and the cameras render at different instants — the
scenario is then only approximately the same scenario.  Here the server is
started *paused* and never runs freely: every advance is an explicit
``multi_step`` of a fixed number of 1 ms physics steps, and every event is
applied at an exact step boundary.  Two runs of one scenario therefore see the
same geometry at the same simulated instant, every time.

Talks to Gazebo Harmonic over gz-transport directly rather than through the ROS
bridge: the bridge on this machine is pinned to a different Gazebo ABI, and one
less asynchronous hop is one less source of run-to-run variation.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gz.transport13 as gz_transport
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.world_stats_pb2 import WorldStatistics

# gz.msgs pixel format enum values we care about
PIXEL_RGB_INT8 = 3
PIXEL_R_FLOAT32 = 13

# Physics steps run after a spawn/remove so gz's UserCommands system drains its
# queue. Fixed on purpose — see SteppedGazebo.spawn.
COMMAND_SETTLE_STEPS = 1


class GazeboError(RuntimeError):
    pass


@dataclass
class Frame:
    topic: str
    sim_time_s: float
    array: np.ndarray


def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0)))


class SteppedGazebo:
    """Own a headless ``gz sim`` server and advance it deterministically."""

    def __init__(
        self,
        world_sdf: Path,
        world_name: str,
        *,
        resource_paths: list[Path],
        partition: str,
        step_size_s: float = 0.001,
        log_path: Path | None = None,
        startup_timeout_s: float = 120.0,
        gui: bool = False,
    ):
        self.world_sdf = Path(world_sdf)
        self.world_name = world_name
        self.resource_paths = [str(p) for p in resource_paths]
        self.partition = partition
        self.step_size_s = float(step_size_s)
        self.log_path = Path(log_path) if log_path else None
        self.startup_timeout_s = float(startup_timeout_s)
        self.gui = bool(gui)

        self._gui_proc: subprocess.Popen | None = None
        self._proc: subprocess.Popen | None = None
        self._node: gz_transport.Node | None = None
        self._lock = threading.Lock()
        self._latest: dict[str, Frame] = {}
        self._stats: dict = {"sim_time_s": None, "iterations": None, "paused": None}
        self._poses: dict = {}
        self._poses_sim_time_s: float | None = None
        self._subscribed: list[str] = []

    # ---------------------------------------------------------------- lifecycle
    def __enter__(self) -> "SteppedGazebo":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        if shutil.which("gz") is None:
            raise GazeboError("`gz` is not on PATH; Gazebo Harmonic is required")
        self._refuse_if_partition_is_occupied()
        env = dict(os.environ)
        env["GZ_SIM_RESOURCE_PATH"] = ":".join(self.resource_paths)
        env["GZ_PARTITION"] = self.partition
        # No `-r`: the server must come up paused so that simulated time zero is
        # ours, not whatever elapsed between spawn and our first service call.
        cmd = ["gz", "sim", "-s", "-v", "1", "--headless-rendering", str(self.world_sdf)]
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        os.environ["GZ_PARTITION"] = self.partition
        self._node = gz_transport.Node()
        self._node.subscribe(WorldStatistics, "/stats", self._on_stats)
        self._node.subscribe(Pose_V, f"/world/{self.world_name}/dynamic_pose/info", self._on_pose_info)
        self._await_server()

        if self.gui:
            # A viewer, not a participant: the GUI is a separate process on the
            # same gz partition, so it shows the scenario being stepped without
            # touching the server's clock. It does compete for the GPU, which can
            # make the sensor renders this loop waits on slower — watch a run with
            # it, take the dataset from a run without it.
            gui_log = open(self.log_path.with_suffix(".gui.log"), "w") if self.log_path \
                else subprocess.DEVNULL
            gui_env = dict(env)
            # PRIME render offload, same as sim/launch/gazebo.launch.py: on this
            # hybrid Intel+NVIDIA machine the window otherwise binds its GL
            # context to the integrated GPU, which is where the EGL dri2 failures
            # come from.
            gui_env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
            gui_env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
            self._gui_proc = subprocess.Popen(
                ["gz", "sim", "-g", "-v", "1"], env=gui_env,
                stdout=gui_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            time.sleep(3.0)  # let the GUI attach before the first step burst

    def _refuse_if_partition_is_occupied(self) -> None:
        """Never adopt somebody else's Gazebo.

        A stale server left behind by an interrupted run keeps advertising on the
        partition it was given. If a new run reuses that partition name, it
        connects to the corpse: the clock is wherever the old run stopped, the
        scenario timeline is nonsense, and nothing about it looks like an error.
        That happened once here — a server orphaned at t=5.2 s was adopted by the
        next run of the same scenario. Partitions now carry the process id, and
        this refuses to start if one is somehow still taken.

        Note for anyone hunting strays by hand: ``gz`` is a Ruby script, so the
        process is named ``ruby``, not ``gz``. ``pgrep -x gz`` finds nothing and
        quietly leaves the server running. Use ``pgrep -af "gz sim"``.
        """
        os.environ["GZ_PARTITION"] = self.partition
        probe = gz_transport.Node()
        services = probe.service_list()
        control = f"/world/{self.world_name}/control"
        if control in services:
            raise GazeboError(
                f"another Gazebo already advertises {control} on partition "
                f"{self.partition!r}; refusing to adopt it. Find it with "
                f"`pgrep -af \"gz sim\"` (the process is named `ruby`, not `gz`)."
            )

    def _await_server(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        control = f"/world/{self.world_name}/control"
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise GazeboError(
                    f"gz sim exited with code {self._proc.returncode} during startup; "
                    f"see {self.log_path}"
                )
            services = self._node.service_list() if self._node else []
            if control in services and self._stats["sim_time_s"] is not None:
                return
            time.sleep(0.5)
        raise GazeboError(
            f"gz sim did not advertise {control} within {self.startup_timeout_s:.0f}s"
        )

    def stop(self) -> None:
        for attr in ("_gui_proc", "_proc"):
            proc = getattr(self, attr, None)
            if proc is None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=10)
                except Exception:
                    pass
            finally:
                setattr(self, attr, None)

    # -------------------------------------------------------------- subscribing
    def subscribe_images(self, topics: list[str]) -> None:
        assert self._node is not None
        for topic in topics:
            if topic in self._subscribed:
                continue
            if not self._node.subscribe(Image, topic, self._make_image_cb(topic)):
                raise GazeboError(f"could not subscribe to {topic}")
            self._subscribed.append(topic)

    def _make_image_cb(self, topic: str):
        def _cb(msg: Image) -> None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            if msg.pixel_format_type == PIXEL_RGB_INT8:
                array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.pixel_format_type == PIXEL_R_FLOAT32:
                array = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            else:
                return
            with self._lock:
                self._latest[topic] = Frame(topic=topic, sim_time_s=stamp, array=array.copy())
        return _cb

    def _on_stats(self, msg: WorldStatistics) -> None:
        with self._lock:
            self._stats = {
                "sim_time_s": msg.sim_time.sec + msg.sim_time.nsec * 1e-9,
                "iterations": int(msg.iterations),
                "paused": bool(msg.paused),
            }

    @property
    def sim_time_s(self) -> float:
        with self._lock:
            value = self._stats["sim_time_s"]
        if value is None:
            raise GazeboError("no world statistics received yet")
        return float(value)

    @property
    def iterations(self) -> int:
        with self._lock:
            value = self._stats["iterations"]
        if value is None:
            raise GazeboError("no world statistics received yet")
        return int(value)

    # ------------------------------------------------------------------ control
    def _request(self, service: str, request, request_type, timeout_ms: int = 8000):
        """Fire a world service.

        The Python binding's reply flag is unreliable here — a command that plainly
        took effect frequently reports ``False`` — so the reply is not trusted.
        Every caller verifies the *effect* instead: the clock advanced, the entity
        appeared, the pose moved.
        """
        assert self._node is not None
        try:
            self._node.request(service, request, request_type, Boolean, timeout_ms)
        except Exception as exc:  # noqa: BLE001
            raise GazeboError(f"request to {service} failed: {exc}") from exc

    def _multi_step(self, n_steps: int) -> None:
        request = WorldControl()
        request.pause = True
        request.multi_step = int(n_steps)
        self._request(f"/world/{self.world_name}/control", request, WorldControl)

    def step(self, n_steps: int, *, timeout_s: float = 180.0, stall_s: float = 25.0,
             max_retries: int = 3) -> float:
        """Advance exactly ``n_steps`` physics steps and wait for the clock to agree.

        A step request is occasionally dropped rather than refused, and the world
        then simply sits still.  Rather than hang, this notices that no iteration
        has happened for ``stall_s`` and re-issues a request for whatever is
        *still* outstanding, so a retry can never over-step.  Landing anywhere
        other than exactly on the target iteration is a hard error: a run that
        quietly gained a millisecond is no longer the scenario it claims to be.
        """
        target = self.iterations + int(n_steps)
        self._multi_step(n_steps)
        deadline = time.monotonic() + timeout_s
        last_iterations, last_progress, retries = self.iterations, time.monotonic(), 0
        while time.monotonic() < deadline:
            current = self.iterations
            if current >= target:
                if current != target:
                    raise GazeboError(
                        f"world overshot: asked for iteration {target}, landed on {current}"
                    )
                return self.sim_time_s
            if current != last_iterations:
                last_iterations, last_progress = current, time.monotonic()
            elif time.monotonic() - last_progress > stall_s:
                if retries >= max_retries:
                    break
                retries += 1
                self._multi_step(target - current)
                last_progress = time.monotonic()
            time.sleep(0.02)
        raise GazeboError(
            f"world did not reach iteration {target} within {timeout_s:.0f}s "
            f"(stuck at {self.iterations} after {retries} retries)"
        )

    def step_to(self, sim_time_s: float, **kwargs) -> float:
        remaining = sim_time_s - self.sim_time_s
        n_steps = int(round(remaining / self.step_size_s))
        if n_steps < 0:
            raise GazeboError(f"cannot step backwards to t={sim_time_s:.3f}s")
        return self.step(n_steps, **kwargs) if n_steps else self.sim_time_s

    # ------------------------------------------------------------------- events
    def spawn(self, model_sdf: Path, entity_name: str, pose: dict, *, timeout_s: float = 30.0) -> None:
        request = EntityFactory()
        request.sdf_filename = str(model_sdf)
        request.name = entity_name
        request.allow_renaming = False
        request.pose.position.x = float(pose["x"])
        request.pose.position.y = float(pose["y"])
        request.pose.position.z = float(pose.get("z", 0.0))
        qx, qy, qz, qw = _quat_from_yaw(float(pose.get("yaw", 0.0)))
        request.pose.orientation.x, request.pose.orientation.y = qx, qy
        request.pose.orientation.z, request.pose.orientation.w = qz, qw
        self._request(f"/world/{self.world_name}/create/blocking", request, EntityFactory)
        # UserCommands only drains its queue inside a simulation update, so the
        # entity does not exist until at least one step has run. That step count
        # is a fixed constant, never "keep stepping until it shows up": an
        # adaptive wait would make the simulated time of a spawn depend on how
        # busy the machine was, which is exactly the non-determinism this class
        # exists to remove.
        self.step(COMMAND_SETTLE_STEPS)
        self._assert_entity(entity_name, present=True)

    def set_pose(self, entity_name: str, pose: dict) -> None:
        request = Pose()
        request.name = entity_name
        request.position.x = float(pose["x"])
        request.position.y = float(pose["y"])
        request.position.z = float(pose.get("z", 0.0))
        qx, qy, qz, qw = _quat_from_yaw(float(pose.get("yaw", 0.0)))
        request.orientation.x, request.orientation.y = qx, qy
        request.orientation.z, request.orientation.w = qz, qw
        self._request(f"/world/{self.world_name}/set_pose", request, Pose)

    def remove(self, entity_name: str, *, timeout_s: float = 30.0) -> None:
        request = Entity()
        request.name = entity_name
        request.type = Entity.MODEL
        self._request(f"/world/{self.world_name}/remove/blocking", request, Entity)
        self.step(COMMAND_SETTLE_STEPS)
        self._assert_entity(entity_name, present=False)

    def _assert_entity(self, entity_name: str, *, present: bool) -> None:
        poses = self.dynamic_poses(required=present)
        if (entity_name in poses) is present:
            return
        state = "present" if present else "absent"
        raise GazeboError(
            f"entity {entity_name!r} is not {state} after {COMMAND_SETTLE_STEPS} settle step(s); "
            f"the world command did not take effect"
        )

    def _on_pose_info(self, msg: Pose_V) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        poses = {}
        for pose in msg.pose:
            q = pose.orientation
            poses[pose.name] = {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
                "yaw": float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))),
            }
        with self._lock:
            self._poses = poses
            self._poses_sim_time_s = stamp

    def dynamic_poses(self, *, required: bool = True, timeout_s: float = 30.0,
                      grace_s: float = 3.0) -> dict:
        """Poses of every non-static model, stamped at the current simulated time.

        Read from ``dynamic_pose/info`` — the simulator's own per-iteration state —
        so a recorded obstacle pose is what the physics engine actually held when
        the cameras rendered, not an echo of the command we sent.

        Deliberately *not* the full ``pose/info``: that topic carries all ~555
        entities of this warehouse on every one of the 200 iterations in a step
        burst, and decoding it in Python starved the interpreter badly enough that
        Gazebo skipped camera renders and dropped step requests outright. The
        scene-info service is not an option either — it returns an empty scene
        through these bindings.

        Gazebo publishes nothing on this topic while no dynamic model exists, so
        ``required=False`` treats silence as "no obstacles", which is the truth
        before the first spawn and after the last remove.
        """
        now = self.sim_time_s
        deadline = time.monotonic() + (timeout_s if required else grace_s)
        while time.monotonic() < deadline:
            with self._lock:
                stamp, poses = self._poses_sim_time_s, self._poses
            if stamp is not None and abs(stamp - now) <= 1.0e-6:
                return poses
            time.sleep(0.02)
        if not required:
            return {}
        raise GazeboError(
            f"no dynamic_pose/info snapshot stamped t={now:.3f}s within {timeout_s:.0f}s"
        )

    def entity_exists(self, entity_name: str) -> bool:
        return entity_name in self.dynamic_poses(required=False)

    def model_pose(self, entity_name: str) -> dict | None:
        """Read a model's pose back out of the simulator (never from our command)."""
        return self.dynamic_poses().get(entity_name)

    # ------------------------------------------------------------------ capture
    def wait_frames(self, topics: list[str], sim_time_s: float, *, required: bool = True,
                    timeout_s: float = 60.0, grace_s: float = 8.0,
                    tolerance_s: float = 1.0e-6) -> dict[str, Frame] | None:
        """Frames stamped ``sim_time_s``, or ``None`` when not ``required``.

        Call this after *every* step burst, not only the ones you keep.  Gazebo
        renders sensors on its own thread and silently skips an update that comes
        due while the previous render is still in flight, so firing the next burst
        the instant the clock arrives is what makes frames disappear.  Blocking
        until the current tick's frames are in hand keeps the renderer in step and
        makes the skip stop happening.
        """
        try:
            return self.capture(topics, sim_time_s,
                                timeout_s=timeout_s if required else grace_s,
                                tolerance_s=tolerance_s)
        except GazeboError:
            if required:
                raise
            return None

    def capture(self, topics: list[str], sim_time_s: float, *, timeout_s: float = 60.0,
                tolerance_s: float = 1.0e-6) -> dict[str, Frame]:
        """Collect one frame per topic, all stamped at exactly ``sim_time_s``.

        Waiting on the stamp rather than on wall-clock is what makes the four
        cameras synchronised: a returned set is four views of one simulated
        instant, not four views of whenever each sensor last fired.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                have = {
                    topic: frame for topic, frame in self._latest.items()
                    if topic in topics and abs(frame.sim_time_s - sim_time_s) <= tolerance_s
                }
            if len(have) == len(topics):
                return have
            time.sleep(0.02)
        with self._lock:
            stamps = {t: (self._latest[t].sim_time_s if t in self._latest else None) for t in topics}
        raise GazeboError(
            f"only got {sum(v is not None and abs(v - sim_time_s) <= tolerance_s for v in stamps.values())}"
            f"/{len(topics)} frames stamped t={sim_time_s:.3f}s within {timeout_s:.0f}s; "
            f"latest stamps: {stamps}"
        )
