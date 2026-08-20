#!/usr/bin/env python3
"""Scenario spec: what changes in the world, and exactly when.

A scenario is a YAML file with a list of events on a simulated-time line plus a
capture schedule.  Loading it here normalises everything onto the sensor tick so
a scenario cannot ask for an event or a frame at an instant the simulator has no
way to produce, and so two people reading the same YAML get the same timeline.

Event kinds
-----------
``spawn``   create an obstacle from the catalogue at a pose
``move``    drive an obstacle to a new pose over ``duration_s`` (0 = teleport)
``stop``    end any motion in progress and hold the current pose
``remove``  delete the obstacle from the world
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

EVENT_KINDS = ("spawn", "move", "stop", "remove")


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    t: float
    kind: str
    entity: str
    model: str | None = None
    pose: dict | None = None
    duration_s: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "t": self.t, "kind": self.kind, "entity": self.entity,
            "model": self.model, "pose": self.pose,
            "duration_s": self.duration_s, "note": self.note,
        }


@dataclass
class Scenario:
    scenario_id: str
    world_file: str
    world_name: str
    description: str
    seed: int
    step_size_s: float
    tick_s: float
    duration_s: float
    target_height_m: float
    grid_resolution_m: float
    events: list[Event]
    capture_times: list[float]
    source_path: Path
    raw: dict = field(default_factory=dict)

    @property
    def entities(self) -> list[str]:
        seen = []
        for event in self.events:
            if event.entity not in seen:
                seen.append(event.entity)
        return seen

    def events_at(self, t: float) -> list[Event]:
        return [e for e in self.events if abs(e.t - t) < 1.0e-9]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "world_file": self.world_file,
            "world_name": self.world_name,
            "description": self.description,
            "seed": self.seed,
            "step_size_s": self.step_size_s,
            "tick_s": self.tick_s,
            "duration_s": self.duration_s,
            "target_height_m": self.target_height_m,
            "grid_resolution_m": self.grid_resolution_m,
            "events": [e.to_dict() for e in self.events],
            "capture_times_s": self.capture_times,
        }


def _snap(value: float, tick: float) -> float:
    """Round onto the sensor tick, so a requested instant is a renderable one."""
    return round(round(float(value) / tick) * tick, 9)


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    required = ("scenario_id", "world_file", "world_name", "events")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ScenarioError(f"{path}: missing required key(s) {missing}")

    tick = float(raw.get("tick_s", 0.2))
    step_size = float(raw.get("step_size_s", 0.001))
    steps_per_tick = tick / step_size
    if abs(steps_per_tick - round(steps_per_tick)) > 1.0e-9:
        raise ScenarioError(
            f"{path}: tick_s={tick} is not a whole number of {step_size}s physics steps"
        )
    duration = float(raw.get("duration_s", 0.0))

    events: list[Event] = []
    for index, item in enumerate(raw["events"]):
        kind = str(item.get("kind", "")).strip()
        if kind not in EVENT_KINDS:
            raise ScenarioError(f"{path}: event {index} has unknown kind {kind!r}; expected {EVENT_KINDS}")
        if "entity" not in item:
            raise ScenarioError(f"{path}: event {index} ({kind}) has no entity")
        t = _snap(item.get("t", 0.0), tick)
        if kind == "spawn" and "model" not in item:
            raise ScenarioError(f"{path}: spawn event {index} has no model")
        if kind in ("spawn", "move") and "pose" not in item:
            raise ScenarioError(f"{path}: {kind} event {index} has no pose")
        pose = dict(item["pose"]) if "pose" in item else None
        if pose is not None:
            pose.setdefault("z", 0.0)
            pose.setdefault("yaw", 0.0)
            if isinstance(pose["yaw"], str):
                pose["yaw"] = {"deg90": math.pi / 2, "deg-90": -math.pi / 2}[pose["yaw"]]
        duration_s = _snap(item.get("duration_s", 0.0), tick) if kind == "move" else 0.0
        events.append(Event(
            t=t, kind=kind, entity=str(item["entity"]),
            model=item.get("model"), pose=pose, duration_s=duration_s,
            note=str(item.get("note", "")),
        ))
    events.sort(key=lambda e: (e.t, EVENT_KINDS.index(e.kind)))

    _validate_event_order(path, events)

    capture_times = _capture_schedule(raw.get("captures", {}), events, tick, duration)
    if not capture_times:
        raise ScenarioError(f"{path}: capture schedule is empty")

    return Scenario(
        scenario_id=str(raw["scenario_id"]),
        world_file=str(raw["world_file"]),
        world_name=str(raw["world_name"]),
        description=str(raw.get("description", "")),
        seed=int(raw.get("seed", 0)),
        step_size_s=step_size,
        tick_s=tick,
        duration_s=max(duration, capture_times[-1]),
        target_height_m=float(raw.get("target_height_m", 0.35)),
        grid_resolution_m=float(raw.get("grid_resolution_m", 0.25)),
        events=events,
        capture_times=capture_times,
        source_path=path,
        raw=raw,
    )


def _validate_event_order(path: Path, events: list[Event]) -> None:
    """Reject timelines a simulator could not act out (move before spawn, etc.)."""
    alive: set[str] = set()
    for event in events:
        if event.kind == "spawn":
            if event.entity in alive:
                raise ScenarioError(f"{path}: {event.entity} is spawned twice without a remove")
            alive.add(event.entity)
        else:
            if event.entity not in alive:
                raise ScenarioError(
                    f"{path}: {event.kind} at t={event.t}s targets {event.entity}, "
                    f"which is not spawned at that point"
                )
            if event.kind == "remove":
                alive.discard(event.entity)


def _capture_schedule(spec: dict, events: list[Event], tick: float, duration: float) -> list[float]:
    """Regular captures, plus one either side of every event.

    The brackets are what make "the oracle changed at the event timestamp"
    checkable: without a frame immediately before and immediately after, a change
    can only be located to within a capture period.
    """
    times: set[float] = set()
    period = float(spec.get("period_s", tick))
    start = _snap(spec.get("start_t", 0.0), tick)
    end = _snap(spec.get("end_t", duration), tick)
    steps = int(round((end - start) / period))
    for index in range(steps + 1):
        times.add(_snap(start + index * period, tick))

    for event in events:
        for t in (event.t - tick, event.t + tick):
            if t >= 0:
                times.add(_snap(t, tick))
        if event.kind == "move" and event.duration_s > 0:
            end_t = event.t + event.duration_s
            times.add(_snap(end_t, tick))
            times.add(_snap(end_t + tick, tick))

    for t in spec.get("extra_times_s", []):
        times.add(_snap(t, tick))

    return sorted(t for t in times if t >= 0)
