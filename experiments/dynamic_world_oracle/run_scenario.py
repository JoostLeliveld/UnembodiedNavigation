#!/usr/bin/env python3
"""Run one dynamic-world scenario and record the evaluation dataset it produces.

Usage
-----
    python3 experiments/dynamic_world_oracle/run_scenario.py \
        --scenario experiments/dynamic_world_oracle/scenarios/s01_box_in_aisle.yaml

What comes out, per capture instant and per camera, is one record carrying the
agreed contract:

    scenario_id, timestamp, camera_id, rgb_path, oracle_depth_path,
    camera_intrinsics, camera_extrinsics, obstacle_state, oracle_visibility_grid

The loop is: apply whatever events are due at this tick, step exactly one sensor
period, then — if this instant is a capture instant — take the four RGB frames
and four depth frames that carry that exact simulated stamp, read the obstacle
poses back out of the simulator, and cast the visibility rays.  Nothing waits on
wall-clock, so the record set is a property of the scenario file rather than of
the machine that ran it.

Everything under ``oracle_depth/`` and ``oracle_visibility/``, and every pose in
``obstacle_state``, is EVALUATION-ONLY: it is simulator ground truth, and using
it as an input to perception, filtering or planning would be marking your own
homework.  The run directory says so too, in ``EVALUATION_ONLY.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "scripts" / "shared"))

import oracle as ora  # noqa: E402
from gz_session import GazeboError, SteppedGazebo  # noqa: E402
from paths import repo_root  # noqa: E402
from scenario import Event, Scenario, load_scenario  # noqa: E402

REPO = repo_root(_HERE)
WORLDS_DIR = REPO / "src" / "sim" / "gazebo_worlds" / "worlds"
MODELS_DIR = REPO / "src" / "sim" / "models"
STAGE_JSON = WORLDS_DIR / "warehouse_full_4cam_dynamic.stage.json"
DEFAULT_OUT_ROOT = REPO / "logs" / "studies" / "dynamic_world_oracle"

EVALUATION_ONLY_NOTE = """# Evaluation-only data

Everything in this directory that is named `oracle_*`, plus every pose in the
`obstacle_state` field of `records.jsonl`, is **simulator ground truth**:

- `oracle_depth/` — the depth buffer the renderer produced, not a sensed depth map
- `oracle_visibility/` — line-of-sight computed from CAD prisms and exact obstacle poses
- `obstacle_state[*].pose` — the pose the physics engine held, read back from `pose/info`

These exist to *score* a visibility or reliability model. No perception node, no
filter and no planner may read them, directly or through a derived artifact. The
only inputs a method may take from this run are `rgb_path`, `camera_intrinsics`
and `camera_extrinsics`.
"""


@dataclass
class Motion:
    """A move event in progress."""

    entity: str
    start_t: float
    end_t: float
    start_pose: dict
    end_pose: dict

    def pose_at(self, t: float) -> dict:
        if self.end_t <= self.start_t:
            return dict(self.end_pose)
        alpha = min(1.0, max(0.0, (t - self.start_t) / (self.end_t - self.start_t)))
        out = {}
        for key in ("x", "y", "z", "yaw"):
            a, b = float(self.start_pose.get(key, 0.0)), float(self.end_pose.get(key, 0.0))
            out[key] = a + alpha * (b - a)
        return out

    def finished(self, t: float) -> bool:
        return t >= self.end_t - 1.0e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gz_version() -> str:
    try:
        return subprocess.run(["gz", "sim", "--version"], capture_output=True, text=True,
                              timeout=20).stdout.strip().splitlines()[0] or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _time_tag(t: float) -> str:
    return f"t{int(round(t * 1000)):07d}ms"


class ScenarioRunner:
    def __init__(self, scenario: Scenario, out_dir: Path, *, partition: str,
                 keep_rgb: bool = True, gui: bool = False):
        self.scenario = scenario
        self.out_dir = out_dir
        self.partition = partition
        self.keep_rgb = keep_rgb
        self.gui = gui

        self.stage = json.loads(STAGE_JSON.read_text(encoding="utf-8"))
        if self.stage["world_file"] != scenario.world_file:
            raise ValueError(
                f"scenario targets {scenario.world_file} but the generated stage descriptor "
                f"describes {self.stage['world_file']}; regenerate with "
                f"scripts/geometry_visibility/make_warehouse_full.py --variant dynamic"
            )
        self.world_sdf = WORLDS_DIR / scenario.world_file
        self.cameras_meta = {c["camera_id"]: c for c in self.stage["cameras"]}
        self.catalogue = {c["model_name"]: c for c in self.stage["obstacle_catalogue"]}

        self.scene = ora.OracleScene.from_world(self.world_sdf, list(self.cameras_meta))
        bounds = self.stage["site_bounds"]
        self.grid = ora.FloorGrid(
            bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"],
            scenario.grid_resolution_m,
        )

        self.rgb_topics = [c["rgb_topic"] for c in self.stage["cameras"]]
        self.depth_topics = [c["oracle_depth_topic"] for c in self.stage["cameras"]]

        self.commanded: dict[str, dict] = {}     # entity -> last commanded pose
        self.models: dict[str, str] = {}         # entity -> catalogue model name
        self.motions: dict[str, Motion] = {}
        self.event_log: list[dict] = []
        self.records: list[dict] = []
        self.skipped_renders: list[float] = []
        self._parts_cache: dict[str, tuple] = {}

    # ------------------------------------------------------------------ layout
    def _prepare_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for camera_id in self.cameras_meta:
            if self.keep_rgb:
                (self.out_dir / "rgb" / camera_id).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "oracle_depth" / camera_id).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "oracle_visibility" / camera_id).mkdir(parents=True, exist_ok=True)
        (self.out_dir / "oracle_visibility" / "any_camera").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "EVALUATION_ONLY.md").write_text(EVALUATION_ONLY_NOTE, encoding="utf-8")

    # ------------------------------------------------------------------ events
    def _apply_event(self, gz: SteppedGazebo, event: Event) -> None:
        before = gz.sim_time_s
        if event.kind == "spawn":
            model_meta = self.catalogue.get(event.model)
            if model_meta is None:
                raise ValueError(f"unknown obstacle model {event.model!r}; "
                                 f"catalogue has {sorted(self.catalogue)}")
            gz.spawn(REPO / model_meta["model_sdf"], event.entity, event.pose)
            self.commanded[event.entity] = dict(event.pose)
            self.models[event.entity] = event.model
        elif event.kind == "move":
            start = self.commanded.get(event.entity)
            if start is None:
                raise ValueError(f"move targets {event.entity!r}, which is not spawned")
            if event.duration_s <= 0:
                gz.set_pose(event.entity, event.pose)
                self.commanded[event.entity] = dict(event.pose)
            else:
                self.motions[event.entity] = Motion(
                    entity=event.entity, start_t=event.t, end_t=event.t + event.duration_s,
                    start_pose=dict(start), end_pose=dict(event.pose),
                )
        elif event.kind == "stop":
            self.motions.pop(event.entity, None)
            held = self.commanded.get(event.entity)
            if held is None:
                raise ValueError(f"stop targets {event.entity!r}, which is not spawned")
            gz.set_pose(event.entity, held)
        elif event.kind == "remove":
            self.motions.pop(event.entity, None)
            gz.remove(event.entity)
            self.commanded.pop(event.entity, None)
            self.models.pop(event.entity, None)
        else:  # pragma: no cover - load_scenario rejects these
            raise ValueError(f"unhandled event kind {event.kind!r}")

        self.event_log.append({
            "scenario_id": self.scenario.scenario_id,
            "requested_sim_time_s": event.t,
            "applied_sim_time_s": before,
            "sim_time_after_command_s": gz.sim_time_s,
            "kind": event.kind,
            "entity": event.entity,
            "model": event.model,
            "pose": event.pose,
            "duration_s": event.duration_s,
            "note": event.note,
        })

    def _advance_motions(self, gz: SteppedGazebo, next_t: float) -> None:
        """Command each moving obstacle to where it should be at the next tick."""
        for entity, motion in list(self.motions.items()):
            pose = motion.pose_at(next_t)
            gz.set_pose(entity, pose)
            self.commanded[entity] = pose
            if motion.finished(next_t):
                self.motions.pop(entity, None)

    # ----------------------------------------------------------------- capture
    def _model_parts(self, model: str):
        """Collision parts of a catalogue model, read once from its generated SDF."""
        if model not in self._parts_cache:
            self._parts_cache[model] = ora.parts_from_model_sdf(
                REPO / self.catalogue[model]["model_sdf"], model)
        return self._parts_cache[model]

    def _obstacle_state(self, gz: SteppedGazebo) -> tuple[list[dict], list]:
        """Where each obstacle is, as the simulator holds it, plus its world prisms."""
        if not self.models:
            return [], []
        poses = gz.dynamic_poses()
        state, prisms = [], []
        for entity, model in self.models.items():
            actual = poses.get(entity)
            if actual is None:
                raise GazeboError(f"{entity!r} is tracked as spawned but absent from pose/info")
            box = ora.place_obstacle(entity, model, self._model_parts(model), actual)
            entry = box.to_dict()
            entry["commanded_pose"] = self.commanded.get(entity)
            entry["moving"] = entity in self.motions
            state.append(entry)
            prisms.extend(box.prisms)
        return state, prisms

    def _write_capture(self, gz: SteppedGazebo, t: float, frames: dict) -> None:
        obstacle_state, obstacle_prisms = self._obstacle_state(gz)
        grids = ora.visibility_grids(
            self.scene.cameras, self.grid, self.scene.static_prisms, obstacle_prisms,
            target_height_m=self.scenario.target_height_m,
        )
        tag = _time_tag(t)

        any_path = self.out_dir / "oracle_visibility" / "any_camera" / f"{tag}.npy"
        np.save(any_path, ora.any_camera_visible(grids))

        for camera_id, meta in self.cameras_meta.items():
            rgb_frame = frames[meta["rgb_topic"]]
            depth_frame = frames[meta["oracle_depth_topic"]]

            rgb_path = None
            if self.keep_rgb:
                rgb_path = self.out_dir / "rgb" / camera_id / f"{tag}.png"
                PILImage.fromarray(rgb_frame.array).save(rgb_path)
            depth_path = self.out_dir / "oracle_depth" / camera_id / f"{tag}.npy"
            np.save(depth_path, depth_frame.array)
            grid_path = self.out_dir / "oracle_visibility" / camera_id / f"{tag}.npy"
            np.save(grid_path, grids[camera_id])

            finite = np.isfinite(depth_frame.array)
            self.records.append({
                "scenario_id": self.scenario.scenario_id,
                "timestamp": round(float(t), 6),
                "camera_id": camera_id,
                "rgb_path": str(rgb_path.relative_to(self.out_dir)) if rgb_path else None,
                "oracle_depth_path": str(depth_path.relative_to(self.out_dir)),
                "camera_intrinsics": meta["intrinsics"],
                "camera_extrinsics": meta["extrinsics"],
                "obstacle_state": obstacle_state,
                "oracle_visibility_grid": {
                    "path": str(grid_path.relative_to(self.out_dir)),
                    "any_camera_path": str(any_path.relative_to(self.out_dir)),
                    "dtype": "uint8",
                    "codes": {str(k): v for k, v in ora.CELL_CODE_MEANING.items()},
                    "target_height_m": self.scenario.target_height_m,
                    **self.grid.to_dict(),
                    **ora.grid_summary(grids[camera_id]),
                },
                # provenance, not part of the contract
                "sim_iterations": gz.iterations,
                "rgb_stamp_s": rgb_frame.sim_time_s,
                "oracle_depth_stamp_s": depth_frame.sim_time_s,
                "oracle_depth_valid_fraction": float(finite.mean()),
            })

    # --------------------------------------------------------------------- run
    def run(self) -> Path:
        scenario = self.scenario
        self._prepare_dirs()
        log_path = self.out_dir / "gz_server.log"
        started_at = datetime.now(timezone.utc).isoformat()
        wall_start = time.monotonic()

        gz = SteppedGazebo(
            world_sdf=self.world_sdf,
            world_name=scenario.world_name,
            resource_paths=[MODELS_DIR, WORLDS_DIR.parent, MODELS_DIR.parent],
            partition=self.partition,
            step_size_s=scenario.step_size_s,
            log_path=log_path,
            gui=self.gui,
        )
        topics = self.rgb_topics + self.depth_topics
        with gz:
            print(f"server     : up at sim t={gz.sim_time_s:.3f}s, iteration {gz.iterations}, "
                  f"paused={gz._stats['paused']}")
            if gz.sim_time_s > 1e-9:
                raise GazeboError(
                    f"the server is already at t={gz.sim_time_s:.3f}s before the first step; "
                    f"a scenario timeline is only meaningful from a world that starts at zero"
                )
            gz.subscribe_images(topics)
            captures = set(scenario.capture_times)
            n_ticks = int(round(scenario.duration_s / scenario.tick_s))
            for index in range(n_ticks):
                t_now = round(index * scenario.tick_s, 9)
                t_next = round((index + 1) * scenario.tick_s, 9)
                for event in scenario.events_at(t_now):
                    self._apply_event(gz, event)
                self._advance_motions(gz, t_next)
                gz.step_to(t_next)
                # Collect this tick's frames whether or not we keep them: leaving
                # the renderer a beat behind is what makes it drop the *next*
                # render, and a dropped render at a capture instant is fatal.
                is_capture = any(abs(t_next - c) < 1.0e-9 for c in captures)
                frames = gz.wait_frames(topics, t_next, required=is_capture)
                if is_capture and frames is not None:
                    self._write_capture(gz, t_next, frames)
                elif not is_capture and frames is None:
                    self.skipped_renders.append(t_next)
            wall_s = time.monotonic() - wall_start

        self._write_outputs(started_at, wall_s)
        return self.out_dir

    def _write_outputs(self, started_at: str, wall_s: float) -> None:
        records_path = self.out_dir / "records.jsonl"
        with open(records_path, "w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        events_path = self.out_dir / "events.csv"
        columns = ["scenario_id", "requested_sim_time_s", "applied_sim_time_s",
                   "sim_time_after_command_s", "kind", "entity", "model", "duration_s", "note"]
        lines = [",".join(columns)]
        for entry in self.event_log:
            lines.append(",".join(
                f'"{entry[c]}"' if isinstance(entry.get(c), str) and "," in str(entry[c])
                else str(entry.get(c, ""))
                for c in columns
            ))
        events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        manifest = {
            "scenario": self.scenario.to_dict(),
            "scenario_file": str(self.scenario.source_path.relative_to(REPO)),
            "world_file": str(self.world_sdf.relative_to(REPO)),
            "world_sha256": _sha256(self.world_sdf),
            "stage_descriptor": str(STAGE_JSON.relative_to(REPO)),
            "stage_sha256": _sha256(STAGE_JSON),
            "obstacle_model_sha256": {
                name: _sha256(REPO / meta["model_sdf"]) for name, meta in self.catalogue.items()
            },
            "code_sha256": {
                path.name: _sha256(path)
                for path in sorted(_HERE.glob("*.py"))
            },
            "grid": self.grid.to_dict(),
            "cell_codes": {str(k): v for k, v in ora.CELL_CODE_MEANING.items()},
            "cameras": list(self.cameras_meta),
            "n_records": len(self.records),
            "n_capture_times": len(self.scenario.capture_times),
            "n_events": len(self.event_log),
            "non_capture_ticks_without_frames": self.skipped_renders,
            "started_at_utc": started_at,
            "wall_clock_s": round(wall_s, 2),
            "gz_version": _gz_version(),
            "python": platform.python_version(),
            "evaluation_only": (
                "oracle_depth/, oracle_visibility/ and obstacle_state are simulator "
                "ground truth; no method may read them. See EVALUATION_ONLY.md."
            ),
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # Checksums of everything a reproducibility check compares.
        digest_lines = []
        for path in sorted(self.out_dir.rglob("*")):
            if not path.is_file() or path.name in {"checksums.sha256", "manifest.json", "gz_server.log"}:
                continue
            digest_lines.append(f"{_sha256(path)}  {path.relative_to(self.out_dir)}")
        (self.out_dir / "checksums.sha256").write_text("\n".join(digest_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="run directory (default: logs/studies/dynamic_world_oracle/<id>/<tag>)")
    parser.add_argument("--run-tag", default="run01")
    parser.add_argument("--partition", default=None,
                        help="GZ_PARTITION for this run; defaults to a per-run unique name")
    parser.add_argument("--no-rgb", action="store_true",
                        help="skip writing RGB PNGs (oracle arrays only)")
    parser.add_argument("--gui", action="store_true",
                        help="open the Gazebo GUI so the scenario can be watched being stepped. "
                             "For inspection only: the GUI competes for the GPU with the sensor "
                             "renders this loop waits on, so take datasets from headless runs.")
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    out_dir = args.out or (DEFAULT_OUT_ROOT / scenario.scenario_id / args.run_tag)
    # The process id keeps a partition from ever being reused by a later run: a
    # server orphaned by an interrupted run would otherwise still be advertising
    # on that name, and the next run would silently adopt its clock.
    partition = args.partition or f"dwo_{scenario.scenario_id}_{args.run_tag}_{os.getpid()}"

    print(f"scenario   : {scenario.scenario_id}  ({scenario.description.strip()})")
    print(f"world      : {scenario.world_file}")
    print(f"timeline   : {scenario.duration_s:.1f}s at {scenario.tick_s:.2f}s ticks, "
          f"{len(scenario.events)} events, {len(scenario.capture_times)} captures")
    print(f"out        : {out_dir}")

    if args.gui:
        print("gui        : ON - a Gazebo window will open; this run is for watching, "
              "not for the record")
    runner = ScenarioRunner(scenario, out_dir, partition=partition,
                            keep_rgb=not args.no_rgb, gui=args.gui)
    runner.run()

    print(f"wrote {len(runner.records)} records over {len(scenario.capture_times)} instants "
          f"x {len(runner.cameras_meta)} cameras")
    print(f"  {out_dir / 'records.jsonl'}")
    print(f"  {out_dir / 'manifest.json'}")
    print(f"  {out_dir / 'events.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
