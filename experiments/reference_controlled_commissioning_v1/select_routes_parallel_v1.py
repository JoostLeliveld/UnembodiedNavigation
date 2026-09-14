#!/usr/bin/env python3
"""Run the frozen v5 route selector once per task in parallel and merge outputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(relative_roots: list[str]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for root in relative_roots
        for path in (REPO / root).rglob("*.py")
        if path.is_file()
    )
    for path in files:
        relative = str(path.relative_to(REPO)).encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _select_one(
    task: str, protocol: dict, temporary_root: Path, selector: Path
) -> tuple[str, Path, str]:
    child_protocol = dict(protocol)
    child_protocol.pop("parallel_wrapper", None)
    child_protocol["task_order"] = [task]
    protocol_path = temporary_root / f"{task}.protocol.json"
    output_path = temporary_root / f"{task}.output"
    _atomic_json(protocol_path, child_protocol)
    completed = subprocess.run(
        [
            sys.executable,
            str(selector),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output_path),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"route selection failed for {task} (exit {completed.returncode}):\n"
            f"{completed.stdout}"
        )
    return task, output_path, completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    wrapper = protocol.get("parallel_wrapper", {})
    if wrapper.get("path") != str(Path(__file__).resolve().relative_to(REPO)):
        raise RuntimeError("protocol does not identify this parallel wrapper")
    if _sha256(Path(__file__).resolve()) != wrapper.get("sha256"):
        raise RuntimeError("parallel wrapper SHA-256 mismatch")
    selector = (REPO / protocol["selector_path"]).resolve()
    if _sha256(selector) != protocol.get("selector_sha256"):
        raise RuntimeError("frozen selector SHA-256 mismatch")
    source_tree = protocol.get("planner_source_tree", {})
    if _tree_sha256(list(source_tree.get("roots", []))) != source_tree.get("sha256"):
        raise RuntimeError("planner source-tree SHA-256 mismatch")
    if output_path.exists():
        raise RuntimeError("route-selection outputs are immutable; choose a new directory")
    tasks = list(protocol["task_order"])
    if not tasks or len(set(tasks)) != len(tasks):
        raise RuntimeError("task_order must be nonempty and unique")

    cache_root = REPO / "logs" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="final_route_selection_", dir=cache_root
    ) as temporary:
        temporary_root = Path(temporary)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(args.jobs, 1), len(tasks))
        ) as pool:
            results = list(pool.map(
                lambda task: _select_one(
                    task, protocol, temporary_root, selector
                ), tasks
            ))
        output_path.mkdir(parents=True)
        selected = {}
        output_files = {}
        for task, child_output, console in results:
            print(console, end="", flush=True)
            child_manifest = json.loads(
                (child_output / "manifest.json").read_text(encoding="utf-8")
            )
            selected[task] = child_manifest["selected"][task]
            for arm, entry in selected[task].items():
                filename = f"{task}__{arm}.json"
                destination = output_path / filename
                shutil.copy2(child_output / filename, destination)
                entry["preselected_route_source_path"] = str(
                    destination.relative_to(REPO)
                )
                entry["preselected_route_source_sha256"] = _sha256(destination)
                output_files[filename] = _sha256(destination)

    manifest = {
        "schema_version": 1,
        "kind": "parallel_stage09_route_selection_manifest",
        "status": "complete",
        "protocol_path": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": _sha256(protocol_path),
        "parallel_wrapper": wrapper,
        "selector_sha256": protocol["selector_sha256"],
        "selected": selected,
        "output_files": dict(sorted(output_files.items())),
    }
    identity = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    manifest["selection_digest"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    _atomic_json(output_path / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
