#!/usr/bin/env python3
"""Copy, hash-verify, manifest, then remove one exact cold-archive source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[dict[str, dict[str, str | int]], int]:
    result: dict[str, dict[str, str | int]] = {}
    total_size = 0
    paths = [root] if root.is_file() or root.is_symlink() else sorted(root.rglob("*"))
    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        relative = "." if root.is_file() or root.is_symlink() else path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            result[relative] = {
                "type": "symlink",
                "target": target,
                "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            }
        elif path.is_file():
            size = path.stat().st_size
            total_size += size
            result[relative] = {"type": "file", "size": size, "sha256": sha256(path)}
        if index % 2000 == 0:
            print(f"inventoried {index}/{len(paths)} paths in {time.monotonic() - started:.1f}s", flush=True)
    return result, total_size


def copy_source(source: Path, staging: Path) -> None:
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, staging, symlinks=True, copy_function=shutil.copy2)
    elif source.is_symlink():
        staging.symlink_to(os.readlink(source))
    else:
        staging.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staging)


def remove_source(source: Path) -> None:
    if source.is_dir() and not source.is_symlink():
        shutil.rmtree(source)
    else:
        source.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--references", default="none")
    parser.add_argument("--central-manifests", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    forbidden = {Path("/"), Path.home(), Path("/home/joostleliveld/Thesis"), Path("/home/joostleliveld/Thesis/UnembodiedNavigation")}
    if source in forbidden or destination in forbidden:
        raise SystemExit("refusing broad source or destination")
    if not source.exists() and not source.is_symlink():
        raise SystemExit(f"missing source: {source}")
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"destination already exists: {destination}")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists() or staging.is_symlink():
        raise SystemExit(f"staging path already exists: {staging}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"hashing source {source}", flush=True)
    source_inventory, total_size = inventory(source)
    print(f"copying {len(source_inventory)} payload entries ({total_size} bytes)", flush=True)
    copy_source(source, staging)
    print("hashing copied payload", flush=True)
    copied_inventory, copied_size = inventory(staging)
    if source_inventory != copied_inventory or total_size != copied_size:
        raise SystemExit("verification failed; source retained and staging copy left for inspection")

    os.replace(staging, destination)
    manifest = {
        "archive_date": date.today().isoformat(),
        "original_path": str(source),
        "archive_path": str(destination),
        "total_size_bytes": total_size,
        "file_or_symlink_count": len(source_inventory),
        "reason": args.reason,
        "claim_or_evidence_references": args.references,
        "restore_command": f"cp -a -- {destination} {source}",
        "verification": "source and archive relative-path/type/size/SHA-256 inventories matched before source removal",
        "payload_manifest_sha256": hashlib.sha256(
            json.dumps(source_inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    local_manifest = destination.parent / f"{destination.name}.archive-manifest.json"
    local_hashes = destination.parent / f"{destination.name}.payload-sha256.json"
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    local_hashes.write_text(json.dumps(source_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.central_manifests.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_manifest, args.central_manifests / local_manifest.name)
    remove_source(source)
    print(f"verified and archived: {source} -> {destination}", flush=True)
    print(f"manifest: {local_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
