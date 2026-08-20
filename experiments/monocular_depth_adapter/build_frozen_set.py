#!/usr/bin/env python3
"""Build (or verify) the frozen image set manifest.

    python3 experiments/monocular_depth_adapter/build_frozen_set.py
    python3 experiments/monocular_depth_adapter/build_frozen_set.py --verify

The manifest is small and tracked; the images it points at stay where the
captures put them, under the ignored ``logs/`` tree.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import frozen_set as fs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=fs.DEFAULT_SET)
    parser.add_argument("--aws-count", type=int, default=12,
                        help="method-development frames from warehouse_aws")
    parser.add_argument("--fourcam-per-camera", type=int, default=3,
                        help="plumbing-only frames per camera from warehouse_full_4cam")
    parser.add_argument("--verify", action="store_true",
                        help="re-hash an existing manifest instead of rebuilding")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing manifest")
    args = parser.parse_args()

    if args.verify:
        problems = fs.verify(args.name)
        if problems:
            print(f"frozen set {args.name}: {len(problems)} problem(s)")
            for p in problems:
                print("  -", p)
            return 1
        frames = fs.load_frames(args.name)
        print(f"frozen set {args.name}: {len(frames)} frames, all hashes match")
        return 0

    path = fs.manifest_path(args.name)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to rebuild "
              "(rebuilding changes what every recorded benchmark number refers to)")
        return 1

    manifest = fs.build_manifest(args.name, aws_count=args.aws_count,
                                 fourcam_per_camera=args.fourcam_per_camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frames = manifest["frames"]
    by_role = Counter(f["role"] for f in frames)
    by_camera = Counter(f"{f['world'].split('.')[0]}/{f['camera_id']}" for f in frames)
    print(f"wrote {path.relative_to(fs.REPO)}  ({len(frames)} frames)")
    for role, n in sorted(by_role.items()):
        print(f"  {role}: {n}")
    for cam, n in sorted(by_camera.items()):
        print(f"  {cam}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
