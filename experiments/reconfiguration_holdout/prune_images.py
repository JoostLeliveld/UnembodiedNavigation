#!/usr/bin/env python3
"""Release a capture's frames once they have been scored, keeping a legible sample.

One environment's grid capture is ~15,000 frames and ~3.5 GB, and the study needs four
of them on a machine with less than 8 GB free.  Everything the analysis reads —
detector outcomes, appearance statistics, the oracle column — is already in the CSVs
by the time this runs, so the frames themselves are only needed for figures and for
somebody who wants to look.

What is kept, deliberately rather than at random:

* every frame the monocular-depth field was computed from, named in that
  environment's ``<env>_fit.json``.  Deleting those would make the adaptive arm
  unreproducible;
* a stratified sample across the four cameras and the six spatial blocks, at one
  heading, so a reader can see what any part of the warehouse looked like;
* nothing else.

Refuses to run unless ``perception_targets.csv`` exists, so a capture can never be
pruned before it has been scored.

    python3 experiments/reconfiguration_holdout/prune_images.py --env L1
    python3 experiments/reconfiguration_holdout/prune_images.py --env L1 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

#: Frames kept per (camera, spatial block).  Four cameras times six blocks times this
#: is the sample that survives: enough to show every region from every camera.
KEEP_PER_CELL = 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True, choices=[e.key for e in C.ENVIRONMENTS])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    env = C.ENV_BY_KEY[args.env]
    if not (env.capture / "perception_targets.csv").is_file():
        raise SystemExit(f"{env.key}: refusing to prune before the frames are scored "
                         f"(no perception_targets.csv in {env.capture})")
    images = env.capture / "images"
    if not images.is_dir():
        print(f"{env.key}: no images directory; nothing to prune")
        return 0

    keep: set[str] = set()

    fit_path = C.OUT_ROOT / "mono_depth" / f"{env.key}_fit.json"
    if fit_path.is_file():
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        for rec in fit.get("frames", {}).values():
            keep.add(str(rec["rgb_path"]))
        print(f"{env.key}: keeping {len(keep)} monocular-depth source frames")
    else:
        print(f"{env.key}: WARNING no {fit_path.name} -- the depth source frames are "
              f"not protected yet. Run mono_depth_field.py --env {env.key} first.")

    rows = list(csv.DictReader((env.capture / "samples.csv").open(encoding="utf-8")))
    by_cell: dict[tuple[str, int], list[str]] = {}
    for row in rows:
        cam = str(row.get("camera_frame") or "")
        theta = float(row.get("theta", 0.0))
        if abs(theta) > 1e-6:
            continue  # one heading is enough for a look-at-it sample
        block = int(C.block_ids(np.array([[float(row["x"]), float(row["y"])]]))[0])
        by_cell.setdefault((cam, block), []).append(str(row["image_path"]))
    for (cam, block), paths in sorted(by_cell.items()):
        step = max(1, len(paths) // KEEP_PER_CELL)
        keep.update(paths[::step][:KEEP_PER_CELL])

    on_disk = sorted(p for p in images.rglob("*.jpg"))
    rel = {str(p.relative_to(env.capture)): p for p in on_disk}
    doomed = [p for r, p in rel.items() if r not in keep]
    freed = sum(p.stat().st_size for p in doomed)
    print(f"{env.key}: {len(on_disk)} frames on disk, keeping {len(rel) - len(doomed)}, "
          f"removing {len(doomed)} ({freed / 1e9:.2f} GB)")
    if args.dry_run:
        print(f"{env.key}: dry run, nothing deleted")
        return 0
    for p in doomed:
        p.unlink()
    (env.capture / "IMAGES_PRUNED.md").write_text(
        f"# Frames pruned\n\n"
        f"This capture's frames were scored into `perception_targets.csv` and\n"
        f"`appearance_features.csv`, then pruned by\n"
        f"`experiments/reconfiguration_holdout/prune_images.py` to fit four\n"
        f"environments on this machine's free disk.\n\n"
        f"- frames captured: {len(on_disk)}\n"
        f"- frames kept: {len(rel) - len(doomed)} "
        f"(every monocular-depth source frame, plus {KEEP_PER_CELL} per camera "
        f"per spatial block at heading 0)\n"
        f"- disk released: {freed / 1e9:.2f} GB\n\n"
        f"Every number the study reports comes from the CSVs, which are intact.\n"
        f"Recapture with `capture_environment.sh {env.key} {env.world_name}.world.sdf 4`.\n",
        encoding="utf-8")
    print(f"{env.key}: wrote IMAGES_PRUNED.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
