#!/usr/bin/env python3
"""Replace spatial q with the frozen constant-per-camera q baseline."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-field", type=Path, required=True)
    parser.add_argument("--constant-q-template", type=Path)
    parser.add_argument("--q-one", action="store_true")
    parser.add_argument("--arm", default="V3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.r_field.resolve(), allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if args.q_one:
        constant_q = np.ones_like(payload["availability"], dtype=float)
        q_metadata = {"availability_arm": "q_equals_one"}
    else:
        if args.constant_q_template is None:
            raise ValueError("supply --constant-q-template or --q-one")
        with np.load(args.constant_q_template.resolve(), allow_pickle=False) as archive:
            constant_q = np.asarray(archive["availability"], dtype=float)
            q_metadata = json.loads(str(archive["metadata_json"].item()))
    if constant_q.shape != payload["availability"].shape:
        raise ValueError("q field shapes differ")
    # The baseline is one measured constant per camera, broadcast over position.
    if not np.allclose(constant_q, constant_q[:, :1, :1]):
        raise ValueError("constant-q template is spatially varying")
    metadata = json.loads(str(payload["metadata_json"].item()))
    metadata.update({
        "arm": args.arm,
        "comparison_role": "q_equals_one_counterpart",
        "availability_arm": "q_equals_one" if args.q_one else "constant_per_camera",
        "availability_source": q_metadata.get("availability_arm", "constant"),
        "paired_spatial_q_arm": "U3" if args.arm.endswith("3") else "U0",
    })
    payload["availability"] = constant_q
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    output.write_bytes(buffer.getvalue())
    print(json.dumps({
        "path": str(output),
        "sha256": sha256(output),
        "q_by_camera": constant_q[:, 0, 0].tolist(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
