#!/usr/bin/env python3
"""Create an immutable controller-only diagnostic from a hash-bound pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("diagnostic configs are immutable; choose a new output")
    config = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    config["study_title"] = (
        f"Controller diagnostic: {args.controller} on frozen T1/C00 route"
    )
    config["study_comparison"] = (
        "Non-inferential controller diagnostic after the turn-then-go tracker "
        "stopped beside bin_office; not a thesis result."
    )
    config["local_controller_type"] = args.controller
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, width=100000), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
