#!/usr/bin/env python3
"""Derive the three-run rate qualification config from the canonical campaign."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = yaml.safe_load(args.campaign.read_text())
    protocol = json.loads(args.protocol.read_text())
    condition = protocol["campaign_condition"]
    task = protocol["task"]
    campaign["conditions"] = {condition: campaign["conditions"][condition]}
    campaign["tasks"] = {
        task: {"conditions": [condition], "seeds": protocol["seeds"]}
    }
    campaign["study_title"] = "Canonical Stage-09 full-stack rate qualification"
    campaign["study_comparison"] = "Three repeated clean pilots; not performance evidence."
    campaign["ros_domain_id_base"] = 90
    temporary = args.output.with_name(args.output.name + ".incomplete")
    if temporary.exists() or args.output.exists():
        raise FileExistsError(args.output if args.output.exists() else temporary)
    temporary.write_text(yaml.safe_dump(campaign, sort_keys=False))
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
