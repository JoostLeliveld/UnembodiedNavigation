#!/usr/bin/env python3
"""Fit covariance candidates after the gate-v2 correction selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fit_visibility_patch_covariance as covariance  # noqa: E402
from run_matched_correction_ladder_v2 import load_capture_v2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--correction-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report_path = args.correction_root.resolve() / "visibility_patch_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("sensor_gate_id") != "commissioning_sensor_gate_v2":
        raise RuntimeError("correction selection did not use sensor gate v2")
    if report.get("selected_correction") != "box_mlp_visibility_residual":
        raise RuntimeError("this covariance fitter requires the selected visibility correction")
    if len(report.get("audit_drive_ids_not_opened", [])) != 6:
        raise RuntimeError("correction report does not verify six sealed audit drives")

    covariance.load_capture = load_capture_v2
    saved_argv = sys.argv
    try:
        sys.argv = [
            str(Path(__file__)),
            "--capture-root", str(args.capture_root.resolve()),
            "--visibility-root", str(args.correction_root.resolve()),
            "--output", str(args.output.resolve()),
        ]
        return covariance.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    raise SystemExit(main())
