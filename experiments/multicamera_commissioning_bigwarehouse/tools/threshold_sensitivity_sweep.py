#!/usr/bin/env python3
"""Sensitivity of the hysteretic release decision to ``min_spatial_trust``.

The pilot released 0 corrections at the asserted 0.45 threshold.  Because that
constant is a copied library default (no data derivation), this sweep replays
the identical M8 pipeline across a threshold grid so the pilot's headline
number is reported as a curve, not a single-point artifact.  Diagnostic only:
it must never be used to tune the frozen protocol threshold post hoc — the
frozen value stays whatever ``config/paper_protocol.yaml``/``study.yaml``
declare, and this figure discloses how the outcome depends on it.

Outputs (CSV + RESULTS.md + figure) go to
``logs/studies/multicamera_commissioning_bigwarehouse/<tag>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for relative in ("src/reliability",):
    location = str(REPO / relative)
    if location not in sys.path:
        sys.path.insert(0, location)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reliability.camera_manager import CameraManagerConfig  # noqa: E402
from reliability.cli import _load_replay_frames  # noqa: E402
from reliability.providers import GridMapReliabilityProvider  # noqa: E402
from reliability.replay import ReplayConfig, ReplayMode, run_replay  # noqa: E402

RUN_ROOT = REPO / "logs/studies/multicamera_commissioning_bigwarehouse/actual_commissioning_20260715"
DEFAULT_FRAMES = RUN_ROOT / "analysis/final_01/replay_export_strict/operational/replay_frames.jsonl"
DEFAULT_GP = REPO / "logs/visibility_comparison/fourcam_actual_20260715/final_02/gp"
DEFAULT_OUT = REPO / "logs/studies/multicamera_commissioning_bigwarehouse/threshold_sensitivity_v1"
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
FROZEN_THRESHOLD = 0.45


def sweep_row(frames, providers, threshold: float) -> dict[str, object]:
    result = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            quality_providers=providers,
            nis_gate=9.21,
            camera_manager_config=CameraManagerConfig(min_spatial_trust=threshold),
        ),
    )
    steps = result.steps
    accepted = sum(len(step.accepted_camera_ids) for step in steps)
    no_eligible = sum(
        "no_eligible_camera" in step.camera_manager_decision.get("reasons", ())
        for step in steps
    )
    switches = sum(bool(step.camera_manager_decision.get("switched")) for step in steps)
    selected = [step.camera_manager_decision.get("selected_camera_id", "") for step in steps]
    row: dict[str, object] = {
        "min_spatial_trust": round(threshold, 3),
        "frames": len(steps),
        "accepted_updates": accepted,
        "no_eligible_camera_frames": no_eligible,
        "handover_switches": switches,
        "nis_rejections": sum(len(step.rejected_camera_ids) for step in steps),
    }
    for camera in CAMERAS:
        row[f"selected_{camera}"] = selected.count(camera)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-jsonl", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--gp-root", type=Path, default=DEFAULT_GP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min", dest="lo", type=float, default=0.05)
    parser.add_argument("--max", dest="hi", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.025)
    args = parser.parse_args()

    frames = _load_replay_frames(args.frames_jsonl)
    providers = {
        camera: GridMapReliabilityProvider.from_npz(
            args.gp_root / camera / "det_hit_expected_kernel_gp.npz",
            camera_id=camera,
            out_of_bounds_policy="clamp",
        )
        for camera in CAMERAS
    }

    thresholds: list[float] = []
    value = args.lo
    while value <= args.hi + 1e-9:
        thresholds.append(round(value, 3))
        value += args.step
    if FROZEN_THRESHOLD not in thresholds:
        thresholds = sorted(set(thresholds) | {FROZEN_THRESHOLD})

    rows = [sweep_row(frames, providers, threshold) for threshold in thresholds]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    release_points = [row for row in rows if int(row["accepted_updates"]) > 0]
    knee = max((float(row["min_spatial_trust"]) for row in release_points), default=None)
    frozen = next(row for row in rows if float(row["min_spatial_trust"]) == FROZEN_THRESHOLD)

    xs = [float(row["min_spatial_trust"]) for row in rows]
    accepted = [int(row["accepted_updates"]) for row in rows]
    blocked = [int(row["no_eligible_camera_frames"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8.4, 4.6), dpi=160)
    axis.plot(xs, accepted, marker="o", color="#2f80ed", label="corrections released")
    axis.plot(xs, blocked, marker="s", color="#ed8a25", label="no-eligible-camera frames")
    axis.axvline(FROZEN_THRESHOLD, color="#c81919", linestyle="--", linewidth=1.5,
                 label=f"frozen threshold = {FROZEN_THRESHOLD}")
    if knee is not None:
        axis.axvline(knee, color="#21a366", linestyle=":", linewidth=1.5,
                     label=f"last releasing threshold = {knee}")
    axis.set_xlabel("min_spatial_trust")
    axis.set_ylabel("count over pilot replay")
    axis.set_title("Pilot release decision vs trust threshold (diagnostic sweep)")
    axis.grid(color="#dce3eb")
    axis.legend(fontsize=9)
    fig.tight_layout()
    figure_path = args.out_dir / "threshold_sweep.png"
    fig.savefig(figure_path, facecolor="white")
    plt.close(fig)

    summary = {
        "frames_jsonl": str(args.frames_jsonl),
        "gp_root": str(args.gp_root),
        "frozen_threshold": FROZEN_THRESHOLD,
        "frozen_accepted_updates": int(frozen["accepted_updates"]),
        "last_releasing_threshold": knee,
        "thresholds": xs,
        "purpose": "disclosure of single-point sensitivity; not a tuning source",
    }
    (args.out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    knee_text = (
        f"the policy would first release corrections at min_spatial_trust <= {knee}"
        if knee is not None
        else "no threshold in the sweep released any correction"
    )
    results = f"""# Trust-threshold sensitivity (pilot replay, diagnostic)

Inputs: `{args.frames_jsonl.relative_to(REPO)}`, expected-kernel GPs in
`{args.gp_root.relative_to(REPO)}`.

The frozen policy threshold {FROZEN_THRESHOLD} released
**{frozen['accepted_updates']} corrections** over {frozen['frames']} replay
frames; {knee_text}.

This sweep exists to disclose how strongly the pilot's "0 corrections
released" headline depends on the asserted (not derived) 0.45 constant.  It is
NOT a tuning source: the protocol threshold stays frozen, and any future
change must be justified from warehouse_aws-era evidence, then pre-registered.

![sweep](threshold_sweep.png)

See `threshold_sweep.csv` for per-threshold selection counts and switches.
"""
    (args.out_dir / "RESULTS.md").write_text(results, encoding="utf-8")
    for row in rows:
        print(row)
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
