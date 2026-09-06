#!/usr/bin/env python3
"""Reader-first gate sensitivity figures for the held-out bbox characterization set.

The gate may use only quantities available at runtime. Ground truth is used after the gate
solely to score the retained camera readings. The ungated figures remain the primary sensor
characterization; this script shows the accuracy/availability trade-off of candidate gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "logs/studies/camera_observation_characterization_20260831"
for rel in ("experiments/deck_figures",):
    value = str((REPO / rel).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import style as D  # noqa: E402
from _fieldscale import drawn_shrink, panel_scale, scale_bar  # noqa: E402

LADDER = (
    ("raw", "Raw box → floor", "#eb6834"),
    ("fixed", "Fixed 30.9 cm", "#2a78d6"),
    ("learned", "Learned linear", "#4a3aa7"),
    ("nn", "Neural net", "#d4267b"),
    ("hull", "Hull reference", "#1baf7a"),
)
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
FOLDER = "06_what_a_gate_costs"
NATIVE_WIDTH = 1280.0
NATIVE_HEIGHT = 720.0
EDGE_MARGIN_PX = 8.0
RANGE_LIMIT_M = 16.0
HIGH_CONFIDENCE = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_title(camera_id: str) -> str:
    return f"Camera {camera_id[-1]}"


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def finite_float(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def selected_point_edge_safe(row: dict[str, str]) -> bool:
    """Scaled equivalent of the frozen 4 px gate at the 640x360 runtime size."""
    u = finite_float(row, "u_bbox_bottom")
    v = finite_float(row, "v_bbox_bottom")
    if u is None or v is None:
        return False
    return min(u, v, NATIVE_WIDTH - u, NATIVE_HEIGHT - v) >= EDGE_MARGIN_PX


def raw_projected_range(row: dict[str, str], camera_xy: dict[str, tuple[float, float]]) -> float:
    """Range of the raw box back-projection, not the evaluation-only true range."""
    x = finite_float(row, "raw_x")
    y = finite_float(row, "raw_y")
    if x is None or y is None:
        return math.inf
    cx, cy = camera_xy[row["camera_id"]]
    return math.hypot(x - cx, y - cy)


def gates(camera_xy: dict[str, tuple[float, float]]) -> tuple[tuple[str, str, Callable], ...]:
    return (
        (
            "all_returns",
            "All selected YOLO returns\n(confidence ≥ 0.25)",
            lambda row: row["raw_valid"] == "1",
        ),
        (
            "edge_safe",
            "Bottom-centre clear of image edge\n(≥8 native pixels)",
            lambda row: row["raw_valid"] == "1" and selected_point_edge_safe(row),
        ),
        (
            "range_16m",
            "Raw projected range ≤ 16 m",
            lambda row: row["raw_valid"] == "1"
            and raw_projected_range(row, camera_xy) <= RANGE_LIMIT_M,
        ),
        (
            "candidate_edge_range",
            "Candidate gate: edge-safe\n+ raw projected range ≤ 16 m",
            lambda row: row["raw_valid"] == "1"
            and selected_point_edge_safe(row)
            and raw_projected_range(row, camera_xy) <= RANGE_LIMIT_M,
        ),
        (
            "candidate_plus_conf50",
            "Sensitivity only: candidate gate\n+ confidence ≥ 0.50",
            lambda row: row["raw_valid"] == "1"
            and selected_point_edge_safe(row)
            and raw_projected_range(row, camera_xy) <= RANGE_LIMIT_M
            and float(row["confidence"]) >= HIGH_CONFIDENCE,
        ),
    )


def values(rows: list[dict[str, str]], method: str, field: str) -> np.ndarray:
    return np.asarray([
        float(row[f"{method}_{field}"])
        for row in rows
        if row[f"{method}_valid"] == "1"
        and math.isfinite(float(row[f"{method}_{field}"]))
    ])


def summarize(data: np.ndarray) -> dict[str, float | int]:
    if not data.size:
        return {"n": 0, "median_m": math.nan, "p90_m": math.nan, "rmse_m": math.nan}
    return {
        "n": int(data.size),
        "median_m": float(np.median(data)),
        "p90_m": float(np.quantile(data, 0.90)),
        "rmse_m": float(np.sqrt(np.mean(data ** 2))),
    }


def build_report(
    rows: list[dict[str, str]],
    gate_defs: tuple[tuple[str, str, Callable], ...],
) -> dict:
    report: dict = {}
    for portion in ("train", "test"):
        opportunities = [row for row in rows if row["split"] == portion]
        detections = [row for row in opportunities if row["raw_valid"] == "1"]
        report[portion] = {
            "opportunities": len(opportunities),
            "detected": len(detections),
            "gates": {},
        }
        for gate_id, label, predicate in gate_defs:
            retained = [row for row in opportunities if predicate(row)]
            by_camera = {
                camera: sum(row["camera_id"] == camera for row in retained)
                for camera in CAMERAS
            }
            report[portion]["gates"][gate_id] = {
                "label": label.replace("\n", " "),
                "retained": len(retained),
                "retained_fraction_of_detections": len(retained) / len(detections),
                "retained_fraction_of_opportunities": len(retained) / len(opportunities),
                "retained_by_camera": by_camera,
                "scores": {
                    method: summarize(values(retained, method, "error_m"))
                    for method, _label, _colour in LADDER
                },
            }
    return report


def draw_tradeoff_and_before_after(report: dict, out: Path) -> None:
    """Availability given up, and the reading error bought, in one figure.

    Merges the former 11_gate_tradeoff and 13_reading_error_before_after: the sweep panels
    carry every gate level, and the two levels the before/after story compares — all
    selected YOLO returns and the candidate gate — are marked hollow and filled so the
    single-step result stays readable inside the full sweep.
    """
    test = report["test"]
    gate_rows = list(test["gates"].items())
    labels = [payload["label"].replace(" + ", "\n+ ") for _key, payload in gate_rows]
    keys = [key for key, _payload in gate_rows]
    y = np.arange(len(gate_rows))[::-1]
    before_slot = y[keys.index("all_returns")]
    after_slot = y[keys.index("candidate_edge_range")]
    candidate = test["gates"]["candidate_edge_range"]

    fig, axes = plt.subplots(
        1, 4, figsize=(23.0, 7.6), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.55, 1.0, 1.0, 1.0)},
    )
    retained_det = [100 * payload["retained_fraction_of_detections"] for _k, payload in gate_rows]
    retained_all = [100 * payload["retained_fraction_of_opportunities"] for _k, payload in gate_rows]
    axes[0].barh(y + 0.16, retained_det, height=0.30, color="#5e8fbf", alpha=0.88,
                 label="of YOLO returns")
    axes[0].barh(y - 0.16, retained_all, height=0.30, color="#edb76c", alpha=0.95,
                 label="of all camera opportunities")
    for slot, detected, all_views in zip(y, retained_det, retained_all):
        axes[0].text(detected + 1.2, slot + 0.16, f"{detected:.1f}%", va="center", fontsize=10)
        axes[0].text(all_views + 1.2, slot - 0.16, f"{all_views:.1f}%", va="center", fontsize=10)
    axes[0].set_xlim(0, 112)
    axes[0].set_xlabel("Camera readings still available (%)")
    axes[0].set_title("What the gate gives up", fontsize=15, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10.5)

    for ax, metric, title in zip(
        axes[1:],
        ("median_m", "p90_m", "rmse_m"),
        ("Median reading error", "90th-percentile reading error", "RMS reading error"),
    ):
        for index, (method, label, colour) in enumerate(LADDER):
            scores = [100 * payload["scores"][method][metric] for _k, payload in gate_rows]
            # methods can land within a centimetre of each other, so alternate the value
            # labels above and below the row to keep them all readable
            label_offset = 13 if index % 2 == 0 else -21
            ax.plot(scores, y, marker="", lw=2, color=colour, label=label, zorder=3)
            for slot, score in zip(y, scores):
                if slot == before_slot:
                    ax.plot(score, slot, marker="o", ms=10, mfc="white", mec=colour, mew=2.2,
                            zorder=5)
                elif slot == after_slot:
                    ax.plot(score, slot, marker="o", ms=10, mfc=colour, mec=colour, mew=2.2,
                            zorder=5)
                    ax.annotate(f"{score:.1f}", (score, slot), xytext=(0, label_offset),
                                textcoords="offset points", ha="center", fontsize=9.5,
                                fontweight="bold", color=colour, zorder=6)
                else:
                    ax.plot(score, slot, marker="o", ms=6, color=colour, zorder=4)
        ax.set_xlabel("Camera-reading error (cm)")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.grid(axis="x", color="#e4e2dc", lw=0.8)
        ax.set_axisbelow(True)

    for ax in axes:
        ax.set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10.5)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    method_handles = [Line2D([0], [0], color=colour, lw=2.4, label=label)
                      for _method, label, colour in LADDER]
    marker_handles = [
        Line2D([0], [0], marker="o", ms=10, mfc="white", mec=D.MUTED, mew=2.2, lw=0,
               label="all selected YOLO returns (before)"),
        Line2D([0], [0], marker="o", ms=10, mfc=D.MUTED, mec=D.MUTED, mew=2.2, lw=0,
               label="candidate gate (after)"),
    ]
    fig.legend(handles=method_handles + marker_handles, loc="lower center", ncol=7,
               fontsize=10.8, frameon=True, bbox_to_anchor=(0.5, -0.055))
    fig.suptitle(
        "The gate trims error tails by giving up camera availability; it does not remove the bias\n"
        f"Held-out tiles: 3,163 YOLO returns before, {candidate['retained']:,} after — "
        f"only {100 * candidate['retained_fraction_of_opportunities']:.1f}% of all camera opportunities remain\n"
        "Every gate uses only the selected box, its confidence, camera calibration and raw projection",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.105,
        "The 16 m rule is a candidate sensitivity check, not a frozen runtime decision. "
        "Ground truth scores errors only after admission.",
        ha="center",
        fontsize=11,
        color=D.MUTED,
    )
    save(fig, out / FOLDER / "15_gate_tradeoff_and_before_after.png")


def draw_signed_ladder(rows: list[dict[str, str]], report: dict, out: Path) -> None:
    candidate = report["test"]["gates"]["candidate_edge_range"]
    spread = np.asarray([
        abs(float(row[f"{method}_{field}"]))
        for row in rows
        for method, _label, _colour in LADDER
        for field in ("along_m", "across_m")
        if row[f"{method}_valid"] == "1"
    ])
    limit = max(0.3, float(np.quantile(spread, 0.93))) if spread.size else 0.5
    fig, axes = plt.subplots(5, 2, figsize=(14.0, 15.0), sharex=True, constrained_layout=True)
    order = list(range(len(LADDER)))[::-1]
    for row_axes, camera in zip(axes, CAMERAS):
        camera_rows = [row for row in rows if row["camera_id"] == camera]
        for ax, field, name in zip(
            row_axes, ("along_m", "across_m"), ("along the ray", "across the ray")
        ):
            for slot, (method, label, colour) in zip(order, LADDER):
                data = values(camera_rows, method, field)
                if not data.size:
                    continue
                low10, low25, mid, high75, high90 = np.quantile(
                    data, [0.10, 0.25, 0.50, 0.75, 0.90]
                )
                ax.plot([low10, high90], [slot, slot], color=colour, lw=1.6, alpha=0.5)
                ax.plot([low25, high75], [slot, slot], color=colour, lw=6.5, alpha=0.85)
                ax.plot([mid], [slot], marker="o", ms=9, color="white", mec=colour, mew=2.4)
            ax.axvline(0.0, color=D.INK, lw=1.2, linestyle="--")
            ax.set_yticks(order)
            ax.set_yticklabels([label for _m, label, _c in LADDER] if ax is row_axes[0]
                               else ["" for _ in LADDER], fontsize=10)
            ax.set_ylim(-0.7, len(LADDER) - 0.3)
            ax.grid(axis="x", color="#e4e2dc", lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xlim(-limit, limit)
            if camera == CAMERAS[0]:
                ax.set_title(f"Signed error {name}", fontsize=14.5, fontweight="bold")
        row_axes[0].text(-0.34, 0.5, camera_title(camera), transform=row_axes[0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=13,
                         fontweight="bold")
    for ax in axes[-1]:
        ax.set_xlabel("Signed camera-reading error (m)")
    handles = [
        Line2D([0], [0], color=D.MUTED, marker="o", ms=9, mfc="white", mec=D.MUTED,
               mew=2.4, lw=0, label="circle = median"),
        Line2D([0], [0], color=D.MUTED, lw=6.5, alpha=0.85,
               label="thick bar = middle half (25–75%)"),
        Line2D([0], [0], color=D.MUTED, lw=1.6, alpha=0.5,
               label="thin bar = 10–90%"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11.5, frameon=True,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "After a candidate operational gate, the learned correction remains centred\n"
        f"Held-out tiles: {candidate['retained']} admitted readings; "
        f"{100 * candidate['retained_fraction_of_detections']:.1f}% of YOLO returns retained",
        fontsize=19,
        fontweight="bold",
    )
    save(fig, out / FOLDER / "16_signed_error_ladder_after_gate.png")


def aggregate_positions(rows: list[dict[str, str]], method: str):
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["position_id"]].append(row)
    result = []
    for group in groups.values():
        first = group[0]
        hits = [row for row in group if row.get("gate_pass") == "1"
                and row[f"{method}_valid"] == "1"]
        if not hits:
            result.append((float(first["robot_x"]), float(first["robot_y"]),
                           math.nan, math.nan, math.nan))
            continue
        result.append((
            float(first["robot_x"]),
            float(first["robot_y"]),
            float(np.median(values(hits, method, "dx"))),
            float(np.median(values(hits, method, "dy"))),
            float(np.median(values(hits, method, "error_m"))),
        ))
    return result


def draw_error_fields(rows: list[dict[str, str]], out: Path) -> None:
    """Admitted-reading residual fields, each rung on its own arrow and colour scale."""
    for camera in CAMERAS:
        camera_rows = [row for row in rows if row["camera_id"] == camera]
        fig, axes = plt.subplots(2, 3, figsize=(17.5, 12.6), constrained_layout=True)
        for ax, (method, label, _colour) in zip(axes.flat, LADDER):
            agg = aggregate_positions(camera_rows, method)
            good = [item for item in agg if math.isfinite(item[2])]
            blank = [item for item in agg if not math.isfinite(item[2])]
            clipped = 0
            scale = panel_scale(np.asarray([item[4] for item in good]))
            D.draw_warehouse(ax, D.layout(), show_cameras=True, camera_labels=True,
                             rack_alpha=0.72)
            if blank:
                ax.scatter([item[0] for item in blank], [item[1] for item in blank],
                           marker="x", s=11, c="#aaa9a4", linewidths=0.7, zorder=3)
            if good:
                gx = np.asarray([item[0] for item in good])
                gy = np.asarray([item[1] for item in good])
                dx = np.asarray([item[2] for item in good])
                dy = np.asarray([item[3] for item in good])
                shrink, clipped = drawn_shrink(dx, dy, scale['gain'])
                scalar = ax.quiver(
                    gx, gy, scale["gain"] * dx * shrink, scale["gain"] * dy * shrink,
                    np.asarray([item[4] for item in good]), cmap="magma",
                    norm=Normalize(0, scale["cap"]), angles="xy", scale_units="xy", scale=1,
                    width=0.0045, headwidth=3.5, headlength=4.2, zorder=5,
                )
                bar = fig.colorbar(scalar, ax=ax, fraction=0.040, pad=0.015, extend="max")
                bar.set_label(f'median error here (m), capped {scale["cap"]:.2f}', fontsize=8.6)
                bar.ax.tick_params(labelsize=8)
                scale_bar(ax, scale["gain"], layout=D.layout(), colour=D.INK)
            tail = f'  ·  {clipped} clipped' if good and clipped else ''
            ax.set_title(
                f'{label}\nmedian {100 * scale["median_m"]:.1f} cm  ·  '
                f'arrows \u00d7{scale["gain"]:.0f}  ·  {scale["n"]} positions{tail}',
                fontsize=12.6, fontweight="bold")
        note = axes.flat[-1]
        note.axis("off")
        note.text(0.02, 0.97, "What changed?", fontsize=16, fontweight="bold", va="top")
        note.text(
            0.02,
            0.86,
            "Only candidate-gate readings remain. Each arrow\n"
            "is the median over admitted headings at that\n"
            "held-out floor position.\n\n"
            "\u26a0 Every panel has its OWN arrow gain and colour\n"
            "cap, both printed above it. Never compare arrow\n"
            "length or colour between panels \u2014 use the printed\n"
            "medians or each panel\u2019s scale bar.\n\n"
            "Grey \u00d7 means no admitted reading at any heading.\n\n"
            "Gate inputs: confidence, selected pixel, camera\n"
            "calibration and raw back-projection. Truth is used\n"
            "only to place and score the arrow offline.",
            fontsize=12.0,
            va="top",
            linespacing=1.38,
        )
        fig.suptitle(
            f"{camera_title(camera)} \u2014 candidate-gated error field on held-out tiles\n"
            "Each panel on its own scale",
            fontsize=19,
            fontweight="bold",
        )
        save(fig, out / FOLDER / "17_admitted_readings_by_camera" /
             f"17_{camera}_admitted_readings.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-fields", action="store_true")
    args = parser.parse_args()

    capture = args.capture.expanduser().resolve()
    table = capture / "bias_update_interpretations.csv"
    bias_manifest_path = capture / "bias_update_interpretations_manifest.json"
    capture_manifest_path = capture / "capture_manifest.json"
    for required in (table, bias_manifest_path, capture_manifest_path):
        if not required.is_file():
            raise RuntimeError(f"Missing required input: {required}")

    bias_manifest = json.loads(bias_manifest_path.read_text(encoding="utf-8"))
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    expected_hash = bias_manifest.get("bias_update_interpretations_sha256")
    if expected_hash != sha256(table):
        raise RuntimeError("bias_update_interpretations.csv no longer matches its manifest")
    camera_xy = {
        item["camera_id"]: tuple(float(value) for value in item["pose_xyz_rpy"][:2])
        for item in capture_manifest["cameras"]
    }
    gate_defs = gates(camera_xy)
    every = list(csv.DictReader(table.open(encoding="utf-8")))
    report = build_report(every, gate_defs)

    out = (args.out or DEFAULT_OUT).expanduser().resolve()
    target = out / FOLDER
    if target.exists() and not args.overwrite:
        raise RuntimeError(f"Output already exists: {target}; pass --overwrite to refresh")
    target.mkdir(parents=True, exist_ok=True)

    test_views = [row for row in every if row["split"] == "test"]
    gate_by_id = {gate_id: predicate for gate_id, _label, predicate in gate_defs}
    candidate = [row for row in test_views if gate_by_id["candidate_edge_range"](row)]
    for row in test_views:
        row["gate_pass"] = "1" if gate_by_id["candidate_edge_range"](row) else "0"

    draw_tradeoff_and_before_after(report, out)
    draw_signed_ladder(candidate, report, out)
    if not args.skip_fields:
        draw_error_fields(test_views, out)

    manifest = {
        "status": "complete",
        "schema": "bbox_gate_sensitivity.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": str(table),
        "source_table_sha256": sha256(table),
        "evaluation_split": "held-out 2 m checkerboard tiles from bias update manifest",
        "model_refit_after_gating": False,
        "model_refit_note": (
            "the existing learned models are evaluated unchanged; the gate is applied "
            "post-detection and no method is retrained on the admitted subset"
        ),
        "ground_truth_firewall": (
            "gate inputs exclude robot_x, robot_y, robot_yaw, camera_range_m and all error "
            "columns; ground truth is used only after admission to score camera readings"
        ),
        "native_image_size": [int(NATIVE_WIDTH), int(NATIVE_HEIGHT)],
        "candidate_gate": {
            "status": "diagnostic proposal; not frozen in runtime",
            "rules": [
                "selected YOLO confidence >= 0.25 (already imposed by detector selection)",
                "bbox bottom-centre at least 8 native pixels from every image edge",
                "horizontal range from camera to raw bbox-floor back-projection <= 16 m",
            ],
            "edge_scaling_note": (
                "8 px at 1280x720 equals frozen usable-observation gate's 4 px at 640x360"
            ),
            "range_note": "computed from raw_x/raw_y and camera calibration, never true range",
        },
        "sensitivity_only": {
            "confidence_threshold": HIGH_CONFIDENCE,
            "note": "exploratory threshold, not selected as the candidate gate",
        },
        "report": report,
        "figures": [str(path.relative_to(target)) for path in sorted(target.rglob("*.png"))],
    }
    manifest_path = target / "gate_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(target),
        "figures": len(manifest["figures"]),
        "test_candidate": report["test"]["gates"]["candidate_edge_range"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
