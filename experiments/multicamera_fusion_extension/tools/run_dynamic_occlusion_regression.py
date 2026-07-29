#!/usr/bin/env python3
"""Run the deterministic D1 moving-occluder regression and write result artifacts.

This is an implementation regression, not a claim about a live warehouse: it
uses a tracked actor crossing one fixed camera's sight line and a deliberately
gate-evading association displacement.  Evaluation truth reaches only
``EvaluationFrame`` after the operational replay has run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability import (  # noqa: E402
    CameraQuality,
    DynamicActorState,
    DynamicOcclusionConfig,
    EvaluationFrame,
    MapObservation,
    ReplayConfig,
    ReplayFrame,
    ReplayMode,
    run_replay,
)


def frames_and_evaluation() -> tuple[tuple[ReplayFrame, ...], tuple[EvaluationFrame, ...]]:
    frames: list[ReplayFrame] = []
    evaluation: list[EvaluationFrame] = []
    for index in range(45):
        timestamp_s = index * 0.2
        truth = (0.04 * index, 2.0)
        blocked = 14 <= index < 30
        actors = (
            DynamicActorState("person_0", (truth[0] * 0.5, -1.0), radius_m=0.35, confidence=0.98),
        ) if blocked else tuple()
        # Simulated detector/association output. The occluder's presence is an
        # operational actor-track input; truth is withheld until scoring.
        observed = (truth[0] + (0.23 if blocked else 0.0), truth[1])
        observation = MapObservation(
            camera_id="camera_A", timestamp_s=timestamp_s, xy_m=observed,
            covariance_m2=((0.08 ** 2, 0.0), (0.0, 0.08 ** 2)),
            quality=CameraQuality(camera_id="camera_A", p_available=0.95),
            source="synthetic_dynamic_regression",
        )
        frames.append(ReplayFrame(
            timestamp_s, (truth[0] + 0.008 * index, truth[1]), (observation,), actors,
        ))
        evaluation.append(EvaluationFrame(timestamp_s, truth))
    return tuple(frames), tuple(evaluation)


def metrics_row(result) -> dict[str, float | int]:
    assert result.metrics is not None
    metrics = result.metrics
    return {
        "rmse_m": metrics.rmse_m,
        "p95_error_m": metrics.p95_error_m,
        "max_error_m": metrics.max_error_m,
        "final_error_m": metrics.final_error_m,
        "mean_nis": metrics.mean_nis,
        "update_acceptance_rate": metrics.update_acceptance_rate,
        "divergence_count": metrics.divergence_count,
    }


def make_showcase(frames, evaluation, baseline, dynamic, output_dir: Path) -> Path:
    """Render the result in a review-friendly four-panel figure."""

    surface, ink, muted, grid = "#fcfcfb", "#151515", "#5e5d58", "#deddd7"
    baseline_colour, dynamic_colour, risk_colour = "#e26d3d", "#187c73", "#5b4bb7"
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted,
    })
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), constrained_layout=True)
    ax_map, ax_error, ax_risk, ax_bars = axes.ravel()
    truth_xy = [item.truth_xy_m for item in evaluation]
    base_xy = [item.mean_xy_m for item in baseline.steps]
    dynamic_xy = [item.mean_xy_m for item in dynamic.steps]
    times = [item.timestamp_s for item in dynamic.steps]
    blocked_start, blocked_end = 14 * 0.2, 30 * 0.2

    # A. Geometry: the actor follows the sight line from the fixed camera.
    ax_map.plot([xy[0] for xy in truth_xy], [xy[1] for xy in truth_xy], color=ink,
                lw=2.2, label="evaluation trajectory", zorder=5)
    ax_map.plot([xy[0] for xy in base_xy], [xy[1] for xy in base_xy], color=baseline_colour,
                lw=1.8, label="M5 sequential fusion", zorder=4)
    ax_map.plot([xy[0] for xy in dynamic_xy], [xy[1] for xy in dynamic_xy], color=dynamic_colour,
                lw=2.0, label="D1 dynamic-aware fusion", zorder=6)
    ax_map.scatter([0.0], [-4.0], marker="v", s=72, color=ink, zorder=7)
    ax_map.plot((0.0, truth_xy[22][0]), (-4.0, truth_xy[22][1]), color=muted,
                lw=1.1, ls=":", alpha=0.75, zorder=1)
    ax_map.annotate("fixed camera", (0.0, -4.0), xytext=(7, -12), textcoords="offset points",
                    color=muted, fontsize=8)
    for index in range(14, 30, 4):
        actor = frames[index].dynamic_actors[0]
        ax_map.add_patch(Circle(actor.xy_m, actor.radius_m, facecolor=risk_colour,
                                edgecolor="none", alpha=0.16, zorder=2))
    ax_map.annotate("tracked person\ncrossing sight line", (0.55, -1.05), color=risk_colour,
                    fontsize=8, ha="center")
    ax_map.set(xlim=(-0.15, 2.25), ylim=(-4.55, 2.45), xlabel="world x (m)", ylabel="world y (m)",
               title="A. Dynamic occluder geometry")
    # This is a long camera-to-robot corridor, so preserve the full geometry
    # while allowing a presentation aspect ratio that keeps the trajectories
    # legible in the dashboard panel.
    ax_map.set_aspect("auto")
    ax_map.legend(loc="upper left", fontsize=7.5)

    # B. Error profile, with the occlusion interval called out directly.
    base_error = [math.dist(step.mean_xy_m, truth.truth_xy_m) for step, truth in zip(baseline.steps, evaluation)]
    dynamic_error = [math.dist(step.mean_xy_m, truth.truth_xy_m) for step, truth in zip(dynamic.steps, evaluation)]
    ax_error.axvspan(blocked_start, blocked_end, color=risk_colour, alpha=0.10, label="actor on sight line")
    ax_error.plot(times, base_error, color=baseline_colour, lw=2.0, label="M5 sequential")
    ax_error.plot(times, dynamic_error, color=dynamic_colour, lw=2.2, label="D1 dynamic-aware")
    ax_error.set(title="B. Localization error during occlusion", xlabel="time (s)", ylabel="position error (m)")
    ax_error.legend(loc="upper left", fontsize=7.5)

    # C. The operational signal which changes the update; no truth is required.
    probability = [step.dynamic_occlusion_diagnostics[0]["occlusion_probability"]
                   for step in dynamic.steps]
    inflation = [step.dynamic_occlusion_diagnostics[0]["covariance_inflation"]
                 for step in dynamic.steps]
    ax_risk.fill_between(times, probability, color=risk_colour, alpha=0.18)
    ax_risk.plot(times, probability, color=risk_colour, lw=2.2, label="occlusion probability")
    ax_risk.set(title="C. Operational dynamic-occlusion response", xlabel="time (s)",
                ylabel="probability", ylim=(0.0, 1.05))
    risk_axis = ax_risk.twinx()
    risk_axis.plot(times, inflation, color=dynamic_colour, lw=1.8, ls="--", label="covariance inflation")
    risk_axis.set_ylabel("covariance ×", color=dynamic_colour)
    risk_axis.tick_params(axis="y", colors=dynamic_colour)
    lines = ax_risk.get_lines() + risk_axis.get_lines()
    ax_risk.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=7.5)

    # D. The two headline error metrics, in cm for fast visual comparison.
    base_metrics, dynamic_metrics = metrics_row(baseline), metrics_row(dynamic)
    labels, xs, width = ("RMSE", "P95 error"), (0, 1), 0.32
    base_values = [100 * base_metrics["rmse_m"], 100 * base_metrics["p95_error_m"]]
    dynamic_values = [100 * dynamic_metrics["rmse_m"], 100 * dynamic_metrics["p95_error_m"]]
    ax_bars.bar([x - width / 2 for x in xs], base_values, width, color=baseline_colour, label="M5 sequential")
    ax_bars.bar([x + width / 2 for x in xs], dynamic_values, width, color=dynamic_colour, label="D1 dynamic-aware")
    for x, value in zip([x - width / 2 for x in xs], base_values):
        ax_bars.text(x, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color=baseline_colour)
    for x, value in zip([x + width / 2 for x in xs], dynamic_values):
        ax_bars.text(x, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color=dynamic_colour)
    ax_bars.set(title="D. Headline error reduction", xticks=xs, xticklabels=labels, ylabel="error (cm)")
    ax_bars.legend(loc="upper right", fontsize=7.5)

    for axis in axes.ravel():
        axis.grid(axis="y", color=grid, lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("D1 dynamic-occlusion-aware camera fusion", fontsize=15, color=ink, fontweight="bold")
    figure.text(0.5, 0.005,
                "Deterministic synthetic regression: actor tracks are operational input; pose truth is evaluation-only.",
                ha="center", color=muted, fontsize=8)
    path = output_dir / "dynamic_occlusion_showcase.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "logs" / "studies" / "multicamera_fusion_extension" / "dynamic_occlusion_regression",
    )
    args = parser.parse_args()
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, evaluation = frames_and_evaluation()
    baseline = run_replay(
        frames, ReplayConfig(mode=ReplayMode.SEQUENTIAL_FUSION, nis_gate=9.21),
        evaluation_frames=evaluation,
    )
    dynamic = run_replay(
        frames,
        ReplayConfig(
            mode=ReplayMode.DYNAMIC_OCCLUSION_AWARE_FUSION,
            nis_gate=9.21,
            dynamic_occlusion_config=DynamicOcclusionConfig(
                camera_xy_m={"camera_A": (0.0, -4.0)},
                ray_margin_m=0.20, covariance_gain=8.0, max_covariance_inflation=12.0,
            ),
        ),
        evaluation_frames=evaluation,
    )
    baseline_row, dynamic_row = metrics_row(baseline), metrics_row(dynamic)
    showcase_path = make_showcase(frames, evaluation, baseline, dynamic, output_dir)
    payload = {
        "study": "D1 deterministic dynamic-occlusion regression",
        "claim_boundary": (
            "Synthetic implementation regression only; it does not establish live Gazebo or physical performance. "
            "Dynamic actor tracks are operational input and truth is evaluation-only."
        ),
        "scenario": {
            "frames": len(frames), "dt_s": 0.2, "camera_id": "camera_A",
            "actor": "one tracked person crossing camera-to-robot line of sight on frames 14..29",
            "association_displacement_m": 0.23,
        },
        "results": {
            ReplayMode.SEQUENTIAL_FUSION.value: baseline_row,
            ReplayMode.DYNAMIC_OCCLUSION_AWARE_FUSION.value: dynamic_row,
        },
        "deltas_dynamic_minus_baseline": {
            key: dynamic_row[key] - baseline_row[key]
            for key in ("rmse_m", "p95_error_m", "max_error_m", "final_error_m")
        },
        "peak_dynamic_occlusion_probability": max(
            diagnostic["occlusion_probability"]
            for step in dynamic.steps for diagnostic in step.dynamic_occlusion_diagnostics
        ),
        "showcase_figure": showcase_path.name,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = "\n".join([
        "# D1 Dynamic-Occlusion Regression Results", "",
        payload["claim_boundary"], "",
        "| Mode | RMSE (m) | P95 error (m) | Max error (m) | Final error (m) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(f"| `{name}` | {row['rmse_m']:.4f} | {row['p95_error_m']:.4f} | "
          f"{row['max_error_m']:.4f} | {row['final_error_m']:.4f} |"
          for name, row in payload["results"].items()),
        "",
        f"Peak operational occlusion probability: {payload['peak_dynamic_occlusion_probability']:.3f}.",
        "",
        "![Dynamic-occlusion showcase](dynamic_occlusion_showcase.png)",
        "",
        "The full machine-readable record is `results.json`.", "",
    ])
    (output_dir / "RESULTS.md").write_text(markdown)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
