#!/usr/bin/env python3
"""Regenerate the paper-vs-current comparison figures.

All media lives under paper_artifacts/figures (single source of truth since the
2026-07-15 consolidation): canonical paired-mechanism PDFs + `_data/` trees at the
root, side-specific renders (previews, gifs, side-only figures) in
`paper_snapshot/` (frozen paper baseline: archived paper GP/config, legacy truth
column) and `current_surface/` (honest campaign, current GP/config, GT columns).
docs/paper_vs_current keeps only the markdown comparison pages and the two frozen
config snapshots; nothing is copied there anymore.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG = ROOT / "paper_artifacts/figures/paper_snapshot"
CURRENT_FIG = ROOT / "paper_artifacts/figures/current_surface"

PAPER_GP = "paper_artifacts/gp/archive/aws_gp_v7b_superseded/yolo_score_raw_gp.npz"
CURRENT_GP = "paper_artifacts/gp/warehouse_visibility_gp_v1/yolo_score_raw_gp.npz"
PAPER_CONFIG = "docs/paper_vs_current/paper/aws_f31b1_final_config.yaml"
CURRENT_CONFIG = "docs/paper_vs_current/current/warehouse_visibility_campaign.yaml"
PAPER_ROBUSTNESS = "logs/visibility_comparison/_paper_runs/robustness_campaign_keepout_lanegraph_v1"
PAPER_METRICS = "paper_artifacts/metrics/archive/robustness_metrics.csv"
CURRENT_ROBUSTNESS = "logs/visibility_comparison/honest_campaign_v1"

PAIRED_RUNS = [
    {
        "out": "paired_mechanism_taskA_PAPER",
        "side": "paper",
        "camp": "_paper_runs/paired_mechanism_clean_verify",
        "task": "route_apron_to_a3_mid",
        "seed": 0,
        "gp": PAPER_GP,
        "config": PAPER_CONFIG,
        "legacy_truth": True,
        "title_a": "(a) Paper C1: short route, weak camera support",
        "title_b": "(b) Paper C2: visible detour",
    },
    {
        "out": "paired_mechanism_taskA_current",
        "side": "current",
        "camp": "paired_mechanism_current_taskA",
        "task": "route_apron_to_a3_mid",
        "seed": 0,
        "gp": CURRENT_GP,
        "config": CURRENT_CONFIG,
        "title_a": "(a) Current C1: short route, recovered localization",
        "title_b": "(b) Current C2: visible detour",
    },
    {
        "out": "paired_mechanism_west_current",
        "side": "current",
        "camp": "paired_mechanism_current_west",
        "task": "route_west_to_a1_upper",
        "seed": 0,
        "gp": CURRENT_GP,
        "config": CURRENT_CONFIG,
        "title_a": "(a) Current C1: west lane failure",
        "title_b": "(b) Current C2: visible detour succeeds",
    },
    {
        "out": "paired_mechanism_taskA_lowlat",
        "side": "current",
        "camp": "honest_campaign_v1",
        "task": "route_apron_to_a3_mid",
        "seed": 0,
        "gp": CURRENT_GP,
        "config": CURRENT_CONFIG,
        "title_a": "(a) Honest C1: apron to A3",
        "title_b": "(b) Honest C2: apron to A3",
    },
    {
        "out": "paired_mechanism_a2mid_lowlat",
        "side": "current",
        "camp": "honest_campaign_v1",
        "task": "route_apron_to_a2_mid",
        "seed": 0,
        "gp": CURRENT_GP,
        "config": CURRENT_CONFIG,
        "title_a": "(a) Honest C1: apron to A2",
        "title_b": "(b) Honest C2: apron to A2",
    },
    {
        "out": "paired_mechanism_west_lowlat",
        "side": "current",
        "camp": "honest_campaign_v1",
        "task": "route_west_to_a1_upper",
        "seed": 1,
        "gp": CURRENT_GP,
        "config": CURRENT_CONFIG,
        "title_a": "(a) Honest C1: west lane safety breach",
        "title_b": "(b) Honest C2: visible detour",
    },
    {
        "out": "paired_mechanism_control_lowlat",
        "side": "current",
        "camp": "honest_campaign_v1",
        "task": "control_west_to_a1_low",
        "seed": 0,
        "gp": CURRENT_GP,
        "config": CURRENT_CONFIG,
        "title_a": "(a) Honest C1: visible control",
        "title_b": "(b) Honest C2: visible control",
    },
]

GIF_RUNS = [
    ("paper", "paired_mechanism_taskA_PAPER.gif", "_paper_runs/paired_mechanism_clean_verify", "F31_b1_apron_a3_mid", 0),
    ("current", "paired_mechanism_taskA_current.gif", "paired_mechanism_current_taskA", "route_apron_to_a3_mid", 0),
    ("current", "paired_mechanism_west_current.gif", "paired_mechanism_current_west", "route_west_to_a1_upper", 0),
    ("current", "paired_mechanism_taskA_lowlat.gif", "honest_campaign_v1", "route_apron_to_a3_mid", 0),
    ("current", "paired_mechanism_a2mid_lowlat.gif", "honest_campaign_v1", "route_apron_to_a2_mid", 0),
    ("current", "paired_mechanism_west_lowlat.gif", "honest_campaign_v1", "route_west_to_a1_upper", 1),
    ("current", "paired_mechanism_control_lowlat.gif", "honest_campaign_v1", "control_west_to_a1_low", 0),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(args: list[str], *, env: dict[str, str] | None = None, cwd: Path = ROOT) -> None:
    merged = os.environ.copy()
    merged.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    if env:
        merged.update(env)
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, env=merged, check=True)


def run_dir_from_campaign(campaign: str, task: str, cond: str, seed: int) -> Path:
    log = ROOT / "logs/visibility_comparison" / campaign / "campaign_log.json"
    data = json.loads(log.read_text(encoding="utf-8"))
    for entry in data.values():
        if str(entry.get("task")) == task and str(entry.get("condition")) == cond and int(entry.get("seed")) == seed:
            return Path(entry["run_dir"])
    raise RuntimeError(f"No run in {log} for {task}/{cond}/seed{seed}")


def latest_run_dir(campaign: str, task: str, cond: str, seed: int) -> Path:
    campaign_root = ROOT / "logs/visibility_comparison" / campaign
    matches = sorted((campaign_root / task / cond / f"seed{seed}").glob("experiment_*"))
    if not matches:
        raise RuntimeError(f"No run directory for {campaign}/{task}/{cond}/seed{seed}")
    return matches[-1]


def regenerate_method_figures() -> None:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    CURRENT_FIG.mkdir(parents=True, exist_ok=True)

    run([
        "python3", "scripts/paper_figures/make_aws_gp_pipeline_figure.py",
        "--gp", PAPER_GP,
        "--out", str(PAPER_FIG / "gp_pipeline_aws.pdf"),
        "--preview", str(PAPER_FIG / "gp_pipeline_aws.png"),
    ])
    run([
        "python3", "scripts/paper_figures/make_aws_gp_pipeline_figure.py",
        "--gp", CURRENT_GP,
        "--out", str(CURRENT_FIG / "gp_pipeline_current.pdf"),
        "--preview", str(CURRENT_FIG / "gp_pipeline_current.png"),
    ])
    run([
        "python3", "scripts/paper_figures/make_aws_problem_setup_figure.py",
        "--split",
        "--out", str(PAPER_FIG / "problem_setup_aws.pdf"),
        "--preview", str(PAPER_FIG / "problem_setup_aws.png"),
    ])
    run([
        "python3", "scripts/paper_figures/make_localization_pathway_figure.py",
        "--out", str(PAPER_FIG / "localization_pathway.pdf"),
        "--preview", str(PAPER_FIG / "localization_pathway.png"),
    ])
    # yolo_training_clarification.png is canonical at paper_artifacts/figures/ and
    # applies to both sides; no per-side copies.
    run(["python3", "scripts/paper_figures/make_yolo_training_clarification.py"])


def regenerate_spreads() -> None:
    run([
        "python3", "scripts/paper_figures/make_robustness_spread.py",
        "--campaign-root", PAPER_ROBUSTNESS,
        "--metrics", PAPER_METRICS,
        "--gp", PAPER_GP,
        "--config", PAPER_CONFIG,
        "--out", str(PAPER_FIG / "robustness_spread.png"),
    ])
    run([
        "python3", "scripts/paper_figures/make_robustness_spread.py",
        "--campaign-root", CURRENT_ROBUSTNESS,
        "--metrics", "/tmp/paper_vs_current_no_metrics.csv",
        "--gp", CURRENT_GP,
        "--config", CURRENT_CONFIG,
        "--out", str(CURRENT_FIG / "robustness_spread_current.png"),
    ])


def convert_pdf_preview(pdf: Path, png_base: Path) -> None:
    run(["pdftoppm", "-png", "-r", "150", "-singlefile", str(pdf), str(png_base)])


def regenerate_paired() -> None:
    for spec in PAIRED_RUNS:
        env = {
            "PAIRED_CAMP": spec["camp"],
            "PAIRED_TASK": spec["task"],
            "PAIRED_SEED": str(spec["seed"]),
            "PAIRED_GP": spec["gp"],
            "PAIRED_CONFIG": spec["config"],
            "PAIRED_OUT": spec["out"],
            "PAIRED_COPY_TO_THESIS": "0",
            "PAIRED_TITLE_A": spec["title_a"],
            "PAIRED_TITLE_B": spec["title_b"],
        }
        if spec.get("legacy_truth"):
            env["PAIRED_ALLOW_LEGACY_TRUTH"] = "1"
        run(["python3", "scripts/paper_figures/make_paired_mechanism.py"], env=env)

        # canonical pdf + provenance + _data stay at paper_artifacts/figures root;
        # only the png preview is rendered into the side directory.
        side_fig = PAPER_FIG if spec["side"] == "paper" else CURRENT_FIG
        side_fig.mkdir(parents=True, exist_ok=True)
        artifact = ROOT / "paper_artifacts/figures" / spec["out"]
        convert_pdf_preview(artifact.with_suffix(".pdf"), side_fig / spec["out"])


def regenerate_gifs() -> None:
    diag = ROOT / "scripts/visibility_comparison/diag/diag_route_animation.py"
    for side, name, campaign, task, seed in GIF_RUNS:
        if campaign == "honest_campaign_v1":
            c1 = run_dir_from_campaign(campaign, task, "C1", seed)
            c2 = run_dir_from_campaign(campaign, task, "C2", seed)
        else:
            c1 = latest_run_dir(campaign, task, "C1", seed)
            c2 = latest_run_dir(campaign, task, "C2", seed)
        out = (PAPER_FIG if side == "paper" else CURRENT_FIG) / name
        run([
            "python3", str(diag),
            f"C1={c1}",
            f"C2={c2}",
            "--out", str(out),
            "--frames", "70",
            "--fps", "10",
        ], cwd=diag.parent)


def main() -> int:
    regenerate_method_figures()
    regenerate_spreads()
    regenerate_paired()
    regenerate_gifs()
    print(f"\nRegenerated paper-vs-current figures under {rel(PAPER_FIG)} and {rel(CURRENT_FIG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
