#!/usr/bin/env python3
"""Create a C1-vs-selected-C2 Monte Carlo comparison config.

Typical flow:
  1. Run the single-seed C2 screening grid.
  2. Plot/rank the screening runs.
  3. Use this script with either --winner-label or --ranking-csv.
  4. Run the generated config with run_model_selection.py.

The generated comparison keeps the selected C2 parameters as the base model and
adds two cells:
  - C1_constant_R: same task/goal/horizon/noise, but planner=constant_R_efe
  - C2_<winner>: selected visibility-aware planner
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SCREENING_CONFIG = Path("scripts/visibility_comparison/c2_taskA_visibility_seek_config.yaml")
DEFAULT_RANKING_CSV = Path(
    "logs/visibility_comparison/c2_taskA_param_grid_v1/run_investigation/run_visibility_ranking.csv"
)
DEFAULT_OUT = Path("scripts/visibility_comparison/c2_taskA_mc_compare_config.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML is not a mapping: {path}")
    return data


def _parse_seeds(raw: str) -> list[int]:
    raw = raw.strip()
    if not raw:
        raise RuntimeError("--seeds cannot be empty")
    if ".." in raw:
        lo_s, hi_s = raw.split("..", 1)
        lo = int(lo_s)
        hi = int(hi_s)
        if hi < lo:
            raise RuntimeError(f"invalid seed range: {raw}")
        return list(range(lo, hi + 1))
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "selected"


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _find_cell(cells: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for cell in cells:
        if str(cell.get("label", "")) == label:
            return cell
    labels = ", ".join(str(cell.get("label", "")) for cell in cells)
    raise RuntimeError(f"winner label not found in screening config: {label!r}. Available: {labels}")


def _choose_from_ranking(path: Path, *, task: str) -> str:
    if not path.is_file():
        raise RuntimeError(f"ranking CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source") != "grid":
                continue
            if task and row.get("task") != task:
                continue
            if not _is_true(row.get("clean_success", "")):
                continue
            label = str(row.get("label", "")).strip()
            if label:
                return label
    raise RuntimeError(f"no clean-success grid row found in ranking CSV: {path}")


def _merged(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    merged.update(copy.deepcopy(overrides or {}))
    return merged


def build_config(
    screening_cfg: dict[str, Any],
    *,
    screening_config_label: str,
    winner_label: str,
    task: str,
    seeds: list[int],
) -> dict[str, Any]:
    base = copy.deepcopy(screening_cfg.get("base"))
    cells = screening_cfg.get("cells")
    if not isinstance(base, dict) or not isinstance(cells, list):
        raise RuntimeError("screening config must contain base and cells")

    winner_cell = _find_cell(cells, winner_label)
    winner_overrides = copy.deepcopy(winner_cell.get("overrides", {}) or {})
    if not winner_overrides:
        raise RuntimeError(f"winner cell has no overrides: {winner_label}")

    selected_base = _merged(base, winner_overrides)
    selected_base["task"] = task
    selected_base["planner"] = "visibility_aware_efe"
    selected_base["seeds"] = seeds
    if "expected_gp_beta" in winner_cell:
        selected_base["expected_gp_beta"] = winner_cell["expected_gp_beta"]

    safe = _safe_label(winner_label)
    return {
        "generated_from": {
            "screening_config": screening_config_label,
            "winner_label": winner_label,
            "winner_axis": winner_cell.get("axis", ""),
            "task": task,
        },
        "expected_cells": 2,
        "base": selected_base,
        "cells": [
            {
                "axis": "monte_carlo_compare",
                "label": "C1_constant_R",
                "overrides": {
                    "planner": "constant_R_efe",
                },
            },
            {
                "axis": "monte_carlo_compare",
                "label": f"C2_{safe}",
                "overrides": {
                    "planner": "visibility_aware_efe",
                },
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-config", default=str(DEFAULT_SCREENING_CONFIG))
    parser.add_argument("--ranking-csv", default=str(DEFAULT_RANKING_CSV))
    parser.add_argument("--winner-label", default="", help="Manual winner cell label from the screening config")
    parser.add_argument("--task", default="shadow_tradeoff_a")
    parser.add_argument("--seeds", default="0..19", help="Comma list or inclusive range, e.g. 0..19")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    screening_path = Path(args.screening_config).expanduser().resolve()
    ranking_path = Path(args.ranking_csv).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    cfg = _load_yaml(screening_path)
    winner = args.winner_label.strip() or _choose_from_ranking(ranking_path, task=args.task)
    seeds = _parse_seeds(args.seeds)
    out_cfg = build_config(
        cfg,
        screening_config_label=str(screening_path),
        winner_label=winner,
        task=args.task,
        seeds=seeds,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out_cfg, sort_keys=False), encoding="utf-8")

    print(f"Winner label: {winner}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} total)" if len(seeds) > 1 else f"Seeds: {seeds}")
    print(f"Wrote: {out_path}")
    print("")
    print("Dry-run it with:")
    print(f"  python3 scripts/visibility_comparison/run_model_selection.py --config {out_path} --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
