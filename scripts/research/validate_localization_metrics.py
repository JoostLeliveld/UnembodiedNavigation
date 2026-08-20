#!/usr/bin/env python3
"""Validate the canonical localization registry against its source artifacts.

The raw evidence under ``logs/`` is intentionally not tracked in every checkout. Missing
sources are warnings by default and failures with ``--require-sources``. Registry/schema and
known misleading active phrases always fail.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "docs/localization_metrics_registry.json"

BANNED_ACTIVE_PHRASES = (
    "camera C's real +78 mm",
    "Camera C carries its measured +78 mm",
    "Running with the deployed v2 calibration",
    "camera with a 77 mm lean can never",
    "The shipped pipeline carries **two** fitted parameters",
    "Break **camera C** first: it is the one with a real 77 mm",
    "honest to within 2 %",
    "honest to within 2%",
    "the two bars should match",
    "should ≈ actual",
    "claims 2.8× more precision than it has",
)

SCAN_ROOTS = (
    REPO / "CLAUDE.md",
    REPO / "docs",
    REPO / "research",
    REPO / "experiments",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(label: str, actual: float, expected: float, errors: list[str]) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
        errors.append(f"{label}: registry={actual!r}, source={expected!r}")


def validate_current(context: dict, source: dict, errors: list[str]) -> None:
    arm = source["arms"]["raw IPM (floor, no correction)"]
    if context["n"] != arm["n"]:
        errors.append(f"PG-IPM-CURRENT.n: registry={context['n']}, source={arm['n']}")
    mapping = {
        "mean_euclidean_error_m": "mean_m",
        "median_euclidean_error_m": "median_m",
        "p95_euclidean_error_m": "p95_m",
        "radial_bias_m": "radial_bias_m",
        "lateral_bias_m": "lateral_bias_m",
    }
    for target, source_key in mapping.items():
        close(f"PG-IPM-CURRENT.pooled.{target}", context["pooled"][target],
              arm[source_key], errors)
    for camera_id, registered in context["per_camera"].items():
        observed = arm["per_camera"][camera_id]
        if registered["n"] != observed["n"]:
            errors.append(
                f"PG-IPM-CURRENT.{camera_id}.n: registry={registered['n']}, "
                f"source={observed['n']}"
            )
        for target, source_key in mapping.items():
            close(f"PG-IPM-CURRENT.{camera_id}.{target}", registered[target],
                  observed[source_key], errors)


def validate_historical_measurement(context: dict, source: dict,
                                    errors: list[str]) -> None:
    if context["n"] != source["n_rows_scored"]:
        errors.append(
            f"MC-DRIVE-V2.n: registry={context['n']}, source={source['n_rows_scored']}"
        )
    registered = context["camera_C_historical"]
    camera_c = source["cross_bearing_gate"]["camera_C"]
    pixel = source["per_camera_all"]["camera_C"]
    pairs = (
        ("signed_cross_bearing_bias_m", camera_c["raw"]["bias_m"]),
        ("signed_cross_bearing_bias_after_mesh_m",
         camera_c["after_cad_model"]["bias_m"]),
        ("pixel_offset_u_contact_px", pixel["du_contact"]["mean"]),
        ("pixel_offset_u_after_mesh_px", pixel["du_mesh"]["mean"]),
    )
    for key, expected in pairs:
        close(f"MC-DRIVE-V2.camera_C_historical.{key}", registered[key], expected, errors)


def validate_belief(context: dict, source: dict, errors: list[str]) -> None:
    source_arms = {
        "trust_every_camera": source["pooled"]["A0_trust_everything"],
        "per_camera_correlation_floor": source["pooled"]["A4_correlation_floor"],
    }
    if context["n"] != source_arms["trust_every_camera"]["n"]:
        errors.append(
            f"BELIEF-V2.n: registry={context['n']}, "
            f"source={source_arms['trust_every_camera']['n']}"
        )
    direct = ("rmse_m", "p95_error_m", "mean_stated_sigma_m", "median_nees")
    for arm_name, registered in context["arms"].items():
        observed = source_arms[arm_name]
        for key in direct:
            close(f"BELIEF-V2.{arm_name}.{key}", registered[key], observed[key], errors)
        close(f"BELIEF-V2.{arm_name}.outside_95_fraction",
              registered["outside_95_fraction"], 1.0 - observed["coverage_95"], errors)


def iter_text_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if path.suffix in {".md", ".py", ".yaml", ".json"} and path.is_file():
            yield path


def validate_language(errors: list[str]) -> None:
    for root in SCAN_ROOTS:
        for path in iter_text_files(root):
            text = path.read_text(encoding="utf-8", errors="replace")
            for phrase in BANNED_ACTIVE_PHRASES:
                if phrase in text:
                    errors.append(f"misleading active phrase in {path.relative_to(REPO)}: {phrase}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-sources", action="store_true",
        help="fail when a raw source artifact under logs/ is unavailable",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    registry = load_json(REGISTRY)
    contexts = registry["contexts"]
    required = {"PG-IPM-CURRENT", "MC-DRIVE-V2", "BELIEF-V2", "HONEST-CAMPAIGN-V1"}
    missing = required - set(contexts)
    if missing:
        errors.append(f"registry is missing contexts: {sorted(missing)}")

    validators = {
        "PG-IPM-CURRENT": validate_current,
        "MC-DRIVE-V2": validate_historical_measurement,
        "BELIEF-V2": validate_belief,
    }
    for context_id, validator in validators.items():
        context = contexts.get(context_id)
        if context is None:
            continue
        source_path = REPO / context["source"]
        if not source_path.is_file():
            message = f"source unavailable: {source_path.relative_to(REPO)}"
            (errors if args.require_sources else warnings).append(message)
            continue
        validator(context, load_json(source_path), errors)

    validate_language(errors)

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"localization metrics contract valid ({len(contexts)} contexts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
