#!/usr/bin/env python3
"""Freeze the incremental static commissioning campaign.

The selector uses only frozen warehouse geometry and the legacy pose manifest.  It never
reads detector outputs, residuals, fitted corrections, or covariance results.  Legacy
commissioning positions are retained as development evidence when their complete
eight-heading footprint remains physically valid.  Fresh validation and final-audit
positions are selected on a 0.10 m lattice with spatially disjoint allocation rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle


REPO = Path(__file__).resolve().parents[2]
for relative in ("src/unav_common",):
    value = str((REPO / relative).resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

from unav_common.occlusion_geometry import (  # noqa: E402
    parse_collision_scene_from_world,
    signed_distance_to_union_xy,
)
from unav_common.rectangular_footprint import RectangularFootprint  # noqa: E402


WORLD = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
PROFILES = REPO / "src/experiments/config/world_profiles.yaml"
MASTER = REPO / "logs/thesis_final_pipeline_v1/master_capture/capture_manifest.json"
RUNTIME_AUDIT = REPO / (
    "logs/thesis_final_pipeline_v1/stage09_navigation/"
    "runtime_yaw_recovery_audit_v1/report.json"
)
COLLISION_INCLUDES = (
    "forklift_parked",
    "pallet_jack",
    "bin_office",
    "pallet_loose_1",
    "pallet_loose_2",
)
VALIDATION_GENERAL_BLOCKS = (
    "B00_04", "B01_02", "B02_02", "B03_01", "B04_01",
    "B04_02", "B05_02", "B05_05", "B07_00", "B07_03",
)
AUDIT_GENERAL_BLOCKS = (
    "B00_03", "B01_04", "B02_01", "B04_03", "B07_04",
)
LEGACY_COMMISSIONING_ROLES = ("commissioning_fit", "final_audit")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def block_id(x: float, y: float) -> str:
    return f"B{math.floor((x + 12.0) / 3.2):02d}_{math.floor((y + 9.6) / 3.2):02d}"


def edge_group_id(x: float, y: float) -> str:
    """A 1.6 m geometry-only tile used to keep new edge holdouts disjoint."""
    ix = math.floor((x + 12.0) / 1.6)
    iy = math.floor((y + 9.6) / 1.6)
    return f"E{ix:02d}_{iy:02d}"


def edge_group_partition(group: str) -> str:
    ix, iy = (int(value) for value in group[1:].split("_"))
    return "commissioning_validation" if (ix + iy) % 2 == 0 else "final_audit"


def inside_traversable(
    points: np.ndarray, regions: list[dict], *, shrink_m: float = 0.0
) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    for region in regions:
        mask |= (
            (points[:, 0] >= float(region["xmin"]) + shrink_m)
            & (points[:, 0] <= float(region["xmax"]) - shrink_m)
            & (points[:, 1] >= float(region["ymin"]) + shrink_m)
            & (points[:, 1] <= float(region["ymax"]) - shrink_m)
        )
    return mask


def outside_expanded_regions(
    points: np.ndarray, regions: list[dict], *, expansion_m: float
) -> np.ndarray:
    valid = np.ones(len(points), dtype=bool)
    for region in regions:
        inside = (
            (points[:, 0] >= float(region["xmin"]) - expansion_m)
            & (points[:, 0] <= float(region["xmax"]) + expansion_m)
            & (points[:, 1] >= float(region["ymin"]) - expansion_m)
            & (points[:, 1] <= float(region["ymax"]) + expansion_m)
        )
        valid &= ~inside
    return valid


def minimum_distance(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    if len(references) == 0:
        return np.full(len(points), np.inf, dtype=float)
    result = np.full(len(points), np.inf, dtype=float)
    # Chunking avoids a large candidate x reference temporary array.
    for start in range(0, len(points), 2048):
        chunk = points[start:start + 2048]
        result[start:start + len(chunk)] = np.min(
            np.linalg.norm(chunk[:, None, :] - references[None, :, :], axis=2),
            axis=1,
        )
    return result


def farthest_select(
    candidates: list[dict],
    count: int,
    references: list[tuple[float, float]],
    *,
    minimum_new_spacing_m: float,
) -> list[dict]:
    """Deterministic farthest-point selection with lexicographic tie breaking."""
    remaining = list(candidates)
    selected: list[dict] = []
    reference_array = np.asarray(references, dtype=float).reshape((-1, 2))
    while len(selected) < count:
        if not remaining:
            raise RuntimeError(f"candidate pool exhausted after {len(selected)} of {count}")
        points = np.asarray([(row["x"], row["y"]) for row in remaining], dtype=float)
        distances = minimum_distance(points, reference_array)
        eligible = np.flatnonzero(distances >= minimum_new_spacing_m - 1e-12)
        if not len(eligible):
            raise RuntimeError(
                f"spacing {minimum_new_spacing_m:.3f} m exhausted the pool after "
                f"{len(selected)} of {count}"
            )
        best_distance = float(np.max(distances[eligible]))
        tied = [
            int(index) for index in eligible
            if math.isclose(float(distances[index]), best_distance, abs_tol=1e-12)
        ]
        best = min(tied, key=lambda index: (remaining[index]["x"], remaining[index]["y"]))
        chosen = remaining.pop(best)
        selected.append(chosen)
        point = np.asarray([[chosen["x"], chosen["y"]]], dtype=float)
        reference_array = np.concatenate((reference_array, point), axis=0)
    return selected


def classify_legacy(
    manifest: dict,
    footprint: RectangularFootprint,
    traversable_regions: list[dict],
    *,
    required_clearance_m: float,
    edge_max_clearance_m: float,
) -> tuple[list[dict], list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in manifest["pose_plan"]:
        if row["dataset_split"] in LEGACY_COMMISSIONING_ROLES:
            grouped[int(row["position_id"])].append(row)
    valid: list[dict] = []
    invalid: list[dict] = []
    for position_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["yaw_idx"]))
        x, y = float(rows[0]["x"]), float(rows[0]["y"])
        centre_valid = bool(inside_traversable(np.asarray([[x, y]]), traversable_regions)[0])
        clearances = [footprint.clearance((x, y, float(row["yaw"]))) for row in rows]
        minimum = float(min(clearances))
        item = {
            "position_id": position_id,
            "position_key": str(rows[0].get("position_key", f"P{position_id:04d}")),
            "legacy_role": str(rows[0]["dataset_split"]),
            "x": x,
            "y": y,
            "block_id": str(rows[0].get("block_id", block_id(x, y))),
            "minimum_body_clearance_m": minimum,
            "heading_count": len(rows),
        }
        if centre_valid and minimum >= required_clearance_m - 1e-12 and len(rows) == 8:
            item["coverage_class"] = (
                "edge" if minimum <= edge_max_clearance_m + 1e-12 else "general"
            )
            valid.append(item)
        else:
            reasons = []
            if not centre_valid:
                reasons.append("centre_outside_declared_traversable_union")
            if minimum < required_clearance_m - 1e-12:
                reasons.append("eight_heading_body_clearance_below_requirement")
            if len(rows) != 8:
                reasons.append("incomplete_heading_set")
            item["exclusion_reasons"] = reasons
            invalid.append(item)
    return valid, invalid


def make_candidate_pool(
    *,
    footprint: RectangularFootprint,
    collision_prisms: tuple,
    traversable_regions: list[dict],
    excluded_regions: list[dict],
    all_legacy_xy: np.ndarray,
    grid_step_m: float,
    legacy_exclusion_m: float,
    required_clearance_m: float,
    edge_max_clearance_m: float,
) -> list[dict]:
    xmin = min(float(row["xmin"]) for row in traversable_regions)
    xmax = max(float(row["xmax"]) for row in traversable_regions)
    ymin = min(float(row["ymin"]) for row in traversable_regions)
    ymax = max(float(row["ymax"]) for row in traversable_regions)
    xs = np.round(np.arange(xmin, xmax + 0.5 * grid_step_m, grid_step_m), 6)
    ys = np.round(np.arange(ymin, ymax + 0.5 * grid_step_m, grid_step_m), 6)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack((gx.ravel(), gy.ravel()))
    # Mirror the capture entry point's declared-region preflight as well as the
    # stronger full-body physical check below.
    traversable = inside_traversable(
        points, traversable_regions, shrink_m=required_clearance_m
    )
    outside_excluded = outside_expanded_regions(
        points, excluded_regions, expansion_m=required_clearance_m
    )
    separated = minimum_distance(points, all_legacy_xy) >= legacy_exclusion_m - 1e-12
    centre_clearance = signed_distance_to_union_xy(collision_prisms, points, keep_in=False)
    conservative_body_clearance = centre_clearance - footprint.radius
    valid = traversable & outside_excluded & separated & (
        conservative_body_clearance >= required_clearance_m - 1e-12
    )
    pool = []
    for (x, y), clearance in zip(points[valid], conservative_body_clearance[valid], strict=True):
        group = edge_group_id(float(x), float(y))
        pool.append({
            "x": float(x),
            "y": float(y),
            "block_id": block_id(float(x), float(y)),
            "edge_group_id": group,
            "edge_group_partition": edge_group_partition(group),
            "minimum_body_clearance_m": float(clearance),
            "coverage_class": (
                "edge" if clearance <= edge_max_clearance_m + 1e-12 else "general"
            ),
        })
    return pool


def expand_poses(positions: list[dict], headings: list[float], *, start_id: int) -> list[dict]:
    poses = []
    for offset, position in enumerate(positions):
        position_id = start_id + offset
        prefix = "V" if position["stratum"] == "commissioning_validation" else "A"
        for heading_id, yaw in enumerate(headings):
            poses.append({
                "x": position["x"],
                "y": position["y"],
                "yaw": yaw,
                "position_id": position_id,
                "position_key": f"{prefix}{position_id:04d}",
                "x_idx": position_id,
                "y_idx": position_id,
                "heading_id": heading_id,
                "heading_degrees": round(math.degrees(yaw) % 360.0, 6),
                "block_id": position["block_id"],
                "stratum": position["stratum"],
                "coverage_class": position["coverage_class"],
                "edge_group_id": position["edge_group_id"],
            })
    return poses


def write_plot(
    path: Path,
    collision_prisms: tuple,
    legacy_valid: list[dict],
    legacy_invalid: list[dict],
    validation: list[dict],
    audit: list[dict],
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 8.2), constrained_layout=True)
    for prism in collision_prisms:
        ax.add_patch(Rectangle(
            (prism.xmin, prism.ymin), prism.xmax - prism.xmin, prism.ymax - prism.ymin,
            facecolor="#d8dde1", edgecolor="#aab2b8", linewidth=0.35, zorder=0,
        ))

    def scatter(rows, **kwargs):
        ax.scatter([row["x"] for row in rows], [row["y"] for row in rows], **kwargs)

    scatter([r for r in legacy_valid if r["coverage_class"] == "general"],
            s=16, color="#7c8791", alpha=0.62, label="Legacy development: general")
    scatter([r for r in legacy_valid if r["coverage_class"] == "edge"],
            s=26, marker="s", color="#48545e", alpha=0.85,
            label="Legacy development: edge")
    scatter([r for r in validation if r["coverage_class"] == "general"],
            s=35, marker="o", color="#238e83", edgecolor="white", linewidth=0.35,
            label="Fresh validation: general")
    scatter([r for r in validation if r["coverage_class"] == "edge"],
            s=46, marker="D", color="#146f67", edgecolor="white", linewidth=0.35,
            label="Fresh validation: edge")
    scatter([r for r in audit if r["coverage_class"] == "general"],
            s=38, marker="^", color="#d79c36", edgecolor="white", linewidth=0.35,
            label="Fresh sealed audit: general")
    scatter([r for r in audit if r["coverage_class"] == "edge"],
            s=48, marker="P", color="#a86818", edgecolor="white", linewidth=0.35,
            label="Fresh sealed audit: edge")
    scatter(legacy_invalid, s=48, marker="x", color="#bd3f3f", linewidth=1.2,
            label="Legacy excluded by current physical geometry")
    ax.set(
        xlim=(-12.2, 12.2), ylim=(-10.1, 10.1), aspect="equal",
        xlabel="World x [m]", ylabel="World y [m]",
        title="Incremental static commissioning: retained evidence and fresh holdouts",
    )
    ax.grid(alpha=0.12, linewidth=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--grid-step-m", type=float, default=0.10)
    parser.add_argument("--legacy-exclusion-m", type=float, default=0.28)
    parser.add_argument("--new-spacing-m", type=float, default=0.30)
    parser.add_argument("--required-clearance-m", type=float, default=0.05)
    parser.add_argument("--edge-max-clearance-m", type=float, default=0.35)
    args = parser.parse_args()

    profile_payload = yaml.safe_load(PROFILES.read_text(encoding="utf-8"))
    profile = profile_payload["worlds"][WORLD.name]
    traversable_regions = [
        row for row in profile["known_2d_regions"]
        if str(row.get("type", "")).strip().lower() == "traversable"
    ]
    excluded_regions = [
        row for row in profile["known_2d_regions"]
        if str(row.get("type", "")).strip().lower()
        not in {"traversable", "site_boundary"}
    ]
    scene = parse_collision_scene_from_world(
        str(WORLD),
        model_names=tuple(profile["collision_model_names"]),
        include_names=COLLISION_INCLUDES,
        robot_z_range=(0.0, 0.55),
    )
    footprint = RectangularFootprint(scene.prisms, length=0.80, width=0.55)
    master = load_json(MASTER)
    runtime_audit = load_json(RUNTIME_AUDIT)
    if not bool(runtime_audit.get("final_audit_previously_accessed")):
        raise RuntimeError("expected the legacy final audit to be recorded as previously accessed")

    legacy_valid, legacy_invalid = classify_legacy(
        master, footprint, traversable_regions,
        required_clearance_m=args.required_clearance_m,
        edge_max_clearance_m=args.edge_max_clearance_m,
    )
    if len(legacy_valid) != 270 or len(legacy_invalid) != 10:
        raise RuntimeError(
            f"legacy geometry changed: expected 270 valid and 10 invalid, got "
            f"{len(legacy_valid)} and {len(legacy_invalid)}"
        )

    all_legacy: dict[int, dict] = {}
    for row in master["pose_plan"]:
        all_legacy.setdefault(int(row["position_id"]), row)
    all_legacy_xy = np.asarray(
        [(float(row["x"]), float(row["y"])) for row in all_legacy.values()], dtype=float
    )
    pool = make_candidate_pool(
        footprint=footprint,
        collision_prisms=scene.prisms,
        traversable_regions=traversable_regions,
        excluded_regions=excluded_regions,
        all_legacy_xy=all_legacy_xy,
        grid_step_m=args.grid_step_m,
        legacy_exclusion_m=args.legacy_exclusion_m,
        required_clearance_m=args.required_clearance_m,
        edge_max_clearance_m=args.edge_max_clearance_m,
    )

    validation_general_pool = [
        row for row in pool
        if row["coverage_class"] == "general" and row["block_id"] in VALIDATION_GENERAL_BLOCKS
    ]
    audit_general_pool = [
        row for row in pool
        if row["coverage_class"] == "general" and row["block_id"] in AUDIT_GENERAL_BLOCKS
    ]
    validation_edge_pool = [
        row for row in pool
        if row["coverage_class"] == "edge"
        and row["edge_group_partition"] == "commissioning_validation"
    ]
    audit_edge_pool = [
        row for row in pool
        if row["coverage_class"] == "edge" and row["edge_group_partition"] == "final_audit"
    ]
    references = [(float(row["x"]), float(row["y"])) for row in all_legacy.values()]
    validation_general = farthest_select(
        validation_general_pool, 84, references,
        minimum_new_spacing_m=args.new_spacing_m,
    )
    references += [(row["x"], row["y"]) for row in validation_general]
    validation_edge = farthest_select(
        validation_edge_pool, 21, references,
        minimum_new_spacing_m=args.new_spacing_m,
    )
    references += [(row["x"], row["y"]) for row in validation_edge]
    audit_general = farthest_select(
        audit_general_pool, 84, references,
        minimum_new_spacing_m=args.new_spacing_m,
    )
    references += [(row["x"], row["y"]) for row in audit_general]
    audit_edge = farthest_select(
        audit_edge_pool, 21, references,
        minimum_new_spacing_m=args.new_spacing_m,
    )

    validation = validation_general + validation_edge
    audit = audit_general + audit_edge
    for row in validation:
        row["stratum"] = "commissioning_validation"
    for row in audit:
        row["stratum"] = "final_audit"

    validation_groups = {row["edge_group_id"] for row in validation_edge}
    audit_groups = {row["edge_group_id"] for row in audit_edge}
    if validation_groups & audit_groups:
        raise RuntimeError("edge validation and audit groups overlap")
    new_xy = np.asarray([(r["x"], r["y"]) for r in validation + audit], dtype=float)
    off_diagonal = np.linalg.norm(new_xy[:, None, :] - new_xy[None, :, :], axis=2)
    off_diagonal += np.eye(len(new_xy)) * 1e9
    observed_new_spacing = float(np.min(off_diagonal))
    if observed_new_spacing < args.new_spacing_m - 1e-12:
        raise RuntimeError("new-position spacing contract was violated")

    first_position_rows: dict[int, dict] = {}
    for row in master["pose_plan"]:
        first_position_rows.setdefault(int(row["position_id"]), row)
    heading_source_id = min(first_position_rows)
    headings = [
        float(row["yaw"])
        for row in sorted(
            [r for r in master["pose_plan"] if int(r["position_id"]) == heading_source_id],
            key=lambda row: int(row["yaw_idx"]),
        )
    ]
    validation_poses = expand_poses(validation, headings, start_id=400)
    audit_poses = expand_poses(audit, headings, start_id=505)

    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    validation_path = out / "static_incremental_validation_poses_v1.json"
    audit_path = out / "static_incremental_audit_poses_v1.json"
    validation_path.write_text(json.dumps({
        "schema": "static_commissioning_pose_plan.v1",
        "role": "commissioning_validation",
        "positions": validation,
        "poses": validation_poses,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps({
        "schema": "static_commissioning_pose_plan.v1",
        "role": "final_audit",
        "sealed_until": "all_static_model_choices_and_thresholds_are_locked",
        "positions": audit,
        "poses": audit_poses,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    legacy_counts = Counter(row["coverage_class"] for row in legacy_valid)
    validation_counts = Counter(row["coverage_class"] for row in validation)
    audit_counts = Counter(row["coverage_class"] for row in audit)
    protocol = {
        "schema": "static_incremental_commissioning_protocol.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_capture",
        "purpose": (
            "Fit the static commissioned observation model across the warehouse, validate "
            "model and filtering choices on fresh spatial support, and reserve a fresh audit."
        ),
        "source_contract": {
            "world": {"path": str(WORLD), "sha256": sha256(WORLD)},
            "world_profiles": {"path": str(PROFILES), "sha256": sha256(PROFILES)},
            "legacy_master_capture": {"path": str(MASTER), "sha256": sha256(MASTER)},
            "legacy_audit_access_report": {
                "path": str(RUNTIME_AUDIT), "sha256": sha256(RUNTIME_AUDIT),
                "final_audit_previously_accessed": True,
            },
        },
        "geometry_contract": {
            "collision_model_names": list(profile["collision_model_names"]),
            "collision_include_names": list(COLLISION_INCLUDES),
            "collision_prism_count": len(scene.prisms),
            "robot_footprint_m": {"length": 0.80, "width": 0.55},
            "legacy_validity": "exact_oriented_rectangle_at_each_of_the_8_captured_headings",
            "new_candidate_validity": "conservative_circumscribed_radius_at_all_headings",
            "required_body_clearance_m": args.required_clearance_m,
            "edge_body_clearance_interval_m": [args.required_clearance_m, args.edge_max_clearance_m],
            "candidate_grid_step_m": args.grid_step_m,
        },
        "partition_contract": {
            "legacy_development": {
                "positions": len(legacy_valid),
                "general": legacy_counts["general"],
                "edge": legacy_counts["edge"],
                "legacy_roles_reclassified_as_development": list(LEGACY_COMMISSIONING_ROLES),
            },
            "fresh_validation": {
                "positions": len(validation),
                "general": validation_counts["general"],
                "edge": validation_counts["edge"],
                "general_blocks": list(VALIDATION_GENERAL_BLOCKS),
                "edge_group_rule": "1.6 m tiles with even (ix + iy)",
            },
            "fresh_sealed_audit": {
                "positions": len(audit),
                "general": audit_counts["general"],
                "edge": audit_counts["edge"],
                "general_blocks": list(AUDIT_GENERAL_BLOCKS),
                "edge_group_rule": "1.6 m tiles with odd (ix + iy)",
            },
            "final_total": {
                "positions": len(legacy_valid) + len(validation) + len(audit),
                "general": legacy_counts["general"] + validation_counts["general"] + audit_counts["general"],
                "edge": legacy_counts["edge"] + validation_counts["edge"] + audit_counts["edge"],
                "headings_per_position": len(headings),
                "cameras_per_pose": 5,
                "opportunities": (len(legacy_valid) + len(validation) + len(audit)) * len(headings) * 5,
            },
        },
        "selection_contract": {
            "uses_observed_detector_or_residual_outputs": False,
            "minimum_distance_from_any_legacy_position_m": args.legacy_exclusion_m,
            "minimum_distance_between_new_positions_m": args.new_spacing_m,
            "observed_minimum_distance_between_new_positions_m": observed_new_spacing,
            "general_holdout_unit": "whole_3.2_m_coarse_block",
            "edge_holdout_unit": "whole_1.6_m_edge_tile",
            "selection_within_pool": "deterministic_farthest_point_then_lexicographic_tie_break",
        },
        "legacy_exclusions": legacy_invalid,
        "fresh_pose_files": {
            "validation": {"path": str(validation_path), "sha256": sha256(validation_path)},
            "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        },
        "audit_firewall": {
            "capture_separately": True,
            "do_not_run_detector_fit_model_selection_or_filter_selection_on_audit": True,
            "open_only_after_static_correction_R_and_filtering_choices_are_frozen": True,
        },
    }
    protocol_path = out / "static_incremental_campaign_v1.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.plot_output is not None:
        write_plot(
            args.plot_output.expanduser().resolve(), scene.prisms,
            legacy_valid, legacy_invalid, validation, audit,
        )
    print(json.dumps({
        "protocol": str(protocol_path),
        "validation_pose_file": str(validation_path),
        "audit_pose_file": str(audit_path),
        "legacy_valid": len(legacy_valid),
        "legacy_invalid": len(legacy_invalid),
        "fresh_validation": dict(validation_counts),
        "fresh_audit": dict(audit_counts),
        "candidate_pool": dict(Counter(row["coverage_class"] for row in pool)),
        "new_minimum_spacing_m": observed_new_spacing,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
