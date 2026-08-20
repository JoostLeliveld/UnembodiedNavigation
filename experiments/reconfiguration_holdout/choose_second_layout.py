#!/usr/bin/env python3
"""Enumerate, externally randomise, and freeze the L2 warehouse layout.

L1 was selected greedily for geometric impact. Repeating that optimisation would
not be an independent replication, while choosing a convenient local seed would
only move the researcher degree of freedom. L2 therefore uses a two-stage rule:

1. Before requesting randomness, enumerate the complete eligible design space.
   A layout has exactly two segments in every west/east x south/middle/north
   stratum: 12 of the 27 structural rack segments, each raised by 0.40 m.
2. After the preregistration and eligibility artifact are frozen, map a public
   NIST Randomness Beacon pulse to one lexicographically enumerated layout using
   the fixed SHA-256/index rule below.

The selection path reads rack names and geometry only. Detector outcomes, captured
images, learned fields, route results, and visibility-impact scores are forbidden.
The geometry oracle runs only *after* the random index is fixed, to document how
strong the selected intervention is and verify floor occupancy is unchanged.

Typical use::

    # Before obtaining the public pulse:
    python3 choose_second_layout.py --enumerate-eligible

    # After freezing SECOND_RECONFIGURATION_PREREGISTRATION.md, save the public
    # beacon response + URL in a small JSON wrapper, then:
    python3 choose_second_layout.py --beacon-record /path/to/beacon_record.json

``--seed`` exposes the exact deterministic mapping for tests and audits, but cannot
write the canonical L2 artifact. Only a persisted beacon record may do that.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, islice, product
import json
from math import prod
from pathlib import Path
import re
from typing import Iterator

import numpy as np

import choose_layout as LAYOUT1


PROTOCOL_LABEL = "reconfiguration-holdout-L2-layout-v1"
STOCK_HEIGHT_M = 0.40
PICKS_PER_STRATUM = 2
HERE = Path(__file__).resolve().parent
OUT = HERE / "layouts/L2_layout.json"
ELIGIBILITY_OUT = HERE / "layouts/L2_eligibility.json"
FIRST_LAYOUT = (
    LAYOUT1.REPO / "logs/studies/reconfiguration_holdout/layout/layout_selected.json"
)
NAME_RE = re.compile(r"^rack_([WE])\d+_(south|mid|north)$")
STRATA = tuple(
    (side, band)
    for band in ("south", "mid", "north")
    for side in ("W", "E")
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segment_stratum(segment: dict) -> tuple[str, str]:
    match = NAME_RE.fullmatch(str(segment["name"]))
    if match is None:
        raise ValueError(f"unrecognised structural rack name: {segment['name']!r}")
    return match.group(1), match.group(2)


def eligible_groups(segments: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups = {stratum: [] for stratum in STRATA}
    names: set[str] = set()
    for segment in segments:
        name = str(segment["name"])
        if name in names:
            raise ValueError(f"duplicate rack segment {name!r}")
        names.add(name)
        groups[segment_stratum(segment)].append(segment)
    for stratum, candidates in groups.items():
        candidates.sort(key=lambda segment: str(segment["name"]))
        if len(candidates) < PICKS_PER_STRATUM:
            raise ValueError(
                f"stratum {stratum!r} has {len(candidates)} candidates, "
                f"needs {PICKS_PER_STRATUM}"
            )
    return groups


def stratum_choices(
    groups: dict[tuple[str, str], list[dict]],
) -> list[list[tuple[str, ...]]]:
    return [
        list(
            combinations(
                (str(segment["name"]) for segment in groups[stratum]),
                PICKS_PER_STRATUM,
            )
        )
        for stratum in STRATA
    ]


def enumerate_layout_names(
    groups: dict[tuple[str, str], list[dict]],
) -> Iterator[tuple[str, ...]]:
    """All eligible layouts in the exact order used by the external draw."""

    for per_stratum in product(*stratum_choices(groups)):
        yield tuple(name for choice in per_stratum for name in choice)


def eligibility_spec(segments: list[dict]) -> dict:
    groups = eligible_groups(segments)
    choices = stratum_choices(groups)
    n_layouts = prod(len(values) for values in choices)
    enumeration_digest = hashlib.sha256()
    observed = 0
    first_layout = last_layout = None
    for layout in enumerate_layout_names(groups):
        if first_layout is None:
            first_layout = layout
        last_layout = layout
        enumeration_digest.update((_canonical_json(list(layout)) + "\n").encode("utf-8"))
        observed += 1
    if observed != n_layouts:
        raise AssertionError("factored eligibility count disagrees with enumeration")

    candidate_rows = [
        {
            key: segment[key]
            for key in ("name", "xmin", "xmax", "ymin", "ymax", "top_z")
        }
        for segment in sorted(segments, key=lambda value: str(value["name"]))
    ]
    return {
        "schema_version": 1,
        "protocol_label": PROTOCOL_LABEL,
        "base_world": str(LAYOUT1.BASE_WORLD.relative_to(LAYOUT1.REPO)),
        "base_world_sha256": _file_sha256(LAYOUT1.BASE_WORLD),
        "stock_height_m": STOCK_HEIGHT_M,
        "selection_constraints": {
            "strata_order": [list(value) for value in STRATA],
            "picks_per_stratum": PICKS_PER_STRATUM,
            "total_segments": len(STRATA) * PICKS_PER_STRATUM,
            "within_stratum_order": "lexicographic segment name",
            "layout_order": "Cartesian product in strata_order; flatten each tuple",
        },
        "candidate_count": len(candidate_rows),
        "candidate_universe": candidate_rows,
        "candidate_universe_sha256": _sha256_text(_canonical_json(candidate_rows)),
        "factored_enumeration": [
            {
                "stratum": list(stratum),
                "candidates": [str(segment["name"]) for segment in groups[stratum]],
                "eligible_choices": [list(value) for value in choices[index]],
            }
            for index, stratum in enumerate(STRATA)
        ],
        "eligible_layout_count": n_layouts,
        "eligible_layout_enumeration_sha256": enumeration_digest.hexdigest(),
        "first_eligible_layout": list(first_layout or ()),
        "last_eligible_layout": list(last_layout or ()),
        "selection_mapping": {
            "digest_preimage": "protocol_label + NUL + lowercase external seed",
            "digest": "SHA-256",
            "index": "big-endian digest integer modulo eligible_layout_count",
        },
        "selection_forbidden_inputs": [
            "detector outcomes",
            "captured images",
            "learned availability fields",
            "route outcomes",
            "visibility-impact scores",
        ],
    }


def render(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def freeze_or_check(path: Path, rendered: str, *, check: bool, label: str) -> None:
    if check:
        if not path.is_file():
            raise SystemExit(f"{label} is not frozen at {path}")
        if path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{label} at {path} differs from deterministic regeneration")
        print(f"[{label}] PASS: {path}")
        return
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"refusing to replace frozen {label} with different bytes: {path}")
        print(f"[{label}] already frozen and identical: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(f"[{label}] wrote {path}")


def _find_values(value: object, normalised_key: str) -> list[object]:
    found: list[object] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_norm = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if key_norm == normalised_key:
                found.append(child)
            found.extend(_find_values(child, normalised_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_values(child, normalised_key))
    return found


def beacon_seed(path: Path) -> tuple[str, dict]:
    """Extract one NIST pulse output while preserving complete public provenance."""

    raw = path.read_bytes()
    record = json.loads(raw)
    if not isinstance(record, dict):
        raise ValueError("beacon record must be a JSON object")
    source_url = str(record.get("source_url", ""))
    if not source_url.startswith("https://beacon.nist.gov/"):
        raise ValueError("beacon record source_url must be an HTTPS beacon.nist.gov URL")
    if "pulse" not in record or not isinstance(record["pulse"], dict):
        raise ValueError("beacon record must preserve the official response under 'pulse'")

    outputs = {
        str(value).strip().lower()
        for value in _find_values(record["pulse"], "outputvalue")
        if isinstance(value, str) and str(value).strip()
    }
    if len(outputs) != 1:
        raise ValueError(f"beacon pulse must contain exactly one outputValue, found {len(outputs)}")
    seed = next(iter(outputs))
    if len(seed) < 64 or len(seed) % 2 or re.fullmatch(r"[0-9a-f]+", seed) is None:
        raise ValueError("beacon outputValue must be an even-length hexadecimal value >= 256 bits")
    provenance = {
        "source_type": "NIST Randomness Beacon 2.0",
        "source_url": source_url,
        "record_file_sha256": _sha256_bytes(raw),
        "output_value_sha256": _sha256_text(seed),
        "record": record,
    }
    return seed, provenance


def select_layout_names(eligibility: dict, groups, seed: str) -> tuple[tuple[str, ...], dict]:
    seed_canonical = str(seed).strip().lower()
    if not seed_canonical:
        raise ValueError("external seed must not be empty")
    digest = _sha256_text(f"{PROTOCOL_LABEL}\0{seed_canonical}")
    n_layouts = int(eligibility["eligible_layout_count"])
    index = int(digest, 16) % n_layouts
    selected = next(islice(enumerate_layout_names(groups), index, index + 1), None)
    if selected is None:
        raise AssertionError(f"selected layout index {index} was not enumerable")
    return selected, {
        "protocol_label": PROTOCOL_LABEL,
        "seed_canonicalisation": "strip surrounding whitespace; lowercase",
        "selection_digest_sha256": digest,
        "eligible_layout_count": n_layouts,
        "selected_zero_based_index": index,
    }


def build_layout(*, seed: str, randomness: dict, eligibility_path: Path) -> dict:
    """Return the selected L2 design and post-selection geometry audit."""

    segments = LAYOUT1.rack_segments(LAYOUT1.BASE_WORLD)
    groups = eligible_groups(segments)
    eligibility = eligibility_spec(segments)
    if not eligibility_path.is_file():
        raise ValueError(f"eligibility artifact is not frozen: {eligibility_path}")
    if eligibility_path.read_text(encoding="utf-8") != render(eligibility):
        raise ValueError("eligibility artifact differs from current deterministic enumeration")

    selected_names_ordered, mapping = select_layout_names(eligibility, groups, seed)
    by_name = {str(segment["name"]): segment for segment in segments}
    selected = [by_name[name] for name in selected_names_ordered]
    selected_names = set(selected_names_ordered)

    # Only now, after the external draw is fixed, evaluate geometry strength.
    scene = LAYOUT1.ora.OracleScene.from_world(
        LAYOUT1.BASE_WORLD, list(LAYOUT1.CAMERAS)
    )
    grid = LAYOUT1.ora.FloorGrid(
        xmin=-11.75,
        xmax=11.75,
        ymin=-9.0,
        ymax=9.0,
        resolution_m=LAYOUT1.GRID_RES_M,
    )
    drive = LAYOUT1.driveable_mask(grid, LAYOUT1.lanes())
    base = LAYOUT1.visible_stack(scene, grid, [])
    eligible_cells = drive & ~LAYOUT1.occupied_any(base)
    base_visible = {
        camera: (base[camera] == LAYOUT1.ora.VISIBLE) & eligible_cells
        for camera in LAYOUT1.CAMERAS
    }
    base_seen = LAYOUT1.fused_seen(base) & eligible_cells
    prisms = [LAYOUT1.stock_prism(segment, STOCK_HEIGHT_M) for segment in selected]
    changed = LAYOUT1.visible_stack(scene, grid, prisms)
    changed_visible = {
        camera: (changed[camera] == LAYOUT1.ora.VISIBLE) & eligible_cells
        for camera in LAYOUT1.CAMERAS
    }
    changed_seen = LAYOUT1.fused_seen(changed) & eligible_cells
    if not np.array_equal(LAYOUT1.occupied_any(base), LAYOUT1.occupied_any(changed)):
        raise AssertionError("L2 changes target-height floor occupancy")
    lowest = min(prism.zmin for prism in prisms)
    if lowest <= 0.5:
        raise AssertionError(f"L2 stock reaches the robot plane at z={lowest:.3f} m")

    first_names: set[str] = set()
    first_layout_sha256 = None
    if FIRST_LAYOUT.is_file():
        first_layout_sha256 = _file_sha256(FIRST_LAYOUT)
        first = json.loads(FIRST_LAYOUT.read_text(encoding="utf-8"))
        first_names = {str(value["name"]) for value in first["restocked_segments"]}

    selected_rows = []
    for segment in selected:
        row = {
            key: segment[key]
            for key in ("name", "xmin", "xmax", "ymin", "ymax", "top_z")
        }
        row.update(
            {
                "stock_height_m": STOCK_HEIGHT_M,
                "selection_stratum": list(segment_stratum(segment)),
            }
        )
        selected_rows.append(row)

    return {
        "schema_version": 2,
        "environment_key": "L2",
        "world_name": "warehouse_full_4cam_recfg2",
        "base_world": str(LAYOUT1.BASE_WORLD.relative_to(LAYOUT1.REPO)),
        "change": "externally randomised balanced rack restock",
        "stock_height_m": STOCK_HEIGHT_M,
        "selection": {
            "outcome_blind": True,
            "eligibility_artifact": str(eligibility_path.relative_to(LAYOUT1.REPO)),
            "eligibility_artifact_sha256": _file_sha256(eligibility_path),
            "eligible_layout_enumeration_sha256": eligibility[
                "eligible_layout_enumeration_sha256"
            ],
            **mapping,
            "external_randomness": randomness,
            "forbidden_inputs": eligibility["selection_forbidden_inputs"],
            "geometry_strength_evaluated_after_selection": True,
        },
        "grid": grid.to_dict(),
        "target_height_m": LAYOUT1.TARGET_HEIGHT_M,
        "spawn_xy": list(LAYOUT1.spawn_xy()),
        "eligible_cells": int(eligible_cells.sum()),
        "fused_coverage_L0": int(base_seen.sum()),
        "fused_coverage_L2": int(changed_seen.sum()),
        "fused_cells_lost": int((base_seen & ~changed_seen).sum()),
        "fused_cells_gained": int((~base_seen & changed_seen).sum()),
        "camera_cell_pairs_lost": int(
            sum(
                int((base_visible[camera] & ~changed_visible[camera]).sum())
                for camera in LAYOUT1.CAMERAS
            )
        ),
        "camera_cell_pairs_gained": int(
            sum(
                int((~base_visible[camera] & changed_visible[camera]).sum())
                for camera in LAYOUT1.CAMERAS
            )
        ),
        "per_camera_visible_L0": {
            LAYOUT1.SHORT[camera]: int(base_visible[camera].sum())
            for camera in LAYOUT1.CAMERAS
        },
        "per_camera_visible_L2": {
            LAYOUT1.SHORT[camera]: int(changed_visible[camera].sum())
            for camera in LAYOUT1.CAMERAS
        },
        "floor_occupancy_identical": True,
        "lowest_added_prism_z": float(lowest),
        "comparison_to_L1_not_used_for_selection": {
            "L1_layout_sha256": first_layout_sha256,
            "overlap_count": len(selected_names & first_names),
            "distinct_from_L1_count": len(selected_names - first_names),
        },
        "restocked_segments": selected_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--eligibility-out", type=Path, default=ELIGIBILITY_OUT)
    parser.add_argument("--enumerate-eligible", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--seed", help="explicit audit/test seed; cannot freeze canonical L2")
    source.add_argument(
        "--beacon-record",
        type=Path,
        help="JSON wrapper containing source_url and official pulse response",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    segments = LAYOUT1.rack_segments(LAYOUT1.BASE_WORLD)
    eligibility = eligibility_spec(segments)
    if args.enumerate_eligible:
        freeze_or_check(
            args.eligibility_out,
            render(eligibility),
            check=args.check,
            label="L2 eligibility",
        )
        print(
            f"[L2 eligibility] {eligibility['candidate_count']} segments, "
            f"{eligibility['eligible_layout_count']} balanced layouts"
        )
        return 0

    if args.seed is None and args.beacon_record is None:
        parser.error("select --enumerate-eligible, --seed, or --beacon-record")
    if args.seed is not None:
        if args.out.resolve() == OUT.resolve():
            parser.error("an explicit seed may not write/check the canonical L2 artifact; use --out")
        seed = args.seed
        randomness = {
            "source_type": "explicit non-beacon audit seed",
            "seed_sha256": _sha256_text(seed.strip().lower()),
        }
    else:
        seed, randomness = beacon_seed(args.beacon_record)

    layout = build_layout(
        seed=seed,
        randomness=randomness,
        eligibility_path=args.eligibility_out,
    )
    freeze_or_check(args.out, render(layout), check=args.check, label="L2 layout")
    print(
        f"[L2 layout] selected index {layout['selection']['selected_zero_based_index']} "
        f"of {layout['selection']['eligible_layout_count']}; "
        f"L1 overlap={layout['comparison_to_L1_not_used_for_selection']['overlap_count']}; "
        f"camera-cell pairs lost={layout['camera_cell_pairs_lost']}; "
        f"fused cells lost={layout['fused_cells_lost']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
