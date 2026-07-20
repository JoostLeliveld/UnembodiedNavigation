#!/usr/bin/env python3
"""Append operational leave-one-camera-out usability labels to opportunity CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import ContractValidationError, LeakageError, reject_evaluation_only_keys  # noqa: E402
from reliability.opportunity import LOOReference, OpportunityRow, label_loo_usability  # noqa: E402


def _read_references(path: Path) -> dict[str, LOOReference]:
    if "evaluation_only" in {part.lower() for part in path.parts}:
        raise LeakageError(f"LOO labeler refuses evaluation-only path: {path}")
    references: dict[str, LOOReference] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ContractValidationError(f"expected object at {path}:{line_number}")
            reference = LOOReference.from_dict(payload)
            if reference.sample_id in references:
                raise ContractValidationError(f"duplicate LOO reference {reference.sample_id!r}")
            references[reference.sample_id] = reference
    return references


def _decode(value: str) -> Any:
    if value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _opportunity_from_csv(row: dict[str, str]) -> OpportunityRow:
    reject_evaluation_only_keys(row, context="opportunity CSV")
    return OpportunityRow(
        sample_id=row["sample_id"],
        camera_id=row["camera_id"],
        run_id=row["run_id"],
        timestamp_s=float(row["timestamp_s"]),
        belief_xy_m=_decode(row["belief_xy_m"]),
        belief_cov_xy_m2=_decode(row["belief_cov_xy_m2"]),
        predicted_uv=_decode(row["predicted_uv"]),
        predicted_cov_uv=_decode(row["predicted_cov_uv"]),
        predicted_height_px=float(row["predicted_height_px"]),
        ellipse_inside_fraction=float(row["ellipse_inside_fraction"]),
        inside_valid_region=str(row["inside_valid_region"]).lower() == "true",
        stream_healthy=str(row["stream_healthy"]).lower() == "true",
        detection_received=str(row["detection_received"]).lower() == "true",
        association_valid=str(row["association_valid"]).lower() == "true",
        raw_confidence=None if row["raw_confidence"] == "" else float(row["raw_confidence"]),
        measured_uv=_decode(row["measured_uv"]),
        availability_label=int(row["availability_label"]),
    )


def label_dataset(
    *, opportunity_csv: Path, reference_jsonl: Path, output_csv: Path, max_residual_px: float
) -> dict[str, int]:
    if "evaluation_only" in {part.lower() for part in opportunity_csv.parts}:
        raise LeakageError(f"LOO labeler refuses evaluation-only path: {opportunity_csv}")
    references = _read_references(reference_jsonl)
    with opportunity_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = [_opportunity_from_csv(row) for row in csv.DictReader(handle)]
    labelled = []
    for row in rows:
        reference = references.get(row.sample_id)
        if reference is None:
            raise ContractValidationError(f"missing LOO reference for {row.sample_id!r}")
        labelled.append(label_loo_usability(row, reference, max_residual_px=max_residual_px))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(labelled[0]) if labelled else []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in labelled:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, list) else value for key, value in row.items()})
    usable = sum(row.get("usable_label") == 1 for row in labelled)
    return {"opportunities": len(labelled), "usable": usable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunity-csv", type=Path, required=True)
    parser.add_argument("--loo-reference-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-residual-px", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(label_dataset(
        opportunity_csv=args.opportunity_csv,
        reference_jsonl=args.loo_reference_jsonl,
        output_csv=args.output,
        max_residual_px=args.max_residual_px,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
