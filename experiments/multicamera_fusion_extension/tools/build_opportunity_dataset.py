#!/usr/bin/env python3
"""Build per-camera opportunity rows from an operational split export.

The sidecar predictions must be generated from operational belief/projection
state.  This command rejects evaluation-only paths and fields rather than
silently turning truth into availability labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import (  # noqa: E402
    ContractValidationError,
    LeakageError,
    OperationalReliabilitySample,
    reject_evaluation_only_keys,
)
from reliability.opportunity import (  # noqa: E402
    OpportunityConfig,
    OpportunityPrediction,
    build_opportunity_row,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if "evaluation_only" in {part.lower() for part in path.parts}:
        raise LeakageError(f"operational opportunity builder refuses evaluation-only path: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractValidationError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ContractValidationError(f"expected object at {path}:{line_number}")
            reject_evaluation_only_keys(payload, context=f"{path}:{line_number}")
            records.append(payload)
    return records


def build_dataset(
    *,
    operational_path: Path,
    prediction_path: Path,
    output_path: Path,
    config: OpportunityConfig,
) -> dict[str, int]:
    samples = [OperationalReliabilitySample.from_dict(item) for item in _read_jsonl(operational_path)]
    predictions = [OpportunityPrediction.from_dict(item) for item in _read_jsonl(prediction_path)]
    by_sample_id: dict[str, OpportunityPrediction] = {}
    for prediction in predictions:
        if prediction.sample_id in by_sample_id:
            raise ContractValidationError(f"duplicate prediction sample_id {prediction.sample_id!r}")
        by_sample_id[prediction.sample_id] = prediction

    rows = []
    missing = []
    for sample in samples:
        prediction = by_sample_id.get(sample.sample_id)
        if prediction is None:
            missing.append(sample.sample_id)
            continue
        row = build_opportunity_row(sample, prediction, config=config)
        if row is not None:
            rows.append(row.to_dict())
    if missing:
        raise ContractValidationError(
            "missing operational prediction sidecars for " + ", ".join(sorted(missing)[:10])
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "camera_id", "run_id", "timestamp_s", "belief_xy_m", "belief_cov_xy_m2",
        "predicted_uv", "predicted_cov_uv", "predicted_height_px",
        "ellipse_inside_fraction", "inside_valid_region", "stream_healthy",
        "detection_received", "association_valid", "raw_confidence", "measured_uv",
        "availability_label",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fields})
    return {"operational_samples": len(samples), "opportunities": len(rows)}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operational-jsonl", type=Path, required=True)
    parser.add_argument("--prediction-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-width-px", type=int, required=True)
    parser.add_argument("--image-height-px", type=int, required=True)
    parser.add_argument("--valid-margin-px", type=float, default=0.0)
    parser.add_argument("--ellipse-sigma", type=float, default=2.0)
    parser.add_argument("--min-ellipse-inside-fraction", type=float, default=0.8)
    parser.add_argument("--min-predicted-height-px", type=float, default=12.0)
    parser.add_argument("--max-measurement-age-s", type=float, default=0.5)
    parser.add_argument("--max-association-delta-s", type=float, default=0.10)
    args = parser.parse_args()
    config = OpportunityConfig(
        image_width_px=args.image_width_px,
        image_height_px=args.image_height_px,
        valid_margin_px=args.valid_margin_px,
        ellipse_sigma=args.ellipse_sigma,
        min_ellipse_inside_fraction=args.min_ellipse_inside_fraction,
        min_predicted_height_px=args.min_predicted_height_px,
        max_measurement_age_s=args.max_measurement_age_s,
        max_association_delta_s=args.max_association_delta_s,
    )
    summary = build_dataset(
        operational_path=args.operational_jsonl,
        prediction_path=args.prediction_jsonl,
        output_path=args.output,
        config=config,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
