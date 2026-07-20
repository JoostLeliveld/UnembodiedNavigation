#!/usr/bin/env python3
"""Train one camera's availability or conditional-usability GP from Plan-02 rows.

This is deliberately a thin, provenance-producing wrapper around the canonical
belief-aware GP fitter. It never reads evaluation-only records and keeps route
IDs intact for the fitter's held-out-run split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.contracts import ContractValidationError, LeakageError, reject_evaluation_only_keys  # noqa: E402


EVENT_FIELDS = ("m_x", "m_y", "S_xx", "S_xy", "S_yy", "det_hit", "yolo_score_raw", "run_id", "camera_id")


def _json_or_none(value: str) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"expected JSON-encoded opportunity field, got {value!r}") from exc


def prepare_events(
    rows: Iterable[dict[str, str]], *, camera_id: str, kind: str
) -> list[dict[str, Any]]:
    """Convert labelled opportunity rows to the canonical GP event schema."""

    if kind not in {"availability", "quality"}:
        raise ContractValidationError("kind must be availability or quality")
    events: list[dict[str, Any]] = []
    label_column = "availability_label" if kind == "availability" else "usable_label"
    for row in rows:
        reject_evaluation_only_keys(row, context="opportunity training row")
        if row.get("camera_id") != camera_id:
            continue
        label_raw = row.get(label_column, "")
        if label_raw in ("", "None", "null"):
            continue
        try:
            label = int(label_raw)
        except ValueError as exc:
            raise ContractValidationError(f"{label_column} must be 0 or 1") from exc
        if label not in {0, 1}:
            raise ContractValidationError(f"{label_column} must be 0 or 1")
        belief_xy = _json_or_none(row.get("belief_xy_m", ""))
        covariance = _json_or_none(row.get("belief_cov_xy_m2", ""))
        if belief_xy is None or covariance is None:
            continue
        if not isinstance(belief_xy, list) or len(belief_xy) != 2:
            raise ContractValidationError("belief_xy_m must be a two-element JSON array")
        if (
            not isinstance(covariance, list)
            or len(covariance) != 2
            or any(not isinstance(item, list) or len(item) != 2 for item in covariance)
        ):
            raise ContractValidationError("belief_cov_xy_m2 must be a 2x2 JSON array")
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            raise ContractValidationError("opportunity row lacks run_id; grouped holdout is impossible")
        raw_score = row.get("raw_confidence", "")
        events.append(
            {
                "m_x": float(belief_xy[0]),
                "m_y": float(belief_xy[1]),
                "S_xx": float(covariance[0][0]),
                "S_xy": float(covariance[0][1]),
                "S_yy": float(covariance[1][1]),
                "det_hit": label,
                "yolo_score_raw": "" if raw_score in ("", "None", "null") else float(raw_score),
                "run_id": run_id,
                "camera_id": camera_id,
            }
        )
    if not events:
        raise ContractValidationError(f"no finite {kind} rows for {camera_id}")
    return events


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(
    *,
    opportunity_csv: Path,
    output_dir: Path,
    camera_id: str,
    kind: str,
    holdout_run_ids: list[str],
    grid_from: Path,
    prior_gp: Path | None,
    modes: str,
) -> dict[str, Any]:
    if "evaluation_only" in {part.lower() for part in opportunity_csv.parts}:
        raise LeakageError(f"GP trainer refuses evaluation-only path: {opportunity_csv}")
    with opportunity_csv.open("r", encoding="utf-8", newline="") as handle:
        events = prepare_events(csv.DictReader(handle), camera_id=camera_id, kind=kind)
    available_runs = {str(event["run_id"]) for event in events}
    requested = sorted(set(holdout_run_ids))
    if not requested:
        raise ContractValidationError("at least one --holdout-run-id is required")
    if not set(requested).issubset(available_runs) or set(requested) == available_runs:
        raise ContractValidationError("held-out run IDs must be present and leave at least one training run")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / f"{camera_id}_{kind}_events.csv"
    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(events)

    canonical_output = output_dir / "gp"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "visibility_comparison" / "fit_belief_aware_gp.py"),
        "--events", str(events_path), "--out", str(canonical_output),
        "--grid-from", str(grid_from), "--target", "hit", "--modes", modes,
    ]
    for run_id in requested:
        command.extend(("--holdout-run-id", run_id))
    if prior_gp is not None:
        command.extend(("--prior-gp", str(prior_gp)))
    subprocess.run(command, cwd=ROOT, check=True)
    model_card = {
        "schema_version": "factorized_reliability_gp.v1",
        "kind": kind,
        "camera_id": camera_id,
        "opportunity_csv": str(opportunity_csv.resolve()),
        "opportunity_csv_sha256": _sha256(opportunity_csv),
        "prepared_events": str(events_path.resolve()),
        "prepared_events_sha256": _sha256(events_path),
        "canonical_fitter": str(command[1]),
        "canonical_fitter_sha256": _sha256(Path(command[1])),
        "holdout_run_ids": requested,
        "modes": modes,
        "artifact_dir": str(canonical_output.resolve()),
    }
    (output_dir / "model_card.json").write_text(json.dumps(model_card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_card


def main(default_kind: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("availability", "quality"), default=default_kind)
    parser.add_argument("--opportunity-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--holdout-run-id", action="append", default=[])
    parser.add_argument("--grid-from", type=Path, required=True)
    parser.add_argument("--prior-gp", type=Path)
    parser.add_argument("--modes", default="naive,uncertainty_weighted,belief_spread,expected_kernel")
    args = parser.parse_args()
    if args.kind is None:
        parser.error("--kind is required")
    card = train(
        opportunity_csv=args.opportunity_csv.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        camera_id=str(args.camera_id), kind=str(args.kind),
        holdout_run_ids=[str(item) for item in args.holdout_run_id],
        grid_from=args.grid_from.expanduser().resolve(),
        prior_gp=None if args.prior_gp is None else args.prior_gp.expanduser().resolve(),
        modes=str(args.modes),
    )
    print(json.dumps(card, sort_keys=True))


if __name__ == "__main__":
    main()
