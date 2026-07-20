#!/usr/bin/env python3
"""Audit whether the four-camera study has earned its paper claims.

This tool is deliberately a *gatekeeper*, not a metric optimiser.  It records
the frozen method hashes and checks the evidence needed for each claim:

* raw/evaluation-only leakage separation;
* per-camera projection calibration and odometry covariance calibration;
* route-disjoint GP evidence and D2 overlap qualification; and
* complete, matched D3 replay manifests.

An absent result is always reported as absent.  In particular, this command
cannot turn the existing sparse pilot into a learned-map or closed-loop result.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import campaign_ledger
from paper_campaign import CAMERAS, DEFAULT_PROTOCOL, DEFAULT_STUDY, discover_runs, qualification_status


REPO = Path(__file__).resolve().parents[3]
STUDY_DIR = REPO / "experiments" / "multicamera_commissioning_bigwarehouse"
DEFAULT_ANALYSIS = STUDY_DIR / "config" / "paper_analysis_plan.yaml"
DEFAULT_OUTPUT = REPO / "logs" / "studies" / "multicamera_commissioning_bigwarehouse" / "paper_readiness_v1"


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a YAML mapping in {path}")
    return payload


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _integer(value: Any) -> int:
    number = _finite(value)
    return int(number) if math.isfinite(number) else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _headers(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return {str(value).strip().lower() for value in next(reader, [])}


def _has_truth_column(columns: Iterable[str]) -> bool:
    forbidden = ("true_", "gt_", "ground_truth", "oracle", "eval_")
    return any(any(column.startswith(prefix) or prefix in column for prefix in forbidden) for column in columns)


def _validated_campaign_runs(
    run_root: Path | None,
    *,
    study_path: Path,
    protocol_path: Path,
    analysis_path: Path,
    method_snapshot: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return only immutable, ledger-validated qualification recordings."""

    failures: list[str] = []
    if run_root is None:
        return [], {
            "pass": False,
            "eligible_runs": 0,
            "failures": ["--run-root with campaign_ledger.json is required"],
        }
    root = run_root.expanduser().resolve()
    ledger_path = root / campaign_ledger.DEFAULT_LEDGER_NAME
    if not ledger_path.is_file():
        return [], {
            "pass": False,
            "eligible_runs": 0,
            "failures": [f"campaign ledger is missing: {ledger_path}"],
        }
    try:
        ledger = campaign_ledger._load_json(ledger_path)
        evaluated = campaign_ledger.evaluate_campaign(root, ledger)
    except campaign_ledger.LedgerError as exc:
        return [], {
            "pass": False,
            "eligible_runs": 0,
            "failures": [f"campaign ledger validation failed: {exc}"],
        }

    frozen = campaign_ledger.input_sha256(evaluated["provenance"])
    for label, path in (("study", study_path), ("protocol", protocol_path)):
        actual = _sha256(path)
        if frozen.get(label) != actual:
            failures.append(f"ledger {label} hash differs from readiness input")
    config_hashes = {
        digest for label, digest in frozen.items() if label.startswith("config:")
    }
    analysis_hash = _sha256(analysis_path)
    if analysis_hash not in config_hashes:
        failures.append("current analysis plan is absent from the ledger's frozen config set")

    completed_rows = [row for row in evaluated["rows"] if row["status"] == "completed"]
    failed_rows = [row for row in evaluated["rows"] if row["status"] == "failed"]
    if failed_rows:
        failures.append(f"{len(failed_rows)} campaign row(s) failed immutable validation")
    completed_dirs = {
        (root / str(row["selected_run_dir"])).resolve(): row
        for row in completed_rows
        if row.get("selected_run_dir")
    }
    declared_attempt_dirs = {
        (root / str(attempt["run_dir"])).resolve()
        for row in evaluated["rows"]
        for attempt in row["attempts"]
    }
    all_runs = discover_runs(root)
    runs_by_dir = {Path(run["raw_dir"]).parent.resolve(): run for run in all_runs}
    untracked = sorted(
        str(path) for path in set(runs_by_dir).difference(declared_attempt_dirs)
    )
    if untracked:
        failures.append(
            f"{len(untracked)} route-manifest run(s) lack a validated completion contract"
        )

    eligible: list[dict[str, Any]] = []
    role_excluded: list[str] = []
    for run_dir, row in sorted(completed_dirs.items(), key=lambda item: str(item[0])):
        row_id = str(row["row_id"])
        run = runs_by_dir.get(run_dir)
        if run is None:
            failures.append(f"completed row {row_id} is not discoverable as a recording")
            continue
        raw_dir = Path(run["raw_dir"])
        operational = _json(raw_dir / "operational_recording_manifest.json") or {}
        truth = _json(raw_dir.parent / "evaluation_only/evaluation_truth_manifest.json") or {}
        expected_role = row.get("expected_evidence_role")
        if operational.get("evidence_role") != expected_role or truth.get(
            "evidence_role"
        ) != expected_role:
            role_excluded.append(str(run["run_id"]))
            failures.append(
                f"{run['run_id']}: recorder role differs from pre-declared row role"
            )
            continue
        if operational.get("method_freeze") != method_snapshot:
            failures.append(f"{run['run_id']}: method-freeze hashes differ from readiness snapshot")
            continue
        run = dict(run)
        run["row_id"] = row_id
        run["row_tuple"] = dict(row["row_tuple"])
        run["phase"] = str(row["row_tuple"]["phase"])
        run["analysis_split"] = str(row["expected_analysis_split"])
        eligible.append(run)

    unfinished = [
        f"{row['row_id']}/{state['attempt_id']}"
        for row in evaluated["rows"]
        for state in row.get("attempt_states", [])
        if state.get("status") in {"in_progress", "invalid"}
    ]
    return eligible, {
        "pass": not failures,
        "ledger": str(ledger_path),
        "completed_rows": len(completed_rows),
        "eligible_runs": len(eligible),
        "role_excluded_runs": sorted(role_excluded),
        "untracked_runs": untracked,
        "unfinished_artifacts": unfinished,
        "frozen_input_sha256": frozen,
        "failures": failures,
    }


def _audit_firewall(runs: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0
    for run in runs:
        raw_dir = Path(run["raw_dir"])
        route = _json(raw_dir / "route_manifest.json")
        operational = _json(raw_dir / "operational_recording_manifest.json")
        checked += 1
        run_id = str(run["run_id"])
        if route is None or route.get("contains_ground_truth") is not False:
            failures.append(f"{run_id}: route manifest does not explicitly forbid ground truth")
        if operational is None or operational.get("contains_ground_truth") is not False:
            failures.append(f"{run_id}: operational manifest does not explicitly forbid ground truth")
        for name in ["experiment.csv", *(f"{camera}_perception.csv" for camera in CAMERAS)]:
            path = raw_dir / name
            columns = _headers(path)
            if not columns:
                failures.append(f"{run_id}: missing or empty raw {name}")
            elif _has_truth_column(columns):
                failures.append(f"{run_id}: evaluation-only column leaked into raw {name}")
    return {"checked_runs": checked, "failures": failures, "pass": bool(checked) and not failures}


def _alignment_summary_errors(
    run: dict[str, Any], summary: dict[str, Any], required_role: str
) -> list[str]:
    raw_dir = Path(run["raw_dir"])
    run_dir = raw_dir.parent
    evaluation_inputs = run_dir / "evaluation_inputs"
    errors: list[str] = []
    operational_path = raw_dir / "operational_recording_manifest.json"
    truth_manifest_path = run_dir / "evaluation_only/evaluation_truth_manifest.json"
    truth_csv = run_dir / "evaluation_only/ground_truth.csv"
    operational = _json(operational_path) or {}
    truth_manifest = _json(truth_manifest_path) or {}
    if summary.get("projection_role") != required_role:
        errors.append(f"projection_role is not {required_role}")
    if summary.get("role_predeclared_by_recorders") is not True:
        errors.append("role was not pre-declared by both recorders")
    if operational.get("evidence_role") != required_role or truth_manifest.get(
        "evidence_role"
    ) != required_role:
        errors.append("recorder evidence roles do not match qualification role")
    source = summary.get("source_contract")
    if not isinstance(source, dict):
        return [*errors, "summary lacks source_contract"]
    attachment = _json(evaluation_inputs / "truth_attachment_manifest.json")
    if (
        not isinstance(attachment, dict)
        or attachment.get("schema_version") != 1
        or attachment.get("status") != "completed"
        or attachment.get("projection_role") != required_role
        or attachment.get("source_contract") != source
    ):
        errors.append("truth attachment completion manifest is missing or inconsistent")
    else:
        declared_outputs = attachment.get("artifacts_sha256")
        if not isinstance(declared_outputs, dict):
            errors.append("truth attachment manifest lacks output hashes")
        else:
            for name in (
                "truth_alignment_summary.json",
                *(f"{camera}_perception.csv" for camera in CAMERAS),
            ):
                path = evaluation_inputs / name
                if not path.is_file() or declared_outputs.get(name) != _sha256(path):
                    errors.append(f"truth attachment output hash mismatch for {name}")
    for field in ("run_id", "plan_row_id", "seed"):
        if source.get(field) != operational.get(field):
            errors.append(f"summary source {field} does not match operational manifest")
    if source.get("evidence_role") != required_role:
        errors.append("summary source role is not qualification")
    for field, path in (
        ("operational_manifest", operational_path),
        ("truth_manifest", truth_manifest_path),
        ("campaign_completion", run_dir / "completion_manifest.json"),
    ):
        record = source.get(field)
        if not isinstance(record, dict) or not path.is_file() or record.get("sha256") != _sha256(path):
            errors.append(f"summary {field} hash does not match immutable source")
    if summary.get("truth_csv_sha256") != _sha256(truth_csv):
        errors.append("summary truth CSV hash does not match source")
    camera_hashes = summary.get("source_camera_csv_sha256")
    if not isinstance(camera_hashes, dict):
        errors.append("summary lacks source camera CSV hashes")
    else:
        for camera in CAMERAS:
            path = raw_dir / f"{camera}_perception.csv"
            if camera_hashes.get(path.name) != _sha256(path):
                errors.append(f"summary source hash mismatch for {camera}")
    detector_hash = (operational.get("detector_runtime") or {}).get("model_sha256")
    projection = operational.get("projection_calibration") or {}
    projection_hash = projection.get("sha256") if isinstance(projection, dict) else None
    if source.get("detector_model_sha256") != detector_hash:
        errors.append("summary detector hash differs from operational manifest")
    if source.get("projection_calibration_sha256") != projection_hash:
        errors.append("summary projection hash differs from operational manifest")

    calibration_path_text = projection.get("path") if isinstance(projection, dict) else None
    calibration_path = Path(str(calibration_path_text)) if calibration_path_text else Path()
    calibration = _json(calibration_path) if calibration_path_text else None
    fit_source = calibration.get("source_contract") if isinstance(calibration, dict) else None
    if not isinstance(fit_source, dict) or fit_source.get("evidence_role") != "fit":
        errors.append("frozen projection calibration lacks a pre-declared fit source")
    elif fit_source.get("plan_row_id") == operational.get("plan_row_id"):
        errors.append("projection fit and qualification use the same campaign row")
    else:
        fit_tuple = fit_source.get("row_tuple")
        qualification_tuple = run.get("row_tuple")
        if (
            not isinstance(fit_tuple, dict)
            or not isinstance(qualification_tuple, dict)
            or fit_tuple.get("route") == qualification_tuple.get("route")
        ):
            errors.append("projection fit route is not disjoint from qualification route")
    return errors


def _audit_projection_calibration(runs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    by_camera: dict[str, dict[str, Any]] = {
        camera: {
            "detected_rows": 0,
            "max_abs_along_bias_m": math.nan,
            "max_abs_cross_bias_m": math.nan,
            "max_p90_projection_error_m": math.nan,
            "max_projection_error_m": math.nan,
            "summary_count": 0,
        }
        for camera in CAMERAS
    }
    missing_summaries: list[str] = []
    for run in runs:
        summary = _json(Path(run["raw_dir"]).parent / "evaluation_inputs" / "truth_alignment_summary.json")
        if summary is None:
            missing_summaries.append(str(run["run_id"]))
            continue
        required_role = str(cfg.get("required_projection_role", "qualification"))
        integrity_errors = _alignment_summary_errors(run, summary, required_role)
        if integrity_errors:
            missing_summaries.append(
                f"{run['run_id']} ({'; '.join(integrity_errors)})"
            )
            continue
        cameras = summary.get("cameras", {})
        if not isinstance(cameras, dict):
            missing_summaries.append(str(run["run_id"]))
            continue
        for camera in CAMERAS:
            item = cameras.get(camera, {})
            if not isinstance(item, dict):
                continue
            out = by_camera[camera]
            out["summary_count"] += 1
            out["detected_rows"] += _integer(item.get("detected_rows_audited"))
            for key, value in (
                ("max_abs_along_bias_m", abs(_finite(item.get("along_bearing_bias_m")))),
                ("max_abs_cross_bias_m", abs(_finite(item.get("cross_bearing_bias_m")))),
                ("max_p90_projection_error_m", _finite(item.get("p90_abs_error_m"))),
                ("max_projection_error_m", _finite(item.get("max_abs_error_m"))),
            ):
                old = out[key]
                out[key] = value if not math.isfinite(old) else max(old, value) if math.isfinite(value) else old

    failures: list[str] = []
    for camera, item in by_camera.items():
        if item["detected_rows"] < int(cfg["min_detected_rows_per_camera"]):
            failures.append(f"{camera}: fewer than {cfg['min_detected_rows_per_camera']} truth-audited detections")
        if (not math.isfinite(item["max_abs_along_bias_m"])
                or item["max_abs_along_bias_m"] > float(cfg["max_abs_along_bias_m"])):
            failures.append(
                f"{camera}: along-bearing bias exceeds {cfg['max_abs_along_bias_m']} m or is unavailable"
            )
        if (not math.isfinite(item["max_abs_cross_bias_m"])
                or item["max_abs_cross_bias_m"] > float(cfg["max_abs_cross_bias_m"])):
            failures.append(
                f"{camera}: cross-bearing bias exceeds {cfg['max_abs_cross_bias_m']} m or is unavailable"
            )
        if not math.isfinite(item["max_p90_projection_error_m"]) or item["max_p90_projection_error_m"] > float(cfg["max_p90_projection_error_m"]):
            failures.append(f"{camera}: p90 projection error exceeds {cfg['max_p90_projection_error_m']} m or is unavailable")
        if not math.isfinite(item["max_projection_error_m"]) or item["max_projection_error_m"] > float(cfg["max_projection_error_m"]):
            failures.append(f"{camera}: maximum projection error exceeds {cfg['max_projection_error_m']} m or is unavailable")
    if missing_summaries:
        failures.append(f"missing evaluation-input alignment summary for runs: {', '.join(sorted(missing_summaries))}")
    return {
        "per_camera": by_camera,
        "missing_summaries": sorted(missing_summaries),
        "failures": failures,
        "pass": bool(runs) and not failures,
    }


def _nearest_truth(stamps: list[float], values: list[tuple[float, float]], stamp: float, max_dt: float) -> tuple[float, float] | None:
    index = bisect.bisect_left(stamps, stamp)
    candidates = [item for item in (index - 1, index) if 0 <= item < len(stamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(stamps[item] - stamp))
    return values[best] if abs(stamps[best] - stamp) <= max_dt else None


def _audit_odometry_covariance(runs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    nees: list[float] = []
    matched = 0
    for run in runs:
        raw_dir = Path(run["raw_dir"])
        truth_rows = _rows(raw_dir.parent / "evaluation_only" / "ground_truth.csv")
        truth = [(_finite(row.get("stamp")), _finite(row.get("gt_x")), _finite(row.get("gt_y"))) for row in truth_rows]
        truth = [item for item in truth if all(math.isfinite(value) for value in item)]
        truth.sort(key=lambda item: item[0])
        stamps = [item[0] for item in truth]
        values = [(item[1], item[2]) for item in truth]
        for row in _rows(raw_dir / "experiment.csv"):
            stamp = _finite(row.get("stamp"))
            x = _finite(row.get("odom_noisy_x"))
            y = _finite(row.get("odom_noisy_y"))
            xx = _finite(row.get("odom_noisy_cov_xx"))
            xy = _finite(row.get("odom_noisy_cov_xy"))
            yy = _finite(row.get("odom_noisy_cov_yy"))
            determinant = xx * yy - xy * xy
            if not all(math.isfinite(value) for value in (stamp, x, y, xx, xy, yy)) or xx <= 0 or yy <= 0 or determinant <= 0:
                continue
            truth_xy = _nearest_truth(stamps, values, stamp, float(cfg["max_truth_alignment_delta_s"]))
            if truth_xy is None:
                continue
            ex, ey = x - truth_xy[0], y - truth_xy[1]
            nees.append((yy * ex * ex - 2.0 * xy * ex * ey + xx * ey * ey) / determinant)
            matched += 1
    mean_nees = sum(nees) / len(nees) if nees else math.nan
    one_sigma = sum(value <= 2.30 for value in nees) / len(nees) if nees else math.nan
    two_sigma = sum(value <= 6.18 for value in nees) / len(nees) if nees else math.nan
    failures: list[str] = []
    if matched < int(cfg["min_matched_samples"]):
        failures.append(f"only {matched} covariance/truth matches; need {cfg['min_matched_samples']}")
    lower, upper = (float(value) for value in cfg["mean_nees_range"])
    if not math.isfinite(mean_nees) or not lower <= mean_nees <= upper:
        failures.append(f"mean NEES {mean_nees!r} outside [{lower}, {upper}]")
    lower, upper = (float(value) for value in cfg["one_sigma_coverage_range"])
    if not math.isfinite(one_sigma) or not lower <= one_sigma <= upper:
        failures.append(f"one-sigma coverage {one_sigma!r} outside [{lower}, {upper}]")
    if not math.isfinite(two_sigma) or two_sigma < float(cfg["min_two_sigma_coverage"]):
        failures.append(f"two-sigma coverage {two_sigma!r} below {cfg['min_two_sigma_coverage']}")
    return {
        "matched_samples": matched,
        "mean_nees": mean_nees,
        "covariance_1sigma_coverage": one_sigma,
        "covariance_2sigma_coverage": two_sigma,
        "failures": failures,
        "pass": bool(runs) and not failures,
    }


def _audit_gp_validation(root: Path | None, cfg: dict[str, Any]) -> dict[str, Any]:
    per_camera: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    required_modes = {str(cfg["primary_model"]), *(str(mode) for mode in cfg["descriptive_baselines"])}
    for camera in CAMERAS:
        path = root / camera / "route_disjoint_validation.csv" if root else Path()
        rows = _rows(path) if root else []
        modes = {str(row.get("mode", "")) for row in rows}
        primary = next((row for row in rows if row.get("mode") == cfg["primary_model"]), None)
        events = _integer(primary.get("heldout_events")) if primary else 0
        missing = sorted(required_modes.difference(modes))
        per_camera[camera] = {"path": str(path) if root else "", "heldout_events": events, "missing_modes": missing}
        if missing:
            failures.append(f"{camera}: missing route-disjoint modes {', '.join(missing)}")
        if events < int(cfg["min_heldout_events_per_camera"]):
            failures.append(f"{camera}: only {events} held-out events; need {cfg['min_heldout_events_per_camera']}")
    return {"per_camera": per_camera, "failures": failures, "pass": bool(root) and not failures}


def _audit_replay(root: Path | None, protocol: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    expected = {
        (str(task["id"]), str(condition), int(seed))
        for task in protocol["navigation_tasks"]
        for condition in protocol["randomization"]["conditions"]
        for seed in protocol["randomization"]["seed_values"]
    }
    manifests = sorted(root.glob("**/replay_manifest.json")) if root and root.exists() else []
    observed: dict[tuple[str, str, int], set[str]] = {}
    bad: list[str] = []
    required_policies = set(protocol["offline_replay"]["policies"])
    for path in manifests:
        manifest = _json(path)
        if manifest is None:
            bad.append(f"invalid manifest: {path}")
            continue
        try:
            key = (str(manifest["task_id"]), str(manifest["condition"]), int(manifest["seed"]))
        except (KeyError, TypeError, ValueError):
            bad.append(f"missing task/condition/seed: {path}")
            continue
        if manifest.get("protocol_id") != protocol.get("protocol_id"):
            bad.append(f"protocol mismatch for {key}: {path}")
        if manifest.get("analysis_plan_id") != analysis.get("analysis_plan_id"):
            bad.append(f"analysis-plan mismatch for {key}: {path}")
        if set(manifest.get("camera_gp_sha256", {})) != set(CAMERAS):
            bad.append(f"missing per-camera posterior hashes for {key}: {path}")
        if manifest.get("source_method_sha256") != _snapshot(analysis)["source_sha256"]:
            bad.append(f"method hash mismatch for {key}: {path}")
        summary = _json(path.parent / "benchmark_summary.json")
        results = summary.get("results", {}) if isinstance(summary, dict) else {}
        policies = set(results) if isinstance(results, dict) else set()
        if not required_policies.issubset(policies):
            bad.append(f"missing policy results for {key}: {path}")
        elif any(results[policy].get("rmse_m") is None for policy in required_policies):
            bad.append(f"evaluation truth missing or unusable for {key}: {path}")
        observed.setdefault(key, set()).update(policies)
    missing = sorted(expected.difference(observed))
    duplicate_count = len(manifests) - len(observed)
    if duplicate_count > 0:
        bad.append(f"{duplicate_count} duplicate replay key(s); each task/condition/seed must be unique")
    return {
        "expected_cases": len(expected),
        "observed_cases": len(observed),
        "missing_cases": len(missing),
        "manifest_count": len(manifests),
        "failures": bad,
        "pass": len(observed) == len(expected) and not bad,
    }


def _snapshot(
    analysis: dict[str, Any], analysis_path: Path = DEFAULT_ANALYSIS
) -> dict[str, Any]:
    files: dict[str, str | None] = {}
    for relative in analysis["method_freeze"]["source_files"]:
        path = (STUDY_DIR / str(relative)).resolve()
        files[str(relative)] = _sha256(path) if path.is_file() else None
    return {
        "analysis_plan_id": analysis["analysis_plan_id"],
        "analysis_plan_sha256": _sha256(analysis_path),
        "source_sha256": files,
    }


def _write_report(path: Path, status: dict[str, Any]) -> None:
    gates = status["gates"]
    rows = [
        ("Operational/evaluation firewall", gates["firewall"]),
        ("Immutable campaign and method freeze", gates["campaign_integrity"]),
        ("D0 projection calibration", gates["calibration"]),
        ("Odometry covariance calibration", gates["odometry_covariance"]),
        ("D1 collection coverage", gates["d1_collection"]),
        ("D1 route-disjoint GP report", gates["d1_gp_validation"]),
        ("D2 overlap qualification", gates["d2_overlap"]),
        ("D3 matched replay", gates["d3_replay"]),
        ("D4 closed-loop confirmation", gates["d4_closed_loop"]),
    ]
    lines = [
        "# Four-camera paper readiness",
        "",
        "## Decision",
        "",
        "**READY FOR CONFIRMATORY PAPER CLAIMS**" if gates["paper_claims_permitted"] else "**NOT READY FOR CONFIRMATORY PAPER CLAIMS.**",
        "",
        "| Requirement | Status |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'NOT YET'} |" for name, passed in rows)
    lines.extend(
        [
            "",
            "## Claim discipline",
            "",
            "The present package is simulation-only. Any missing gate is a missing result, not a negative result; it may not be replaced by a pilot, a pooled map, or an oracle baseline.",
            "",
            "## Next required action",
            "",
            status["next_action"],
            "",
            "Detailed machine-readable audit: `readiness_status.json`. The source hashes in `method_snapshot.json` identify the frozen implementation this audit evaluated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--validation-root", type=Path, default=None)
    parser.add_argument("--replay-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    allowed_root = (REPO / "logs" / "studies" / "multicamera_commissioning_bigwarehouse").resolve()
    if out_dir != allowed_root and allowed_root not in out_dir.parents:
        raise RuntimeError(f"--out-dir must stay under {allowed_root}")
    study_path = args.study.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    analysis_path = args.analysis.expanduser().resolve()
    study = _yaml(study_path)
    protocol = _yaml(protocol_path)
    analysis = _yaml(analysis_path)
    if analysis.get("protocol_id") != protocol.get("protocol_id"):
        raise RuntimeError("analysis plan and paper protocol IDs do not match")
    method_snapshot = _snapshot(analysis, analysis_path)
    runs, campaign_integrity = _validated_campaign_runs(
        args.run_root,
        study_path=study_path,
        protocol_path=protocol_path,
        analysis_path=analysis_path,
        method_snapshot=method_snapshot,
    )
    qualification = qualification_status(study, protocol, runs)
    firewall = _audit_firewall(runs)
    d0_phases = {
        str(value)
        for value in analysis.get("data_integrity", {}).get(
            "d0_qualification_phases", []
        )
    }
    d0_runs = [run for run in runs if str(run.get("phase")) in d0_phases]
    calibration = _audit_projection_calibration(
        d0_runs, analysis["calibration_gate"]
    )
    calibration["qualification_phases"] = sorted(d0_phases)
    calibration["qualification_run_count"] = len(d0_runs)
    odometry = _audit_odometry_covariance(
        d0_runs, analysis["odometry_covariance_gate"]
    )
    odometry["qualification_phases"] = sorted(d0_phases)
    odometry["qualification_run_count"] = len(d0_runs)
    gp = _audit_gp_validation(args.validation_root.expanduser().resolve() if args.validation_root else None, analysis["d1_route_disjoint_gp"])
    replay = _audit_replay(
        args.replay_root.expanduser().resolve() if args.replay_root else None,
        protocol,
        analysis,
    )
    d4 = False  # This audit cannot activate a policy; it is earned only by a dedicated active campaign.
    gates = {
        "firewall": firewall["pass"],
        "campaign_integrity": campaign_integrity["pass"],
        "calibration": calibration["pass"],
        "odometry_covariance": odometry["pass"],
        "d1_collection": qualification["gates"]["mapping_collection_complete"],
        "d1_gp_validation": gp["pass"],
        "d2_overlap": qualification["gates"]["overlap_complete"],
        "d3_replay": replay["pass"],
        "d4_closed_loop": d4,
    }
    gates["paper_claims_permitted"] = all(gates.values())
    status = {
        "analysis_plan_id": analysis["analysis_plan_id"],
        "protocol_id": protocol["protocol_id"],
        "scope": analysis["scope"],
        "gates": gates,
        "collection_qualification": qualification,
        "campaign_integrity": campaign_integrity,
        "firewall": firewall,
        "projection_calibration": calibration,
        "odometry_covariance": odometry,
        "route_disjoint_gp": gp,
        "paired_replay": replay,
        "next_action": "Collect D1 with separate evaluation-only truth and propagated covariance, then repair any camera that fails D0 projection calibration before fitting or combining camera maps.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readiness_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "method_snapshot.json").write_text(json.dumps(method_snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(out_dir / "PAPER_EVIDENCE.md", status)
    print(json.dumps({"out_dir": str(out_dir), "gates": gates}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
