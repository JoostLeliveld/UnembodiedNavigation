#!/usr/bin/env python3
"""Commissioning-only robustness audit for the Stage-09 nuisance-yaw adapter."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from reliability.commissioned_measurement import CommissionedMeasurementModel
from reliability.projection import camera_model_from_world
from reliability.silhouette_observation import (
    equivalent_position_measurement,
    plausibility_reasons,
    select_plausible_nuisance_yaw,
)
from unav_common.robot_hull import VISUAL_HULL, silhouette_box


REPO = Path(__file__).resolve().parents[2]
CAMERAS = tuple(f"camera_{letter}" for letter in "ABCDE")
CHI2 = {"50": 1.38629436112, "90": 4.60517018599,
        "95": 5.99146454711, "99": 9.21034037198}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = (len(ordered) - 1) * probability
    lower, upper = int(math.floor(index)), int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def error_summary(values) -> dict:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0, "median_m": None, "rms_m": None, "p90_m": None,
                "p95_m": None, "maximum_m": None, "above_0_25_m": 0,
                "above_0_25_fraction": None}
    return {
        "n": len(values),
        "median_m": statistics.median(values),
        "rms_m": math.sqrt(sum(value * value for value in values) / len(values)),
        "p90_m": percentile(values, 0.90),
        "p95_m": percentile(values, 0.95),
        "maximum_m": max(values),
        "above_0_25_m": sum(value > 0.25 for value in values),
        "above_0_25_fraction": sum(value > 0.25 for value in values) / len(values),
    }


def covariance_summary(d2_r, d2_operational) -> dict:
    def containments(values):
        return {
            probability: sum(value <= threshold for value in values) / max(len(values), 1)
            for probability, threshold in CHI2.items()
        }
    return {
        "n": len(d2_r),
        "R2C_only_containment": containments(d2_r),
        "operational_Pxy_plus_R2C_containment": containments(d2_operational),
        "R2C_only_above_99_fraction": (
            sum(value > CHI2["99"] for value in d2_r) / max(len(d2_r), 1)),
        "operational_above_99_fraction": (
            sum(value > CHI2["99"] for value in d2_operational)
            / max(len(d2_operational), 1)),
    }


def mahalanobis(error: np.ndarray, covariance) -> float:
    matrix = np.asarray(covariance, dtype=float)
    return float(error @ np.linalg.solve(matrix, error))


def camera_models():
    world = REPO / "src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf"
    includes = (
        "external_camera", "external_camera_b", "external_camera_c",
        "external_camera_d", "external_camera_e",
    )
    return {
        camera_id: camera_model_from_world(world, include_name=include)
        for camera_id, include in zip(CAMERAS, includes, strict=True)
    }


def evaluate(record, camera, prior, covariance, model, protocol, *, nuisance: bool):
    box = record["best_box_xyxy"]
    raw = record["raw_ground_xy"]
    reasons = []
    if box is None:
        return {"admitted": False, "reasons": ["no_detection"], "recovered": False}
    if float(record["raw_best_confidence"]) < 0.25:
        reasons.append("confidence")
    if raw is None:
        reasons.append("projection_unavailable")
    selected_yaw = float(prior[2])
    selection = None
    if not reasons and nuisance:
        selection = select_plausible_nuisance_yaw(
            tuple(box), camera, *prior, covariance,
            interval_z=float(protocol["adapter"]["yaw_interval_z"]),
            samples=int(protocol["adapter"]["yaw_samples"]),
            minimum_passing_fraction=float(
                protocol["adapter"].get("minimum_passing_fraction", 0.0)),
        )
        if selection is None:
            reasons.extend(plausibility_reasons(tuple(box), camera, *prior))
        else:
            selected_yaw = selection.yaw
    elif not reasons:
        reasons.extend(plausibility_reasons(tuple(box), camera, *prior))
    if reasons:
        return {"admitted": False, "reasons": reasons, "recovered": False}

    equivalent = equivalent_position_measurement(
        (float(raw[0]), float(raw[1])), ((1.0, 0.0), (0.0, 1.0)),
        camera, (prior[0], prior[1]), selected_yaw,
    )
    predicted = silhouette_box(
        camera, prior[0], prior[1], selected_yaw, VISUAL_HULL)
    if equivalent is None or predicted is None:
        return {"admitted": False, "reasons": ["equivalent_unavailable"],
                "recovered": False}
    stage07 = model.apply(
        record["camera_id"], raw, equivalent[0], box, predicted,
        float(record["raw_best_confidence"]), (prior[0], prior[1]),
        selected_yaw, camera,
    )
    if stage07 is None:
        return {"admitted": False, "reasons": ["stage07_unavailable"],
                "recovered": False}
    point, r2c = stage07
    truth = np.asarray([record["robot_x"], record["robot_y"]], dtype=float)
    error_vector = np.asarray(point, dtype=float) - truth
    pxy = np.asarray(covariance, dtype=float)[:2, :2]
    innovation = np.asarray(point, dtype=float) - np.asarray(prior[:2], dtype=float)
    return {
        "admitted": True,
        "reasons": [],
        "recovered": bool(selection is not None and selection.recovered),
        "yaw_delta_rad": (
            float(selection.delta_rad) if selection is not None else 0.0),
        "stage07_error_m": float(np.linalg.norm(error_vector)),
        "d2_r2c": mahalanobis(error_vector, r2c),
        "d2_operational": mahalanobis(innovation, np.asarray(r2c) + pxy),
    }


def summarize(rows, regime_id: str, arm: str) -> dict:
    selected = [row for row in rows if row["regime_id"] == regime_id]
    reference = [row["reference_positive"] for row in selected]
    admitted = [row[f"{arm}_admitted"] for row in selected]
    tp = sum(keep and target for keep, target in zip(admitted, reference, strict=True))
    fp = sum(keep and not target for keep, target in zip(admitted, reference, strict=True))
    errors = [row.get(f"{arm}_stage07_error_m") for row in selected
              if row[f"{arm}_admitted"]]
    d2_r = [row[f"{arm}_d2_r2c"] for row in selected if row[f"{arm}_admitted"]]
    d2_op = [row[f"{arm}_d2_operational"] for row in selected if row[f"{arm}_admitted"]]
    return {
        "opportunities": len(selected),
        "reference_positive": sum(reference),
        "admitted": sum(admitted),
        "true_admitted": tp,
        "false_admitted": fp,
        "precision": tp / max(tp + fp, 1),
        "reference_recall": tp / max(sum(reference), 1),
        "recovered": sum(row.get(f"{arm}_recovered", False) for row in selected),
        "stage07_error": error_summary(errors),
        "covariance": covariance_summary(d2_r, d2_op),
        "refusal_reasons": dict(sorted(Counter(
            reason for row in selected for reason in row[f"{arm}_reasons"]
        ).items())),
    }


def subgroup_summaries(rows, regime_id: str, arm: str, field: str) -> dict:
    values = sorted({row[field] for row in rows if row["regime_id"] == regime_id})
    output = {}
    for value in values:
        subset = [row for row in rows if row["regime_id"] == regime_id and row[field] == value]
        # summarize() filters only by regime, so give each subset a private common ID.
        remapped = [dict(row, regime_id="subset") for row in subset]
        output[str(value)] = summarize(remapped, "subset", arm)
    return output


def named_sliver_check(cameras, protocol) -> dict:
    challenge_path = (
        REPO / "logs/thesis_final_pipeline_v1/stage06_detector_gate/"
        "camera_D_sliver_challenge_result/report.json")
    challenge = json.loads(challenge_path.read_text())
    camera_row = next(row for row in challenge["cameras"] if row["camera_id"] == "camera_D")
    pose = json.loads((REPO / "experiments/thesis_pipeline_lock/generated/"
                       "camera_D_sliver_challenge_v1.json").read_text())["poses"][0]
    # The detector did not return this 20x30-pixel fragment at confidence 0.25.
    # Feed the semantic fragment itself to the geometry adapter, which is the
    # stronger diagnostic: even an upstream false return must not be admitted.
    covariance = np.diag([0.25 ** 2, 0.25 ** 2, math.radians(13.0) ** 2])
    selection = select_plausible_nuisance_yaw(
        tuple(camera_row["mask_box_xyxy"]), cameras["camera_D"],
        pose["x"], pose["y"], pose["yaw"], covariance,
        interval_z=float(protocol["adapter"]["yaw_interval_z"]),
        samples=int(protocol["adapter"]["yaw_samples"]),
        minimum_passing_fraction=float(
            protocol["adapter"].get("minimum_passing_fraction", 0.0)),
    )
    return {
        "source": str(challenge_path.relative_to(REPO)),
        "source_sha256": sha256(challenge_path),
        "semantic_fragment_box_xyxy": camera_row["mask_box_xyxy"],
        "visible_width_ratio_at_truth": camera_row["reference_metrics"]["visible_width_ratio"],
        "adapter_admitted": selection is not None,
        "required": "refused",
        "pass": selection is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage06-inference", type=Path, required=True)
    parser.add_argument("--stage07-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") not in {
        "frozen_before_commissioning_robustness_audit",
        "frozen_before_v2_commissioning_robustness_audit",
        "frozen_before_v3_commissioning_robustness_audit",
        "frozen_before_v4_commissioning_robustness_audit",
    }:
        raise RuntimeError("runtime-yaw recovery protocol is not frozen")
    inference = args.stage06_inference.resolve()
    inference_manifest_path = inference / "manifest.json"
    inference_manifest = json.loads(inference_manifest_path.read_text())
    records_path = inference / inference_manifest["records"]
    if sha256(records_path) != inference_manifest["records_sha256"]:
        raise RuntimeError("Stage-06 detector record identity drift")
    with records_path.open() as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != 9600:
        raise RuntimeError("Stage-06 commissioning population must contain 9,600 opportunities")
    model_path = args.stage07_model.resolve()
    model = CommissionedMeasurementModel(model_path)
    cameras = camera_models()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    ledger_path = output / "opportunity_ledger.jsonl"
    rows = []
    with ledger_path.open("x") as ledger:
        for regime in protocol["perturbations"]["regimes"]:
            regime_id = regime["id"]
            for record in records:
                for draw in range(protocol["perturbations"]["draws_per_regime_per_opportunity"]):
                    key = (f"{protocol['perturbations']['seed_namespace']}|{regime_id}|"
                           f"{record['pose_id']}|{record['camera_id']}|{draw}").encode()
                    seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2 ** 32)
                    rng = np.random.default_rng(seed)
                    xy_error = np.clip(
                        rng.normal(0.0, regime["xy_error_sigma_m"], 2),
                        -regime["xy_error_clip_m_per_axis"],
                        regime["xy_error_clip_m_per_axis"],
                    )
                    yaw_clip = math.radians(regime["yaw_error_clip_deg"])
                    yaw_error = float(np.clip(
                        rng.normal(0.0, math.radians(regime["yaw_error_sigma_deg"])),
                        -yaw_clip, yaw_clip,
                    ))
                    prior = (
                        float(record["robot_x"]) + float(xy_error[0]),
                        float(record["robot_y"]) + float(xy_error[1]),
                        float(record["robot_yaw"]) + yaw_error,
                    )
                    covariance = np.diag([
                        float(regime["declared_xy_sigma_m"]) ** 2,
                        float(regime["declared_xy_sigma_m"]) ** 2,
                        math.radians(float(regime["declared_yaw_sigma_deg"])) ** 2,
                    ])
                    point = evaluate(record, cameras[record["camera_id"]], prior,
                                     covariance, model, protocol, nuisance=False)
                    nuisance = evaluate(record, cameras[record["camera_id"]], prior,
                                        covariance, model, protocol, nuisance=True)
                    row = {
                        "regime_id": regime_id, "draw": draw,
                        "pose_id": record["pose_id"], "position_id": record["position_id"],
                        "block_id": record["block_id"], "camera_id": record["camera_id"],
                        "reference_positive": record["reference_class"] == "positive",
                        "reference_class": record["reference_class"],
                        "detector_return_at_0_25": bool(record["detector_return_at_0.25"]),
                        "prior_error_xy_m": [float(xy_error[0]), float(xy_error[1])],
                        "prior_error_yaw_rad": yaw_error,
                    }
                    for arm, outcome in (("point", point), ("nuisance", nuisance)):
                        for field, value in outcome.items():
                            row[f"{arm}_{field}"] = value
                    rows.append(row)
                    ledger.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    sliver = named_sliver_check(cameras, protocol)
    regimes = {}
    failures = []
    rules = protocol["advance_rules"]
    for regime in protocol["perturbations"]["regimes"]:
        regime_id = regime["id"]
        point = summarize(rows, regime_id, "point")
        nuisance = summarize(rows, regime_id, "nuisance")
        selected = [row for row in rows if row["regime_id"] == regime_id]
        added_false = sum(
            row["nuisance_admitted"] and not row["point_admitted"]
            and not row["reference_positive"] for row in selected)
        added_false_fraction = added_false / len(selected)
        regimes[regime_id] = {
            "point_yaw": point,
            "nuisance_yaw": nuisance,
            "added_false_admissions": added_false,
            "added_false_admission_fraction_of_all_opportunities": added_false_fraction,
            "by_camera": subgroup_summaries(rows, regime_id, "nuisance", "camera_id"),
            "by_spatial_block": subgroup_summaries(rows, regime_id, "nuisance", "block_id"),
        }
        if nuisance["precision"] < rules["each_regime_precision_min"]:
            failures.append(f"{regime_id}:precision")
        if nuisance["stage07_error"]["above_0_25_fraction"] > rules["each_regime_catastrophic_fraction_max"]:
            failures.append(f"{regime_id}:catastrophic_fraction")
        if added_false_fraction > rules["each_regime_added_false_admission_fraction_of_all_opportunities_max"]:
            failures.append(f"{regime_id}:added_false_admissions")
        p95_limit = rules[f"{regime_id}_stage07_p95_m_max"]
        if nuisance["stage07_error"]["p95_m"] > p95_limit:
            failures.append(f"{regime_id}:stage07_p95")
        operational = nuisance["covariance"]["operational_Pxy_plus_R2C_containment"]
        if operational["95"] < rules["each_regime_operational_95_containment_min"]:
            failures.append(f"{regime_id}:operational_95_undercoverage")
        if operational["95"] > rules["each_regime_operational_95_containment_max"]:
            failures.append(f"{regime_id}:operational_95_overcoverage")
        if nuisance["covariance"]["operational_above_99_fraction"] > rules["each_regime_operational_above_99_fraction_max"]:
            failures.append(f"{regime_id}:operational_99_tail")
    if not sliver["pass"]:
        failures.append("camera_D_named_sliver")

    report = {
        "schema": "thesis_stage09_runtime_yaw_recovery_audit.v1",
        "status": "pass" if not failures else "fail",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": protocol["data_boundary"]["audit_role"],
        "advance_pass": not failures,
        "advance_failures": failures,
        "regimes": regimes,
        "named_camera_D_sliver": sliver,
        "protocol": str(protocol_path.relative_to(REPO)),
        "protocol_sha256": sha256(protocol_path),
        "stage06_inference_manifest_sha256": sha256(inference_manifest_path),
        "stage06_records_sha256": sha256(records_path),
        "stage07_measurement_model": str(model_path.relative_to(REPO)),
        "stage07_measurement_model_sha256": sha256(model_path),
        "opportunity_ledger": ledger_path.name,
        "opportunity_ledger_sha256": sha256(ledger_path),
        "opportunity_rows": len(rows),
        "final_audit_previously_accessed": True,
        "implementation": str(Path(__file__).resolve().relative_to(REPO)),
        "implementation_sha256": sha256(Path(__file__).resolve()),
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "advance_failures": failures,
        "opportunity_rows": len(rows),
        "regimes": {key: {
            "precision": value["nuisance_yaw"]["precision"],
            "recall": value["nuisance_yaw"]["reference_recall"],
            "p95_m": value["nuisance_yaw"]["stage07_error"]["p95_m"],
            "recovered": value["nuisance_yaw"]["recovered"],
            "added_false": value["added_false_admissions"],
        } for key, value in regimes.items()},
        "camera_D_sliver_pass": sliver["pass"],
        "report": str(report_path),
    }, indent=2))
    return 0 if report["advance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
