"""Offline usable-observation dataset exporter (P2).

Builds one :class:`~reliability.observation_opportunity.ObservationOpportunity` per camera
per synchronized opportunity (including misses) from existing operational logs, evaluates
the frozen gate, and writes a versioned dataset + manifest with a GT-firewall audit.

Only the single-camera ``warehouse_aws`` honest-campaign adapter is implemented here
(method-development corpus, per the two-world rule). The adapter reads a strict whitelist
of operational columns; ground-truth / evaluation columns (``true_*``, ``gt_*``,
``localization_error_*``) are never read into a record. Belief position is the canonical
``planner_belief_x/y`` (STATE→BELIEF), joined to each perception frame by nearest stamp via
the self-checking ``campaign_metrics`` loader; ``truth`` is deliberately *not* required to
keep a row (the loader's own GT requirement is relaxed here on purpose).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import glob
import hashlib
import json
import math
import pathlib
import subprocess
from typing import Any, Iterator

from reliability.contracts import EVALUATION_ONLY_TOKENS, _contains_evaluation_key
from reliability.observation_gates import (
    UsableObservationGateConfig,
    evaluate_observation_opportunity,
)
from reliability.observation_opportunity import ObservationOpportunity

# Strict operational whitelist read from perception.csv. NO true_*/gt_*/localization_* here.
PERCEPTION_WHITELIST = (
    "log_stamp",
    "detected",
    "yolo_score_raw",
    "yolo_detected_after_threshold",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "obs_u",
    "obs_v",
    "pixel_pose_available",
    "pred_world_x",
)

# perception.csv columns that are GT / evaluation-only and must never be read as features.
PERCEPTION_FORBIDDEN = (
    "true_available",
    "true_x",
    "true_y",
    "true_yaw",
    "localization_error_m",
    "localization_error_calibrated_m",
    "state_pos_error",
)


@dataclass(frozen=True)
class ExporterConfig:
    """Dataset-mapping parameters (recorded in the manifest, separate from the gate)."""

    camera_id: str = "external_camera_aws"
    world: str = "warehouse_aws"
    detection_floor: float = 0.05  # yolo_score_raw >= this => a real robot detection exists
    stamp_tol_s: float = 0.3       # perception->belief nearest-stamp join tolerance
    min_spacing_s: float = 0.0     # temporal downsampling within a run (0 = keep all)
    holdout_routes: tuple[str, ...] = field(default_factory=tuple)  # -> split="test"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _run_id_from_path(perception_csv: str) -> tuple[str, str]:
    parts = pathlib.Path(perception_csv).parts
    idx = parts.index("honest_campaign_v1")
    route, cond, seed, exp = parts[idx + 1], parts[idx + 2], parts[idx + 3], parts[idx + 4]
    return route, f"{route}/{cond}/{seed}/{exp}"


def _campaign_metrics_dir(start: pathlib.Path) -> str:
    """Walk up from ``start`` to find scripts/geometry_visibility/campaign_metrics.py."""
    for parent in [start] + list(start.parents):
        candidate = parent / "scripts" / "geometry_visibility" / "campaign_metrics.py"
        if candidate.is_file():
            return str(candidate.parent)
    raise RuntimeError(f"could not locate scripts/geometry_visibility above {start}")


def _load_belief(experiment_csv: str) -> dict[str, Any]:
    """Canonical belief (planner_belief_x/y) + yaw, asserted by campaign_metrics."""
    import sys

    root = _campaign_metrics_dir(pathlib.Path(experiment_csv).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import campaign_metrics as cm  # canonical, self-checking loader

    run = cm.load_run(experiment_csv)  # asserts ||planner_belief - gt|| == belief_error_gt_m
    yaw = []
    for row in csv.DictReader(open(experiment_csv)):
        yaw.append(_f(row.get("planner_belief_yaw", "nan")))
    return {"stamp": run["stamp"], "bx": run["belief_x"], "by": run["belief_y"], "yaw": yaw}


def iter_honest_campaign_raw_records(
    campaign_dir: str, cfg: ExporterConfig, stats: dict[str, int] | None = None
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield (run_id, route_id, raw_record) for every single-camera opportunity.

    ``stats`` (if given) accumulates transparent drop accounting: a perception frame is
    dropped when it has no operational state to attach (belief NaN) or falls outside the
    join tolerance. On this corpus belief-NaN coincides with projection failure
    (pixel_pose_available=0) — dropping is correct (no state s -> no p_use(s) record) but
    must be visible.
    """
    import numpy as np

    if stats is not None:
        for key in ("rows_seen", "kept", "dropped_belief_nan", "dropped_stamp_tol", "dropped_min_spacing"):
            stats.setdefault(key, 0)

    pattern = str(pathlib.Path(campaign_dir) / "*/*/*/*/perception.csv")
    for perception_csv in sorted(glob.glob(pattern)):
        route_id, run_id = _run_id_from_path(perception_csv)
        experiment_csv = str(pathlib.Path(perception_csv).parent / "experiment.csv")
        belief = _load_belief(experiment_csv)
        est = belief["stamp"]
        last_kept_t = None
        for row in csv.DictReader(open(perception_csv)):
            # firewall: never let a forbidden GT/eval column into the record
            if row.get("detected") not in ("0", "1"):
                continue
            t = _f(row.get("log_stamp"))
            if not math.isfinite(t) or len(est) == 0:
                continue
            if stats is not None:
                stats["rows_seen"] += 1
            j = int(np.argmin(np.abs(est - t)))
            if abs(est[j] - t) > cfg.stamp_tol_s:
                if stats is not None:
                    stats["dropped_stamp_tol"] += 1
                continue
            if not math.isfinite(belief["bx"][j]):
                if stats is not None:
                    stats["dropped_belief_nan"] += 1
                continue
            if cfg.min_spacing_s > 0.0 and last_kept_t is not None and (t - last_kept_t) < cfg.min_spacing_s:
                if stats is not None:
                    stats["dropped_min_spacing"] += 1
                continue
            last_kept_t = t
            if stats is not None:
                stats["kept"] += 1

            score = _f(row.get("yolo_score_raw"))
            detection_received = math.isfinite(score) and score >= cfg.detection_floor
            proj_valid = row.get("pixel_pose_available") == "1" and math.isfinite(_f(row.get("pred_world_x")))
            bx_min, by_min = _f(row.get("bbox_xmin")), _f(row.get("bbox_ymin"))
            bx_max, by_max = _f(row.get("bbox_xmax")), _f(row.get("bbox_ymax"))
            yaw_j = belief["yaw"][j] if j < len(belief["yaw"]) and math.isfinite(belief["yaw"][j]) else 0.0

            raw: dict[str, Any] = {
                "timestamp": t,
                "run_id": run_id,
                "route_id": route_id,
                "camera_id": cfg.camera_id,
                "state_x": float(belief["bx"][j]),
                "state_y": float(belief["by"][j]),
                "state_yaw": float(yaw_j),
                "state_source": "BELIEF",
                "frame_expected": True,
                "frame_received": True,
                "frame_age_ms": None,
                "detection_received": bool(detection_received),
                "detector_class": "robot" if detection_received else None,
                "detector_confidence": (score if math.isfinite(score) else None),
                "bbox_xmin": bx_min if math.isfinite(bx_min) else None,
                "bbox_ymin": by_min if math.isfinite(by_min) else None,
                "bbox_xmax": bx_max if math.isfinite(bx_max) else None,
                "bbox_ymax": by_max if math.isfinite(by_max) else None,
                "selected_pixel_u": _f(row.get("obs_u")) if math.isfinite(_f(row.get("obs_u"))) else None,
                "selected_pixel_v": _f(row.get("obs_v")) if math.isfinite(_f(row.get("obs_v"))) else None,
                "projection_attempted": True,
                "projection_valid": bool(proj_valid),
                # disabled checks: stored as best-available proxies, NOT evaluated by the gate
                "association_attempted": False,
                "association_valid": False,
                "tracking_available": False,
                "tracking_valid": False,
                "accepted_by_localizer": bool(proj_valid),
            }
            yield run_id, route_id, raw


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _gt_firewall_audit() -> dict[str, Any]:
    """Prove the read whitelist and the output schema contain no GT/eval columns."""
    whitelist_ok = [c for c in PERCEPTION_WHITELIST if _contains_evaluation_key(c)]
    out_fields = list(ObservationOpportunity.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    out_eval = [c for c in out_fields if _contains_evaluation_key(c)]
    return {
        "whitelist_columns_read": list(PERCEPTION_WHITELIST),
        "forbidden_columns_excluded": list(PERCEPTION_FORBIDDEN),
        "eval_tokens": list(EVALUATION_ONLY_TOKENS),
        "whitelist_leak_hits": whitelist_ok,   # must be []
        "output_schema_leak_hits": out_eval,   # must be []
        "state_source": "BELIEF",
        "passed": (not whitelist_ok) and (not out_eval),
    }


def export_observation_dataset(
    campaign_dir: str,
    gate_config: UsableObservationGateConfig,
    exporter_config: ExporterConfig,
    output_dir: str,
    *,
    write_csv: bool = True,
) -> dict[str, Any]:
    """Export the dataset + manifest. Returns the manifest dict."""
    import pandas as pd

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    row_accounting: dict[str, int] = {}
    for _run_id, _route_id, raw in iter_honest_campaign_raw_records(
        campaign_dir, exporter_config, stats=row_accounting
    ):
        opp = evaluate_observation_opportunity(raw, gate_config)  # firewalls the raw record
        record = opp.to_dict()
        record.pop("source_labels", None)  # constant map; keep the table narrow
        rows.append(record)

    if not rows:
        raise RuntimeError(f"no observation opportunities found under {campaign_dir}")

    frame = pd.DataFrame(rows)
    frame["split"] = frame["route_id"].apply(
        lambda r: "test" if r in exporter_config.holdout_routes else "train"
    )
    # split integrity: a run may not straddle splits
    per_run_splits = frame.groupby("run_id")["split"].nunique()
    assert (per_run_splits == 1).all(), "a run spans multiple splits"

    parquet_path = out / "observations.parquet"
    frame.to_parquet(parquet_path, index=False)
    if write_csv:
        frame.to_csv(out / "observations.csv", index=False)

    # firewall on the written table itself
    output_leak = [c for c in frame.columns if _contains_evaluation_key(c)]
    firewall = _gt_firewall_audit()
    firewall["output_table_leak_hits"] = output_leak
    firewall["passed"] = firewall["passed"] and not output_leak
    assert firewall["passed"], f"GT firewall FAILED: {firewall}"

    def _rate(mask_col: str, denom_mask=None) -> float:
        sub = frame if denom_mask is None else frame[denom_mask]
        return float(sub[mask_col].mean()) if len(sub) else float("nan")

    det_mask = frame["detection_label"] == 1
    manifest: dict[str, Any] = {
        "schema_version": rows[0]["schema_version"],
        "git_commit": _git_commit(),
        "created_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "campaign_dir": campaign_dir,
        "world": exporter_config.world,
        "cameras": [exporter_config.camera_id],
        "state_source": "BELIEF",
        "gate_id": gate_config.gate_id,
        "gate_config": gate_config.to_dict(),
        "gate_config_hash": gate_config.config_hash(),
        "exporter_config": exporter_config.to_dict(),
        "exporter_config_hash": exporter_config.config_hash(),
        "confidence_threshold": gate_config.confidence_threshold,
        "detection_floor": exporter_config.detection_floor,
        "min_spacing_s": exporter_config.min_spacing_s,
        "routes": sorted(frame["route_id"].unique().tolist()),
        "runs": int(frame["run_id"].nunique()),
        "row_count": int(len(frame)),
        "row_accounting": row_accounting,
        "class_balance": {
            "p_det_mean": _rate("detection_label"),
            "p_qual_mean_given_det": _rate("quality_label", det_mask),
            "p_use_mean": _rate("usable_label"),
        },
        "failure_reason_counts": frame["failure_reason"].value_counts().to_dict(),
        "rows_per_route": frame.groupby("route_id").size().to_dict(),
        "rows_per_run": frame.groupby("run_id").size().to_dict(),
        "split_counts": frame.groupby("split").size().to_dict(),
        "split_grouping_key": "route_id",
        "holdout_routes": list(exporter_config.holdout_routes),
        "gt_firewall_audit": firewall,
        "artifacts": {
            "parquet": str(parquet_path),
            "csv": str(out / "observations.csv") if write_csv else None,
        },
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return manifest
