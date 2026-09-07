"""Historical exporter API, retired at the operational-data boundary.

The receipt-only warehouse_aws corpus has no physical acquisition identity,
pre-event belief provenance, or explicit frame-reception ledger. Existing frozen
artifacts remain readable, but these files cannot certify a new operational dataset.
The whitelist audit checks field names only; it never proves causal availability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
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


LEGACY_EXPORT_ERROR = (
    'receipt-only honest_campaign_v1 export is unsupported: physical image/opportunity '
    'identity, explicit acquisition outcomes and pre-event operational belief provenance '
    'are absent. A new manifest-bound event adapter is required; legacy ticks cannot '
    'be certified as independent operational observations.'
)


def iter_honest_campaign_raw_records(
    campaign_dir: str, cfg: ExporterConfig, stats: dict[str, int] | None = None
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Refuse an unidentifiable legacy population before reading or emitting rows."""
    raise RuntimeError(LEGACY_EXPORT_ERROR)


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
    """Refuse before creating output; archived artifacts are not modified."""
    raise RuntimeError(LEGACY_EXPORT_ERROR)
