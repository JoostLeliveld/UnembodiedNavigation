"""Ground anchoring and visibility inference from monocular depth.

The method: a fixed camera with known calibration already knows the depth of
every pixel whose ray lands on the warehouse floor. Those pixels are a free,
deployment-legal ruler. Fit a monocular prediction onto them robustly, and the
rest of the image becomes metric -- including the box in the aisle that the
floor model cannot explain. Back-project it, raycast to the robot body, and
report where the camera can and cannot see, with an explicit ``unknown``
wherever the single view had no evidence.

Typical use::

    from ground_anchoring import (
        CameraCalibration, DepthPrediction, DepthConvention, estimate_visibility,
    )

    result = estimate_visibility(prediction, calib, xs, ys, drivable=aisles)
    if result.is_valid:
        p_vis = result.visibility.p_visible

Boundary: nothing here reads oracle depth, simulator obstacle poses, or an
oracle visibility grid. Those are for scoring the output.
"""

from .contracts import (  # noqa: F401
    AnchorConfig,
    CameraCalibration,
    ContractViolation,
    DepthConvention,
    DepthConventionError,
    DepthPrediction,
    FitConfig,
    FloorPlane,
    Footprint,
    FrameStatus,
    GroundFit,
    MethodConfig,
    MetricDepth,
    RaycastConfig,
    TargetVolume,
    VisibilityField,
    VisibilityResult,
    covered_by_any,
)
from .conventions import (  # noqa: F401
    depth_from_fit_space,
    fit_space_for,
    sigma_to_depth,
    target_in_fit_space,
    to_optical_axis,
)
from .floor_anchors import (  # noqa: F401
    FloorAnchors,
    analytic_plane_depth,
    select_floor_anchors,
)
from .ground_fit import fit_ground_affine, predicted_depth_sigma  # noqa: F401
from .heightmap import (  # noqa: F401
    HeightMap,
    back_project,
    ground_visibility_mask,
    rasterize_heights,
)
from .io_contract import (  # noqa: F401
    METHOD_VISIBLE_KEYS,
    ORACLE_ONLY_KEYS,
    OracleAccessError,
    assert_no_oracle_access,
    calibration_from_parts,
    camera_from_record,
    load_prediction,
    method_visible_record,
    prediction_index,
    save_result,
)
from .pipeline import (  # noqa: F401
    METHOD_VERSION,
    config_fingerprint,
    estimate_visibility,
    estimate_visibility_batch,
)
from .raycast import SightlineField, line_of_sight_field  # noqa: F401
from .temporal import TemporalAnchorConfig, TemporalGroundAnchorFilter  # noqa: F401

__all__ = [
    "METHOD_VERSION",
    "METHOD_VISIBLE_KEYS",
    "ORACLE_ONLY_KEYS",
    "AnchorConfig",
    "CameraCalibration",
    "ContractViolation",
    "DepthConvention",
    "DepthConventionError",
    "DepthPrediction",
    "FitConfig",
    "FloorAnchors",
    "FloorPlane",
    "Footprint",
    "FrameStatus",
    "GroundFit",
    "HeightMap",
    "MethodConfig",
    "MetricDepth",
    "OracleAccessError",
    "RaycastConfig",
    "SightlineField",
    "TargetVolume",
    "TemporalAnchorConfig",
    "TemporalGroundAnchorFilter",
    "VisibilityField",
    "VisibilityResult",
    "analytic_plane_depth",
    "assert_no_oracle_access",
    "back_project",
    "calibration_from_parts",
    "camera_from_record",
    "config_fingerprint",
    "covered_by_any",
    "depth_from_fit_space",
    "estimate_visibility",
    "estimate_visibility_batch",
    "fit_ground_affine",
    "fit_space_for",
    "ground_visibility_mask",
    "line_of_sight_field",
    "load_prediction",
    "method_visible_record",
    "predicted_depth_sigma",
    "prediction_index",
    "rasterize_heights",
    "save_result",
    "select_floor_anchors",
    "sigma_to_depth",
    "target_in_fit_space",
    "to_optical_axis",
]
