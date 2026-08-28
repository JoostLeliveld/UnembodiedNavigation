#!/usr/bin/env python3
"""Live shadow-mode wrapper around :class:`reliability.camera_manager.CameraManager`.

Runs the SAME operational selection pipeline as the offline M8 replay —
contract parsing, ground-plane projection, GP quality providers, hysteretic
selection, handover covariance inflation — against the live four-camera
detector streams, and logs every decision WITHOUT authority.

Authority model (research_story ch.09 gate: "active planner handover stays
disabled" until the mapping/overlap/covariance/replay gates pass):

* ``authority: shadow`` (default) — publish the decision JSON on
  ``~decision_topic`` and the selected, handover-adjusted observation on
  ``~selected_topic``.  Nothing in the estimator or planner graph consumes
  these topics; this is the step-6 "log decisions alongside the existing
  estimator" rung.
* ``authority: active`` — additionally republish the selected observation as a
  ``PoseWithCovarianceStamped`` on ``~active_output_topic`` (default
  ``/state/bev``, the contract produced by ``pixel_to_bev_state_node`` in the
  single-camera stack).  Position-only: the orientation is identity with a
  non-informative yaw variance.  This mode is for the gated closed-loop
  campaign only and must stay off until the release gates pass.

Inputs are operational-only: detector contracts, calibration-derived
projection, GP trust artifacts.  Ground truth cannot enter this node
(leakage firewall).
"""

from __future__ import annotations

import collections
from collections import deque
from dataclasses import replace
import json
import math
from pathlib import Path

from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.contracts import CameraObservation, ContractValidationError
from reliability.fusion import (
    MapObservation,
    SequentialFusionResult,
    distance_angle_weighted_fusion_2d,
    independent_measurement_fusion_2d,
    map_observations_to_json,
    network_pooled_fusion_2d,
    select_smallest_covariance,
)
from reliability.handover import HandoverUncertaintyConfig, handover_adjusted_observation
from reliability.projection import (
    camera_model_from_world,
    project_observation_to_world,
    project_observation_to_world_with_covariance,
)
from reliability.silhouette_observation import (
    equivalent_position_measurement,
    plausibility_reasons,
)
from reliability.providers import GridMapReliabilityProvider
from reliability.replay import ReplayConfig, ReplayMode, _with_provider_quality

try:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - keeps the library importable without ROS
    rclpy = None
    Node = object
    PoseWithCovarianceStamped = None
    String = None


DEFAULT_CAMERA_IDS = ["camera_A", "camera_B", "camera_C", "camera_D"]
DEFAULT_MODEL_INCLUDES = [
    "external_camera",
    "external_camera_b",
    "external_camera_c",
    "external_camera_d",
]
# Yaw variance for the position-only active output: deliberately
# non-informative so no consumer can mistake the selection for a heading fix.
NONINFORMATIVE_YAW_VAR = float(math.pi**2)
#: The commissioned sensor: ONE pixel-noise number, measured once on the frozen detector and
#: pushed through each camera's own geometry. Distance and viewing angle are already inside
#: the result, because they are what changes that geometry -- so nothing here penalises range
#: or angle by hand, and no per-camera covariance is fitted. Its stated covariance is the
#: claim under test, so nothing downstream may floor or inflate it except the commissioned
#: bias floor.
COMMISSIONED_COVARIANCE = "commissioned_sigma_px"
SUPPORTED_COVARIANCE_PROFILES = (COMMISSIONED_COVARIANCE,)
#: How a detector's box is turned into a statement about where the robot is.
OBSERVATION_MODEL_HULL = "hull"                  # predict the box from the robot's shape
OBSERVATION_MODEL_RAW_BOX = "raw_box"            # the box bottom-centre IS the robot
OBSERVATION_MODEL_FIXED_OFFSET = "fixed_offset"  # ... plus one fixed push away from the camera
SUPPORTED_OBSERVATION_MODELS = (
    OBSERVATION_MODEL_HULL,
    OBSERVATION_MODEL_RAW_BOX,
    OBSERVATION_MODEL_FIXED_OFFSET,
)


def load_commissioned_sigma_px(calibration_path: str) -> float:
    """The frozen detector noise, in pixels, read from the commissioning artifact.

    Read rather than typed in, so an arm cannot be driven against a remembered number: the
    covariance every fusion rule is judged on comes from this one value.
    """

    payload = json.loads(Path(calibration_path).read_text())
    sigma_px = float(payload["calibration"]["sigma_px"])
    if not math.isfinite(sigma_px) or sigma_px <= 0.0:
        raise ValueError(f"sigma_px in {calibration_path} is not a positive number")
    return sigma_px


def load_commissioned_sigma_px_by_camera(calibration_path: str) -> dict[str, float]:
    """Each camera's own detector noise, in pixels, if commissioning measured it.

    Commissioning has always produced these -- the runtime simply pooled them into one number
    and threw the rest away. Measured on the drives, one camera needs about three times the
    variance of another, so the pooled number cannot be right for both.
    """

    payload = json.loads(Path(calibration_path).read_text())
    table = payload.get("calibration", {}).get("sigma_px_by_camera") or {}
    out = {}
    for camera_id, value in table.items():
        sigma = float(value)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma_px for {camera_id} in {calibration_path} is not positive")
        out[str(camera_id)] = sigma
    return out


def commissioned_pixel_covariance(sigma_px: float):
    """``R_pix = sigma_px^2 I`` -- isotropic in pixels, identical for every camera."""

    variance = float(sigma_px) ** 2
    return ((variance, 0.0), (0.0, variance))


def offset_away_from_camera(world_xy, camera_position, offset_m: float):
    """Push a ground reading away from its camera by a fixed distance.

    The fixed-offset observation model, and the whole of it: the box's bottom edge is the
    robot's nearest point to the camera, so its floor projection lands short, along the
    viewing bearing. One number cannot follow the 11 cm that gap swings as the robot turns --
    which is what this arm exists to demonstrate rather than to hide.
    """

    dx = float(world_xy[0]) - float(camera_position[0])
    dy = float(world_xy[1]) - float(camera_position[1])
    norm = math.hypot(dx, dy)
    if norm <= 1.0e-9:
        return (float(world_xy[0]), float(world_xy[1]))
    scale = float(offset_m) / norm
    return (float(world_xy[0]) + dx * scale, float(world_xy[1]) + dy * scale)


def _nearest_state_pose(
    history: list[tuple[float, tuple[float, float, float]]] | deque,
    timestamp_s: float,
    *,
    max_delta_s: float,
) -> tuple[float, float, float] | None:
    """The operational belief pose nearest a camera capture time, yaw included.

    Returns ``None`` when the history is empty or the nearest entry is too old, and also
    when the entry predates the yaw-carrying history format -- an observation function
    evaluated at a guessed heading is not a correction.
    """

    if (
        not history
        or not math.isfinite(timestamp_s)
        or not math.isfinite(max_delta_s)
        or max_delta_s < 0.0
    ):
        return None
    nearest_stamp, nearest_pose = min(
        history, key=lambda row: abs(float(row[0]) - timestamp_s)
    )
    if abs(float(nearest_stamp) - timestamp_s) > max_delta_s:
        return None
    if len(nearest_pose) < 3:
        return None
    return float(nearest_pose[0]), float(nearest_pose[1]), float(nearest_pose[2])


def _nearest_state_xy(
    history: list[tuple[float, tuple[float, float]]] | deque,
    timestamp_s: float,
    *,
    max_delta_s: float,
) -> tuple[float, float] | None:
    """Return the operational state prediction nearest a camera capture time."""

    if (
        not history
        or not math.isfinite(timestamp_s)
        or not math.isfinite(max_delta_s)
        or max_delta_s < 0.0
    ):
        return None
    nearest_stamp, nearest_xy = min(
        history, key=lambda row: abs(float(row[0]) - timestamp_s)
    )
    if abs(float(nearest_stamp) - timestamp_s) > max_delta_s:
        return None
    return float(nearest_xy[0]), float(nearest_xy[1])


def propagate_correction_to_now(xy, covariance_m2, pose_then, pose_now, *,
                                drift_std_m_per_s: float, dt_s: float,
                                residual_interval_s: float = 0.0):
    """Carry a correction forward from the pose it describes to the pose it will be used on.

    A camera correction describes where the robot WAS when the frame was taken. Measured on 24
    drives, the correction the filter holds is ~400 ms old and the error it appears to have is
    almost entirely that staleness: scored against the pose 0.35 s earlier the same corrections
    are accurate to 2.3 cm rather than 8.2 cm, and the effect is a displacement BEHIND the
    robot, proportional to speed. That is a bias, and inflating the covariance cannot fix a
    bias -- so the correction is moved instead of being distrusted.

    The displacement comes from the filter's own recent motion, so an error in the belief's
    absolute position cancels in the difference. What does not cancel is the motion estimate's
    own error over the interval, which is added to the covariance as an isotropic term.
    """

    dx = float(pose_now[0]) - float(pose_then[0])
    dy = float(pose_now[1]) - float(pose_then[1])
    dt = max(float(dt_s), 0.0)
    grown = max(float(drift_std_m_per_s) * dt, 0.0) ** 2
    cxx = float(covariance_m2[0][0]) + grown
    cxy = float(covariance_m2[0][1])
    cyy = float(covariance_m2[1][1]) + grown

    # A correction is carried to the instant it is PUBLISHED, and consumed a little later. That
    # leftover interval leaves a displacement along the direction of travel, proportional to
    # speed -- measured at -1.72 cm on the repaired pipeline, against 1.89 cm of spread. It is a
    # bias, so it does not average away, and a filter fusing many corrections will shrink
    # straight through it: the belief's ellipse reaches 1.12 cm on the axis where the error is
    # 2.63 cm. It cannot be subtracted (its size depends on a delay nobody measures per
    # correction), so it is declared as uncertainty ALONG TRAVEL, which is where it acts.
    speed = math.hypot(dx, dy) / dt if dt > 0.0 else 0.0
    residual = max(float(residual_interval_s), 0.0) * speed
    if residual > 0.0 and (dx or dy):
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        along = residual ** 2
        cxx += along * ux * ux
        cxy += along * ux * uy
        cyy += along * uy * uy

    covariance = ((cxx, cxy), (cxy, cyy))
    return (float(xy[0]) + dx, float(xy[1]) + dy), covariance, (dx, dy)


def _fusion_report_covariance(covariance_m2, *, common_mode_std_m: float = 0.0):
    """Add back the part of the error the cameras make TOGETHER, and nothing else.

    Every fusion rule here treats the cameras as making independent mistakes, so combining them shrinks
    the stated ellipse as though the shared part shrank too -- and it does not. Measured on
    the drives of 2026-08-27: each camera's own stated ellipse is about right, and the moment
    two are combined the result is 3.8x overconfident. Adding the shared part back after the
    combination is the correction, and unlike a per-camera inflation it cannot be washed out
    by adding more cameras.

    This is a COMMISSIONED constant, measured once against ground truth on a commissioning
    drive, exactly as the detector's pixel noise is. It is not read from any live signal.
    """

    shared = max(float(common_mode_std_m), 0.0) ** 2
    return (
        (float(covariance_m2[0][0]) + shared, float(covariance_m2[0][1])),
        (float(covariance_m2[1][0]), float(covariance_m2[1][1]) + shared),
    )


FUSION_RULE_LEGACY = "legacy"
FUSION_RULE_BEST_SINGLE = "best_single"
FUSION_RULE_DISTANCE_ANGLE = "distance_angle"
FUSION_RULE_INDEPENDENT = "independent"
FUSION_RULE_NETWORK = "network"
#: The four arms of the fusion comparison, plus the historical default. Every rule sees the
#: same admitted observations and the same disagreement gate, so the rule is the only thing
#: that differs between arms -- the gate is shared method, not a treatment.
SUPPORTED_FUSION_RULES = (
    FUSION_RULE_LEGACY,
    FUSION_RULE_BEST_SINGLE,
    FUSION_RULE_DISTANCE_ANGLE,
    FUSION_RULE_INDEPENDENT,
    FUSION_RULE_NETWORK,
)


def _combine_by_rule(accepted, *, rule: str, camera_positions_m):
    """Turn the accepted observations into one measurement, by the arm's rule."""

    if rule == FUSION_RULE_BEST_SINGLE:
        chosen = select_smallest_covariance(accepted)
        return chosen.xy_m, chosen.covariance_m2, (chosen.camera_id,)
    used = tuple(observation.camera_id for observation in accepted)
    if rule == FUSION_RULE_DISTANCE_ANGLE:
        if camera_positions_m is None:
            raise ValueError("the distance_angle rule needs camera positions")
        mean, covariance = distance_angle_weighted_fusion_2d(accepted, camera_positions_m)
    elif rule == FUSION_RULE_NETWORK:
        mean, covariance = network_pooled_fusion_2d(accepted)
    elif rule in (FUSION_RULE_INDEPENDENT, FUSION_RULE_LEGACY):
        mean, covariance = independent_measurement_fusion_2d(accepted)
    else:
        raise ValueError(f"unsupported fusion_rule {rule!r}")
    return mean, covariance, used


def _gated_fusion(
    observations: list[MapObservation],
    *,
    disagreement_gate_m: float,
    rule: str = FUSION_RULE_LEGACY,
    camera_positions_m=None,
    belief_floors=None,
) -> SequentialFusionResult:
    """Gate around the robust centre, then combine what survives by ``rule``.

    Prior-free by construction: the cameras are combined into one measurement and the
    planner's own filter is the only thing holding a belief. Filtering here as well would
    count each camera twice.

    ``belief_floors`` maps camera id to that sighting's bias floor. The floors of the
    cameras that were USED are combined and applied to the combined covariance, which is
    the only place a floor does what it is for: added to each camera's R instead, a
    persistent error would be treated as fresh noise and shrink like 1/N from a larger
    start. ``None`` (the default, and what an uncommissioned floor slope yields) leaves the
    combined covariance untouched.
    """

    gate = float(disagreement_gate_m)
    if not math.isfinite(gate) or gate <= 0.0:
        raise ValueError("disagreement_gate_m must be finite and positive")
    if not observations:
        raise ValueError("at least one observation is required")

    def median(values):
        ordered = sorted(float(value) for value in values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    centre = (
        median(observation.xy_m[0] for observation in observations),
        median(observation.xy_m[1] for observation in observations),
    )
    accepted = []
    rejected = []
    # How far each camera sits from what the others saw, in metres -- the quantity this gate
    # actually decides on, reported so a reader can see why a camera was dropped. It is a
    # distance, not a normalised innovation: no prior is involved here.
    residuals = {}
    for observation in observations:
        residual = math.hypot(
            float(observation.xy_m[0]) - centre[0],
            float(observation.xy_m[1]) - centre[1],
        )
        residuals[observation.camera_id] = residual
        if residual <= gate:
            accepted.append(observation)
        else:
            rejected.append(observation.camera_id)
    if not accepted:
        # The caller treats an empty accepted set as no measurement.
        return SequentialFusionResult(
            mean_xy=centre,
            covariance_m2=((1.0, 0.0), (0.0, 1.0)),
            accepted_camera_ids=(),
            rejected_camera_ids=tuple(rejected),
            nis_by_camera=residuals,
        )
    mean, covariance, used = _combine_by_rule(
        accepted, rule=rule, camera_positions_m=camera_positions_m)
    if belief_floors:
        from reliability.bias_floor import apply_belief_floor, combine_floors

        chosen = [belief_floors[cam] for cam in used if cam in belief_floors]
        if chosen:
            covariance = apply_belief_floor(covariance, combine_floors(chosen))
    return SequentialFusionResult(
        mean_xy=mean,
        covariance_m2=covariance,
        accepted_camera_ids=tuple(used),
        rejected_camera_ids=tuple(rejected),
        nis_by_camera=residuals,
    )


def _synchronous_fusion_candidates(
    manager: CameraManager,
    *,
    now_s: float,
    observations: list[MapObservation],
    max_timestamp_spread_s: float,
) -> tuple[
    list[MapObservation],
    dict[str, float],
    dict[str, tuple[str, ...]],
    list[str],
]:
    """Apply manager eligibility and discard old cached camera frames."""
    if not math.isfinite(max_timestamp_spread_s) or max_timestamp_spread_s < 0.0:
        raise ValueError("max_timestamp_spread_s must be finite and non-negative")
    candidates, scores, rejected = manager.eligible_observations(
        timestamp_s=now_s, observations=observations
    )
    if not candidates:
        return [], scores, rejected, []
    newest_stamp_s = max(float(obs.timestamp_s) for obs in candidates.values())
    synchronous = [
        obs for obs in candidates.values()
        if newest_stamp_s - float(obs.timestamp_s) <= max_timestamp_spread_s
    ]
    time_skew_rejected = sorted(
        camera_id for camera_id, obs in candidates.items()
        if newest_stamp_s - float(obs.timestamp_s) > max_timestamp_spread_s
    )
    return synchronous, scores, rejected, time_skew_rejected


class CameraManagerNode(Node):
    """Shadow-mode (optionally active) live camera selection."""

    def __init__(self) -> None:
        super().__init__("camera_manager_node")
        self.declare_parameter("camera_ids", DEFAULT_CAMERA_IDS)
        self.declare_parameter(
            "observation_topic_template", "/perception/camera_observation/{camera_id}"
        )
        self.declare_parameter("world_sdf", "")
        self.declare_parameter("camera_model_includes", DEFAULT_MODEL_INCLUDES)
        self.declare_parameter("gp_artifacts", [""])
        # Launch-friendly alternative to the aligned list: one string with a
        # {camera_id} placeholder, e.g.
        # ".../final_02/gp/{camera_id}/det_hit_expected_kernel_gp.npz".
        self.declare_parameter("gp_artifact_template", "")
        self.declare_parameter("decision_rate_hz", 5.0)
        # Projection takes no parameters: it is inverse perspective mapping, the
        # box-bottom ray intersected with the floor plane.  The contact-plane
        # constant and the projection_calibration artifact were deleted 2026-08-07
        # because every fitted correction measured worse than none -- see
        # logs/studies/pixel_ground_path/e7_ipm_zero_parameter/RESULTS.md.
        self.declare_parameter("require_gp_artifacts", True)
        self.declare_parameter("frame_id", "map_bev")
        self.declare_parameter("authority", "shadow")
        self.declare_parameter("decision_topic", "/reliability/camera_manager/decision")
        # read only to measure how far the robot moved while a correction was in flight
        self.declare_parameter("odometry_topic", "/odom_noisy")
        self.declare_parameter("selected_topic", "/reliability/camera_manager/selected_observation")
        self.declare_parameter("active_output_topic", "/state/bev")
        # GP reliability is a property of the predicted robot location, not of
        # the camera measurement being scored.  Query a timestamp-matched
        # operational belief; the measurement is used only during bootstrap.
        self.declare_parameter("reliability_query_topic", "/planner_belief")
        self.declare_parameter("reliability_query_max_time_delta_s", 0.35)
        # Operational gates: defaults mirror CameraManagerConfig; override from
        # the frozen study/protocol config in the launch file, never here.
        defaults = CameraManagerConfig()
        # Covariance-weighted fusion of ALL in-view cameras (Joseph sequential
        # update with NIS/disagreement gating) instead of hard single-camera
        # selection. Default False preserves the selection path.
        self.declare_parameter("fusion_mode", False)
        # Publish the PER-CAMERA map observations alongside the fused pose, so a
        # consumer can fold them into its own filter one at a time instead of
        # receiving a single pre-fused pose. Purely additive: the fused/selected
        # outputs are unchanged, so this is safe to leave on.
        self.declare_parameter("publish_map_observations", False)
        self.declare_parameter(
            "map_observations_topic", "/reliability/camera_manager/map_observations"
        )
        self.declare_parameter("fusion_disagreement_gate_m", 0.6)
        # Bias floor, DESIGN_LOCK D5 layer 3, in metres of along-ray bias per metre of
        # range. 0.0 disables it, which is the historical behaviour and stays the default:
        # a floor changes what the filter is permitted to believe, so it is switched on by
        # a run's configuration and never inherited silently.
        #
        # The measured value is 0.0016 -- the bound over every (camera, range) cell of the
        # frozen detector reading (D22). Without a floor the fused belief measured 2.5x
        # overconfident on held-out poses and 6.9x within 9 m; at 0.0016 it measured 1.17
        # against a chi-square-2 median of 1.386. The range DEPENDENCE is not yet
        # demonstrated -- a constant floor of similar size scores the same in every
        # stratum -- so treat this as a well-sized floor, not a validated shape.
        # The observation function. False keeps the deployed assumption that the
        # detector's bottom-centre pixel back-projects to the robot's CENTRE; True states
        # the silhouette function instead -- the pixel back-projects to where the projected
        # visual hull's bottom-centre lands, which is between the robot and the camera.
        # Measured, the deployed assumption costs a 7.78 cm typical miss of which 6.34 cm
        # is systematic and heading-dependent, so a filter fed it needs 8.4x the covariance
        # a zero-mean filter would need. The correction is evaluated at the filter's OWN
        # prior pose and yaw, needs no ground truth, and is exactly equivalent to a
        # general-H update (`reliability.silhouette_observation`). Default True: with the
        # correction off, every covariance this node states is a covariance for a
        # measurement that is not the quantity the filter thinks it is.
        self.declare_parameter("silhouette_observation_correction", True)
        self.declare_parameter("bias_floor_along_slope_m_per_m", 0.0)
        self.declare_parameter("bias_floor_across_slope_m_per_m", 0.00035)
        self.declare_parameter("fusion_max_timestamp_spread_s", 0.25)
        # commissioned_sigma_px is the only profile: R_pix = sigma_px^2 I from the frozen
        # calibration, with each camera's own geometry turning it into an ellipse on the
        # floor. Kept as a parameter so a run's provenance states which sensor model it used.
        self.declare_parameter("covariance_profile", COMMISSIONED_COVARIANCE)
        # The number is read from the artifact rather than typed in; set commissioned_sigma_px
        # only to override it deliberately, and say so in the run's provenance.
        self.declare_parameter("commissioned_calibration_path", "")
        self.declare_parameter("commissioned_sigma_px", 0.0)
        # Use each camera's own commissioned noise instead of the pooled one. Commissioning
        # already measures both; pooling them is a choice, and on these cameras a wrong one.
        self.declare_parameter("commissioned_per_camera_sigma", False)
        # The part of the error the cameras make TOGETHER, in metres, added back after they
        # are combined. 0 reproduces the previous behaviour, in which fusing two cameras
        # produced an answer 3.8x more confident than it had any right to be.
        self.declare_parameter("fusion_common_mode_std_m", 0.0)
        # What the detector's box is taken to mean. hull = predict the box from the robot's
        # shape (the frozen method); raw_box = the box bottom-centre IS the robot;
        # fixed_offset = the same point pushed a fixed distance away from the camera.
        # Which rule turns several cameras into one measurement. "legacy" keeps the
        # historical behaviour of this node exactly; the other four are the fusion arms.
        # A correction describes where the robot WAS. Off (the historical behaviour) it is
        # applied as if it described now, which measured 8.2 cm of lag bias at 0.22 m/s.
        self.declare_parameter("correction_timestamp_compensation", False)
        # Uncertainty the propagation itself adds, per second of age. The interval is ~0.35 s,
        # so at 0.05 m/s this contributes under 2 cm -- deliberately small, because the motion
        # over that interval is short and the belief's absolute error cancels in the difference.
        self.declare_parameter("correction_propagation_drift_std_m_per_s", 0.05)
        # How long after being carried forward a correction is actually consumed. It leaves a
        # displacement of `speed * this` along the direction of travel, which is a bias rather
        # than noise, so it is declared as uncertainty in that direction instead of being
        # pretended away. Measured on this pipeline: ~50 ms.
        self.declare_parameter("correction_residual_interval_s", 0.05)
        self.declare_parameter("fusion_rule", FUSION_RULE_LEGACY)
        # The admission check. It compares the detected box against the box the robot's own
        # shape predicts -- tall enough, right width, bottom edge where the contact point should
        # be, not touching the frame edge -- and it needs no ground truth. Commissioning ran it
        # and reported 1.44 cm readings; the live pipeline did not, and swallowed the 30% it
        # refuses, whose median error is 24 cm and whose worst is 122 cm. Default true: a
        # correction built from a robot whose feet are hidden is not a measurement.
        self.declare_parameter("admission_gate", True)
        self.declare_parameter("observation_model", OBSERVATION_MODEL_HULL)
        self.declare_parameter("fixed_offset_m", 0.0)
        self.declare_parameter("min_spatial_trust", defaults.min_spatial_trust)
        self.declare_parameter("min_association_confidence", defaults.min_association_confidence)
        self.declare_parameter("max_measurement_age_s", defaults.max_measurement_age_s)
        self.declare_parameter("age_decay_s", defaults.age_decay_s)
        self.declare_parameter("candidate_score_margin", defaults.candidate_score_margin)
        self.declare_parameter(
            "required_consecutive_better_frames", defaults.required_consecutive_better_frames
        )
        self.declare_parameter(
            "max_cross_camera_disagreement_m", defaults.max_cross_camera_disagreement_m
        )
        self.declare_parameter("max_overlap_time_delta_s", defaults.max_overlap_time_delta_s)
        self.declare_parameter(
            "require_consistency_when_source_available",
            defaults.require_consistency_when_source_available,
        )
        self.declare_parameter("fallback_on_active_camera_loss", defaults.fallback_on_active_camera_loss)

        self.camera_ids = [str(item) for item in self.get_parameter("camera_ids").value]
        template = str(self.get_parameter("observation_topic_template").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.authority = str(self.get_parameter("authority").value).strip().lower()
        if self.authority not in ("shadow", "active"):
            raise ValueError(f"authority must be 'shadow' or 'active', got {self.authority!r}")

        world_sdf = str(self.get_parameter("world_sdf").value)
        includes = [str(item) for item in self.get_parameter("camera_model_includes").value]
        if not world_sdf:
            raise ValueError("world_sdf parameter is required for projection")
        if len(includes) != len(self.camera_ids):
            raise ValueError("camera_model_includes must align with camera_ids")
        self.camera_models = {
            camera_id: camera_model_from_world(world_sdf, include_name=include)
            for camera_id, include in zip(self.camera_ids, includes)
        }
        artifacts = [str(item) for item in self.get_parameter("gp_artifacts").value]
        if artifacts == [""]:
            artifacts = []
        template_artifact = str(self.get_parameter("gp_artifact_template").value)
        if not artifacts and template_artifact:
            artifacts = [
                template_artifact.format(camera_id=camera_id) for camera_id in self.camera_ids
            ]
        if artifacts and len(artifacts) != len(self.camera_ids):
            raise ValueError("gp_artifacts must be empty or align with camera_ids")
        if bool(self.get_parameter("require_gp_artifacts").value) and len(artifacts) != len(self.camera_ids):
            raise ValueError("one frozen GP artifact per camera is required by the commissioning contract")
        providers = {}
        for camera_id, artifact in zip(self.camera_ids, artifacts):
            if artifact:
                providers[camera_id] = GridMapReliabilityProvider.from_npz(
                    Path(artifact), camera_id=camera_id, out_of_bounds_policy="min"
                )
        # One ReplayConfig instance carries the provider set and the exact
        # score->covariance constants of the offline M8 pipeline, so shadow
        # decisions stay comparable to replay decisions.
        self.replay_config = ReplayConfig(
            mode=ReplayMode.HYSTERETIC_HANDOVER_SELECTION,
            quality_providers=providers,
        )

        self.manager = CameraManager(
            CameraManagerConfig(
                min_spatial_trust=float(self.get_parameter("min_spatial_trust").value),
                min_association_confidence=float(
                    self.get_parameter("min_association_confidence").value
                ),
                max_measurement_age_s=float(self.get_parameter("max_measurement_age_s").value),
                age_decay_s=float(self.get_parameter("age_decay_s").value),
                candidate_score_margin=float(self.get_parameter("candidate_score_margin").value),
                required_consecutive_better_frames=int(
                    self.get_parameter("required_consecutive_better_frames").value
                ),
                max_cross_camera_disagreement_m=float(
                    self.get_parameter("max_cross_camera_disagreement_m").value
                ),
                max_overlap_time_delta_s=float(
                    self.get_parameter("max_overlap_time_delta_s").value
                ),
                require_consistency_when_source_available=bool(
                    self.get_parameter("require_consistency_when_source_available").value
                ),
                fallback_on_active_camera_loss=bool(
                    self.get_parameter("fallback_on_active_camera_loss").value
                ),
                allowed_camera_ids=tuple(self.camera_ids),
            )
        )
        self.handover_config = HandoverUncertaintyConfig()
        self._latest: dict[str, CameraObservation] = {}
        self._belief_query_history = deque(maxlen=400)
        #: Odometry, kept purely to measure how far the robot moved between the pose a
        #: correction describes and the pose it is used on. The belief can be used for this
        #: too, but it is corrected and therefore jumps; odometry is smooth over the ~0.3 s
        #: that matters, and its absolute drift cancels in the difference. Measured: belief
        #: deltas left ~2 cm on the table that odometry deltas do not.
        self._odom_history = deque(maxlen=600)
        self._reliability_query_source_by_camera: dict[str, str] = {}
        self._previous_camera_id: str | None = None
        self._previous_observation: MapObservation | None = None

        self.decision_pub = self.create_publisher(
            String, str(self.get_parameter("decision_topic").value), 10
        )
        self.selected_pub = self.create_publisher(
            PoseWithCovarianceStamped, str(self.get_parameter("selected_topic").value), 10
        )
        self.active_pub = None
        if self.authority == "active":
            self.active_pub = self.create_publisher(
                PoseWithCovarianceStamped, str(self.get_parameter("active_output_topic").value), 10
            )
            self.get_logger().warn(
                "authority=active: selected observations WILL be published to "
                f"{self.get_parameter('active_output_topic').value}. This is only "
                "valid after the ch.09 release gates pass."
            )

        self.fusion_mode = bool(self.get_parameter("fusion_mode").value)
        self.map_observations_pub = None
        if bool(self.get_parameter("publish_map_observations").value):
            self.map_observations_pub = self.create_publisher(
                String, str(self.get_parameter("map_observations_topic").value), 10
            )
        self.fusion_disagreement_gate_m = float(self.get_parameter("fusion_disagreement_gate_m").value)
        self.silhouette_correction = bool(
            self.get_parameter("silhouette_observation_correction").value)
        self._silhouette_status_by_camera: dict[str, str] = {}
        # What the detector itself said about each reading, kept beside the reading so the
        # log can ask whether the detector's own confidence predicts how wrong it was.
        # Confidence is NOT used to weight anything here -- it is recorded and nothing else.
        self._detection_extras_by_camera: dict[str, dict[str, float]] = {}
        self.bias_floor_along_slope = float(
            self.get_parameter("bias_floor_along_slope_m_per_m").value)
        self.bias_floor_across_slope = float(
            self.get_parameter("bias_floor_across_slope_m_per_m").value)
        if self.bias_floor_along_slope < 0.0 or self.bias_floor_across_slope < 0.0:
            raise RuntimeError("bias floor slopes must be non-negative")
        self.fusion_max_timestamp_spread_s = float(
            self.get_parameter("fusion_max_timestamp_spread_s").value
        )
        if (
            not math.isfinite(self.fusion_max_timestamp_spread_s)
            or self.fusion_max_timestamp_spread_s < 0.0
        ):
            raise ValueError("fusion_max_timestamp_spread_s must be finite and non-negative")
        self.covariance_profile = str(
            self.get_parameter("covariance_profile").value
        ).strip().lower()
        if self.covariance_profile not in SUPPORTED_COVARIANCE_PROFILES:
            raise ValueError(
                "covariance_profile must be one of "
                f"{SUPPORTED_COVARIANCE_PROFILES}, got {self.covariance_profile!r}"
            )
        self.commissioned_sigma_px = 0.0
        self.fusion_common_mode_std_m = float(
            self.get_parameter("fusion_common_mode_std_m").value)
        # The commissioned sensor model, the only one: R_pix from the frozen calibration.
        self.commissioned_pixel_cov_by_camera: dict[str, tuple] = {}
        self.per_camera_sigma = bool(
            self.get_parameter("commissioned_per_camera_sigma").value)
        stated = float(self.get_parameter("commissioned_sigma_px").value)
        path = str(self.get_parameter("commissioned_calibration_path").value).strip()
        if stated > 0.0:
            self.commissioned_sigma_px = stated
            source = "the commissioned_sigma_px parameter"
        elif path:
            self.commissioned_sigma_px = load_commissioned_sigma_px(path)
            source = path
        else:
            raise ValueError(
                "covariance_profile=commissioned_sigma_px needs either "
                "commissioned_calibration_path or commissioned_sigma_px; refusing to "
                "invent the detector's noise"
            )
        self.commissioned_pixel_cov = commissioned_pixel_covariance(
            self.commissioned_sigma_px)
        self.commissioned_pixel_cov_by_camera = {}
        if self.per_camera_sigma and path:
            for camera_id, sigma in load_commissioned_sigma_px_by_camera(path).items():
                self.commissioned_pixel_cov_by_camera[camera_id] = (
                    commissioned_pixel_covariance(sigma))
            if self.commissioned_pixel_cov_by_camera:
                listing = ", ".join(
                    f"{k}={v:.3f}" for k, v in
                    sorted(load_commissioned_sigma_px_by_camera(path).items()))
                self.get_logger().info(
                    f"per-camera detector noise in use: {listing} px "
                    f"(the pooled {self.commissioned_sigma_px:.4f} px is the fallback "
                    f"for any camera commissioning did not measure)")
        self.get_logger().info(
            f"covariance_profile=commissioned_sigma_px: R_pix = "
            f"({self.commissioned_sigma_px:.4f} px)^2 I from {source}, pushed through "
            f"each camera's geometry; no floor and no handover inflation"
        )

        self.timestamp_compensation = bool(
            self.get_parameter("correction_timestamp_compensation").value)
        self.propagation_drift_std = float(
            self.get_parameter("correction_propagation_drift_std_m_per_s").value)
        self.correction_residual_interval_s = float(
            self.get_parameter("correction_residual_interval_s").value)
        if self.correction_residual_interval_s < 0.0:
            raise ValueError("correction_residual_interval_s must be non-negative")
        if self.propagation_drift_std < 0.0:
            raise ValueError("correction_propagation_drift_std_m_per_s must be non-negative")
        if self.timestamp_compensation:
            self.get_logger().info(
                "correction_timestamp_compensation=true: corrections are carried forward from "
                "the pose they describe to the pose they are used on")
        self._propagation_status = ""

        self.fusion_rule = str(self.get_parameter("fusion_rule").value).strip().lower()
        if self.fusion_rule not in SUPPORTED_FUSION_RULES:
            raise ValueError(
                f"fusion_rule must be one of {SUPPORTED_FUSION_RULES}, "
                f"got {self.fusion_rule!r}"
            )
        if self.fusion_rule != FUSION_RULE_LEGACY and not self.fusion_mode:
            raise ValueError(
                f"fusion_rule={self.fusion_rule} needs fusion_mode=true; with selection "
                "only one camera is ever used and the rule would never run"
            )

        self.admission_gate = bool(self.get_parameter("admission_gate").value)
        if not self.admission_gate:
            self.get_logger().warn(
                "admission_gate=false: every detection becomes a correction, including ones "
                "whose contact point is hidden. This reproduces the ungated pipeline.")
        self._gate_rejections = collections.Counter()

        self.observation_model = str(
            self.get_parameter("observation_model").value
        ).strip().lower()
        if self.observation_model not in SUPPORTED_OBSERVATION_MODELS:
            raise ValueError(
                "observation_model must be one of "
                f"{SUPPORTED_OBSERVATION_MODELS}, got {self.observation_model!r}"
            )
        self.fixed_offset_m = float(self.get_parameter("fixed_offset_m").value)
        if self.observation_model == OBSERVATION_MODEL_FIXED_OFFSET:
            if not math.isfinite(self.fixed_offset_m) or self.fixed_offset_m <= 0.0:
                raise ValueError(
                    "observation_model=fixed_offset needs a positive fixed_offset_m; it is a "
                    "commissioned distance, not a default"
                )
        # One source of truth: the hull correction is on exactly when the observation model
        # is the hull. Previously this was a separate boolean that could disagree with it.
        if self.observation_model == OBSERVATION_MODEL_HULL:
            if not self.silhouette_correction:
                raise ValueError(
                    "observation_model=hull with silhouette_observation_correction=false "
                    "asks for the hull model with the hull correction switched off"
                )
        else:
            self.silhouette_correction = False
            self.get_logger().warn(
                f"observation_model={self.observation_model}: the hull prediction is OFF, so "
                f"every reading is treated as the robot's centre"
                + (f" after a fixed {self.fixed_offset_m * 100:.1f} cm push away from the camera"
                   if self.observation_model == OBSERVATION_MODEL_FIXED_OFFSET else "")
            )

        if self.fusion_mode:
            self.get_logger().warn("fusion_mode=true: publishing covariance-weighted FUSION of all in-view cameras to /state/bev")

        self.reliability_query_max_time_delta_s = float(
            self.get_parameter("reliability_query_max_time_delta_s").value
        )
        if (
            not math.isfinite(self.reliability_query_max_time_delta_s)
            or self.reliability_query_max_time_delta_s < 0.0
        ):
            raise ValueError(
                "reliability_query_max_time_delta_s must be finite and non-negative"
            )
        self.create_subscription(
            PoseWithCovarianceStamped,
            str(self.get_parameter("reliability_query_topic").value),
            self._belief_query_callback,
            20,
        )
        if self.timestamp_compensation:
            from nav_msgs.msg import Odometry  # noqa: PLC0415
            self.create_subscription(
                Odometry, str(self.get_parameter("odometry_topic").value),
                self._odom_callback, 50)

        for camera_id in self.camera_ids:
            topic = template.format(camera_id=camera_id)
            self.create_subscription(String, topic, self._observation_callback(camera_id), 10)

        rate = max(0.1, float(self.get_parameter("decision_rate_hz").value))
        self.create_timer(1.0 / rate, self._decide)

    def _observation_callback(self, expected_camera_id: str):
        def callback(message) -> None:
            try:
                observation = CameraObservation.from_json(message.data)
            except (ContractValidationError, ValueError, TypeError) as exc:
                self.get_logger().warn(f"{expected_camera_id}: rejected observation: {exc}")
                return
            if observation.camera_id != expected_camera_id:
                self.get_logger().warn(
                    f"{expected_camera_id}: ignoring observation labelled {observation.camera_id!r}"
                )
                return
            self._latest[expected_camera_id] = observation

        return callback

    def _odom_callback(self, message) -> None:
        """Odometry, used only for the length of the correction's propagation interval."""

        stamp = (float(message.header.stamp.sec)
                 + 1.0e-9 * float(message.header.stamp.nanosec))
        self._odom_history.append(
            (stamp, (float(message.pose.pose.position.x),
                     float(message.pose.pose.position.y), 0.0)))

    def _belief_query_callback(self, message) -> None:
        if message.header.frame_id and message.header.frame_id != self.frame_id:
            self.get_logger().warn(
                "ignoring reliability query state in frame "
                f"{message.header.frame_id!r}; expected {self.frame_id!r}"
            )
            return
        timestamp_s = (
            float(message.header.stamp.sec)
            + 1.0e-9 * float(message.header.stamp.nanosec)
        )
        # Yaw is carried alongside x/y because the observation function needs it: the
        # silhouette's bottom edge is generated by whichever body part is nearest the
        # camera, so where the reading lands swings with heading. Consumers that only want
        # position keep indexing [0] and [1] of the same tuple.
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (float(orientation.w) * float(orientation.z)
                   + float(orientation.x) * float(orientation.y)),
            1.0 - 2.0 * (float(orientation.y) ** 2 + float(orientation.z) ** 2),
        )
        pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw,
        )
        if math.isfinite(timestamp_s) and all(math.isfinite(value) for value in pose):
            if (
                not self._belief_query_history
                or timestamp_s != self._belief_query_history[-1][0]
            ):
                self._belief_query_history.append((timestamp_s, pose))

    def _map_observations(self, now_s: float) -> list[MapObservation]:
        observations: list[MapObservation] = []
        for camera_id, contract in self._latest.items():
            # The detector's noise is one commissioned number in PIXELS, identical for
            # every camera. Whatever pixel covariance the contract arrived with is
            # replaced by it here, so no arm can be driven against a remembered value,
            # and the geometry alone decides what it is worth in centimetres.
            stated = replace(contract, conditional_cov_uv=(
                self.commissioned_pixel_cov_by_camera.get(
                    camera_id, self.commissioned_pixel_cov)))
            projected = project_observation_to_world_with_covariance(
                stated, self.camera_models[camera_id]
            )
            if projected is None:
                continue
            world_xy, covariance_m2 = projected
            source = "live_contract:commissioned_sigma_px"
            prior_pose = _nearest_state_pose(
                self._belief_query_history,
                contract.timestamp_s,
                max_delta_s=self.reliability_query_max_time_delta_s,
            )

            # Is this detection usable at all? Every test compares the detected box against the
            # box predicted from the robot's own shape at the believed pose, so none of it needs
            # ground truth. A detection whose contact point is hidden reads long along the
            # viewing ray -- that is where the 100-122 cm readings came from.
            if self.admission_gate and contract.bbox_xyxy is None:
                # Fail closed. The check is the only thing standing between a hidden robot and
                # a metre-wrong correction, so a detection that cannot be checked is refused
                # rather than waved through.
                self._gate_rejections["no_bounding_box"] += 1
                self._silhouette_status_by_camera[camera_id] = "refused_no_bounding_box"
                continue
            if self.admission_gate:
                if prior_pose is None:
                    # Bootstrap. The check compares a detection against a box PREDICTED from
                    # the believed pose, and at start-up there is no belief yet -- the belief
                    # is what this first correction creates. Refusing here deadlocks the robot:
                    # no correction, so no belief, so no prediction, so no correction. So the
                    # first readings pass unchecked and are counted, and the check applies from
                    # the moment a pose exists. The same readings are also uncorrected for the
                    # hull at this point, which is the pre-existing bootstrap behaviour.
                    self._gate_rejections["bootstrap_unchecked"] += 1
                else:
                  reasons = plausibility_reasons(
                      contract.bbox_xyxy, self.camera_models[camera_id],
                      prior_pose[0], prior_pose[1], prior_pose[2])
                  if reasons:
                      for reason in reasons:
                          self._gate_rejections[reason] += 1
                      self._silhouette_status_by_camera[camera_id] = (
                          "refused:" + ",".join(reasons))
                      continue
            # The reading is the bottom-centre of the robot's SILHOUETTE, not its centre.
            # Rewriting it as the position measurement an H = I filter wants needs a pose
            # to predict from, and the only pose available online is the filter's own
            # prior -- including its yaw, which no position-only query carries. Without a
            # timestamp-matched belief the correction is skipped rather than guessed: at an
            # unknown heading the prediction can point the wrong way, which is worse than
            # the uncorrected reading it would replace.
            if self.silhouette_correction and prior_pose is not None:
                corrected = equivalent_position_measurement(
                    world_xy,
                    covariance_m2,
                    self.camera_models[camera_id],
                    (prior_pose[0], prior_pose[1]),
                    prior_pose[2],
                )
                if corrected is None:
                    silhouette_status = "skipped_prediction_unavailable"
                else:
                    world_xy, covariance_m2 = corrected[0], corrected[1]
                    source = f"{source}:silhouette_corrected"
                    silhouette_status = "applied"
            elif self.silhouette_correction:
                silhouette_status = "skipped_no_timestamp_matched_belief_pose"
            elif self.observation_model == OBSERVATION_MODEL_FIXED_OFFSET:
                world_xy = offset_away_from_camera(
                    world_xy, self.camera_models[camera_id].cam_pos, self.fixed_offset_m)
                source = f"{source}:fixed_offset"
                silhouette_status = "fixed_offset_applied"
            else:
                silhouette_status = "disabled_robot_centre_assumption"
            self._silhouette_status_by_camera[camera_id] = silhouette_status
            bbox = contract.bbox_xyxy
            cam_pos = self.camera_models[camera_id].cam_pos
            self._detection_extras_by_camera[camera_id] = {
                "conf": float(contract.detector_score),
                "conf_raw": float(contract.detector_score_raw),
                "bbox_h_px": (float(bbox[3] - bbox[1]) if bbox is not None else float("nan")),
                "bbox_w_px": (float(bbox[2] - bbox[0]) if bbox is not None else float("nan")),
                "range_m": float(math.hypot(world_xy[0] - float(cam_pos[0]),
                                            world_xy[1] - float(cam_pos[1]))),
            }
            base = MapObservation(
                camera_id=camera_id,
                timestamp_s=contract.timestamp_s,
                xy_m=world_xy,
                covariance_m2=covariance_m2,
                quality=_contract_quality(contract),
                source=source,
            )
            if prior_pose is None:
                query_xy = world_xy
                query_source = "measurement_bootstrap"
            else:
                query_xy = (prior_pose[0], prior_pose[1])
                query_source = "timestamp_matched_planner_belief"
            self._reliability_query_source_by_camera[camera_id] = query_source
            observations.append(
                _with_provider_quality(base, self.replay_config, query_xy, now_s)
            )
        return observations

    def _decide(self) -> None:
        now_s = self.get_clock().now().nanoseconds * 1.0e-9
        observations = self._map_observations(now_s)
        self._publish_map_observations(observations)
        if self.fusion_mode and self.active_pub is not None:
            self._decide_fused(now_s, observations)
            return
        decision = self.manager.select(timestamp_s=now_s, observations=observations)
        # The handover switch is REPORTED, never applied: the commissioned covariance is the
        # claim under test, so a camera switch may not quietly widen it. The diagnostic says
        # when a switch happened and by how much it would have been inflated.
        _, diagnostic = handover_adjusted_observation(
            previous_camera_id=self._previous_camera_id,
            selected_observation=decision.selected_observation,
            candidate_observations=tuple(observations),
            previous_observation=self._previous_observation,
            config=self.handover_config,
        )
        selected = decision.selected_observation
        if selected is not None:
            self._previous_camera_id = selected.camera_id
            self._previous_observation = selected

        payload = decision.to_dict()
        payload["authority"] = self.authority
        payload["covariance_profile"] = self.covariance_profile
        payload["handover_diagnostic"] = diagnostic.to_dict()
        payload["gp_query_source_by_camera"] = dict(
            self._reliability_query_source_by_camera
        )
        payload["silhouette_correction_by_camera"] = dict(
            self._silhouette_status_by_camera
        )
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.decision_pub.publish(message)

        if selected is None:
            return
        pose = self._pose_message(selected)
        self.selected_pub.publish(pose)
        if self.active_pub is not None:
            self.active_pub.publish(pose)

    def _publish_map_observations(self, observations: list[MapObservation]) -> None:
        """Emit the per-camera observations for a downstream sequential filter.

        Each keeps its own covariance. Collapsing them into one fused pose first
        (a) discards the per-camera measurement model and (b) makes the consumer
        run a second filter on top of this node's -- which has no motion model
        and seeds from a median with an identity prior.
        """
        if self.map_observations_pub is None:
            return
        message = String()
        message.data = map_observations_to_json(observations, frame_id=self.frame_id)
        self.map_observations_pub.publish(message)

    def _bias_floors(self, observations):
        """Per-camera bias floors for this batch, or None when the floor is disabled.

        Each floor is built from the sighting's own range and bearing, taken from the
        camera's surveyed position and the observation it produced, because the measured
        bias tracks the camera's line of sight rather than the world frame.
        """
        if self.bias_floor_along_slope <= 0.0:
            return None
        from reliability.bias_floor import bias_floor_matrix, ray_bearing_rad

        floors = {}
        for observation in observations:
            model = self.camera_models.get(observation.camera_id)
            if model is None:
                continue
            camera_xy = (float(model.cam_pos[0]), float(model.cam_pos[1]))
            target_xy = (float(observation.xy_m[0]), float(observation.xy_m[1]))
            try:
                bearing = ray_bearing_rad(camera_xy, target_xy)
            except Exception:                                  # noqa: BLE001
                continue
            range_m = math.hypot(target_xy[0] - camera_xy[0], target_xy[1] - camera_xy[1])
            floors[observation.camera_id] = bias_floor_matrix(
                range_m, bearing,
                along_slope=self.bias_floor_along_slope,
                across_slope=self.bias_floor_across_slope,
            )
        return floors or None

    def _decide_fused(self, now_s: float, observations: list[MapObservation]) -> None:
        # Camera callbacks retain one latest observation each. Fuse only views
        # that satisfy the same operational gates as selection and whose stamps
        # occupy one time neighbourhood; otherwise an old cached view is
        # incorrectly treated as a simultaneous measurement of the robot.
        fresh, scores, rejected, time_skew_rejected = _synchronous_fusion_candidates(
            self.manager,
            now_s=now_s,
            observations=observations,
            max_timestamp_spread_s=self.fusion_max_timestamp_spread_s,
        )
        if not fresh:
            payload = {"authority": self.authority, "fusion_mode": True,
                       "accepted_camera_ids": [], "reasons": ["no_eligible_synchronous_observations"],
                       "scores_by_camera": scores,
                       "rejected_by_camera": {key: list(value) for key, value in rejected.items()},
                       "time_skew_rejected_camera_ids": time_skew_rejected,
                       "gp_query_source_by_camera": dict(
                           self._reliability_query_source_by_camera
                       ),
                       "silhouette_correction_by_camera": dict(
                           self._silhouette_status_by_camera
                       )}
            msg = String(); msg.data = json.dumps(payload, sort_keys=True)
            self.decision_pub.publish(msg)
            return
        result = _gated_fusion(
            fresh,
            disagreement_gate_m=self.fusion_disagreement_gate_m,
            rule=self.fusion_rule,
            camera_positions_m={
                camera_id: model.cam_pos
                for camera_id, model in self.camera_models.items()
            },
            belief_floors=self._bias_floors(fresh),
        )
        if not result.accepted_camera_ids:
            payload = {
                "authority": self.authority,
                "fusion_mode": True,
                "accepted_camera_ids": [],
                "rejected_camera_ids": list(result.rejected_camera_ids),
                "reasons": ["all_synchronous_observations_rejected"],
                "scores_by_camera": scores,
                "rejected_by_camera": {
                    key: list(value) for key, value in rejected.items()
                },
                "time_skew_rejected_camera_ids": time_skew_rejected,
                "covariance_profile": self.covariance_profile,
                "gp_query_source_by_camera": dict(
                    self._reliability_query_source_by_camera
                ),
                "silhouette_correction_by_camera": dict(
                    self._silhouette_status_by_camera
                ),
            }
            decision_message = String()
            decision_message.data = json.dumps(payload, sort_keys=True)
            self.decision_pub.publish(decision_message)
            return
        ts = max(float(o.timestamp_s) for o in fresh)

        # The fused correction describes where the robot was when those frames were taken.
        # Carry it forward to now, or it arrives as a systematic displacement backwards along
        # the direction of travel -- measured at 8.2 cm median on 24 drives, collapsing to
        # 2.3 cm once propagated. See logs/studies/fusion_on_fixed_routes/latency/.
        mean_xy, propagated_cov = result.mean_xy, result.covariance_m2
        self._propagation_status = "disabled"
        if self.timestamp_compensation:
            history = self._odom_history if len(self._odom_history) > 2 \
                else self._belief_query_history
            pose_then = _nearest_state_pose(
                history, ts, max_delta_s=self.reliability_query_max_time_delta_s)
            pose_now = _nearest_state_pose(
                history, now_s, max_delta_s=self.reliability_query_max_time_delta_s)
            if pose_then is None or pose_now is None:
                # Without both ends of the interval the motion is unknown. Publishing the
                # uncorrected reading and saying so beats inventing a displacement.
                self._propagation_status = "skipped_no_pose_history"
            else:
                mean_xy, propagated_cov, delta = propagate_correction_to_now(
                    result.mean_xy, result.covariance_m2, pose_then, pose_now,
                    drift_std_m_per_s=self.propagation_drift_std,
                    dt_s=max(now_s - ts, 0.0),
                    residual_interval_s=self.correction_residual_interval_s)
                self._propagation_status = (
                    f"applied age={max(now_s - ts, 0.0):.3f}s "
                    f"dx={delta[0]:+.3f} dy={delta[1]:+.3f}")
                # it now describes NOW, so it is stamped now
                ts = now_s

        message = PoseWithCovarianceStamped()
        message.header.stamp.sec = int(ts)
        message.header.stamp.nanosec = int(round((ts - int(ts)) * 1.0e9))
        message.header.frame_id = self.frame_id
        message.pose.pose.position.x = float(mean_xy[0])
        message.pose.pose.position.y = float(mean_xy[1])
        message.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        report_covariance = _fusion_report_covariance(
            propagated_cov, common_mode_std_m=self.fusion_common_mode_std_m
        )
        cov[0] = report_covariance[0][0]
        cov[1] = report_covariance[0][1]
        cov[6] = report_covariance[1][0]
        cov[7] = report_covariance[1][1]
        cov[35] = NONINFORMATIVE_YAW_VAR
        message.pose.covariance = cov
        self.selected_pub.publish(message)
        self.active_pub.publish(message)
        payload = {"authority": self.authority, "fusion_mode": True,
                   "accepted_camera_ids": list(result.accepted_camera_ids),
                   "rejected_camera_ids": list(result.rejected_camera_ids),
                   "fused_xy": [float(mean_xy[0]), float(mean_xy[1])],
                   "fused_xy_before_propagation": [float(result.mean_xy[0]),
                                                   float(result.mean_xy[1])],
                   "propagation": self._propagation_status,
                   "gate_rejections": dict(self._gate_rejections),
                   "n_fresh": len(fresh),
                   # The cameras that were ON THE TABLE at this instant, before the arm's rule
                   # chose among them. accepted_camera_ids is what the rule USED, which for a
                   # single-best rule is always one camera -- so the two are not the same axis,
                   # and the fusion comparison has to be read against this one.
                   "synchronous_camera_ids": [str(o.camera_id) for o in fresh],
                   # Every camera's own answer at this instant -- where it put the robot and
                   # how sure it was -- beside the one the rule produced from them. Without
                   # this the fusion is a number with no visible mechanism: you can see that
                   # arms differ but not WHERE each camera landed or what the rule did with
                   # them. Small: five cameras at 5 Hz.
                   "observations": [
                       {"camera": str(o.camera_id),
                        # When the CAMERA saw it, not when the log wrote it. Without this the
                        # only truth a reading can be scored against is the truth at logging
                        # time, which is later by the whole detector-plus-manager delay -- and
                        # that delay reads as measurement error identically on every camera.
                        "obs_stamp": float(o.timestamp_s),
                        "xy": [float(o.xy_m[0]), float(o.xy_m[1])],
                        "cov": [[float(o.covariance_m2[0][0]), float(o.covariance_m2[0][1])],
                                [float(o.covariance_m2[1][0]), float(o.covariance_m2[1][1])]],
                        "used": str(o.camera_id) in set(result.accepted_camera_ids),
                        **self._detection_extras_by_camera.get(str(o.camera_id), {})}
                       for o in fresh],
                   "fused_cov": [[float(report_covariance[0][0]), float(report_covariance[0][1])],
                                 [float(report_covariance[1][0]), float(report_covariance[1][1])]],
                   "scores_by_camera": scores,
                   "rejected_by_camera": {key: list(value) for key, value in rejected.items()},
                   "time_skew_rejected_camera_ids": time_skew_rejected,
                   "covariance_profile": self.covariance_profile,
                   "common_mode_std_m": self.fusion_common_mode_std_m,
                   "max_timestamp_spread_s": self.fusion_max_timestamp_spread_s}
        payload["gp_query_source_by_camera"] = dict(
            self._reliability_query_source_by_camera
        )
        payload["silhouette_correction_by_camera"] = dict(
            self._silhouette_status_by_camera
        )
        dmsg = String(); dmsg.data = json.dumps(payload, sort_keys=True)
        self.decision_pub.publish(dmsg)

    def _pose_message(self, observation: MapObservation):
        message = PoseWithCovarianceStamped()
        seconds = int(observation.timestamp_s)
        message.header.stamp.sec = seconds
        message.header.stamp.nanosec = int(round((observation.timestamp_s - seconds) * 1.0e9))
        message.header.frame_id = self.frame_id
        message.pose.pose.position.x = float(observation.xy_m[0])
        message.pose.pose.position.y = float(observation.xy_m[1])
        message.pose.pose.orientation.w = 1.0
        covariance = [0.0] * 36
        covariance[0] = float(observation.covariance_m2[0][0])
        covariance[1] = float(observation.covariance_m2[0][1])
        covariance[6] = float(observation.covariance_m2[1][0])
        covariance[7] = float(observation.covariance_m2[1][1])
        covariance[35] = NONINFORMATIVE_YAW_VAR
        message.pose.covariance = covariance
        return message


def _contract_quality(contract: CameraObservation):
    from reliability.contracts import CameraQuality

    return CameraQuality(
        camera_id=contract.camera_id,
        p_available=contract.availability_probability,
        conditional_cov_uv=contract.conditional_cov_uv,
        association_confidence=contract.association_probability,
        epistemic_score=0.0,
        stale=False,
        source_model="live_contract",
    )


def main(args=None) -> int:
    if rclpy is None:
        raise SystemExit("rclpy is required to run camera_manager_node")
    rclpy.init(args=args)
    node = CameraManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
