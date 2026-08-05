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

import json
import math
from pathlib import Path

from reliability.camera_manager import CameraManager, CameraManagerConfig
from reliability.contracts import CameraObservation, ContractValidationError
from reliability.fusion import (
    MapObservation,
    SequentialFusionResult,
    independent_measurement_fusion_2d,
    map_observations_to_json,
    sequential_kalman_update_2d,
)
from reliability.handover import HandoverUncertaintyConfig, handover_adjusted_observation
from reliability.projection import (
    camera_model_from_world,
    load_projection_calibration,
    project_observation_to_world,
    project_observation_to_world_with_covariance,
    projection_kwargs_for_camera,
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
LEGACY_MULTICAM_COVARIANCE = "legacy_fixed_metric"
PAPER1_HISTORICAL_COVARIANCE = "paper1_historical"
SUPPORTED_COVARIANCE_PROFILES = (
    LEGACY_MULTICAM_COVARIANCE,
    PAPER1_HISTORICAL_COVARIANCE,
)


def _fusion_report_covariance(
    covariance_m2,
    *,
    covariance_profile: str,
    report_std_m: float,
):
    """Apply only the covariance policy selected for the experiment arm."""

    if covariance_profile == PAPER1_HISTORICAL_COVARIANCE:
        return (
            (float(covariance_m2[0][0]), float(covariance_m2[0][1])),
            (float(covariance_m2[1][0]), float(covariance_m2[1][1])),
        )
    if covariance_profile != LEGACY_MULTICAM_COVARIANCE:
        raise ValueError(f"unsupported covariance_profile {covariance_profile!r}")
    rvar = float(report_std_m) ** 2
    return (
        (max(float(covariance_m2[0][0]), rvar), 0.0),
        (0.0, max(float(covariance_m2[1][1]), rvar)),
    )


def _handover_profile_observation(
    selected_observation: MapObservation | None,
    inflated_observation: MapObservation | None,
    *,
    covariance_profile: str,
) -> MapObservation | None:
    """Keep the historical arm free of multicamera-only switch inflation."""

    if covariance_profile == PAPER1_HISTORICAL_COVARIANCE:
        return selected_observation
    if covariance_profile == LEGACY_MULTICAM_COVARIANCE:
        return inflated_observation
    raise ValueError(f"unsupported covariance_profile {covariance_profile!r}")


def _paper1_precision_fusion_with_disagreement_gate(
    observations: list[MapObservation],
    *,
    disagreement_gate_m: float,
) -> SequentialFusionResult:
    """Gate around the robust centre, then form a prior-free fused measurement."""

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
    residuals = {}
    for observation in observations:
        residual = math.hypot(
            float(observation.xy_m[0]) - centre[0],
            float(observation.xy_m[1]) - centre[1],
        )
        residuals[observation.camera_id] = math.nan
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
    mean, covariance = independent_measurement_fusion_2d(accepted)
    return SequentialFusionResult(
        mean_xy=mean,
        covariance_m2=covariance,
        accepted_camera_ids=tuple(item.camera_id for item in accepted),
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
        self.declare_parameter("contact_z_m", 0.05)
        # Optional projection_calibration.json (per-camera along-bearing
        # offsets, commissioning constants) — same file the recorder consumes.
        self.declare_parameter("projection_calibration", "")
        self.declare_parameter("require_projection_calibration", True)
        self.declare_parameter("require_gp_artifacts", True)
        self.declare_parameter("frame_id", "map_bev")
        self.declare_parameter("authority", "shadow")
        self.declare_parameter("decision_topic", "/reliability/camera_manager/decision")
        self.declare_parameter("selected_topic", "/reliability/camera_manager/selected_observation")
        self.declare_parameter("active_output_topic", "/state/bev")
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
        self.declare_parameter("fusion_max_timestamp_spread_s", 0.25)
        # legacy_fixed_metric preserves the first multicam implementation.
        # paper1_historical consumes the detector's 2.5/40 px precision blend,
        # propagates it through the complete projection, and disables the
        # multicam-only handover/report-floor covariance inflation.
        self.declare_parameter("covariance_profile", LEGACY_MULTICAM_COVARIANCE)
        # Reported per-correction std (m) sent to the planner EKF. The shipped
        # 0.08 m makes the EKF over-trust noisy far-range oblique detections and
        # chase the noise; ~0.3 m lets it smooth. Floor on the fused covariance.
        self.declare_parameter("fusion_report_std_m", 0.3)
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
        self.contact_z_m = float(self.get_parameter("contact_z_m").value)
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
        calibration_path = str(self.get_parameter("projection_calibration").value)
        if bool(self.get_parameter("require_projection_calibration").value) and not calibration_path:
            raise ValueError(
                "projection_calibration is required by the fail-closed commissioning contract"
            )
        self.projection_calibrations = (
            load_projection_calibration(calibration_path) if calibration_path else {}
        )

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
                    Path(artifact), camera_id=camera_id, out_of_bounds_policy="clamp"
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
        self.fusion_report_std_m = float(self.get_parameter("fusion_report_std_m").value)
        if not math.isfinite(self.fusion_report_std_m) or self.fusion_report_std_m < 0.0:
            raise ValueError("fusion_report_std_m must be finite and non-negative")
        if self.fusion_mode:
            self.get_logger().warn("fusion_mode=true: publishing covariance-weighted FUSION of all in-view cameras to /state/bev")
        if self.covariance_profile == PAPER1_HISTORICAL_COVARIANCE:
            self.get_logger().warn(
                "covariance_profile=paper1_historical: using projected 2.5/40 px "
                "precision-blend covariance with no handover or reporting-floor inflation"
            )

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

    def _map_observations(self, now_s: float) -> list[MapObservation]:
        observations: list[MapObservation] = []
        for camera_id, contract in self._latest.items():
            projection_kwargs = projection_kwargs_for_camera(
                self.projection_calibrations, camera_id, contact_z_m=self.contact_z_m
            )
            if self.covariance_profile == PAPER1_HISTORICAL_COVARIANCE:
                projected = project_observation_to_world_with_covariance(
                    contract, self.camera_models[camera_id], **projection_kwargs
                )
                if projected is None:
                    continue
                world_xy, covariance_m2 = projected
            else:
                world_xy = project_observation_to_world(
                    contract, self.camera_models[camera_id], **projection_kwargs
                )
                if world_xy is None:
                    continue
                covariance_m2 = self.replay_config.fixed_measurement_cov_m2
            base = MapObservation(
                camera_id=camera_id,
                timestamp_s=contract.timestamp_s,
                xy_m=world_xy,
                covariance_m2=covariance_m2,
                quality=_contract_quality(contract),
                source=(
                    "live_contract:paper1_projected_covariance"
                    if self.covariance_profile == PAPER1_HISTORICAL_COVARIANCE
                    else "live_contract:legacy_fixed_metric_covariance"
                ),
            )
            # Offline replay queries the provider at the filter mean; the
            # shadow node carries no filter, so the measurement position is the
            # query point. Documented divergence, conservative in practice.
            observations.append(
                _with_provider_quality(base, self.replay_config, world_xy, now_s)
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
        inflated, diagnostic = handover_adjusted_observation(
            previous_camera_id=self._previous_camera_id,
            selected_observation=decision.selected_observation,
            candidate_observations=tuple(observations),
            previous_observation=self._previous_observation,
            config=self.handover_config,
        )
        adjusted = _handover_profile_observation(
            decision.selected_observation,
            inflated,
            covariance_profile=self.covariance_profile,
        )
        if adjusted is not None:
            self._previous_camera_id = adjusted.camera_id
            self._previous_observation = decision.selected_observation

        payload = decision.to_dict()
        payload["authority"] = self.authority
        payload["covariance_profile"] = self.covariance_profile
        payload["handover_covariance_inflation_applied"] = (
            self.covariance_profile != PAPER1_HISTORICAL_COVARIANCE
        )
        payload["handover_diagnostic"] = diagnostic.to_dict()
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.decision_pub.publish(message)

        if adjusted is None:
            return
        pose = self._pose_message(adjusted)
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
                       "time_skew_rejected_camera_ids": time_skew_rejected}
            msg = String(); msg.data = json.dumps(payload, sort_keys=True)
            self.decision_pub.publish(msg)
            return
        if self.covariance_profile == PAPER1_HISTORICAL_COVARIANCE:
            result = _paper1_precision_fusion_with_disagreement_gate(
                fresh,
                disagreement_gate_m=self.fusion_disagreement_gate_m,
            )
        else:
            xs = sorted(float(o.xy_m[0]) for o in fresh)
            ys = sorted(float(o.xy_m[1]) for o in fresh)
            seed = (xs[len(xs) // 2], ys[len(ys) // 2])
            result = sequential_kalman_update_2d(
                seed, ((1.0, 0.0), (0.0, 1.0)), fresh,
                nis_gate=9.21, disagreement_gate_m=self.fusion_disagreement_gate_m,
            )
        if (
            self.covariance_profile == PAPER1_HISTORICAL_COVARIANCE
            and not result.accepted_camera_ids
        ):
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
            }
            decision_message = String()
            decision_message.data = json.dumps(payload, sort_keys=True)
            self.decision_pub.publish(decision_message)
            return
        ts = max(float(o.timestamp_s) for o in fresh)
        message = PoseWithCovarianceStamped()
        message.header.stamp.sec = int(ts)
        message.header.stamp.nanosec = int(round((ts - int(ts)) * 1.0e9))
        message.header.frame_id = self.frame_id
        message.pose.pose.position.x = float(result.mean_xy[0])
        message.pose.pose.position.y = float(result.mean_xy[1])
        message.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        report_covariance = _fusion_report_covariance(
            result.covariance_m2,
            covariance_profile=self.covariance_profile,
            report_std_m=self.fusion_report_std_m,
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
                   "fused_xy": [float(result.mean_xy[0]), float(result.mean_xy[1])],
                   "n_fresh": len(fresh),
                   "scores_by_camera": scores,
                   "rejected_by_camera": {key: list(value) for key, value in rejected.items()},
                   "time_skew_rejected_camera_ids": time_skew_rejected,
                   "covariance_profile": self.covariance_profile,
                   "reporting_floor_applied":
                       self.covariance_profile != PAPER1_HISTORICAL_COVARIANCE,
                   "max_timestamp_spread_s": self.fusion_max_timestamp_spread_s}
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
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
