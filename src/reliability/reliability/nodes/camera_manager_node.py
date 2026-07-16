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
from reliability.fusion import MapObservation
from reliability.handover import HandoverUncertaintyConfig, handover_adjusted_observation
from reliability.projection import (
    camera_model_from_world,
    load_projection_calibration,
    project_observation_to_world,
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
        self.declare_parameter("frame_id", "map_bev")
        self.declare_parameter("authority", "shadow")
        self.declare_parameter("decision_topic", "/reliability/camera_manager/decision")
        self.declare_parameter("selected_topic", "/reliability/camera_manager/selected_observation")
        self.declare_parameter("active_output_topic", "/state/bev")
        # Operational gates: defaults mirror CameraManagerConfig; override from
        # the frozen study/protocol config in the launch file, never here.
        defaults = CameraManagerConfig()
        self.declare_parameter("min_spatial_trust", defaults.min_spatial_trust)
        self.declare_parameter("min_association_confidence", defaults.min_association_confidence)
        self.declare_parameter("max_measurement_age_s", defaults.max_measurement_age_s)
        self.declare_parameter("candidate_score_margin", defaults.candidate_score_margin)
        self.declare_parameter(
            "required_consecutive_better_frames", defaults.required_consecutive_better_frames
        )
        self.declare_parameter(
            "max_cross_camera_disagreement_m", defaults.max_cross_camera_disagreement_m
        )
        self.declare_parameter("max_overlap_time_delta_s", defaults.max_overlap_time_delta_s)

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
            world_xy = project_observation_to_world(
                contract,
                self.camera_models[camera_id],
                contact_z_m=self.contact_z_m,
                along_bearing_offset_m=self.projection_calibrations.get(camera_id, {}).get(
                    "intercept_m", 0.0
                ),
                along_bearing_slope_per_m=self.projection_calibrations.get(camera_id, {}).get(
                    "slope_per_m", 0.0
                ),
            )
            if world_xy is None:
                continue
            base = MapObservation(
                camera_id=camera_id,
                timestamp_s=contract.timestamp_s,
                xy_m=world_xy,
                covariance_m2=self.replay_config.fixed_measurement_cov_m2,
                quality=_contract_quality(contract),
                source="live_contract",
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
        decision = self.manager.select(timestamp_s=now_s, observations=observations)
        adjusted, diagnostic = handover_adjusted_observation(
            previous_camera_id=self._previous_camera_id,
            selected_observation=decision.selected_observation,
            candidate_observations=tuple(observations),
            previous_observation=self._previous_observation,
            config=self.handover_config,
        )
        if adjusted is not None:
            self._previous_camera_id = adjusted.camera_id
            self._previous_observation = decision.selected_observation

        payload = decision.to_dict()
        payload["authority"] = self.authority
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
