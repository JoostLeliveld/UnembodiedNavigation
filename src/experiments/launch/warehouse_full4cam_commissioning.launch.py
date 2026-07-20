"""Passive four-camera commissioning launch for ``warehouse_full_4cam``.

Retargeted 2026-07-15 from an archived two-camera commissioning launch.
Cameras A–D are the wall-mounted
``external_camera``/``_b``/``_c``/``_d`` models; their bridges come from the
generic simulator launcher (``bridge_camera_b/c/d``), so no study-owned bridge
is needed. Study routes/spawns are defined in
``experiments/multicamera_commissioning_bigwarehouse/config/study.yaml`` and
must be (re)designed for this world's geometry before collection.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _detector_params(
    *,
    camera_id: str,
    image_topic: str,
    pixel_topic: str,
    diagnostics_topic: str,
    observation_topic: str,
    device_arg: str,
) -> dict:
    return {
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "model_path": LaunchConfiguration("yolo_model"),
        # A bare launch value such as ``0`` is otherwise emitted into the
        # temporary ROS parameter YAML as an integer.  The detector declares
        # ``device`` as a string, so preserve the CLI token's type explicitly
        # (``0``, ``cpu`` and ``cuda:0`` are all valid Ultralytics selectors).
        "device": ParameterValue(LaunchConfiguration(device_arg), value_type=str),
        "cpu_num_threads": LaunchConfiguration(f"yolo_cpu_num_threads_{camera_id.lower()}"),
        "cpu_num_interop_threads": LaunchConfiguration("yolo_cpu_num_interop_threads"),
        "image_size": LaunchConfiguration("yolo_imgsz"),
        "confidence_threshold": LaunchConfiguration("yolo_conf_threshold"),
        "iou_threshold": LaunchConfiguration("yolo_iou_threshold"),
        "class_name": LaunchConfiguration("yolo_target_class"),
        "class_id": LaunchConfiguration("yolo_class_id"),
        "use_masks": LaunchConfiguration("yolo_use_masks"),
        "mask_min_area": LaunchConfiguration("yolo_min_mask_area_px"),
        "mask_bottom_band_px": LaunchConfiguration("yolo_mask_bottom_band_px"),
        "min_bbox_area_px": LaunchConfiguration("yolo_min_bbox_area_px"),
        "debug_frame_dir": LaunchConfiguration("yolo_debug_frame_dir"),
        "use_torchscript": LaunchConfiguration("yolo_use_torchscript"),
        "warmup_iters": LaunchConfiguration("yolo_warmup_iters"),
        "inference_in_callback": LaunchConfiguration("yolo_inference_in_callback"),
        "publish_camera_observation_json": True,
        "image_topic": image_topic,
        "pixel_pose_topic": pixel_topic,
        "diagnostics_topic": diagnostics_topic,
        "camera_observation_topic": observation_topic,
        "camera_id": camera_id,
        "camera_calibration_id": f"warehouse_full_4cam_{camera_id}",
        "camera_image_frame_id": camera_id,
        "camera_observation_r_visible_uv": LaunchConfiguration("camera_observation_r_visible_uv"),
        "camera_observation_r_miss_uv": LaunchConfiguration("camera_observation_r_miss_uv"),
    }


def _detector_node(camera_id: str, image_topic: str) -> Node:
    return Node(
        package="perception",
        executable="yolo_robot_detector_node",
        name=f"yolo_robot_detector_{camera_id.lower()}",
        output="screen",
        condition=UnlessCondition(LaunchConfiguration("yolo_batched_four_camera")),
        parameters=[
            _detector_params(
                camera_id=camera_id,
                image_topic=image_topic,
                pixel_topic=f"/perception/{camera_id}/pixel_pose",
                diagnostics_topic=f"/perception/{camera_id}/detection_diagnostics",
                observation_topic=f"/perception/camera_observation/{camera_id}",
                device_arg=f"yolo_device_{camera_id.lower()}",
            )
        ],
    )


def _batched_detector_node() -> Node:
    """One model allocation and one deterministic A--D inference batch."""

    return Node(
        package="perception",
        executable="batched_four_camera_yolo_node",
        name="batched_four_camera_yolo",
        output="screen",
        condition=IfCondition(LaunchConfiguration("yolo_batched_four_camera")),
        # A detector contract/inference fault exits this process.  Propagate
        # that exit to the complete launch so an unsupervised route cannot keep
        # collecting plausible-looking rows after perception has failed.
        on_exit=Shutdown(reason="batched four-camera detector exited"),
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "model_path": LaunchConfiguration("yolo_model"),
            "device": ParameterValue(
                LaunchConfiguration("yolo_batched_device"), value_type=str
            ),
            "cpu_num_threads": LaunchConfiguration("yolo_batched_cpu_num_threads"),
            "cpu_num_interop_threads": LaunchConfiguration("yolo_cpu_num_interop_threads"),
            "opencv_num_threads": LaunchConfiguration("yolo_opencv_num_threads"),
            "image_size": LaunchConfiguration("yolo_imgsz"),
            "confidence_threshold": LaunchConfiguration("yolo_conf_threshold"),
            "iou_threshold": LaunchConfiguration("yolo_iou_threshold"),
            "class_name": LaunchConfiguration("yolo_target_class"),
            "class_id": LaunchConfiguration("yolo_class_id"),
            "use_masks": LaunchConfiguration("yolo_use_masks"),
            "mask_min_area": LaunchConfiguration("yolo_min_mask_area_px"),
            "mask_bottom_band_px": LaunchConfiguration("yolo_mask_bottom_band_px"),
            "min_bbox_area_px": LaunchConfiguration("yolo_min_bbox_area_px"),
            "debug_frame_dir": LaunchConfiguration("yolo_debug_frame_dir"),
            "pixel_noise_sigma": LaunchConfiguration("yolo_pixel_noise_sigma"),
            "seed": LaunchConfiguration("yolo_seed"),
            "warmup_iters": LaunchConfiguration("yolo_warmup_iters"),
            "max_batch_stamp_skew_s": LaunchConfiguration("yolo_max_batch_stamp_skew_s"),
            "max_pending_wall_s": LaunchConfiguration("yolo_max_pending_wall_s"),
            "camera_observation_r_visible_uv": LaunchConfiguration(
                "camera_observation_r_visible_uv"
            ),
            "camera_observation_r_miss_uv": LaunchConfiguration(
                "camera_observation_r_miss_uv"
            ),
        }],
    )


def generate_launch_description() -> LaunchDescription:
    sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("sim"),
                "launch",
                "bringup_sim.launch.py",
            ])
        ),
        launch_arguments={
            "world": LaunchConfiguration("world"),
            "world_name": LaunchConfiguration("world_name"),
            "headless": LaunchConfiguration("headless"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "reset_world": LaunchConfiguration("reset_world"),
            "use_lidar": "false",
            "bridge_scan": "false",
            "bridge_camera_b": "true",
            "bridge_camera_c": "true",
            "bridge_camera_d": "true",
            "bridge_contacts": LaunchConfiguration("bridge_contacts"),
            "spawn_x": LaunchConfiguration("spawn_x"),
            "spawn_y": LaunchConfiguration("spawn_y"),
            "spawn_z": LaunchConfiguration("spawn_z"),
            "spawn_yaw": LaunchConfiguration("spawn_yaw"),
        }.items(),
    )

    # The recorder stores the same independent encoder-odometry stream used by
    # the replay tooling.  It is deliberately a passive side channel: the
    # collection driver follows Gazebo's /odom, while /odom_noisy is only an
    # operational input to later localisation replay.
    encoder_noise = Node(
        package="sim",
        executable="encoder_noise_node",
        name="fullwarehouse_encoder_noise",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "enabled": LaunchConfiguration("use_encoder_noise"),
            "input_topic": "/odom",
            "output_topic": "/odom_noisy",
            "seed": LaunchConfiguration("encoder_noise_seed"),
        }],
    )

    # Step-6 shadow rung (research_story ch.09): run the hysteretic manager
    # against the live detector streams and log every decision WITHOUT
    # authority.  Off by default so pure collection runs stay byte-identical.
    # manager_authority=active republishes the selected observation to
    # /state/bev and is only legitimate after the ch.09 release gates pass.
    shadow_manager = Node(
        package="reliability",
        executable="camera_manager_node",
        name="camera_manager_shadow",
        output="screen",
        condition=IfCondition(LaunchConfiguration("shadow_manager")),
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "world_sdf": PathJoinSubstitution([
                FindPackageShare("sim"), "gazebo_worlds", "worlds",
                LaunchConfiguration("world"),
            ]),
            "authority": LaunchConfiguration("manager_authority"),
            "gp_artifact_template": LaunchConfiguration("manager_gp_artifact_template"),
            "decision_rate_hz": LaunchConfiguration("manager_decision_rate_hz"),
            "projection_calibration": LaunchConfiguration("manager_projection_calibration"),
            "require_projection_calibration": LaunchConfiguration("manager_require_projection_calibration"),
            "require_gp_artifacts": LaunchConfiguration("manager_require_gp_artifacts"),
            "min_spatial_trust": LaunchConfiguration("manager_min_spatial_trust"),
            "min_association_confidence": LaunchConfiguration("manager_min_association_confidence"),
            "max_measurement_age_s": LaunchConfiguration("manager_max_measurement_age_s"),
            "age_decay_s": LaunchConfiguration("manager_age_decay_s"),
            "candidate_score_margin": LaunchConfiguration("manager_candidate_score_margin"),
            "required_consecutive_better_frames": LaunchConfiguration(
                "manager_required_consecutive_better_frames"
            ),
            "max_cross_camera_disagreement_m": LaunchConfiguration(
                "manager_max_cross_camera_disagreement_m"
            ),
            "max_overlap_time_delta_s": LaunchConfiguration("manager_max_overlap_time_delta_s"),
            "require_consistency_when_source_available": LaunchConfiguration(
                "manager_require_consistency_when_source_available"
            ),
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="warehouse_full_4cam.world.sdf"),
        DeclareLaunchArgument("world_name", default_value="warehouse_full_4cam"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("reset_world", default_value="false"),
        DeclareLaunchArgument(
            "bridge_contacts", default_value="true",
            description="Bridge all world contact sensors; disable only for non-evidence timing ablations",
        ),
        # Central-aisle default; route-specific spawns come from study.yaml once
        # the full_4cam routes are designed.
        DeclareLaunchArgument("spawn_x", default_value="0.0"),
        DeclareLaunchArgument("spawn_y", default_value="0.0"),
        DeclareLaunchArgument("spawn_z", default_value="0.05"),
        DeclareLaunchArgument("spawn_yaw", default_value="1.5708"),
        DeclareLaunchArgument("use_encoder_noise", default_value="true"),
        DeclareLaunchArgument("encoder_noise_seed", default_value="0"),
        DeclareLaunchArgument("yolo_model", default_value="", description="Local path to the trained YOLO model"),
        DeclareLaunchArgument("yolo_device", default_value=""),
        # One native model and a four-image batch is the P2000-safe study
        # default.  The legacy four-process allocation remains available with
        # yolo_batched_four_camera:=false for an explicit timing comparison or
        # a host with more GPU memory.
        DeclareLaunchArgument(
            "yolo_batched_four_camera", default_value="true",
            description="Use one native YOLO model for strict A-D batches",
        ),
        DeclareLaunchArgument(
            "yolo_batched_device", default_value="0",
            description="String Ultralytics device for the shared batched model",
        ),
        DeclareLaunchArgument("yolo_batched_cpu_num_threads", default_value="2"),
        DeclareLaunchArgument("yolo_opencv_num_threads", default_value="1"),
        DeclareLaunchArgument("yolo_max_batch_stamp_skew_s", default_value="0.10"),
        DeclareLaunchArgument("yolo_max_pending_wall_s", default_value="0.50"),
        DeclareLaunchArgument("yolo_pixel_noise_sigma", default_value="0.0"),
        DeclareLaunchArgument("yolo_seed", default_value="0"),
        # Four concurrent GPU detector processes (~760 MiB each) exceed the 4
        # GiB card.  In separate-process fallback, camera A therefore stays on
        # a bounded CPU allocation and only B--D share the GPU.
        DeclareLaunchArgument(
            "yolo_device_camera_a", default_value="cpu",
            description="Separate-process fallback device for camera_A",
        ),
        DeclareLaunchArgument("yolo_device_camera_b", default_value=LaunchConfiguration("yolo_device")),
        DeclareLaunchArgument("yolo_device_camera_c", default_value=LaunchConfiguration("yolo_device")),
        DeclareLaunchArgument("yolo_device_camera_d", default_value=LaunchConfiguration("yolo_device")),
        DeclareLaunchArgument(
            "yolo_cpu_num_threads_camera_a", default_value="2",
            description="Bound CPU camera A so it cannot starve Gazebo and GPU detector callbacks",
        ),
        DeclareLaunchArgument("yolo_cpu_num_threads_camera_b", default_value="1"),
        DeclareLaunchArgument("yolo_cpu_num_threads_camera_c", default_value="1"),
        DeclareLaunchArgument("yolo_cpu_num_threads_camera_d", default_value="1"),
        DeclareLaunchArgument("yolo_cpu_num_interop_threads", default_value="1"),
        DeclareLaunchArgument("yolo_imgsz", default_value="640"),
        DeclareLaunchArgument("yolo_conf_threshold", default_value="0.05"),
        DeclareLaunchArgument("yolo_iou_threshold", default_value="0.45"),
        DeclareLaunchArgument("yolo_target_class", default_value="robot"),
        DeclareLaunchArgument("yolo_class_id", default_value="0"),
        DeclareLaunchArgument("yolo_use_masks", default_value="false"),
        DeclareLaunchArgument("yolo_min_mask_area_px", default_value="12.0"),
        DeclareLaunchArgument("yolo_mask_bottom_band_px", default_value="3.0"),
        DeclareLaunchArgument("yolo_min_bbox_area_px", default_value="0.0"),
        DeclareLaunchArgument("yolo_debug_frame_dir", default_value=""),
        # The bundled TorchScript export was produced at a fixed 640-square
        # shape and therefore cannot consume the 1280x720 Gazebo streams.  The
        # native checkpoint preserves Ultralytics' resize path and is the
        # commissioning-safe default.  TorchScript remains an explicit opt-in
        # for a correctly re-exported dynamic-shape model.
        DeclareLaunchArgument("yolo_use_torchscript", default_value="false"),
        DeclareLaunchArgument("yolo_warmup_iters", default_value="3"),
        DeclareLaunchArgument("yolo_inference_in_callback", default_value="true"),
        DeclareLaunchArgument("camera_observation_r_visible_uv", default_value="2.5"),
        DeclareLaunchArgument("camera_observation_r_miss_uv", default_value="40.0"),
        DeclareLaunchArgument(
            "shadow_manager", default_value="false",
            description="Run the hysteretic camera manager in shadow mode (no authority)",
        ),
        DeclareLaunchArgument(
            "manager_authority", default_value="shadow",
            description="'shadow' logs decisions only; 'active' republishes the "
                        "selected observation to /state/bev (gated: ch.09 release gates)",
        ),
        DeclareLaunchArgument(
            "manager_gp_artifact_template", default_value="",
            description="Optional per-camera GP npz path template with {camera_id}",
        ),
        DeclareLaunchArgument("manager_decision_rate_hz", default_value="5.0"),
        DeclareLaunchArgument(
            "manager_projection_calibration", default_value="",
            description="projection_calibration.json shared with the recorder; required in paper shadow runs",
        ),
        DeclareLaunchArgument("manager_require_projection_calibration", default_value="true"),
        DeclareLaunchArgument("manager_require_gp_artifacts", default_value="true"),
        DeclareLaunchArgument("manager_min_spatial_trust", default_value="0.45"),
        DeclareLaunchArgument("manager_min_association_confidence", default_value="0.70"),
        DeclareLaunchArgument("manager_max_measurement_age_s", default_value="0.15"),
        DeclareLaunchArgument("manager_age_decay_s", default_value="0.15"),
        DeclareLaunchArgument("manager_candidate_score_margin", default_value="0.08"),
        DeclareLaunchArgument("manager_required_consecutive_better_frames", default_value="3"),
        DeclareLaunchArgument("manager_max_cross_camera_disagreement_m", default_value="0.30"),
        DeclareLaunchArgument("manager_max_overlap_time_delta_s", default_value="0.05"),
        DeclareLaunchArgument("manager_require_consistency_when_source_available", default_value="true"),
        sim_bringup,
        encoder_noise,
        shadow_manager,
        _batched_detector_node(),
        _detector_node("camera_A", "/external_camera/image_raw"),
        _detector_node("camera_B", "/external_camera_b/image_raw"),
        _detector_node("camera_C", "/external_camera_c/image_raw"),
        _detector_node("camera_D", "/external_camera_d/image_raw"),
    ])
