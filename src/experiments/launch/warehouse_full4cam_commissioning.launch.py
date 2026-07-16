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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _detector_params(
    *,
    camera_id: str,
    image_topic: str,
    pixel_topic: str,
    diagnostics_topic: str,
    observation_topic: str,
) -> dict:
    return {
        "use_sim_time": LaunchConfiguration("use_sim_time"),
        "model_path": LaunchConfiguration("yolo_model"),
        "device": LaunchConfiguration("yolo_device"),
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
        parameters=[
            _detector_params(
                camera_id=camera_id,
                image_topic=image_topic,
                pixel_topic=f"/perception/{camera_id}/pixel_pose",
                diagnostics_topic=f"/perception/{camera_id}/detection_diagnostics",
                observation_topic=f"/perception/camera_observation/{camera_id}",
            )
        ],
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
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="warehouse_full_4cam.world.sdf"),
        DeclareLaunchArgument("world_name", default_value="warehouse_full_4cam"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("reset_world", default_value="false"),
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
        sim_bringup,
        encoder_noise,
        shadow_manager,
        _detector_node("camera_A", "/external_camera/image_raw"),
        _detector_node("camera_B", "/external_camera_b/image_raw"),
        _detector_node("camera_C", "/external_camera_c/image_raw"),
        _detector_node("camera_D", "/external_camera_d/image_raw"),
    ])
