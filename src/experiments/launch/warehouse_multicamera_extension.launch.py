from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _detector_params(
    camera_id: str,
    image_topic: str,
    pixel_topic: str,
    diagnostics_topic: str,
    observation_topic: str,
):
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


def _detector_node(camera_letter: str, image_topic: str) -> Node:
    camera_id = f"camera_{camera_letter}"
    return Node(
        package="perception",
        executable="yolo_robot_detector_node",
        name=f"yolo_robot_detector_camera_{camera_letter.lower()}",
        output="screen",
        parameters=[
            _detector_params(
                camera_id,
                image_topic,
                f"/perception/{camera_id}/pixel_pose",
                f"/perception/{camera_id}/detection_diagnostics",
                f"/perception/camera_observation/{camera_id}",
            )
        ],
    )


def generate_launch_description():
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

    detectors = [
        _detector_node("A", "/external_camera/image_raw"),
        _detector_node("B", "/external_camera_b/image_raw"),
        _detector_node("C", "/external_camera_c/image_raw"),
        _detector_node("D", "/external_camera_d/image_raw"),
    ]

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="warehouse_full_4cam.world.sdf"),
        DeclareLaunchArgument("world_name", default_value="warehouse_full_4cam"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("reset_world", default_value="false"),
        DeclareLaunchArgument("spawn_x", default_value="-1.0"),
        DeclareLaunchArgument("spawn_y", default_value="-7.8"),
        DeclareLaunchArgument("spawn_z", default_value="0.05"),
        DeclareLaunchArgument("spawn_yaw", default_value="1.5708"),
        DeclareLaunchArgument("yolo_model", default_value="", description="Local path to the trained YOLO model"),
        DeclareLaunchArgument("yolo_device", default_value=""),
                # 960, matching every trained model. Until 2026-08-21 this defaulted to 640
        # while all five checkpoints were trained at imgsz 960, so inference ran at a
        # resolution the weights had never seen. Measured on 200 real val frames with
        # the frozen four-camera detector: the median bottom-edge error against the
        # ground-truth box improves from 3.24 to 2.52 px (mask) and 2.88 to 2.40 px
        # (box) going from 640 to 960, and recall is unchanged. 1280 is better again
        # (2.83 / 2.02) but a five-image batch costs 164 ms against the 200 ms the
        # 5 Hz cameras allow, leaving nothing for the rest of the stack; 960 costs
        # 99 ms. The measurement matters because the MEASUREMENT is the mask's bottom
        # edge, so inference resolution is a measurement parameter, not a speed knob.
        DeclareLaunchArgument("yolo_imgsz", default_value="960"),
        DeclareLaunchArgument("yolo_conf_threshold", default_value="0.05"),
        DeclareLaunchArgument("yolo_iou_threshold", default_value="0.45"),
        DeclareLaunchArgument("yolo_target_class", default_value="robot"),
        DeclareLaunchArgument("yolo_class_id", default_value="0"),
        DeclareLaunchArgument("yolo_use_masks", default_value="false"),
        DeclareLaunchArgument("yolo_min_mask_area_px", default_value="12.0"),
        DeclareLaunchArgument("yolo_mask_bottom_band_px", default_value="3.0"),
        DeclareLaunchArgument("yolo_min_bbox_area_px", default_value="0.0"),
        DeclareLaunchArgument("yolo_debug_frame_dir", default_value=""),
        DeclareLaunchArgument("yolo_use_torchscript", default_value="false"),
        DeclareLaunchArgument("yolo_warmup_iters", default_value="3"),
        DeclareLaunchArgument("yolo_inference_in_callback", default_value="true"),
        DeclareLaunchArgument("camera_observation_r_visible_uv", default_value="2.5"),
        DeclareLaunchArgument("camera_observation_r_miss_uv", default_value="40.0"),
        sim_bringup,
        *detectors,
    ])
