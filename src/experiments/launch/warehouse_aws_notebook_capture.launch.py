"""Single-camera notebook capture: the AWS warehouse, one ceiling camera, one detector.

This is the one-camera counterpart of `warehouse_full4cam_commissioning.launch.py`. It
exists because neither existing launch does what the filter notebooks need in this world:

  * `warehouse_visibility_capture.launch.py` runs the detector into
    `pixel_to_bev_state_node` and never publishes the `camera_observation` JSON the
    notebook recorder consumes, and brings up no encoder-noise stream, so there is no
    `/odom_noisy` for the filter to dead-reckon on;
  * `warehouse_full4cam_commissioning.launch.py` publishes exactly the right streams but
    is wired for four cameras in the frozen flagship world.

So: `bringup_sim` for the AWS world (which also bridges `/ground_truth_tf`), the same
`encoder_noise_node` the four-camera study uses, and ONE `yolo_robot_detector_node`
configured to publish `camera_observation` JSON on
`/perception/camera_observation/camera_A`.

`camera_A` is this world's only camera, `external_camera` in the world file. The name is
kept because every downstream loader, scorer and figure in `experiments/filter_notebook`
is keyed on `camera_<X>`; calling it anything else would fork those.

Two-world rule: this world is where method development belongs
(`research/06_world_camera_design.md`), which is the point of running the notebook here.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

CAMERA_ID = "camera_A"
IMAGE_TOPIC = "/external_camera/image_raw"


def _launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration("world").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)

    bringup_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("sim"), "launch", "bringup_sim.launch.py"])
        ),
        launch_arguments={
            "use_sim_time": "true",
            "use_lidar": "false",
            "bridge_scan": "false",
            "headless": LaunchConfiguration("headless").perform(context),
            "world": world,
            "world_name": world_name,
            "spawn_x": LaunchConfiguration("spawn_x").perform(context),
            "spawn_y": LaunchConfiguration("spawn_y").perform(context),
            "spawn_z": "0.01",
            "spawn_yaw": LaunchConfiguration("spawn_yaw").perform(context),
            # Opt-in marker disks. Default off, so every existing capture is unchanged;
            # a capture that wants the keypoint reading scored on the SAME frames as the
            # box bottom passes show_pose_markers:=true.
            "show_pose_markers": LaunchConfiguration("show_pose_markers").perform(context),
        }.items(),
    )

    wait_for_odom = Node(
        package="sim", executable="wait_for_odom", name="wait_for_odom", output="screen",
        parameters=[{"topic": "/odom", "timeout_s": 30.0, "min_messages": 3,
                     "require_pose_match": False}],
    )

    # The passive operational odometry the filter actually consumes. The route driver
    # follows Gazebo's clean /odom; nothing steers on this stream.
    encoder_noise = Node(
        package="sim", executable="encoder_noise_node", name="aws_notebook_encoder_noise",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "enabled": LaunchConfiguration("use_encoder_noise"),
            "input_topic": "/odom",
            "output_topic": "/odom_noisy",
            "seed": LaunchConfiguration("encoder_noise_seed"),
        }],
    )

    detector = Node(
        package="perception", executable="yolo_robot_detector_node",
        name=f"yolo_robot_detector_{CAMERA_ID.lower()}", output="screen",
        parameters=[{
            "use_sim_time": True,
            "model_path": LaunchConfiguration("yolo_model"),
            # Ultralytics accepts "0", "cpu" and "cuda:0"; the node declares this as a
            # string, so the CLI token's type has to be preserved explicitly.
            "device": ParameterValue(LaunchConfiguration("yolo_device"), value_type=str),
            "image_size": LaunchConfiguration("yolo_imgsz"),
            "confidence_threshold": LaunchConfiguration("yolo_conf_threshold"),
            "iou_threshold": LaunchConfiguration("yolo_iou_threshold"),
            "class_name": LaunchConfiguration("yolo_target_class"),
            "class_id": LaunchConfiguration("yolo_class_id"),
            # Box bottom-centre, matching the deployed path the notebooks model. Masks
            # would change what the observation IS, not just how well it is found.
            "use_masks": False,
            "use_torchscript": False,
            "pixel_noise_sigma": 0.0,
            "seed": LaunchConfiguration("seed"),
            "publish_camera_observation_json": True,
            "image_topic": IMAGE_TOPIC,
            "pixel_pose_topic": f"/perception/{CAMERA_ID}/pixel_pose",
            "diagnostics_topic": f"/perception/{CAMERA_ID}/detection_diagnostics",
            "camera_observation_topic": f"/perception/camera_observation/{CAMERA_ID}",
            "camera_id": CAMERA_ID,
            "camera_calibration_id": f"warehouse_aws_{CAMERA_ID}",
            "camera_image_frame_id": CAMERA_ID,
            "camera_observation_r_visible_uv": LaunchConfiguration(
                "camera_observation_r_visible_uv"),
            "camera_observation_r_miss_uv": LaunchConfiguration(
                "camera_observation_r_miss_uv"),
        }],
    )

    return [bringup_sim, wait_for_odom, encoder_noise, detector]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="warehouse_aws.world.sdf"),
        DeclareLaunchArgument("world_name", default_value="warehouse_aws"),
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument("spawn_x", default_value="0.0"),
        DeclareLaunchArgument("spawn_y", default_value="-4.0"),
        DeclareLaunchArgument("spawn_yaw", default_value="1.5708"),
        DeclareLaunchArgument("show_pose_markers", default_value="false"),
        DeclareLaunchArgument("use_encoder_noise", default_value="true"),
        DeclareLaunchArgument("encoder_noise_seed", default_value="0"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("yolo_model", default_value=""),
        DeclareLaunchArgument("yolo_device", default_value="0"),
        DeclareLaunchArgument("yolo_imgsz", default_value="960"),
        DeclareLaunchArgument("yolo_conf_threshold", default_value="0.05"),
        DeclareLaunchArgument("yolo_iou_threshold", default_value="0.45"),
        DeclareLaunchArgument("yolo_target_class", default_value="robot"),
        DeclareLaunchArgument("yolo_class_id", default_value="-1"),
        DeclareLaunchArgument("camera_observation_r_visible_uv", default_value="2.5"),
        DeclareLaunchArgument("camera_observation_r_miss_uv", default_value="40.0"),
        OpaqueFunction(function=_launch_setup),
    ])
