"""Full dual-arm Nero bring-up.

Launches:
  * robot_state_publisher with the composed two-arm URDF
  * agx_arm_ctrl single-arm driver pushed into /left namespace
  * agx_arm_ctrl single-arm driver pushed into /right namespace
  * nero_webapp browser UI node (6 camera streams max — fisheye + color +
    depth × 2 sides — WebRTC'd to the browser)

All per-device defaults point at the stable names maintained by the
udev rules + /usr/local/bin/nero-detect-pika: `can_left`, `can_right`,
`/dev/pika_{side}`, `/dev/fisheye_{side}`. Re-plug any cable on any hub
and the launch still works without edits.

Setting any device path to an empty string cleanly disables that
specific device — useful for bringing up one arm at a time during
hardware work.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    left_can_arg = DeclareLaunchArgument(
        "left_can", default_value="can_left",
        description="CAN interface for the left arm (stable udev name).",
    )
    right_can_arg = DeclareLaunchArgument(
        "right_can", default_value="can_right",
        description="CAN interface for the right arm (stable udev name).",
    )
    http_port_arg = DeclareLaunchArgument(
        "http_port", default_value="8000",
        description="HTTP port for the nero_webapp UI.",
    )
    http_host_arg = DeclareLaunchArgument(
        "http_host", default_value="0.0.0.0",
        description="HTTP bind address for the nero_webapp UI.",
    )
    # Pika grippers (CH340 USB-serial). Symlinks maintained by
    # nero-detect-pika. Empty path disables that side's gripper.
    left_pika_serial_arg = DeclareLaunchArgument(
        "left_pika_serial", default_value="/dev/pika_left",
        description="USB-serial device for the left Pika gripper "
                    "(empty to disable).",
    )
    right_pika_serial_arg = DeclareLaunchArgument(
        "right_pika_serial", default_value="/dev/pika_right",
        description="USB-serial device for the right Pika gripper "
                    "(empty to disable).",
    )
    # Per-side Pika fisheye cameras (UVC). Symlinks maintained by
    # nero-detect-pika.
    left_fisheye_device_arg = DeclareLaunchArgument(
        "left_fisheye_device", default_value="/dev/fisheye_left",
        description="V4L2 path for the left Pika fisheye (empty to disable).",
    )
    right_fisheye_device_arg = DeclareLaunchArgument(
        "right_fisheye_device", default_value="/dev/fisheye_right",
        description="V4L2 path for the right Pika fisheye (empty to disable).",
    )
    # Per-side RealSense D405 serials. Empty → webapp_node falls back
    # to /etc/nero/d405_sides.conf, which the detector already manages.
    left_realsense_serial_arg = DeclareLaunchArgument(
        "left_realsense_serial", default_value="",
        description="RealSense D405 serial for the left arm (empty "
                    "= fall back to /etc/nero/d405_sides.conf).",
    )
    right_realsense_serial_arg = DeclareLaunchArgument(
        "right_realsense_serial", default_value="",
        description="RealSense D405 serial for the right arm (empty "
                    "= fall back to /etc/nero/d405_sides.conf).",
    )

    # Torso dimensions — single source of truth. Flows to both the
    # xacro (URDF model) and the webapp (3D rendering).
    torso_width_arg = DeclareLaunchArgument(
        "torso_width", default_value="0.1126",
        description="Distance between the two arm base-link origins (metres).",
    )
    torso_depth_arg = DeclareLaunchArgument(
        "torso_depth", default_value="0.10",
        description="Torso depth front-to-back (metres).",
    )
    torso_height_arg = DeclareLaunchArgument(
        "torso_height", default_value="0.60",
        description="Torso height (metres).",
    )
    shoulder_tilt_arg = DeclareLaunchArgument(
        "shoulder_tilt_deg", default_value="20.0",
        description="Roll of each arm about its own +X axis (deg). "
                    "Right arm tilts +shoulder_tilt_deg, left tilts -shoulder_tilt_deg.",
    )

    quest_host_arg = DeclareLaunchArgument(
        "quest_host", default_value="0.0.0.0",
        description="Bind host for the Quest teleop-xr WSS server.",
    )
    quest_port_arg = DeclareLaunchArgument(
        "quest_port", default_value="4443",
        description="Port for the Quest teleop-xr WSS server.",
    )
    quest_debug_arg = DeclareLaunchArgument(
        "quest_debug", default_value="false",
        description="If true, quest_teleop_node fabricates sinusoidal poses "
                    "instead of running the WebXR server (no headset needed).",
    )

    # Composed two-arm URDF via xacro — pass torso dims as args.
    xacro_file = PathJoinSubstitution([
        FindPackageShare("nero_dual_description"),
        "urdf", "two_nero.urdf.xacro",
    ])
    robot_description = {
        "robot_description": ParameterValue(
            Command([
                "xacro ", xacro_file,
                " torso_width:=", LaunchConfiguration("torso_width"),
                " torso_depth:=", LaunchConfiguration("torso_depth"),
                " torso_height:=", LaunchConfiguration("torso_height"),
                " shoulder_tilt_deg:=", LaunchConfiguration("shoulder_tilt_deg"),
            ]), value_type=str
        ),
    }

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # Each IncludeLaunchDescription needs its OWN PythonLaunchDescriptionSource —
    # in Jazzy, sharing a source between includes raises "executed more than once".
    single_arm_launch_path = PathJoinSubstitution([
        FindPackageShare("agx_arm_ctrl"),
        "launch", "start_single_agx_arm.launch.py",
    ])

    left_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_arm_launch_path),
        launch_arguments={
            "namespace":     "left",
            "can_port":      LaunchConfiguration("left_can"),
            "arm_type":      "nero",
            "effector_type": "none",
        }.items(),
    )

    right_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(single_arm_launch_path),
        launch_arguments={
            "namespace":     "right",
            "can_port":      LaunchConfiguration("right_can"),
            "arm_type":      "nero",
            "effector_type": "none",
        }.items(),
    )

    webapp_node = Node(
        package="nero_webapp",
        executable="webapp_node",
        name="nero_webapp",
        output="screen",
        parameters=[{
            "left_ns":   "left",
            "right_ns":  "right",
            "http_host": LaunchConfiguration("http_host"),
            "http_port": LaunchConfiguration("http_port"),
            "left_pika_serial":       LaunchConfiguration("left_pika_serial"),
            "right_pika_serial":      LaunchConfiguration("right_pika_serial"),
            "left_fisheye_device":    LaunchConfiguration("left_fisheye_device"),
            "right_fisheye_device":   LaunchConfiguration("right_fisheye_device"),
            "left_realsense_serial":  LaunchConfiguration("left_realsense_serial"),
            "right_realsense_serial": LaunchConfiguration("right_realsense_serial"),
            "torso_width":       LaunchConfiguration("torso_width"),
            "torso_depth":       LaunchConfiguration("torso_depth"),
            "torso_height":      LaunchConfiguration("torso_height"),
            "shoulder_tilt_deg": LaunchConfiguration("shoulder_tilt_deg"),
        }],
    )

    # Gravity compensation nodes (one per arm). Start DISABLED —
    # the webapp's Float button toggles them via parameter service.
    nero_urdf = PathJoinSubstitution([
        FindPackageShare("nero_webapp"), "static", "nero_with_gripper.urdf",
    ])
    gravity_comp_left = Node(
        package="nero_webapp",
        executable="gravity_comp",
        name="gravity_comp_left",
        output="screen",
        parameters=[{
            "side": "left",
            "urdf_path": nero_urdf,
            "gravity_vector": [-9.81, 0.0, 0.0],
        }],
    )
    gravity_comp_right = Node(
        package="nero_webapp",
        executable="gravity_comp",
        name="gravity_comp_right",
        output="screen",
        parameters=[{
            "side": "right",
            "urdf_path": nero_urdf,
            "gravity_vector": [-9.81, 0.0, 0.0],
        }],
    )

    quest_teleop = Node(
        package="nero_webapp",
        executable="quest_teleop_node",
        name="quest_teleop_node",
        output="screen",
        parameters=[{
            "host": LaunchConfiguration("quest_host"),
            "port": ParameterValue(LaunchConfiguration("quest_port"), value_type=int),
            "debug_sinusoidal": ParameterValue(LaunchConfiguration("quest_debug"), value_type=bool),
        }],
    )
    # FK/IK is done in-process via Pinocchio on the single-arm URDF
    # (same one gravity_comp loads). No MoveIt / move_group required —
    # avoids TF-tree conflicts against the composed dual-arm publisher.
    quest_leader_left = Node(
        package="nero_webapp",
        executable="quest_leader_node",
        name="quest_leader_left",
        output="screen",
        parameters=[{
            "side": "left",
            "ee_link": "gripper_flange",
            "urdf_path": nero_urdf,
        }],
    )
    quest_leader_right = Node(
        package="nero_webapp",
        executable="quest_leader_node",
        name="quest_leader_right",
        output="screen",
        parameters=[{
            "side": "right",
            "ee_link": "gripper_flange",
            "urdf_path": nero_urdf,
        }],
    )

    return LaunchDescription([
        left_can_arg,
        right_can_arg,
        http_port_arg,
        http_host_arg,
        left_pika_serial_arg,
        right_pika_serial_arg,
        left_fisheye_device_arg,
        right_fisheye_device_arg,
        left_realsense_serial_arg,
        right_realsense_serial_arg,
        torso_width_arg,
        torso_depth_arg,
        torso_height_arg,
        shoulder_tilt_arg,
        quest_host_arg,
        quest_port_arg,
        quest_debug_arg,
        rsp_node,
        left_arm,
        right_arm,
        webapp_node,
        gravity_comp_left,
        gravity_comp_right,
        quest_teleop,
        quest_leader_left,
        quest_leader_right,
    ])
