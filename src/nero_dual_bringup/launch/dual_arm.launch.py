"""Full dual-arm Nero bring-up.

Launches:
  * robot_state_publisher with the composed two-arm URDF
  * agx_arm_ctrl single-arm driver pushed into /left namespace
  * agx_arm_ctrl single-arm driver pushed into /right namespace
  * nero_webapp browser UI node

Before running: bring both CAN links up, e.g.
    sudo ip link set can0 up type can bitrate 1000000
    sudo ip link set can1 up type can bitrate 1000000
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    left_can_arg = DeclareLaunchArgument(
        "left_can", default_value="can0",
        description="CAN interface for the left arm.",
    )
    right_can_arg = DeclareLaunchArgument(
        "right_can", default_value="can1",
        description="CAN interface for the right arm.",
    )
    http_port_arg = DeclareLaunchArgument(
        "http_port", default_value="8080",
        description="HTTP port for the nero_webapp UI.",
    )
    http_host_arg = DeclareLaunchArgument(
        "http_host", default_value="0.0.0.0",
        description="HTTP bind address for the nero_webapp UI.",
    )
    # Pika grippers are USB-serial (/dev/ttyACM*), not on the arm's CAN bus.
    # Leave empty to disable that side's gripper cleanly.
    left_pika_serial_arg = DeclareLaunchArgument(
        "left_pika_serial", default_value="",
        description="USB-serial device path for the left Pika gripper (empty to disable).",
    )
    right_pika_serial_arg = DeclareLaunchArgument(
        "right_pika_serial", default_value="",
        description="USB-serial device path for the right Pika gripper (empty to disable).",
    )

    # Composed two-arm URDF via xacro.
    xacro_file = PathJoinSubstitution([
        FindPackageShare("nero_dual_description"),
        "urdf", "two_nero.urdf.xacro",
    ])
    robot_description = {
        "robot_description": Command(["xacro ", xacro_file]),
    }

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    single_arm_launch = PythonLaunchDescriptionSource(
        PathJoinSubstitution([
            FindPackageShare("agx_arm_ctrl"),
            "launch", "start_single_agx_arm.launch.py",
        ])
    )

    left_arm = IncludeLaunchDescription(
        single_arm_launch,
        launch_arguments={
            "namespace":     "left",
            "can_port":      LaunchConfiguration("left_can"),
            "arm_type":      "nero",
            "effector_type": "none",
        }.items(),
    )

    right_arm = IncludeLaunchDescription(
        single_arm_launch,
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
            "left_pika_serial":  LaunchConfiguration("left_pika_serial"),
            "right_pika_serial": LaunchConfiguration("right_pika_serial"),
        }],
    )

    return LaunchDescription([
        left_can_arg,
        right_can_arg,
        http_port_arg,
        http_host_arg,
        left_pika_serial_arg,
        right_pika_serial_arg,
        rsp_node,
        left_arm,
        right_arm,
        webapp_node,
    ])
