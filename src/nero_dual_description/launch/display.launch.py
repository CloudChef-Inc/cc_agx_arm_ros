"""Hardware-free URDF sanity check.

Loads two_nero.urdf.xacro into robot_state_publisher, spins up
joint_state_publisher_gui so you can wiggle every joint, and opens RViz.

    ros2 launch nero_dual_description display.launch.py
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("nero_dual_description")
    default_xacro = PathJoinSubstitution([pkg_share, "urdf", "two_nero.urdf.xacro"])

    model_arg = DeclareLaunchArgument(
        "model",
        default_value=default_xacro,
        description="Path to the composed dual-arm xacro",
    )

    robot_description = {
        "robot_description": Command(
            [FindExecutable(name="xacro"), " ", LaunchConfiguration("model")]
        )
    }

    return LaunchDescription([
        model_arg,
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
        ),
    ])
