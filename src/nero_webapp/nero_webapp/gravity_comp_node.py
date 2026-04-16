#!/usr/bin/env python3
"""Gravity compensation node for the AgileX Nero 7-DOF arm.

Reads joint positions from feedback, computes the gravity torque
vector G(q) via Pinocchio, and publishes MIT-mode commands with
kp=0 (no position stiffness) so the arm floats freely — the user
can physically guide it by hand.

Start DISABLED by default (safety). Enable via:
  ros2 param set /gravity_comp enabled true

The arm's mounting orientation on the torso is critical: the URDF
models the arm standing upright, but it's actually side-mounted
with a rotation. The `mount_rpy` parameter transforms the gravity
vector into the arm's base frame so the compensation pushes in the
right direction.
"""
from __future__ import annotations

import threading
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

try:
    import pinocchio as pin
except ImportError:
    pin = None

try:
    from agx_arm_msgs.msg import MoveMITMsg
except ImportError:
    MoveMITMsg = None


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
N_ARM_JOINTS = 7


class GravityCompNode(Node):
    def __init__(self, node_name: str = "gravity_comp") -> None:
        super().__init__(node_name)

        if pin is None:
            self.get_logger().fatal(
                "pinocchio not installed — run: "
                "pip install --break-system-packages pinocchio"
            )
            raise SystemExit(1)
        if MoveMITMsg is None:
            self.get_logger().fatal("agx_arm_msgs not found")
            raise SystemExit(1)

        self.declare_parameter("side", "right")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("rate", 200.0)
        self.declare_parameter("enabled", False)
        # Damping gains per joint — light damping keeps the arm from
        # oscillating when the user lets go mid-motion.
        self.declare_parameter(
            "kd", [0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.1])
        # Maximum torque per joint (N·m). Safety clamp.
        self.declare_parameter("torque_limit", 5.0)
        # Arm mounting orientation on the torso (roll, pitch, yaw in
        # radians, URDF extrinsic-XYZ convention). The URDF models
        # the arm upright; this rotation transforms the gravity
        # vector into the arm's base frame.
        # From two_nero.urdf.xacro: right = pi/2 0 0, left = -pi/2 0 0.
        side = self.get_parameter("side").get_parameter_value().string_value
        default_rpy = [1.5708, 0.0, 0.0] if side == "right" else [-1.5708, 0.0, 0.0]
        self.declare_parameter("mount_rpy", default_rpy)
        # Stale-data timeout: if we haven't received feedback in this
        # many seconds, stop sending torques (safety).
        self.declare_parameter("feedback_timeout", 0.1)
        urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        rate = self.get_parameter("rate").get_parameter_value().double_value
        self.kd = list(
            self.get_parameter("kd").get_parameter_value().double_array_value)
        self.torque_limit = (
            self.get_parameter("torque_limit").get_parameter_value().double_value)
        mount_rpy = list(
            self.get_parameter("mount_rpy").get_parameter_value().double_array_value)
        self.feedback_timeout = (
            self.get_parameter("feedback_timeout").get_parameter_value().double_value)

        if not urdf_path:
            self.get_logger().fatal("urdf_path parameter is required")
            raise SystemExit(1)

        # Load Pinocchio model from URDF.
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.get_logger().info(
            f"Pinocchio model loaded: nq={self.model.nq} nv={self.model.nv} "
            f"njoints={self.model.njoints}")

        # Transform gravity into the arm's base frame. The URDF's
        # base_link frame has Z pointing along the arm chain — but the
        # arm is mounted on the torso with a rotation, so world gravity
        # (-Z) isn't along the arm's -Z.
        self._set_gravity_from_mount(mount_rpy)

        # Map arm joint names → Pinocchio model indices.
        self.pin_q_indices: List[int] = []
        self.pin_v_indices: List[int] = []
        for name in ARM_JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                self.get_logger().error(f"joint {name} not in model")
                continue
            self.pin_q_indices.append(int(self.model.idx_qs[jid]))
            self.pin_v_indices.append(int(self.model.idx_vs[jid]))
        if len(self.pin_q_indices) != N_ARM_JOINTS:
            self.get_logger().fatal(
                f"expected {N_ARM_JOINTS} joints, found {len(self.pin_q_indices)}")
            raise SystemExit(1)

        # Current joint state (updated by feedback subscriber).
        self._q = np.zeros(self.model.nq)
        self._q_lock = threading.Lock()
        self._last_fb_time = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            JointState, f"/{side}/feedback/joint_states",
            self._on_feedback, qos)
        self._pub = self.create_publisher(
            MoveMITMsg, f"/{side}/control/move_mit", qos)

        self.create_timer(1.0 / rate, self._control_tick)

        # Dynamic enable/disable via parameter callback.
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"Gravity comp for {side} arm at {rate} Hz — "
            f"DISABLED (set 'enabled' param to true to activate)")

    def _set_gravity_from_mount(self, rpy: List[float]) -> None:
        """Transform world gravity [0, 0, -9.81] into the arm's base
        frame using the mount RPY, and write it into the Pinocchio
        model's gravity field."""
        from scipy.spatial.transform import Rotation as R
        # URDF RPY is extrinsic XYZ (fixed-axis): R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        # scipy's uppercase "XYZ" matches this convention.
        rot = R.from_euler("XYZ", rpy)
        g_world = np.array([0.0, 0.0, -9.81])
        g_arm = rot.inv().apply(g_world)
        self.model.gravity = pin.Motion(
            np.concatenate([g_arm, [0.0, 0.0, 0.0]]))
        self.get_logger().info(
            f"Gravity in arm frame: [{g_arm[0]:.3f}, {g_arm[1]:.3f}, "
            f"{g_arm[2]:.3f}] m/s²")

    def _on_param_change(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "enabled":
                was = self.get_parameter("enabled").get_parameter_value().bool_value
                self.get_logger().info(
                    f"gravity comp: {'ENABLED' if p.value else 'DISABLED'} "
                    f"(was {'enabled' if was else 'disabled'})")
        return SetParametersResult(successful=True)

    def _on_feedback(self, msg: JointState) -> None:
        by_name = dict(zip(msg.name, msg.position))
        with self._q_lock:
            for i, name in enumerate(ARM_JOINT_NAMES):
                if name in by_name:
                    self._q[self.pin_q_indices[i]] = by_name[name]
            self._last_fb_time = self.get_clock().now()

    def _control_tick(self) -> None:
        if not self.get_parameter("enabled").get_parameter_value().bool_value:
            return
        if self._last_fb_time is None:
            return
        age = (self.get_clock().now() - self._last_fb_time).nanoseconds / 1e9
        if age > self.feedback_timeout:
            return

        with self._q_lock:
            q = self._q.copy()

        pin.computeGeneralizedGravity(self.model, self.data, q)
        tau_g = self.data.g.copy()

        msg = MoveMITMsg()
        msg.joint_index = list(range(1, N_ARM_JOINTS + 1))
        msg.p_des = [0.0] * N_ARM_JOINTS
        msg.v_des = [0.0] * N_ARM_JOINTS
        msg.kp = [0.0] * N_ARM_JOINTS
        msg.kd = self.kd[:N_ARM_JOINTS]
        torques = []
        for i in range(N_ARM_JOINTS):
            t = float(tau_g[self.pin_v_indices[i]])
            t = max(-self.torque_limit, min(self.torque_limit, t))
            torques.append(t)
        msg.torque = torques
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    try:
        # Use side-specific node name so the webapp can target each
        # arm's parameter service independently (gravity_comp_left,
        # gravity_comp_right).
        import sys
        side = "right"
        for i, arg in enumerate(sys.argv):
            if "side:=" in arg:
                side = arg.split(":=")[1]
        node = GravityCompNode(node_name=f"gravity_comp_{side}")
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
