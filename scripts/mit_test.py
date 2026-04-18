#!/usr/bin/env python3
"""mit_test.py — monitor J1 during gravity comp zero-torque test.

Prints J1 position, torque, and drift every 0.5s.
Also subscribes to the MIT command topic to confirm gravity comp is active.

Run:
  source ~/Desktop/CloudChef/cc_agx_arm_ros/install/setup.bash
  python3 ~/Desktop/CloudChef/cc_agx_arm_ros/scripts/mit_test.py
"""
import rclpy, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from agx_arm_msgs.msg import MoveMITMsg


class MitTest(Node):
    def __init__(self):
        super().__init__("mit_test")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            JointState, "/left/feedback/joint_states", self.on_fb, qos)
        self.create_subscription(
            MoveMITMsg, "/left/control/move_mit", self.on_mit, qos)

        self.initial_pos = None
        self.last_print = 0
        self.mit_active = False
        self.mit_count = 0

        self.get_logger().info("=== Watching J1 (left arm) — prints every 0.5s ===")

    def on_mit(self, msg):
        self.mit_count += 1
        if not self.mit_active:
            self.mit_active = True
            t = msg.torque[0] if msg.torque else 0
            kp = msg.kp[0] if msg.kp else 0
            kd = msg.kd[0] if msg.kd else 0
            p = msg.p_des[0] if msg.p_des else 0
            self.get_logger().info(
                f">>> MIT commands started — "
                f"J1: kp={kp:.2f} kd={kd:.2f} t_ff={t:.4f} p_des={p:.4f}")

    def on_fb(self, msg):
        try:
            idx = msg.name.index("joint1")
        except ValueError:
            return

        pos = msg.position[idx]
        torque = msg.effort[idx] if idx < len(msg.effort) else 0.0

        if self.initial_pos is None:
            self.initial_pos = pos

        now = time.time()
        if now - self.last_print < 0.5:
            return
        self.last_print = now

        drift = pos - self.initial_pos
        mit = f"  mit_msgs={self.mit_count}" if self.mit_count else ""
        self.get_logger().info(
            f"J1  pos={pos:+.4f}  drift={drift:+.4f}  "
            f"torque={torque:+.4f}{mit}")


def main():
    rclpy.init()
    try:
        rclpy.spin(MitTest())
    except KeyboardInterrupt:
        print()

if __name__ == "__main__":
    main()
