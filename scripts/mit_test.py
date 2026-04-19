#!/usr/bin/env python3
"""mit_test.py — monitor J1 during gravity comp zero-torque test.

Prints J1 pos/torque/drift every 0.5s, plus event logs when:
  - J1 position jumps (webapp Send detected)
  - MIT commands start/stop flowing (gravity comp toggled)
  - MIT t_ff value changes (gravity_scale changed)
  - J1 drifts significantly (arm falling = MIT working)

Usage:
  python3 scripts/mit_test.py          # default: left arm
  python3 scripts/mit_test.py right    # right arm
"""
import rclpy, sys, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from agx_arm_msgs.msg import MoveMITMsg


class MitTest(Node):
    def __init__(self):
        super().__init__("mit_test")
        side = sys.argv[1] if len(sys.argv) > 1 else "left"
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            JointState, f"/{side}/feedback/joint_states", self.on_fb, qos)
        self.create_subscription(
            MoveMITMsg, f"/{side}/control/move_mit", self.on_mit, qos)
        self.side = side

        self.prev_pos = None
        self.pos_at_mit_start = None
        self.last_print = 0
        self.last_mit_time = 0
        self.mit_active = False
        self.mit_count = 0
        self.last_tff = None
        self.drift_logged = False

        self.get_logger().info(f"=== Watching J1 ({side} arm) — 0.5s intervals + events ===")

    def on_mit(self, msg):
        self.mit_count += 1
        self.last_mit_time = time.time()
        t = msg.torque[0] if msg.torque else 0
        kp = msg.kp[0] if msg.kp else 0
        kd = msg.kd[0] if msg.kd else 0
        p = msg.p_des[0] if msg.p_des else 0

        if not self.mit_active:
            self.mit_active = True
            self.pos_at_mit_start = self.prev_pos
            self.drift_logged = False
            self.get_logger().info(
                f"EVENT: gravity comp ENABLED — MIT commands started")
            self.get_logger().info(
                f"       J1: kp={kp:.2f} kd={kd:.2f} t_ff={t:.4f} p_des={p:.4f}")

        if self.last_tff is not None and abs(t - self.last_tff) > 0.01:
            self.get_logger().info(
                f"EVENT: t_ff changed {self.last_tff:.4f} → {t:.4f}")
        self.last_tff = t

    def on_fb(self, msg):
        try:
            idx = msg.name.index("joint1")
        except ValueError:
            return

        pos = msg.position[idx]
        torque = msg.effort[idx] if idx < len(msg.effort) else 0.0

        # Detect position jump (webapp Send)
        if self.prev_pos is not None and abs(pos - self.prev_pos) > 0.05:
            self.get_logger().info(
                f"EVENT: J1 position jumped {self.prev_pos:+.4f} → {pos:+.4f} "
                f"(webapp Send?)")
        self.prev_pos = pos

        # Detect MIT stopped
        if self.mit_active and time.time() - self.last_mit_time > 0.5:
            self.mit_active = False
            self.get_logger().info("EVENT: gravity comp DISABLED — MIT commands stopped")

        # Detect arm falling
        if (self.mit_active and self.pos_at_mit_start is not None
                and not self.drift_logged
                and abs(pos - self.pos_at_mit_start) > 0.02):
            self.drift_logged = True
            drift = pos - self.pos_at_mit_start
            self.get_logger().info(
                f"EVENT: ARM FELL — drift={drift:+.4f} rad from enable point")

        # Periodic status
        now = time.time()
        if now - self.last_print < 0.5:
            return
        self.last_print = now

        drift = pos - self.pos_at_mit_start if self.pos_at_mit_start is not None else 0.0
        mit_str = "YES" if self.mit_active else "no"
        self.get_logger().info(
            f"J1  pos={pos:+.4f}  torque={torque:+.4f}  "
            f"drift={drift:+.4f}  mit={mit_str}")


def main():
    rclpy.init()
    try:
        rclpy.spin(MitTest())
    except KeyboardInterrupt:
        print()

if __name__ == "__main__":
    main()
