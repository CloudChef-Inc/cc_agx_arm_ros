#!/usr/bin/env python3
"""vel_monitor.py — only prints when velocity deviates from zero."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

THRESHOLD = 1e-6

class VelMon(Node):
    def __init__(self):
        super().__init__("vel_mon")
        self.sub = self.create_subscription(
            JointState, "/left/feedback/joint_states", self.cb, 1)
        self.n = 0

    def cb(self, msg):
        self.n += 1
        nonzero = [
            (i + 1, v) for i, v in enumerate(msg.velocity[:7])
            if abs(v) > THRESHOLD
        ]
        if nonzero:
            parts = " ".join(f"J{j}:{v:+.4f}" for j, v in nonzero)
            self.get_logger().info(f"NON-ZERO: {parts}")
        elif self.n % 200 == 0:
            self.get_logger().info(f"{self.n} samples — all zero so far")

def main():
    rclpy.init()
    print("Watching /left/feedback/joint_states — will print ONLY non-zero velocities.")
    print("Move joints from the webapp. Ctrl-C to stop.\n")
    try:
        rclpy.spin(VelMon())
    except KeyboardInterrupt:
        print()

if __name__ == "__main__":
    main()
