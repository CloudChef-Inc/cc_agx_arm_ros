#!/usr/bin/env python3
"""mit_test.py — monitor J1 position + torque during gravity comp test.

Verifies the test sequence by tracking state transitions:
  [WAITING]  → waiting for J1 to move to a non-zero position (webapp Send)
  [ARMED]    → J1 at target, waiting for MIT commands (gravity comp enable)
  [MIT]      → MIT commands detected, watching for arm to fall

Run:  source ~/cc-nero/install/setup.bash && python3 scripts/mit_test.py
"""
import rclpy, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

try:
    from agx_arm_ctrl_msgs.msg import MoveMITMsg
    HAS_MIT_MSG = True
except ImportError:
    HAS_MIT_MSG = False

MOVE_THRESHOLD = 0.15   # rad — J1 must move at least this much from zero
DRIFT_THRESHOLD = 0.02  # rad — arm "fell" if it drifts this much after MIT


class MitTest(Node):
    def __init__(self):
        super().__init__("mit_test")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            JointState, "/left/feedback/joint_states", self.on_fb, qos)

        if HAS_MIT_MSG:
            self.create_subscription(
                MoveMITMsg, "/left/control/move_mit", self.on_mit, qos)
        else:
            self.get_logger().warn(
                "MoveMITMsg not found — won't detect MIT commands. "
                "Source the workspace first.")

        self.state = "WAITING"
        self.start_pos = None       # J1 position when script starts
        self.armed_pos = None       # J1 position when gravity comp enabled
        self.last_print = 0
        self.mit_count = 0
        self.fell = False

        self.log("=== MIT MODE TEST — watching J1 (left arm) ===")
        self.log("")
        self.log("  1. Move J1 slider to ~0.5 rad in webapp, click Send")
        self.log("  2. ros2 param set /gravity_comp_left gravity_scale 0.0")
        self.log("  3. Click Gravity Comp (left) in webapp")
        self.log("")
        self.log("[WAITING] for J1 to move away from start position...")

    def log(self, msg):
        self.get_logger().info(msg)

    def on_mit(self, msg):
        self.mit_count += 1
        if self.state == "ARMED":
            t_j1 = msg.torque[0] if msg.torque else None
            kp_j1 = msg.kp[0] if msg.kp else None
            kd_j1 = msg.kd[0] if msg.kd else None
            self.state = "MIT"
            self.armed_pos = self._last_pos
            self.log("")
            self.log(f"[MIT] Gravity comp ENABLED — MIT commands flowing")
            self.log(f"      J1: kp={kp_j1}  kd={kd_j1}  t_ff={t_j1}")
            if t_j1 is not None and abs(t_j1) > 0.01:
                self.log(f"      WARNING: t_ff is NOT zero — "
                         f"did you set gravity_scale to 0?")
            else:
                self.log(f"      t_ff ≈ 0 — good, gravity_scale is 0")
            self.log(f"      J1 position at enable: {self.armed_pos:+.4f} rad")
            self.log(f"      Watching for arm to fall...")
            self.log("")

    def on_fb(self, msg):
        try:
            idx = msg.name.index("joint1")
        except ValueError:
            return

        pos = msg.position[idx]
        torque = msg.effort[idx] if idx < len(msg.effort) else 0.0
        vel = msg.velocity[idx] if idx < len(msg.velocity) else 0.0
        self._last_pos = pos

        if self.start_pos is None:
            self.start_pos = pos
            self.log(f"J1 start position: {pos:+.4f} rad")
            return

        now = time.time()

        # State: waiting for user to Send a position from webapp
        if self.state == "WAITING":
            if abs(pos - self.start_pos) > MOVE_THRESHOLD:
                self.state = "ARMED"
                self.log("")
                self.log(f"[ARMED] J1 moved to {pos:+.4f} rad "
                         f"(Δ={pos - self.start_pos:+.4f})")
                self.log(f"        Now set gravity_scale=0 and "
                         f"enable gravity comp")
            return

        # State: MIT active — monitor drift and torque
        if self.state == "MIT" and now - self.last_print >= 0.5:
            self.last_print = now
            drift = pos - self.armed_pos
            self.log(f"  pos={pos:+.4f}  drift={drift:+.4f}  "
                     f"torque={torque:+.4f}  vel={vel:+.4f}")

            if not self.fell and abs(drift) > DRIFT_THRESHOLD:
                self.fell = True
                self.log("")
                self.log(f"  >>> ARM FELL — drift={drift:+.4f} rad")
                self.log(f"  >>> MIT mode IS working through the ROS stack")
                self.log("")


def main():
    rclpy.init()
    try:
        rclpy.spin(MitTest())
    except KeyboardInterrupt:
        print()

if __name__ == "__main__":
    main()
