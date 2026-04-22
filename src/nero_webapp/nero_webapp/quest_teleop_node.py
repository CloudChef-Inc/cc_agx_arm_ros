#!/usr/bin/env python3
"""quest_teleop_node — bridges teleop-xr (Quest controller tracking) to ROS 2.

Runs the teleop-xr HTTPS/WSS server in a background thread. Each update
from the Quest carries the headset + controller devices; we filter to
`role == "controller"` and republish the gripPose (or pose) as a
geometry_msgs/PoseStamped on:

    /quest/left_controller
    /quest/right_controller

A Bool is also published per side on:

    /quest/left_controller/trigger
    /quest/right_controller/trigger

(button index 0 = trigger; used later to hook into the gripper).

SSL bootstrap (one-time): see Controller_tracking/experiments/teleop_xr/SETUP.md
The Quest browser must visit `https://<host-ip>:<port>` and accept the
self-signed cert before WebXR can connect.

Debug mode: `--debug-sinusoidal` fabricates poses so the downstream
pipeline (quest_leader_node, webapp preview) can be exercised without a
headset.
"""
from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class QuestTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("quest_teleop_node")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 4443)
        self.declare_parameter("debug_sinusoidal", False)
        self.declare_parameter("frame_id", "quest_world")

        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.debug = bool(self.get_parameter("debug_sinusoidal").value)
        self.frame_id = self.get_parameter("frame_id").value

        self._pubs = {
            "left": self.create_publisher(PoseStamped, "/quest/left_controller", 10),
            "right": self.create_publisher(PoseStamped, "/quest/right_controller", 10),
        }
        self._trig_pubs = {
            "left": self.create_publisher(Bool, "/quest/left_controller/trigger", 10),
            "right": self.create_publisher(Bool, "/quest/right_controller/trigger", 10),
        }

        if self.debug:
            self.get_logger().warn("DEBUG sinusoidal mode: fabricating controller poses")
            self._t0 = time.time()
            self.create_timer(1.0 / 50.0, self._debug_tick)
            return

        try:
            from teleop_xr import Teleop
            from teleop_xr.config import TeleopSettings
        except ImportError as e:
            self.get_logger().error(
                "teleop-xr not installed. `pip install teleop-xr` in the ROS env. "
                f"Import error: {e}"
            )
            raise

        settings = TeleopSettings(host=self.host, port=self.port)
        self._teleop = Teleop(settings=settings)
        self._teleop.subscribe(self._on_xr_update)

        self._server_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._server_thread.start()
        self.get_logger().info(
            f"teleop-xr server running on https://{self.host}:{self.port}  "
            "(open this URL on Quest browser and accept the cert)"
        )

    def _on_xr_update(self, pose, xr_state: dict) -> None:
        data = xr_state.get("data", xr_state)
        devices = data.get("devices", [])
        now = self.get_clock().now().to_msg()
        for dev in devices:
            if dev.get("role") != "controller":
                continue
            hand = dev.get("handedness")
            if hand not in ("left", "right"):
                continue
            pose_data = dev.get("gripPose") or dev.get("pose")
            if not pose_data:
                continue
            pos = pose_data.get("position", {})
            ori = pose_data.get("orientation", {})

            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = float(pos.get("x", 0.0))
            msg.pose.position.y = float(pos.get("y", 0.0))
            msg.pose.position.z = float(pos.get("z", 0.0))
            msg.pose.orientation.w = float(ori.get("w", 1.0))
            msg.pose.orientation.x = float(ori.get("x", 0.0))
            msg.pose.orientation.y = float(ori.get("y", 0.0))
            msg.pose.orientation.z = float(ori.get("z", 0.0))
            self._pubs[hand].publish(msg)

            gamepad = dev.get("gamepad") or {}
            buttons = gamepad.get("buttons", [])
            trig_pressed = bool(buttons[0].get("pressed")) if buttons else False
            self._trig_pubs[hand].publish(Bool(data=trig_pressed))

    def _debug_tick(self) -> None:
        t = time.time() - self._t0
        now = self.get_clock().now().to_msg()
        for hand, sign in (("left", +1.0), ("right", -1.0)):
            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = 0.0
            msg.pose.position.y = sign * 0.3 + 0.05 * math.sin(2 * math.pi * 0.2 * t)
            msg.pose.position.z = 1.2 + 0.05 * math.sin(2 * math.pi * 0.3 * t)
            msg.pose.orientation.w = 1.0
            self._pubs[hand].publish(msg)
            self._trig_pubs[hand].publish(Bool(data=False))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QuestTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
