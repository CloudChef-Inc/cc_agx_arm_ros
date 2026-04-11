"""ROS 2 node that serves a browser UI to command two AgileX Nero arms.

Architecture
------------
- Main thread runs uvicorn (FastAPI HTTP + WebSocket).
- rclpy spins on a background daemon thread.
- Browser <-> node: one WebSocket at /ws
    Browser -> node : {"side": "left"|"right", "positions": [7 floats]}
    Node    -> browser (20 Hz):
                       {"type": "state", "data": {
                           "left":  {"names": [...], "positions": [...]},
                           "right": {"names": [...], "positions": [...]}
                       }}

Topic wiring (matches agilexrobotics/agx_arm_ros > agx_arm_ctrl single-arm driver)
- Command : /<side>/control/joint_states      (sensor_msgs/JointState)
- Feedback: /<side>/feedback/joint_states     (sensor_msgs/JointState)

Server-side safety: commanded positions are clamped to the Nero URDF
joint limits from nero_description.urdf before being published.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Dict, List, Set

from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import uvicorn


# Default Nero joint names (7 revolute). Match nero_description.urdf.
DEFAULT_JOINT_NAMES: List[str] = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
]

# Hard limits from nero_description.urdf, in radians.
JOINT_LIMITS: List[tuple[float, float]] = [
    (-2.70526,    2.70526),   # joint1
    (-1.74,       1.74),      # joint2
    (-2.75,       2.75),      # joint3
    (-1.01,       2.14),      # joint4
    (-2.75,       2.75),      # joint5
    (-0.73,       0.95),      # joint6
    (-1.5707963,  1.5707963), # joint7
]


class WebappNode(Node):
    def __init__(self) -> None:
        super().__init__("nero_webapp")

        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("left_ns",  "left")
        self.declare_parameter("right_ns", "right")
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 8080)

        self.joint_names: List[str] = (
            self.get_parameter("joint_names").get_parameter_value().string_array_value
            or DEFAULT_JOINT_NAMES
        )
        left_ns  = self.get_parameter("left_ns").get_parameter_value().string_value
        right_ns = self.get_parameter("right_ns").get_parameter_value().string_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_left = self.create_publisher(
            JointState, f"/{left_ns}/control/joint_states", qos
        )
        self.pub_right = self.create_publisher(
            JointState, f"/{right_ns}/control/joint_states", qos
        )

        self.create_subscription(
            JointState, f"/{left_ns}/feedback/joint_states",
            lambda msg: self._on_state("left", msg), qos,
        )
        self.create_subscription(
            JointState, f"/{right_ns}/feedback/joint_states",
            lambda msg: self._on_state("right", msg), qos,
        )

        self._latest: Dict[str, Dict] = {
            "left":  {"names": list(self.joint_names), "positions": [0.0] * len(self.joint_names)},
            "right": {"names": list(self.joint_names), "positions": [0.0] * len(self.joint_names)},
        }
        self._latest_lock = threading.Lock()

        self.get_logger().info(
            f"nero_webapp: publish /{left_ns}/control/joint_states "
            f"and /{right_ns}/control/joint_states; "
            f"subscribe /{left_ns}/feedback/joint_states and /{right_ns}/feedback/joint_states"
        )

    # ---- feedback cache ------------------------------------------------
    def _on_state(self, side: str, msg: JointState) -> None:
        # Only keep the 7 arm joints, in the canonical order. The upstream
        # driver may also publish gripper width joints; filter them out.
        names_in = list(msg.name)
        positions_in = list(msg.position)
        by_name = dict(zip(names_in, positions_in))
        filtered_names: List[str] = []
        filtered_positions: List[float] = []
        for jn in self.joint_names:
            if jn in by_name:
                filtered_names.append(jn)
                filtered_positions.append(float(by_name[jn]))
        if not filtered_names:
            # Driver published under unexpected names — fall back to raw.
            filtered_names = names_in
            filtered_positions = positions_in
        with self._latest_lock:
            self._latest[side] = {
                "names": filtered_names,
                "positions": filtered_positions,
            }

    def snapshot(self) -> Dict[str, Dict]:
        with self._latest_lock:
            return {k: dict(v) for k, v in self._latest.items()}

    # ---- command path --------------------------------------------------
    def command(self, side: str, positions: List[float]) -> None:
        if len(positions) != len(self.joint_names):
            self.get_logger().warn(
                f"ignoring {side} command: expected {len(self.joint_names)} "
                f"positions, got {len(positions)}"
            )
            return

        clamped = [
            max(lo, min(hi, float(p)))
            for p, (lo, hi) in zip(positions, JOINT_LIMITS)
        ]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = clamped

        if side == "left":
            self.pub_left.publish(msg)
        elif side == "right":
            self.pub_right.publish(msg)
        else:
            self.get_logger().warn(f"unknown side: {side}")


# ---- FastAPI app -----------------------------------------------------
def build_app(node: WebappNode, static_dir: Path) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text())

    @app.get("/joint_limits")
    async def joint_limits() -> Dict:
        return {"names": node.joint_names, "limits": JOINT_LIMITS}

    clients: Set[WebSocket] = set()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        clients.add(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                side = data.get("side")
                positions = data.get("positions")
                if side in ("left", "right") and isinstance(positions, list):
                    node.command(side, positions)
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    async def broadcaster() -> None:
        while True:
            await asyncio.sleep(0.05)
            if not clients:
                continue
            payload = json.dumps({"type": "state", "data": node.snapshot()})
            dead: List[WebSocket] = []
            for ws in list(clients):
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

    @app.on_event("startup")
    async def _start_broadcaster() -> None:
        asyncio.create_task(broadcaster())

    return app


def main() -> None:
    rclpy.init()
    node = WebappNode()

    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    static_dir = Path(get_package_share_directory("nero_webapp")) / "static"
    host = node.get_parameter("http_host").get_parameter_value().string_value
    port = node.get_parameter("http_port").get_parameter_value().integer_value

    app = build_app(node, static_dir)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
