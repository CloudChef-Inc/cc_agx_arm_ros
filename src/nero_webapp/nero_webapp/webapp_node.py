"""ROS 2 node that serves a browser UI to command two AgileX Nero arms.

Architecture
------------
- Main thread runs uvicorn (FastAPI HTTP + WebSocket).
- rclpy spins on a background daemon thread.
- Browser <-> node: one WebSocket at /ws
    Browser -> node : {"side": "left"|"right", "positions": [8 floats — 7 arm + gripper]}
    Node    -> browser (20 Hz):
                       {"type": "state", "data": {
                           "left":  {"names": [...], "positions": [...]},
                           "right": {"names": [...], "positions": [...]}
                       }}

Arm vs gripper split
--------------------
Arm joints (joint1..joint7) go out as sensor_msgs/JointState on
/<side>/control/joint_states, handled by agilexrobotics' agx_arm_ctrl.

The "gripper" slider drives an AgileX Pika — which is USB-serial
(/dev/ttyACM*), NOT on the arm's CAN bus. We drive it directly via
pika_sdk from this node. Per-side device paths come from parameters
`left_pika_serial` / `right_pika_serial`; empty string disables that
side's gripper cleanly (webapp still runs arm-only).

Server-side safety: commanded positions are clamped to the Nero URDF
joint limits from nero_description.urdf before being published.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Set

from ament_index_python.packages import get_package_share_directory
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

# pika_sdk is optional — absence just means gripper control is unavailable.
try:
    from pika.gripper import Gripper as PikaGripper  # type: ignore
except Exception:  # noqa: BLE001
    PikaGripper = None  # type: ignore
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
import uvicorn

from .cameras import FisheyeCamera, RealSenseCamera
from .webrtc import setup_webrtc_routes


# Default Nero joint names (7 revolute + gripper). Match nero_description.urdf.
DEFAULT_JOINT_NAMES: List[str] = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "gripper",
]

# Hard limits from nero_description.urdf (radians) + gripper (metres, 0–0.1 m).
JOINT_LIMITS: List[tuple[float, float]] = [
    (-2.70526,    2.70526),   # joint1
    (-1.74,       1.74),      # joint2
    (-2.75,       2.75),      # joint3
    (-1.01,       2.14),      # joint4
    (-2.75,       2.75),      # joint5
    (-0.73,       0.95),      # joint6
    (-1.5707963,  1.5707963), # joint7
    ( 0.0,        0.1),       # gripper width (metres)
]


class WebappNode(Node):
    def __init__(self) -> None:
        super().__init__("nero_webapp")

        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("left_ns",  "left")
        self.declare_parameter("right_ns", "right")
        self.declare_parameter("http_host", "0.0.0.0")
        self.declare_parameter("http_port", 8080)
        # Per-side Pika USB-serial device (e.g. /dev/ttyACM0). Empty = disabled.
        self.declare_parameter("left_pika_serial",  "")
        self.declare_parameter("right_pika_serial", "")
        # Cameras. Each can be disabled independently by emptying its device
        # parameter. The three feeds are: the Pika's fisheye webcam, and the
        # RealSense D405's color + colorized depth streams.
        self.declare_parameter("fisheye_device", "")
        self.declare_parameter("fisheye_width",  1280)
        self.declare_parameter("fisheye_height", 720)
        self.declare_parameter("fisheye_fps",    30)
        # Auto-detect the visible image circle and crop to it — removes
        # the black vignette, saves encode bandwidth, and makes the tile
        # fill more useful area with content pixels.
        self.declare_parameter("fisheye_circle_crop", True)
        self.declare_parameter("realsense_serial",  "")
        self.declare_parameter("realsense_enable",  True)
        self.declare_parameter("realsense_color_w", 1280)
        self.declare_parameter("realsense_color_h", 720)
        self.declare_parameter("realsense_depth_w", 1280)
        self.declare_parameter("realsense_depth_h", 720)
        self.declare_parameter("realsense_fps",     30)

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

        # ---- Pika grippers (USB-serial, per side, optional) ----------------
        self._pika: Dict[str, object] = {"left": None, "right": None}
        self._pika_lock = threading.Lock()
        self._init_pika(
            "left",
            self.get_parameter("left_pika_serial").get_parameter_value().string_value,
        )
        self._init_pika(
            "right",
            self.get_parameter("right_pika_serial").get_parameter_value().string_value,
        )

        # Camera workers — constructed here, started explicitly in start_cameras().
        self.fisheye: "FisheyeCamera | None" = None
        self.realsense: "RealSenseCamera | None" = None

        self.get_logger().info(
            f"nero_webapp: publish /{left_ns}/control/joint_states "
            f"and /{right_ns}/control/joint_states; "
            f"subscribe /{left_ns}/feedback/joint_states and /{right_ns}/feedback/joint_states"
        )

    # ---- cameras (V4L2 fisheye + RealSense) ----------------------------
    def start_cameras(self) -> None:
        fish_dev = self.get_parameter("fisheye_device").get_parameter_value().string_value
        if fish_dev:
            cam = FisheyeCamera(
                device_path=fish_dev,
                width=self.get_parameter("fisheye_width").get_parameter_value().integer_value,
                height=self.get_parameter("fisheye_height").get_parameter_value().integer_value,
                fps=self.get_parameter("fisheye_fps").get_parameter_value().integer_value,
                auto_circle_crop=self.get_parameter("fisheye_circle_crop").get_parameter_value().bool_value,
            )
            if cam.start():
                self.fisheye = cam
            else:
                self.get_logger().error(f"fisheye: failed to open {fish_dev}; disabling")
                cam.stop()

        if self.get_parameter("realsense_enable").get_parameter_value().bool_value:
            rs_cam = RealSenseCamera(
                serial=self.get_parameter("realsense_serial").get_parameter_value().string_value,
                color_w=self.get_parameter("realsense_color_w").get_parameter_value().integer_value,
                color_h=self.get_parameter("realsense_color_h").get_parameter_value().integer_value,
                depth_w=self.get_parameter("realsense_depth_w").get_parameter_value().integer_value,
                depth_h=self.get_parameter("realsense_depth_h").get_parameter_value().integer_value,
                fps=self.get_parameter("realsense_fps").get_parameter_value().integer_value,
            )
            if rs_cam.start():
                self.realsense = rs_cam
            else:
                self.get_logger().error("realsense: pipeline failed to start; disabling")
                rs_cam.stop()

    def stop_cameras(self) -> None:
        if self.fisheye is not None:
            self.fisheye.stop()
            self.fisheye = None
        if self.realsense is not None:
            self.realsense.stop()
            self.realsense = None

    def camera_slots(self) -> list:
        """Ordered (name, LatestFrame) list for WebRTC track publishing.

        Order matters: browser pairs it with pc.getTransceivers() order.
        """
        slots = []
        if self.fisheye is not None:
            slots.append(("fisheye", self.fisheye.frames))
        if self.realsense is not None:
            slots.append(("color", self.realsense.color))
            slots.append(("depth", self.realsense.depth))
        return slots

    # ---- Pika gripper (USB-serial) -------------------------------------
    def _init_pika(self, side: str, serial_path: str) -> None:
        if not serial_path:
            return
        if PikaGripper is None:
            self.get_logger().warn(
                f"{side} Pika serial set to {serial_path} but pika_sdk isn't installed; "
                "install with `pip install git+https://github.com/agilexrobotics/pika_sdk.git` "
                "to enable gripper control"
            )
            return
        try:
            g = PikaGripper(serial_path)
            if not g.connect():
                self.get_logger().error(f"{side} Pika: failed to connect on {serial_path}")
                return
            if not g.enable():
                self.get_logger().error(f"{side} Pika: connected but enable() failed")
                return
            self._pika[side] = g
            self.get_logger().info(f"{side} Pika connected on {serial_path}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"{side} Pika init error: {e}")

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
            snap = {k: dict(v) for k, v in self._latest.items()}
        # Overlay live Pika gripper width onto each side if connected.
        for side in ("left", "right"):
            g = self._pika.get(side)
            if g is None:
                continue
            try:
                with self._pika_lock:
                    dist_mm = g.get_gripper_distance()
                width_m = float(dist_mm) / 1000.0
            except Exception:
                continue
            names = list(snap[side].get("names", []))
            positions = list(snap[side].get("positions", []))
            if "gripper" in names:
                positions[names.index("gripper")] = width_m
            else:
                names.append("gripper")
                positions.append(width_m)
            snap[side] = {"names": names, "positions": positions}
        return snap

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

        # Arm joints (everything except "gripper") go out over ROS; the
        # Pika gripper is USB-serial and handled separately below.
        arm_names: List[str] = []
        arm_positions: List[float] = []
        gripper_width: float | None = None
        for name, value in zip(self.joint_names, clamped):
            if name == "gripper":
                gripper_width = value
            else:
                arm_names.append(name)
                arm_positions.append(value)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = arm_names
        msg.position = arm_positions

        if side == "left":
            self.pub_left.publish(msg)
        elif side == "right":
            self.pub_right.publish(msg)
        else:
            self.get_logger().warn(f"unknown side: {side}")
            return

        # Pika takes millimeters, range 0–100 mm. URDF limit is 0–0.1 m.
        if gripper_width is not None:
            g = self._pika.get(side)
            if g is not None:
                dist_mm = max(0.0, min(100.0, gripper_width * 1000.0))
                try:
                    with self._pika_lock:
                        g.set_gripper_distance(dist_mm)
                except Exception as e:  # noqa: BLE001
                    self.get_logger().warn(f"{side} Pika set_gripper_distance failed: {e}")


# ---- FastAPI app -----------------------------------------------------
def build_app(node: WebappNode, static_dir: Path) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Expose the upstream agx_arm_description share dir so the browser can
    # fetch the Nero URDF + meshes directly (DAE visual meshes render the
    # real arm geometry client-side via urdf-loader).
    agx_share = Path(get_package_share_directory("agx_arm_description"))
    app.mount(
        "/pkg/agx_arm_description",
        StaticFiles(directory=str(agx_share)),
        name="agx_pkg",
    )

    # Hand-merged flat URDF (arm + Pika gripper), shipped in our static dir.
    # Loaded once at startup so the GET /nero_urdf path is a cheap memory read.
    nero_urdf_xml = (static_dir / "nero_with_gripper.urdf").read_text()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text())

    @app.get("/nero_urdf")
    async def nero_urdf() -> Response:
        return Response(nero_urdf_xml, media_type="application/xml")

    @app.get("/joint_limits")
    async def joint_limits() -> Dict:
        return {"names": node.joint_names, "limits": JOINT_LIMITS}

    @app.get("/stats")
    async def stats() -> Dict:
        """Per-camera capture stats (fps, last-frame shape).

        Client-side fetches this once a second and pairs it with
        RTCPeerConnection.getStats() for a full end-to-end picture.
        """
        out: Dict[str, Dict] = {}
        for name, slot in node.camera_slots():
            out[name] = slot.info()
        return {"server_time": time.time(), "cameras": out}

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

    # WebRTC /offer endpoint for live camera streams. Only mounted when
    # at least one camera was started successfully.
    slots = node.camera_slots()
    if slots:
        setup_webrtc_routes(app, slots)

    return app


def main() -> None:
    rclpy.init()
    node = WebappNode()

    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # Start camera threads BEFORE build_app so camera_slots() reflects
    # which cameras actually opened successfully.
    node.start_cameras()

    static_dir = Path(get_package_share_directory("nero_webapp")) / "static"
    host = node.get_parameter("http_host").get_parameter_value().string_value
    port = node.get_parameter("http_port").get_parameter_value().integer_value

    app = build_app(node, static_dir)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        node.stop_cameras()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
