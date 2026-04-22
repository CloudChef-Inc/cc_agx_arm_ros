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
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect

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
        # Per-side camera device paths (the Pika fisheye is a UVC camera,
        # the D405 is addressed by serial). Empty string disables that
        # side's camera cleanly so single-arm testing works.
        self.declare_parameter("left_fisheye_device",  "")
        self.declare_parameter("right_fisheye_device", "")
        # Shared fisheye config — both sides use identical resolution/
        # fps/exposure/crop for a consistent UI.
        self.declare_parameter("fisheye_width",  640)
        self.declare_parameter("fisheye_height", 480)
        self.declare_parameter("fisheye_fps",    15)
        self.declare_parameter("fisheye_circle_crop", True)
        self.declare_parameter("fisheye_exposure", 200)
        # Per-side RealSense D405 serials. If empty, fall back to
        # /etc/nero/d405_sides.conf (maintained by nero-detect-pika) —
        # the detector already knows the side↔serial mapping, so the
        # webapp picks it up without duplicating config.
        self.declare_parameter("left_realsense_serial",  "")
        self.declare_parameter("right_realsense_serial", "")
        # 848×480 @ 15fps for RealSense — keeps total sw-encode
        # load manageable with 6 streams on aiortc/libx264.
        self.declare_parameter("realsense_color_w", 848)
        self.declare_parameter("realsense_color_h", 480)
        self.declare_parameter("realsense_depth_w", 848)
        self.declare_parameter("realsense_depth_h", 480)
        self.declare_parameter("realsense_fps",     15)
        # Torso dimensions — used by the 3D rendering in the browser.
        # Set from the launch file (same source as the xacro).
        self.declare_parameter("torso_width",  0.185)
        self.declare_parameter("torso_depth",  0.10)
        self.declare_parameter("torso_height", 0.60)

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

        # Quest leader ghost + status (for UI preview rendering).
        self._quest_ghost: Dict[str, Dict] = {"left": {}, "right": {}}
        self._quest_status: Dict[str, str] = {"left": "IDLE", "right": "IDLE"}
        for _side, _ns in (("left", left_ns), ("right", right_ns)):
            self.create_subscription(
                JointState, f"/{_ns}/quest_leader/target_joint_states",
                lambda msg, s=_side: self._on_quest_ghost(s, msg), qos,
            )
        from std_msgs.msg import String as _String
        for _side, _ns in (("left", left_ns), ("right", right_ns)):
            self.create_subscription(
                _String, f"/{_ns}/quest_leader/status",
                lambda msg, s=_side: self._on_quest_status(s, msg), qos,
            )

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

        # Camera workers — per-side, started explicitly in start_cameras().
        self.fisheye: Dict[str, "FisheyeCamera | None"] = {"left": None, "right": None}
        self.realsense: Dict[str, "RealSenseCamera | None"] = {"left": None, "right": None}

        self.get_logger().info(
            f"nero_webapp: publish /{left_ns}/control/joint_states "
            f"and /{right_ns}/control/joint_states; "
            f"subscribe /{left_ns}/feedback/joint_states and /{right_ns}/feedback/joint_states"
        )

    # ---- cameras (per-side V4L2 fisheye + RealSense) ------------------
    def _read_d405_sides_conf(self) -> Dict[str, List[str]]:
        """Read /etc/nero/d405_sides.conf if present. Returns {side: [serials]}.

        Parsing is forgiving — comments (#) and blank lines are skipped,
        same side can appear multiple times (useful when the D405
        firmware flips between device and ASIC serials — both can be
        listed and we try each until one opens).
        """
        result: Dict[str, List[str]] = {}
        try:
            with open("/etc/nero/d405_sides.conf") as f:
                for line in f:
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    side, serial = parts
                    result.setdefault(side, []).append(serial)
        except OSError:
            pass
        return result

    def start_cameras(self) -> None:
        width  = self.get_parameter("fisheye_width").get_parameter_value().integer_value
        height = self.get_parameter("fisheye_height").get_parameter_value().integer_value
        fps    = self.get_parameter("fisheye_fps").get_parameter_value().integer_value
        crop   = self.get_parameter("fisheye_circle_crop").get_parameter_value().bool_value
        expo   = self.get_parameter("fisheye_exposure").get_parameter_value().integer_value
        rs_cw  = self.get_parameter("realsense_color_w").get_parameter_value().integer_value
        rs_ch  = self.get_parameter("realsense_color_h").get_parameter_value().integer_value
        rs_dw  = self.get_parameter("realsense_depth_w").get_parameter_value().integer_value
        rs_dh  = self.get_parameter("realsense_depth_h").get_parameter_value().integer_value
        rs_fps = self.get_parameter("realsense_fps").get_parameter_value().integer_value

        d405_conf = self._read_d405_sides_conf()

        for side in ("left", "right"):
            dev = self.get_parameter(f"{side}_fisheye_device").get_parameter_value().string_value
            if dev:
                cam = FisheyeCamera(
                    device_path=dev, width=width, height=height, fps=fps,
                    auto_circle_crop=crop, exposure_manual_value=expo,
                    overlay_label=f"fisheye_{side}",
                )
                if cam.start():
                    self.fisheye[side] = cam
                else:
                    self.get_logger().error(f"{side} fisheye: failed to open {dev}")
                    cam.stop()

            # Figure out which D405 serial(s) to try for this side:
            # explicit ROS param wins, otherwise fall back to the
            # detector's config file.
            param_serial = self.get_parameter(
                f"{side}_realsense_serial").get_parameter_value().string_value
            if param_serial:
                candidates = [param_serial]
            else:
                candidates = d405_conf.get(side, [])
            for serial in candidates:
                rs_cam = RealSenseCamera(
                    serial=serial,
                    color_w=rs_cw, color_h=rs_ch,
                    depth_w=rs_dw, depth_h=rs_dh,
                    fps=rs_fps,
                    color_label=f"color_{side}",
                    depth_label=f"depth_{side}",
                )
                if rs_cam.start():
                    self.realsense[side] = rs_cam
                    break
                rs_cam.stop()
            else:
                if candidates:
                    self.get_logger().error(
                        f"{side} RealSense: none of {candidates} opened")

    def stop_cameras(self) -> None:
        for side in ("left", "right"):
            if self.fisheye[side] is not None:
                self.fisheye[side].stop()
                self.fisheye[side] = None
            if self.realsense[side] is not None:
                self.realsense[side].stop()
                self.realsense[side] = None

    def camera_slots(self) -> list:
        """Ordered (name, LatestFrame) list for WebRTC track publishing.

        Fixed order — left arm first, right arm second, fisheye/color/
        depth within each side. The browser uses the server-returned
        `cameras` list to pair incoming tracks with the matching video
        element, so disabled sides simply drop out of this list without
        shifting the others.
        """
        slots = []
        for side in ("left", "right"):
            if self.fisheye[side] is not None:
                slots.append((f"fisheye_{side}", self.fisheye[side].frames))
            if self.realsense[side] is not None:
                slots.append((f"color_{side}", self.realsense[side].color))
                slots.append((f"depth_{side}", self.realsense[side].depth))
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

    def _on_quest_ghost(self, side: str, msg: JointState) -> None:
        with self._latest_lock:
            self._quest_ghost[side] = {
                "names": list(msg.name),
                "positions": [float(p) for p in msg.position],
            }

    def _on_quest_status(self, side: str, msg) -> None:
        with self._latest_lock:
            self._quest_status[side] = str(msg.data)

    def snapshot(self) -> Dict[str, Dict]:
        with self._latest_lock:
            snap = {k: dict(v) for k, v in self._latest.items()}
            for side in ("left", "right"):
                snap[side]["quest_ghost"] = dict(self._quest_ghost.get(side) or {})
                snap[side]["quest_status"] = self._quest_status.get(side, "IDLE")
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

    # Starlette's StaticFiles refuses to serve files whose realpath escapes
    # the mount directory. With `colcon build --symlink-install`, the files
    # in install/share/.../static/ are symlinks to src/.../static/, which
    # trips that check. Mount the resolved parent so the realpath lands
    # inside the mount root in both symlink-install and plain builds.
    resolved_static = static_dir
    probe = static_dir / "index.html"
    if probe.is_symlink() or probe.exists():
        resolved_static = probe.resolve().parent
    app.mount("/static", StaticFiles(directory=str(resolved_static)), name="static")

    # Expose the upstream agx_arm_description share dir so the browser can
    # fetch the Nero URDF + meshes directly (DAE visual meshes render the
    # real arm geometry client-side via urdf-loader).
    agx_share = Path(get_package_share_directory("agx_arm_description"))
    app.mount(
        "/pkg/agx_arm_description",
        StaticFiles(directory=str(agx_share)),
        name="agx_pkg",
    )

    # Serve Pika gripper meshes so the browser can load the STL files.
    try:
        pika_share = Path(get_package_share_directory("pika_gripper_description"))
        app.mount(
            "/pkg/pika_gripper_description",
            StaticFiles(directory=str(pika_share)),
            name="pika_pkg",
        )
    except Exception:  # noqa: BLE001
        pass  # pika_gripper_description not installed — gripper won't render

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

    @app.get("/torso_config")
    async def torso_config() -> Dict:
        return {
            "torso_width":  node.get_parameter("torso_width").get_parameter_value().double_value,
            "torso_depth":  node.get_parameter("torso_depth").get_parameter_value().double_value,
            "torso_height": node.get_parameter("torso_height").get_parameter_value().double_value,
        }

    # ---- gravity comp toggle (calls the gravity_comp node's param) ----
    # Lazy-create parameter clients per side on first use so the
    # webapp doesn't crash if the gravity_comp node isn't running.
    _param_clients: Dict[str, object] = {}

    def _get_param_client(side: str):
        if side not in _param_clients:
            from rcl_interfaces.srv import SetParameters
            name = f"/gravity_comp_{side}/set_parameters"
            _param_clients[side] = node.create_client(SetParameters, name)
        return _param_clients[side]

    @app.post("/gravity_comp")
    async def gravity_comp(request: Request) -> Dict:
        import asyncio
        from rcl_interfaces.msg import Parameter as ParamMsg
        from rcl_interfaces.msg import ParameterValue, ParameterType
        from rcl_interfaces.srv import SetParameters

        body = await request.json()
        side = body.get("side", "right")
        enabled = bool(body.get("enabled", False))

        client = _get_param_client(side)
        if not client.wait_for_service(timeout_sec=1.0):
            return {"ok": False, "error": f"gravity_comp_{side} not running"}

        req = SetParameters.Request()
        p = ParamMsg()
        p.name = "enabled"
        p.value = ParameterValue()
        p.value.type = ParameterType.PARAMETER_BOOL
        p.value.bool_value = enabled
        req.parameters = [p]

        future = client.call_async(req)
        # Poll until done — can't use spin_until_future_complete
        # because the node is already spinning on a background thread.
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            await asyncio.sleep(0.05)
        if not future.done():
            return {"ok": False, "error": "parameter set timed out"}
        resp = future.result()
        if resp and resp.results and resp.results[0].successful:
            return {"ok": True, "side": side, "enabled": enabled}
        return {"ok": False, "error": "parameter set failed"}

    # ---- teach mode toggle (calls agx_arm_ctrl's teach_mode service) ----
    _teach_clients: Dict[str, object] = {}

    def _get_teach_client(side: str):
        if side not in _teach_clients:
            from std_srvs.srv import SetBool
            name = f"/{side}/teach_mode"
            _teach_clients[side] = node.create_client(SetBool, name)
        return _teach_clients[side]

    @app.post("/teach_mode")
    async def teach_mode(request: Request) -> Dict:
        import asyncio
        from std_srvs.srv import SetBool

        body = await request.json()
        side = body.get("side", "right")
        enabled = bool(body.get("enabled", False))

        client = _get_teach_client(side)
        if not client.wait_for_service(timeout_sec=1.0):
            return {"ok": False, "error": f"teach_mode service for {side} not running"}

        req = SetBool.Request()
        req.data = enabled

        future = client.call_async(req)
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            await asyncio.sleep(0.05)
        if not future.done():
            return {"ok": False, "error": "teach_mode service timed out"}
        resp = future.result()
        if resp and resp.success:
            return {"ok": True, "side": side, "enabled": enabled}
        return {"ok": False, "error": resp.message if resp else "service call failed"}

    # ---- Quest leader mode (calls quest_leader_node services) ----
    _quest_clients: Dict[str, Dict[str, object]] = {"left": {}, "right": {}}

    def _get_quest_client(side: str, kind: str):
        cache = _quest_clients.setdefault(side, {})
        if kind not in cache:
            if kind == "calibrate":
                from std_srvs.srv import Trigger
                cache[kind] = node.create_client(Trigger, f"/{side}/quest_leader/calibrate")
            else:
                from std_srvs.srv import SetBool
                cache[kind] = node.create_client(SetBool, f"/{side}/quest_leader/{kind}")
        return cache[kind]

    async def _call_quest(side: str, kind: str, req):
        import asyncio as _asyncio
        client = _get_quest_client(side, kind)
        if not client.wait_for_service(timeout_sec=1.0):
            return {"ok": False, "error": f"quest_leader/{kind} for {side} not running"}
        future = client.call_async(req)
        deadline = time.time() + 3.0
        while not future.done() and time.time() < deadline:
            await _asyncio.sleep(0.05)
        if not future.done():
            return {"ok": False, "error": f"{kind} timed out"}
        resp = future.result()
        return {"ok": bool(resp and resp.success),
                "message": getattr(resp, "message", ""),
                "side": side}

    @app.post("/quest_leader/calibrate")
    async def quest_calibrate(request: Request) -> Dict:
        from std_srvs.srv import Trigger
        body = await request.json()
        side = body.get("side", "right")
        return await _call_quest(side, "calibrate", Trigger.Request())

    @app.post("/quest_leader/preview")
    async def quest_preview(request: Request) -> Dict:
        from std_srvs.srv import SetBool
        body = await request.json()
        side = body.get("side", "right")
        req = SetBool.Request()
        req.data = bool(body.get("enabled", False))
        return await _call_quest(side, "preview", req)

    @app.post("/quest_leader/follow")
    async def quest_follow(request: Request) -> Dict:
        from std_srvs.srv import SetBool
        body = await request.json()
        side = body.get("side", "right")
        req = SetBool.Request()
        req.data = bool(body.get("enabled", False))
        return await _call_quest(side, "follow", req)

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
