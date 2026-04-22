#!/usr/bin/env python3
"""quest_leader_node — Quest controller → IK → joint targets for one arm.

State machine per side: IDLE → CALIBRATING → READY → ACTIVE.

Calibration: drive the arm to CALIBRATION_JOINT_POSE_{LEFT,RIGHT}, wait
for convergence, run a 5-second countdown (so the operator can match the
controller to the pose), then snapshot `ctrl_ref` (Quest pose) and
`ee_ref` (feedback/tcp_pose). Transition to READY.

Preview (SetBool true): start the 30 Hz loop — compute target EE from
delta, call /compute_ik, publish to `quest_leader/target_joint_states`.
Webapp renders this as a ghost.

Send (SetBool true): additionally mirror the IK solution to
`control/joint_states`. Requires Preview already on. Auto-disarms if the
Quest stream stalls > 0.5 s.

Delta math:
    dp = ctrl_pos - ctrl_ref_pos              (world-frame translation)
    target_pos = ee_ref_pos + dp
    dR = ctrl_ref_R^-1 · ctrl_R               (rotation delta in ctrl frame)
    target_R = ee_ref_R · dR                  (apply in EE ref frame)

Per-frame delta clamp: 5 cm, 15° — bounds the command if the stream hiccups.

Launch one per side (namespace=left, namespace=right).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool

try:
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.msg import PositionIKRequest, RobotState
except ImportError:
    GetPositionIK = None  # handled at runtime

from scipy.spatial.transform import Rotation as R


# Placeholder calibration poses. Tune on the bench.
# TODO: Atish to tune.
CALIBRATION_JOINT_POSE_LEFT: List[float] = [0.0, 0.5, 0.0, -1.2, 0.0, 1.0, 0.0]
CALIBRATION_JOINT_POSE_RIGHT: List[float] = [0.0, 0.5, 0.0, -1.2, 0.0, 1.0, 0.0]

CONVERGE_TOL = 0.02       # rad, per joint
CONVERGE_TIMEOUT = 5.0    # s
COUNTDOWN_S = 5
IK_RATE_HZ = 30.0
MAX_DP = 0.05             # m per frame
MAX_DR_DEG = 15.0         # deg per frame
STREAM_STALL_S = 0.5      # auto-disarm threshold

STATE_IDLE = "IDLE"
STATE_CALIBRATING = "CALIBRATING"
STATE_READY = "READY"
STATE_ACTIVE = "ACTIVE"


@dataclass
class Pose6:
    pos: np.ndarray           # (3,)
    rot: R

    @staticmethod
    def from_msg(p: PoseStamped) -> "Pose6":
        q = p.pose.orientation
        return Pose6(
            pos=np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z]),
            rot=R.from_quat([q.x, q.y, q.z, q.w]),
        )


def _clamp_delta(dp: np.ndarray, dr: R) -> tuple[np.ndarray, R]:
    n = float(np.linalg.norm(dp))
    if n > MAX_DP:
        dp = dp * (MAX_DP / n)
    rv = dr.as_rotvec()
    ang = float(np.linalg.norm(rv))
    max_ang = math.radians(MAX_DR_DEG)
    if ang > max_ang:
        rv = rv * (max_ang / ang)
        dr = R.from_rotvec(rv)
    return dp, dr


class QuestLeaderNode(Node):
    def __init__(self) -> None:
        super().__init__("quest_leader_node")

        self.declare_parameter("side", "right")  # "left" or "right"
        self.declare_parameter("ee_link", "gripper_flange")
        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("ik_service", "/compute_ik")

        self.side: str = self.get_parameter("side").value
        self.ee_link: str = self.get_parameter("ee_link").value
        self.group: str = self.get_parameter("planning_group").value
        ik_srv: str = self.get_parameter("ik_service").value

        if self.side not in ("left", "right"):
            raise ValueError(f"side must be left|right, got {self.side}")

        self._state = STATE_IDLE
        self._lock = threading.Lock()
        self._cb_group = ReentrantCallbackGroup()

        # Latest inputs.
        self._latest_ctrl: Optional[Pose6] = None
        self._latest_ctrl_ts: float = 0.0
        self._latest_tcp: Optional[Pose6] = None
        self._latest_joint_state: Optional[JointState] = None

        # References captured at calibration.
        self._ctrl_ref: Optional[Pose6] = None
        self._ee_ref: Optional[Pose6] = None
        self._last_ik_solution: Optional[List[float]] = None
        self._last_ik_fail_log = 0.0

        self._preview_on = False
        self._send_on = False
        self._countdown_remaining = 0

        # Topics.
        ns = f"/{self.side}"
        self.create_subscription(
            PoseStamped, f"/quest/{self.side}_controller",
            self._ctrl_cb, 10, callback_group=self._cb_group,
        )
        self.create_subscription(
            PoseStamped, f"{ns}/feedback/tcp_pose",
            self._tcp_cb, 10, callback_group=self._cb_group,
        )
        self.create_subscription(
            JointState, f"{ns}/feedback/joint_states",
            self._js_cb, 10, callback_group=self._cb_group,
        )

        self._ghost_pub = self.create_publisher(
            JointState, f"{ns}/quest_leader/target_joint_states", 10,
        )
        self._cmd_pub = self.create_publisher(
            JointState, f"{ns}/control/joint_states", 10,
        )
        self._status_pub = self.create_publisher(
            String, f"{ns}/quest_leader/status", 10,
        )

        # Services.
        self.create_service(
            Trigger, f"{ns}/quest_leader/calibrate",
            self._calibrate_srv, callback_group=self._cb_group,
        )
        self.create_service(
            SetBool, f"{ns}/quest_leader/preview",
            self._preview_srv, callback_group=self._cb_group,
        )
        self.create_service(
            SetBool, f"{ns}/quest_leader/send",
            self._send_srv, callback_group=self._cb_group,
        )

        # IK client.
        if GetPositionIK is None:
            self.get_logger().error("moveit_msgs not available — IK disabled")
            self._ik_client = None
        else:
            self._ik_client = self.create_client(
                GetPositionIK, ik_srv, callback_group=self._cb_group,
            )

        # Periodic loops.
        self.create_timer(1.0 / IK_RATE_HZ, self._ik_tick, callback_group=self._cb_group)
        self.create_timer(0.2, self._publish_status, callback_group=self._cb_group)

        self.get_logger().info(f"quest_leader_node up (side={self.side}, ee={self.ee_link})")

    # ---------- subscriptions ----------
    def _ctrl_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_ctrl = Pose6.from_msg(msg)
            self._latest_ctrl_ts = time.time()

    def _tcp_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_tcp = Pose6.from_msg(msg)

    def _js_cb(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = msg

    # ---------- services ----------
    def _calibrate_srv(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().info(f"[{self.side}] calibration requested")
        self._preview_on = False
        self._send_on = False
        self._state = STATE_CALIBRATING

        # Drive to calibration pose.
        target_q = (CALIBRATION_JOINT_POSE_LEFT if self.side == "left"
                    else CALIBRATION_JOINT_POSE_RIGHT)
        if not self._latest_joint_state:
            self._state = STATE_IDLE
            response.success = False
            response.message = "no feedback/joint_states yet"
            return response

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self._latest_joint_state.name)
        js.position = list(target_q)[: len(js.name)]
        self._cmd_pub.publish(js)

        # Wait for convergence.
        t0 = time.time()
        while time.time() - t0 < CONVERGE_TIMEOUT:
            with self._lock:
                cur = list(self._latest_joint_state.position) if self._latest_joint_state else []
            if cur and len(cur) >= len(target_q):
                if max(abs(a - b) for a, b in zip(cur[: len(target_q)], target_q)) < CONVERGE_TOL:
                    break
            time.sleep(0.05)

        # 5-second countdown.
        for n in range(COUNTDOWN_S, 0, -1):
            self._countdown_remaining = n
            time.sleep(1.0)
        self._countdown_remaining = 0

        # Snapshot references.
        with self._lock:
            ctrl = self._latest_ctrl
            tcp = self._latest_tcp
            ctrl_fresh = (time.time() - self._latest_ctrl_ts) < 0.5

        if not ctrl or not ctrl_fresh or not tcp:
            self._state = STATE_IDLE
            response.success = False
            response.message = (
                f"missing refs at snapshot (ctrl={bool(ctrl)}, "
                f"fresh={ctrl_fresh}, tcp={bool(tcp)})"
            )
            return response

        self._ctrl_ref = ctrl
        self._ee_ref = tcp
        self._last_ik_solution = None
        self._state = STATE_READY
        self.get_logger().info(f"[{self.side}] calibrated; state=READY")
        response.success = True
        response.message = "calibrated"
        return response

    def _preview_srv(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            if self._state not in (STATE_READY, STATE_ACTIVE):
                response.success = False
                response.message = f"not calibrated (state={self._state})"
                return response
            self._preview_on = True
            if self._state == STATE_READY:
                self._state = STATE_ACTIVE
        else:
            self._preview_on = False
            self._send_on = False
            if self._state == STATE_ACTIVE:
                self._state = STATE_READY
        response.success = True
        response.message = f"preview={'on' if self._preview_on else 'off'}"
        return response

    def _send_srv(self, request: SetBool.Request, response: SetBool.Response):
        if request.data and not self._preview_on:
            response.success = False
            response.message = "enable preview first"
            return response
        self._send_on = bool(request.data)
        response.success = True
        response.message = f"send={'on' if self._send_on else 'off'}"
        return response

    # ---------- main loop ----------
    def _ik_tick(self) -> None:
        if not self._preview_on or self._state != STATE_ACTIVE:
            return
        if self._ctrl_ref is None or self._ee_ref is None:
            return

        with self._lock:
            ctrl = self._latest_ctrl
            ctrl_age = time.time() - self._latest_ctrl_ts
            js = self._latest_joint_state

        if ctrl is None:
            return

        # Auto-disarm Send on stream stall.
        if self._send_on and ctrl_age > STREAM_STALL_S:
            self.get_logger().warn(
                f"[{self.side}] Quest stream stalled ({ctrl_age:.2f}s) — disarming Send"
            )
            self._send_on = False
            return

        # Delta.
        dp = ctrl.pos - self._ctrl_ref.pos
        dr = self._ctrl_ref.rot.inv() * ctrl.rot
        dp, dr = _clamp_delta(dp, dr)

        target_pos = self._ee_ref.pos + dp
        target_rot = self._ee_ref.rot * dr

        # IK.
        if self._ik_client is None or not self._ik_client.service_is_ready():
            return
        if js is None:
            return

        req = GetPositionIK.Request()
        ikr = PositionIKRequest()
        ikr.group_name = self.group
        ikr.ik_link_name = self.ee_link
        ikr.avoid_collisions = False
        ikr.timeout.sec = 0
        ikr.timeout.nanosec = 20_000_000  # 20 ms

        seed = RobotState()
        seed.joint_state.name = list(js.name)
        if self._last_ik_solution and len(self._last_ik_solution) == len(js.name):
            seed.joint_state.position = list(self._last_ik_solution)
        else:
            seed.joint_state.position = list(js.position)
        ikr.robot_state = seed

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "base_link"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = target_pos
        qx, qy, qz, qw = target_rot.as_quat()
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        ikr.pose_stamped = ps
        req.ik_request = ikr

        future = self._ik_client.call_async(req)
        future.add_done_callback(lambda f: self._on_ik_done(f, js.name))

    def _on_ik_done(self, future, joint_names: List[str]) -> None:
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().warn(f"[{self.side}] IK exception: {e}")
            return
        if resp is None or resp.error_code.val != 1:  # SUCCESS==1
            now = time.time()
            if now - self._last_ik_fail_log > 1.0:
                code = resp.error_code.val if resp else "?"
                self.get_logger().warn(f"[{self.side}] IK failed (code={code})")
                self._last_ik_fail_log = now
            return

        sol_names = list(resp.solution.joint_state.name)
        sol_pos = list(resp.solution.joint_state.position)
        # Reorder into joint_names order.
        idx = {n: i for i, n in enumerate(sol_names)}
        try:
            ordered = [sol_pos[idx[n]] for n in joint_names if n in idx]
        except KeyError:
            ordered = sol_pos
        if len(ordered) < len(joint_names):
            return

        self._last_ik_solution = ordered

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(joint_names)
        out.position = ordered[: len(joint_names)]
        self._ghost_pub.publish(out)
        if self._send_on:
            self._cmd_pub.publish(out)

    def _publish_status(self) -> None:
        parts = [f"state={self._state}", f"preview={self._preview_on}",
                 f"send={self._send_on}"]
        if self._state == STATE_CALIBRATING and self._countdown_remaining > 0:
            parts.append(f"countdown={self._countdown_remaining}")
        self._status_pub.publish(String(data="; ".join(parts)))


def main(args=None) -> None:
    rclpy.init(args=args)
    from rclpy.executors import MultiThreadedExecutor
    node = QuestLeaderNode()
    exec_ = MultiThreadedExecutor(num_threads=3)
    exec_.add_node(node)
    try:
        exec_.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
