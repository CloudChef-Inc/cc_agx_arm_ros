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
    from moveit_msgs.srv import GetPositionIK, GetPositionFK
    from moveit_msgs.msg import PositionIKRequest, RobotState
except ImportError:
    GetPositionIK = None  # handled at runtime
    GetPositionFK = None

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
        # Blank → derive from side as "{side}_gripper_flange"/"{side}_".
        self.declare_parameter("ee_link", "")
        self.declare_parameter("joint_prefix", "")
        self.declare_parameter("planning_group", "arm")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("fk_service", "/compute_fk")

        self.side: str = self.get_parameter("side").value
        # agx_arm_moveit is a single-arm config (unprefixed joints,
        # tip=tcp_link). Both quest_leader_{left,right} share one
        # move_group instance; kinematics are symmetric.
        self.ee_link: str = self.get_parameter("ee_link").value or "tcp_link"
        self.joint_prefix: str = self.get_parameter("joint_prefix").value
        self.group: str = self.get_parameter("planning_group").value
        self.base_frame: str = self.get_parameter("base_frame").value
        ik_srv: str = self.get_parameter("ik_service").value
        fk_srv: str = self.get_parameter("fk_service").value

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
        self._follow_on = False
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
            SetBool, f"{ns}/quest_leader/follow",
            self._follow_srv, callback_group=self._cb_group,
        )

        # IK / FK clients.
        if GetPositionIK is None:
            self.get_logger().error("moveit_msgs not available — IK disabled")
            self._ik_client = None
            self._fk_client = None
        else:
            self._ik_client = self.create_client(
                GetPositionIK, ik_srv, callback_group=self._cb_group,
            )
            self._fk_client = self.create_client(
                GetPositionFK, fk_srv, callback_group=self._cb_group,
            )

        # Periodic loops.
        self.create_timer(1.0 / IK_RATE_HZ, self._ik_tick, callback_group=self._cb_group)
        self.create_timer(0.2, self._publish_status, callback_group=self._cb_group)

        self.get_logger().info(
            f"quest_leader_node up (side={self.side}, ee={self.ee_link}, "
            f"joint_prefix={self.joint_prefix!r}, base_frame={self.base_frame})"
        )

    # Prefixed names used when talking to MoveIt (dual-arm URDF).
    def _prefixed(self, names: List[str]) -> List[str]:
        return [
            n if n.startswith(self.joint_prefix) else f"{self.joint_prefix}{n}"
            for n in names
        ]

    def _unprefixed(self, names: List[str]) -> List[str]:
        p = self.joint_prefix
        return [n[len(p):] if n.startswith(p) else n for n in names]

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
        """Sim-only calibration. The real arm does NOT move.

        We publish the calibration pose to the ghost topic so the UI
        shows where to hold the controller, compute ee_ref via /compute_fk
        from that joint pose (instead of feedback/tcp_pose, which would
        reflect the real arm's current — wrong — pose), run the 5 s
        countdown, and snapshot the controller pose as ctrl_ref.
        """
        self.get_logger().info(f"[{self.side}] calibration requested (sim-only)")
        self._preview_on = False
        self._follow_on = False
        self._state = STATE_CALIBRATING

        target_q = (CALIBRATION_JOINT_POSE_LEFT if self.side == "left"
                    else CALIBRATION_JOINT_POSE_RIGHT)
        if not self._latest_joint_state:
            self._state = STATE_IDLE
            response.success = False
            response.message = "no feedback/joint_states yet"
            return response
        joint_names = list(self._latest_joint_state.name)

        # Publish the calibration pose as the ghost (UI preview target) —
        # unprefixed names, since the webapp + arm driver speak unprefixed.
        ghost = JointState()
        ghost.header.stamp = self.get_clock().now().to_msg()
        ghost.name = joint_names
        ghost.position = list(target_q)[: len(joint_names)]
        self._ghost_pub.publish(ghost)

        # Derive ee_ref via FK on the calibration joint pose — MoveIt
        # expects prefixed names (left_joint1.., right_joint1..).
        ee_ref = self._fk_from_joints(self._prefixed(joint_names), ghost.position)
        if ee_ref is None:
            self._state = STATE_IDLE
            response.success = False
            response.message = "FK failed for calibration pose"
            return response

        # 5-second countdown.
        for n in range(COUNTDOWN_S, 0, -1):
            self._countdown_remaining = n
            time.sleep(1.0)
        self._countdown_remaining = 0

        with self._lock:
            ctrl = self._latest_ctrl
            ctrl_fresh = (time.time() - self._latest_ctrl_ts) < 0.5

        if not ctrl or not ctrl_fresh:
            self._state = STATE_IDLE
            response.success = False
            response.message = f"missing ctrl ref at snapshot (fresh={ctrl_fresh})"
            return response

        self._ctrl_ref = ctrl
        self._ee_ref = ee_ref
        self._last_ik_solution = list(ghost.position)
        self._state = STATE_READY
        self.get_logger().info(f"[{self.side}] calibrated (sim); state=READY")
        response.success = True
        response.message = "calibrated"
        return response

    def _fk_from_joints(self, names: List[str], positions: List[float]) -> Optional[Pose6]:
        if self._fk_client is None:
            self.get_logger().error(f"[{self.side}] FK client not constructed")
            return None
        if not self._fk_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(f"[{self.side}] /compute_fk service not available")
            return None
        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.fk_link_names = [self.ee_link]
        req.robot_state.joint_state.name = list(names)
        req.robot_state.joint_state.position = list(positions)
        self.get_logger().info(
            f"[{self.side}] FK request: link={self.ee_link} "
            f"names={list(names)} positions={list(positions)}"
        )

        future = self._fk_client.call_async(req)
        deadline = time.time() + 2.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().error(f"[{self.side}] FK call timed out")
            return None
        resp = future.result()
        if resp is None:
            self.get_logger().error(f"[{self.side}] FK response was None")
            return None
        if resp.error_code.val != 1:
            self.get_logger().error(
                f"[{self.side}] FK error_code={resp.error_code.val} "
                f"(1=SUCCESS; see moveit_msgs/MoveItErrorCodes)"
            )
            return None
        if not resp.pose_stamped:
            self.get_logger().error(f"[{self.side}] FK returned no pose_stamped")
            return None
        ps = resp.pose_stamped[0]
        q = ps.pose.orientation
        return Pose6(
            pos=np.array([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z]),
            rot=R.from_quat([q.x, q.y, q.z, q.w]),
        )

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
            self._follow_on = False
            if self._state == STATE_ACTIVE:
                self._state = STATE_READY
        response.success = True
        response.message = f"preview={'on' if self._preview_on else 'off'}"
        return response

    def _follow_srv(self, request: SetBool.Request, response: SetBool.Response):
        if request.data and not self._preview_on:
            response.success = False
            response.message = "enable preview first"
            return response
        self._follow_on = bool(request.data)
        response.success = True
        response.message = f"follow={'on' if self._follow_on else 'off'}"
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

        # Auto-disarm Follow on stream stall.
        if self._follow_on and ctrl_age > STREAM_STALL_S:
            self.get_logger().warn(
                f"[{self.side}] Quest stream stalled ({ctrl_age:.2f}s) — disarming Follow"
            )
            self._follow_on = False
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

        # Seed with prefixed names so MoveIt recognizes the joints.
        seed = RobotState()
        unprefixed_js_names = list(js.name)
        seed.joint_state.name = self._prefixed(unprefixed_js_names)
        if self._last_ik_solution and len(self._last_ik_solution) == len(unprefixed_js_names):
            seed.joint_state.position = list(self._last_ik_solution)
        else:
            seed.joint_state.position = list(js.position)
        ikr.robot_state = seed

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = target_pos
        qx, qy, qz, qw = target_rot.as_quat()
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        ikr.pose_stamped = ps
        req.ik_request = ikr

        future = self._ik_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_ik_done(f, unprefixed_js_names)
        )

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

        # MoveIt returns prefixed names; reorder into the unprefixed
        # (arm-driver / webapp) order.
        sol_names = list(resp.solution.joint_state.name)
        sol_pos = list(resp.solution.joint_state.position)
        prefixed_target = self._prefixed(joint_names)
        idx = {n: i for i, n in enumerate(sol_names)}
        ordered = [sol_pos[idx[n]] for n in prefixed_target if n in idx]
        if len(ordered) < len(joint_names):
            return

        self._last_ik_solution = ordered

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(joint_names)  # unprefixed — arm driver expects this
        out.position = ordered[: len(joint_names)]
        self._ghost_pub.publish(out)
        if self._follow_on:
            self._cmd_pub.publish(out)

    def _publish_status(self) -> None:
        parts = [f"state={self._state}", f"preview={self._preview_on}",
                 f"follow={self._follow_on}"]
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
