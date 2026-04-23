#!/usr/bin/env python3
"""quest_leader_node — Quest controller → IK → joint targets for one arm.

State machine per side: IDLE → CALIBRATING → READY → ACTIVE.

FK and IK are computed in-process with Pinocchio on the single-arm URDF
(same lib/URDF as gravity_comp). We do NOT talk to MoveIt — that path hit
TF-tree conflicts against the composed dual-arm publisher.

Calibration (sim-only — real arm does not move):
  publish the calibration pose to the ghost topic, compute ee_ref via
  FK on that pose, run a 5 s countdown, snapshot ctrl_ref. → READY.

Preview: 30 Hz loop — compute target EE from delta, solve IK, publish
`quest_leader/target_joint_states`. Webapp renders ghost.

Follow: additionally mirror IK solution to `control/joint_states`. Requires
Preview. Auto-disarms if the Quest stream stalls > 0.5 s.

Delta math:
    dp = ctrl_pos - ctrl_ref_pos            (world-frame translation)
    target_pos = ee_ref_pos + dp
    dR = ctrl_ref_R^-1 · ctrl_R             (rotation delta in ctrl frame)
    target_R = ee_ref_R · dR                (apply in EE ref frame)

Per-frame delta clamp: 5 cm, 15°.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool

import pinocchio as pin
from scipy.spatial.transform import Rotation as R


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
N_ARM = 7

# Calibration joint poses (radians, 7-DOF arm) + gripper width (metres).
# Bench-tuned values: operator holds Quest controllers out in front at
# shoulder height with palms facing inward, fingers wrapping the grip.
_D2R = math.pi / 180.0
CALIBRATION_JOINT_POSE_LEFT: List[float] = [
    -90 * _D2R, 70 * _D2R, -45 * _D2R, 0.0, 135 * _D2R, 0.0, 0.0,
]
CALIBRATION_JOINT_POSE_RIGHT: List[float] = [
    90 * _D2R, 70 * _D2R, 45 * _D2R, 0.0, -135 * _D2R, 0.0, 0.0,
]
CALIBRATION_GRIPPER_WIDTH: float = 0.1  # metres, fully open

COUNTDOWN_S = 5
IK_RATE_HZ = 30.0
# Cumulative clamp on (ctrl - ctrl_ref). 0.05 m was far too tight —
# it kept the target pinned within 5 cm of the calibration EE even
# for intentional 20–30 cm moves. Workspace-reasonable bound.
MAX_DP = 0.50
MAX_DR_DEG = 90.0
STREAM_STALL_S = 0.5

# DLS IK params.
IK_MAX_ITERS = 30
IK_EPS = 1e-3
IK_DAMPING = 1e-4

STATE_IDLE = "IDLE"
STATE_CALIBRATING = "CALIBRATING"
STATE_READY = "READY"
STATE_ACTIVE = "ACTIVE"


@dataclass
class Pose6:
    pos: np.ndarray
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

        self.declare_parameter("side", "right")
        self.declare_parameter("ee_link", "gripper_flange")
        self.declare_parameter("urdf_path", "")
        # Shoulder tilt about the arm's +X axis (degrees). Combined with
        # the ±π/2 yaw the arm is mounted at, this fully describes the
        # rotation from the dual-arm torso frame (= robot world) to the
        # single-arm URDF base_link. Used to rotate the controller
        # delta `dp` from quest/robot-world into base_link before adding
        # to the FK'd `ee_ref`.
        self.declare_parameter("shoulder_tilt_deg", 20.0)

        self.side: str = self.get_parameter("side").value
        self.ee_link: str = self.get_parameter("ee_link").value
        urdf_path: str = self.get_parameter("urdf_path").value
        shoulder_tilt_deg: float = float(
            self.get_parameter("shoulder_tilt_deg").value
        )

        if self.side not in ("left", "right"):
            raise ValueError(f"side must be left|right, got {self.side}")

        # Mount rotation (robot-world → base_link). Matches the xacro:
        #   right: rpy = "(pi/2 + tilt) 0 0"   — roll +110° about +X
        #   left : rpy = "-(pi/2 + tilt) 0 0"  — roll −110° about +X
        # URDF rpy = fixed-axis (extrinsic) xyz → scipy 'xyz'.
        shoulder_tilt = math.radians(shoulder_tilt_deg)
        roll = (math.pi / 2.0) + shoulder_tilt
        if self.side == "left":
            roll = -roll
        self._R_mount: R = R.from_euler("xyz", [roll, 0.0, 0.0])
        self._R_mount_inv: R = self._R_mount.inv()
        self.get_logger().info(
            f"[{self.side}] mount rpy = ({math.degrees(roll):.2f}, 0, 0) deg"
        )

        if not urdf_path:
            urdf_path = str(
                (get_package_share_directory("nero_webapp") + "/static/nero_with_gripper.urdf")
            )

        # Pinocchio model (full URDF — gripper joints included).
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.get_logger().info(
            f"Pinocchio model loaded from {urdf_path}: "
            f"nq={self.model.nq} nv={self.model.nv} njoints={self.model.njoints}"
        )

        self._q_indices: List[int] = []
        self._v_indices: List[int] = []
        for name in ARM_JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise RuntimeError(f"joint {name} not in model")
            self._q_indices.append(int(self.model.idx_qs[jid]))
            self._v_indices.append(int(self.model.idx_vs[jid]))

        if not self.model.existFrame(self.ee_link):
            raise RuntimeError(f"ee_link '{self.ee_link}' not found in URDF")
        self._ee_fid = self.model.getFrameId(self.ee_link)

        # Neutral pose used for non-arm joints.
        self._q_template = pin.neutral(self.model)

        self._state = STATE_IDLE
        self._lock = threading.Lock()
        self._cb_group = ReentrantCallbackGroup()

        self._latest_ctrl: Optional[Pose6] = None
        self._latest_ctrl_ts: float = 0.0
        self._latest_joint_state: Optional[JointState] = None

        self._ctrl_ref: Optional[Pose6] = None
        self._ee_ref: Optional[Pose6] = None
        self._last_ik_solution: Optional[np.ndarray] = None
        self._last_ik_fail_log = 0.0

        self._preview_on = False
        self._follow_on = False
        self._countdown_remaining = 0
        self._calibration_start = 0.0
        self._pending_ee_ref: Optional[Pose6] = None
        self._pending_target_q: Optional[np.ndarray] = None

        ns = f"/{self.side}"
        self.create_subscription(
            PoseStamped, f"/quest/{self.side}_controller",
            self._ctrl_cb, 10, callback_group=self._cb_group,
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

        self.create_timer(1.0 / IK_RATE_HZ, self._ik_tick, callback_group=self._cb_group)
        self.create_timer(0.2, self._publish_status, callback_group=self._cb_group)
        self.create_timer(0.2, self._calibration_tick, callback_group=self._cb_group)

        self.get_logger().info(
            f"quest_leader_node up (side={self.side}, ee={self.ee_link}, "
            f"in-process Pinocchio FK/IK)"
        )

    # ---------- subscriptions ----------
    def _ctrl_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest_ctrl = Pose6.from_msg(msg)
            self._latest_ctrl_ts = time.time()

    def _js_cb(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = msg

    # ---------- FK / IK (Pinocchio, in-process) ----------
    def _q_full(self, arm_q: np.ndarray) -> np.ndarray:
        q = self._q_template.copy()
        for i, qi in enumerate(self._q_indices):
            q[qi] = arm_q[i]
        return q

    def _fk(self, arm_q: np.ndarray) -> Pose6:
        q = self._q_full(arm_q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self._ee_fid)
        oMf = self.data.oMf[self._ee_fid]
        return Pose6(
            pos=np.array(oMf.translation, dtype=float),
            rot=R.from_matrix(np.array(oMf.rotation)),
        )

    def _ik(
        self,
        target_pos: np.ndarray,
        target_rot: R,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, bool, float]:
        """Damped least-squares IK. Returns (arm_q, success, final_err_norm)."""
        q_arm = np.array(q_seed, dtype=float).copy()
        target_se3 = pin.SE3(target_rot.as_matrix(), target_pos)
        err_norm = float("inf")
        for _ in range(IK_MAX_ITERS):
            q_full = self._q_full(q_arm)
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacement(self.model, self.data, self._ee_fid)
            oMf = self.data.oMf[self._ee_fid]
            err = pin.log(oMf.actInv(target_se3)).vector  # 6-vector in EE frame
            err_norm = float(np.linalg.norm(err))
            if err_norm < IK_EPS:
                return q_arm, True, err_norm
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q_full, self._ee_fid, pin.LOCAL
            )
            J = J_full[:, self._v_indices]
            # Damped least squares: dq = J^T (J J^T + λ²I)^-1 err
            JJt = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)
            dq_arm = J.T @ np.linalg.solve(JJt, err)
            q_arm = q_arm + dq_arm
        return q_arm, False, err_norm

    # ---------- services ----------
    def _calibrate_srv(self, request: Trigger.Request, response: Trigger.Response):
        """Kick off a non-blocking 5 s calibration sequence.

        Service returns immediately; the tick timer drives the countdown
        and final snapshot. Keeps status-publish + IK-loop timers live
        (a blocking sleep here would starve them on MultiThreadedExecutor).
        """
        self.get_logger().info(f"[{self.side}] calibration requested (sim-only)")
        self._preview_on = False
        self._follow_on = False

        target_q = np.array(
            CALIBRATION_JOINT_POSE_LEFT if self.side == "left"
            else CALIBRATION_JOINT_POSE_RIGHT,
            dtype=float,
        )

        ghost = JointState()
        ghost.header.stamp = self.get_clock().now().to_msg()
        # Include the gripper width so the webapp can drive it to "fully
        # open" for the calibration pose too. FK/IK only consume the
        # arm joints — the extra name is a pass-through for the UI.
        ghost.name = list(ARM_JOINT_NAMES) + ["gripper"]
        ghost.position = target_q.tolist() + [CALIBRATION_GRIPPER_WIDTH]
        self._ghost_pub.publish(ghost)

        try:
            ee_ref = self._fk(target_q)
        except Exception as e:
            self._state = STATE_IDLE
            response.success = False
            response.message = f"FK exception: {e}"
            self.get_logger().error(f"[{self.side}] FK failed: {e}")
            return response

        self.get_logger().info(
            f"[{self.side}] calibration FK: pos={ee_ref.pos.tolist()} "
            f"quat={ee_ref.rot.as_quat().tolist()}"
        )

        self._pending_ee_ref = ee_ref
        self._pending_target_q = target_q.copy()
        self._calibration_start = time.time()
        self._countdown_remaining = COUNTDOWN_S
        self._state = STATE_CALIBRATING
        response.success = True
        response.message = "calibration started"
        return response

    def _calibration_tick(self) -> None:
        """Runs at 5 Hz. Advances the countdown and finalizes at T=0."""
        if self._state != STATE_CALIBRATING:
            return
        elapsed = time.time() - self._calibration_start
        remaining = max(0, int(math.ceil(COUNTDOWN_S - elapsed)))
        self._countdown_remaining = remaining
        if elapsed < COUNTDOWN_S:
            return

        # T=0: snapshot controller pose, transition to READY.
        with self._lock:
            ctrl = self._latest_ctrl
            ctrl_ts = self._latest_ctrl_ts
        age = time.time() - ctrl_ts if ctrl_ts else float("inf")
        ctrl_fresh = ctrl is not None and age < 0.5
        if not ctrl_fresh:
            self.get_logger().error(
                f"[{self.side}] calibration failed: no fresh controller pose "
                f"(have_ctrl={ctrl is not None}, age={age:.2f}s)"
            )
            self._state = STATE_IDLE
            return

        self._ctrl_ref = ctrl
        self._ee_ref = self._pending_ee_ref
        self._last_ik_solution = self._pending_target_q.copy()
        self._countdown_remaining = 0
        self._state = STATE_READY
        self.get_logger().info(f"[{self.side}] calibrated (sim); state=READY")

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

        if ctrl is None:
            return

        if self._follow_on and ctrl_age > STREAM_STALL_S:
            self.get_logger().warn(
                f"[{self.side}] Quest stream stalled ({ctrl_age:.2f}s) — disarming Follow"
            )
            self._follow_on = False
            return

        # Delta in robot-world (= quest-world for now: no transform
        # applied between the two; the Quest stream is treated as
        # already in robot-world coordinates).
        dp_world = ctrl.pos - self._ctrl_ref.pos
        dr_world = ctrl.rot * self._ctrl_ref.rot.inv()
        dp_world, dr_world = _clamp_delta(dp_world, dr_world)

        # Rotate delta into this arm's base_link frame, because
        # self._ee_ref comes from FK on the single-arm URDF and is
        # expressed in base_link.
        dp_base = self._R_mount_inv.apply(dp_world)
        # Conjugation: world rotation expressed in base_link.
        dr_base = self._R_mount_inv * dr_world * self._R_mount

        target_pos = self._ee_ref.pos + dp_base
        target_rot = dr_base * self._ee_ref.rot

        seed = (self._last_ik_solution
                if self._last_ik_solution is not None
                else np.zeros(N_ARM))
        q_sol, ok, err_norm = self._ik(target_pos, target_rot, seed)
        if not ok:
            now = time.time()
            if now - self._last_ik_fail_log > 1.0:
                seed_ee = self._fk(seed)
                self.get_logger().warn(
                    f"[{self.side}] IK did not converge (err={err_norm:.4f}) "
                    f"target_pos={target_pos.tolist()} "
                    f"ee_ref_pos={self._ee_ref.pos.tolist()} "
                    f"dp_world={dp_world.tolist()} "
                    f"dp_base={dp_base.tolist()} "
                    f"seed_ee_pos={seed_ee.pos.tolist()} "
                    f"seed_q={seed.tolist()}"
                )
                self._last_ik_fail_log = now
            return

        self._last_ik_solution = q_sol

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(ARM_JOINT_NAMES)
        out.position = q_sol.tolist()
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
