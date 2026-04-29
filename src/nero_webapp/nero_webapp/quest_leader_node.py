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

No delta clamp — the 7-DOF arm has human-scale workspace and the IK
is iter-bounded, so out-of-reach targets self-throttle (IK returns
not-ok and we hold the last good solution) without needing input
limits.
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

# Debug-mode circle yaw about world Z (radians). MUST match
# main.js::DEBUG_CIRCLE_YAW_RAD and quest_teleop_node::DEBUG_CIRCLE_YAW_RAD.
# Used when debug_orient_to_circle is enabled to force the EE to point
# along the circle's normal (which is otherwise undefined — the
# debug controller stream publishes identity orientation).
DEBUG_CIRCLE_YAW_RAD = {
    "left":  -math.pi / 6,
    "right": +math.pi / 6,
}

COUNTDOWN_S = 5
IK_RATE_HZ = 30.0
STREAM_STALL_S = 0.5

# Singularity-robust IK params.
#   The 7-DOF nero arm has a wrist singularity at q6 = 0 (joint5 and
#   joint7 axes align → J loses one rank). Plain DLS with a fixed tiny
#   damping fails to converge there: the step in the singular direction
#   shrinks but doesn't vanish, the solver burns iterations, and we
#   return non-converged. Solution: SVD-based pseudoinverse with
#   damping that ramps up as the smallest singular value of J shrinks
#   (Wampler-Nakamura SR-inverse). Away from singularities the damping
#   is at IK_DAMPING_MIN (≈ unweighted pinv); at the singularity the
#   damping smoothly rises to IK_DAMPING_MAX, trading exact tracking
#   in the rank-deficient direction (where motion is impossible anyway)
#   for numerical stability.
#
# IK_MAX_ITERS:
#   Tracking-IK budget. We do NOT need full Newton convergence in one
#   tick — at 30 Hz the controller delta between ticks is small, so a
#   handful of damped Newton steps is enough; the next tick continues
#   from the previous solution. 50 iters was ~400 ms in worst case
#   (unreachable target, no convergence possible) which throttled the
#   tick to 6 Hz and made teleop lag visibly. 12 caps worst-case at
#   ~50 ms so eff_hz stays close to IK_RATE_HZ even when the operator
#   commands an unreachable pose.
IK_MAX_ITERS = 12
IK_EPS = 5e-3                # ≈ 5 mm / 0.3°: tracking-grade convergence
IK_DAMPING_MIN = 1e-4        # well-conditioned damping
IK_DAMPING_MAX = 1e-1        # damping at the singularity
IK_SIGMA_THRESH = 1e-2       # σ_min below this triggers damping ramp
# Even when the loop stops short of IK_EPS, the last iterate is usable
# for teleop tracking as long as the residual is small. This threshold
# lets near-singular frames still drive the ghost instead of dropping.
IK_TRACK_TOL = 5e-2          # ~5 cm / 3° max acceptable for tracking
# Cap ‖dq‖ per iteration (rad). Even SR-DLS can produce a step big
# enough to skip across the singular manifold into a different IK
# branch; clamping keeps the solver local to the seed.
IK_MAX_STEP = 0.3

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
        # Debug-only: force the IK target orientation to point the EE
        # normal to the debug circle plane (see DEBUG_CIRCLE_YAW_RAD).
        # Bypasses the usual controller-delta path for orientation,
        # since the debug teleop publishes identity orientation.
        self.declare_parameter("debug_orient_to_circle", False)
        # Rotation from Quest stream frame → robot-world, as extrinsic-
        # xyz rpy in radians. teleop_xr publishes in ROS / REP-103
        # convention (X=forward, Y=left, Z=up), NOT raw WebXR. Robot
        # world is (X=right, Y=front, Z=up). For an operator standing
        # behind the robot facing the same direction, R_z(+π/2) maps
        # one onto the other (rpy = [0, 0, +π/2]).
        #
        # Default of [0,0,0] here is identity, used by debug mode
        # (synthetic poses already authored in robot-world). The launch
        # file substitutes [0, 0, +π/2] for real-Quest mode.
        self.declare_parameter("quest_to_world_rpy", [0.0, 0.0, 0.0])
        # Diagnostic flags. With debug_freeze_orientation=True, the IK
        # target orientation is held at the calibration EE orientation
        # (controller orientation is ignored) — useful to bisect a
        # frame-mapping bug down to the position channel only. With
        # debug_freeze_position=True, the IK target position is held
        # at the calibration EE position — useful for the inverse.
        self.declare_parameter("debug_freeze_orientation", False)
        self.declare_parameter("debug_freeze_position", False)

        self.side: str = self.get_parameter("side").value
        self.ee_link: str = self.get_parameter("ee_link").value
        urdf_path: str = self.get_parameter("urdf_path").value
        shoulder_tilt_deg: float = float(
            self.get_parameter("shoulder_tilt_deg").value
        )
        self._debug_orient_to_circle: bool = bool(
            self.get_parameter("debug_orient_to_circle").value
        )
        self._debug_freeze_orient: bool = bool(
            self.get_parameter("debug_freeze_orientation").value
        )
        self._debug_freeze_pos: bool = bool(
            self.get_parameter("debug_freeze_position").value
        )
        q2w_rpy = list(self.get_parameter("quest_to_world_rpy").value)
        if len(q2w_rpy) != 3:
            raise ValueError(
                f"quest_to_world_rpy must be length 3 (rpy in rad), got {q2w_rpy}"
            )
        self._R_q2w: R = R.from_euler("xyz", q2w_rpy)
        self._R_q2w_inv: R = self._R_q2w.inv()
        self.get_logger().info(
            f"[quest→world] rpy(rad)={q2w_rpy}  "
            f"(identity rotation iff all zeros — debug-mode default)"
        )

        if self.side not in ("left", "right"):
            raise ValueError(f"side must be left|right, got {self.side}")

        # Mount rotation (world → base_link). Matches two_nero.urdf.xacro
        # and main.js::buildArms() (both now share the same convention,
        # so this single formula is authoritative):
        #   right shoulder rpy = ( π + tilt, -π/2, 0 )
        #   left  shoulder rpy = (   -tilt , -π/2, 0 )
        # URDF rpy is extrinsic xyz: matrix = Rz(y)·Ry(p)·Rx(r).
        shoulder_tilt = math.radians(shoulder_tilt_deg)
        if self.side == "right":
            rpy = (math.pi + shoulder_tilt, -math.pi / 2.0, 0.0)
        else:
            rpy = (-shoulder_tilt, -math.pi / 2.0, 0.0)
        # scipy 'xyz' (lower-case) = extrinsic xyz, matching URDF rpy.
        self._R_mount: R = R.from_euler("xyz", rpy)
        self._R_mount_inv: R = self._R_mount.inv()
        self.get_logger().info(
            f"[{self.side}] mount rpy (rad) = "
            f"({rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f})"
        )

        # Debug mode: the IK target orientation is the calibration EE
        # orientation yawed by α about world Z — nothing else. Keeps
        # the gripper's roll/pitch exactly as calibrated and just
        # turns it to face the (yawed) ring. Cache the base-frame
        # conjugation once: R_yaw_base = R_mount⁻¹ · Rz(α) · R_mount.
        yaw = DEBUG_CIRCLE_YAW_RAD[self.side]
        Rz_world = R.from_euler("z", yaw)
        self._R_yaw_base: R = self._R_mount_inv * Rz_world * self._R_mount

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

        # Per-arm-joint position limits, pulled from the URDF once.
        # Used inside the DLS loop to clamp solutions — without this,
        # near-singular iterations can wind q up by multiple 2π and
        # FK is mod-2π so the solver happily reports "converged" on a
        # kinematic branch the hardware can't actually reach.
        qlo = self.model.lowerPositionLimit
        qhi = self.model.upperPositionLimit
        self._q_lower = np.array([float(qlo[i]) for i in self._q_indices])
        self._q_upper = np.array([float(qhi[i]) for i in self._q_indices])

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
        self._last_ik_ok_log = 0.0

        # Rolling profiling buffers — last ~2 s at 30 Hz. Surfaced on the
        # 1 Hz diagnostic log so we can tell where teleop lag comes from:
        #   ik_ms        : IK solve duration. > 33 ms ⇒ IK is the bottleneck.
        #   iters        : actual loop count (≤ IK_MAX_ITERS). High = near singular.
        #   tick_gap_ms  : wall time between successive _ik_tick fires.
        #                  > 33 ms ⇒ executor not keeping up with IK_RATE_HZ.
        #   ctrl_age_ms  : age of latest /quest/<side>_controller msg at tick.
        #                  High ⇒ Quest/network upstream is slow, not us.
        self._prof_ik_ms: List[float] = []
        self._prof_iters: List[int] = []
        self._prof_tick_gaps_ms: List[float] = []
        self._prof_ctrl_age_ms: List[float] = []
        self._prof_last_tick_t: float = 0.0

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
        # Controller pose for the webapp marker. Position = dp_world
        # (controller offset from calibration anchor, in robot-world);
        # orientation = dr_world (controller rotation since calibration,
        # also in robot-world). The browser anchors the marker at the
        # ghost's gripper_flange world position captured at calibration,
        # so this is "where the controller has moved to" relative to
        # the operator's starting hand pose.
        self._ctrl_viz_pub = self.create_publisher(
            PoseStamped, f"{ns}/quest_leader/controller_viz", 10,
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
    ) -> tuple[np.ndarray, bool, float, int]:
        """Singularity-robust IK (SR-DLS). Returns (arm_q, success, err_norm, iters).

        success is True when err_norm < IK_EPS (fully converged) OR
        err_norm < IK_TRACK_TOL with a finished iteration budget — the
        latter so frames near a singularity still produce a usable
        tracking solution rather than dropping out.

        iters is the actual number of iterations executed (1..IK_MAX_ITERS),
        used by the profiling log to flag near-singular ticks.
        """
        q_arm = np.array(q_seed, dtype=float).copy()
        target_se3 = pin.SE3(target_rot.as_matrix(), target_pos)
        err_norm = float("inf")
        for it in range(IK_MAX_ITERS):
            q_full = self._q_full(q_arm)
            # Single FK pass per iteration. computeJointJacobians runs FK
            # internally and caches the joint Jacobians; getFrameJacobian
            # below pulls the EE Jacobian from that cache without redoing
            # FK. Replaces the old forwardKinematics + computeFrameJacobian
            # pattern, which did FK twice.
            pin.computeJointJacobians(self.model, self.data, q_full)
            pin.updateFramePlacement(self.model, self.data, self._ee_fid)
            oMf = self.data.oMf[self._ee_fid]
            err = pin.log(oMf.actInv(target_se3)).vector  # 6-vector in EE frame
            err_norm = float(np.linalg.norm(err))
            if err_norm < IK_EPS:
                return q_arm, True, err_norm, it + 1

            J_full = pin.getFrameJacobian(
                self.model, self.data, self._ee_fid, pin.LOCAL
            )
            J = J_full[:, self._v_indices]  # (6, N_ARM)

            # SVD-based singularity-robust pseudoinverse. Damping is
            # adapted to the smallest singular value of J: at
            # well-conditioned configs (σ_min ≥ IK_SIGMA_THRESH) it
            # stays at IK_DAMPING_MIN; as J approaches rank deficiency
            # it ramps quadratically up to IK_DAMPING_MAX, sacrificing
            # exactness in the rank-deficient direction (where the EE
            # cannot move anyway) for numerical stability.
            U, S, Vt = np.linalg.svd(J, full_matrices=False)  # 6×6, 6, 6×N
            sigma_min = float(S[-1])
            ratio2 = max(0.0, 1.0 - (sigma_min / IK_SIGMA_THRESH) ** 2)
            lam2 = (IK_DAMPING_MIN ** 2) + ratio2 * (IK_DAMPING_MAX ** 2)
            sigma_inv = S / (S * S + lam2)         # damped reciprocals
            # J^+ = V Σ_damped U^T  (shape: N_ARM × 6)
            dq_arm = (Vt.T * sigma_inv) @ (U.T @ err)

            # Step cap — even SR-DLS can leap across the singular
            # manifold into a wound-up IK branch on a single iteration.
            step_n = float(np.linalg.norm(dq_arm))
            if step_n > IK_MAX_STEP:
                dq_arm = dq_arm * (IK_MAX_STEP / step_n)
            q_arm = q_arm + dq_arm
            # Project onto joint limits — stops the solver reporting
            # solutions the hardware can't reach.
            q_arm = np.clip(q_arm, self._q_lower, self._q_upper)

        # Out of iterations. Accept the iterate as a tracking solution
        # if it's close enough — otherwise the ghost would freeze every
        # time the arm passes near a wrist-singular config.
        return q_arm, err_norm < IK_TRACK_TOL, err_norm, IK_MAX_ITERS

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
        ctrl_eul = ctrl.rot.as_euler("xyz", degrees=True).tolist()
        ee_eul = self._ee_ref.rot.as_euler("xyz", degrees=True).tolist()
        self.get_logger().info(
            f"[{self.side}] calibrated; state=READY\n"
            f"  ctrl_ref.pos (quest)   = {ctrl.pos.tolist()}\n"
            f"  ctrl_ref.rot (deg xyz) = {[round(x, 1) for x in ctrl_eul]}\n"
            f"  ee_ref.pos   (base)    = {self._ee_ref.pos.tolist()}\n"
            f"  ee_ref.rot   (deg xyz) = {[round(x, 1) for x in ee_eul]}"
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
        # Profiling — measure tick-to-tick gap before any early returns
        # so we see the true rate the executor is firing this timer at.
        tick_start = time.perf_counter()
        if self._prof_last_tick_t > 0.0:
            self._prof_tick_gaps_ms.append((tick_start - self._prof_last_tick_t) * 1000.0)
            if len(self._prof_tick_gaps_ms) > 60:
                self._prof_tick_gaps_ms = self._prof_tick_gaps_ms[-60:]
        self._prof_last_tick_t = tick_start

        # Publish controller viz whenever we have a calibration anchor —
        # independent of preview/follow, so the operator sees the marker
        # at READY too (before they enable Preview).
        self._publish_controller_viz()

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

        self._prof_ctrl_age_ms.append(ctrl_age * 1000.0)
        if len(self._prof_ctrl_age_ms) > 60:
            self._prof_ctrl_age_ms = self._prof_ctrl_age_ms[-60:]

        # Delta in WebXR (Quest) frame, then rotated into robot-world.
        # WebXR (per spec): +X right, +Y up, +Z back-toward-user — but
        # whose right/up/back? The local-floor reference space's axes
        # depend on the user agent (on Quest, typically the guardian's
        # forward direction, NOT the user's current facing). So the
        # rotation R_q2w must be measured for the actual setup, not
        # assumed. The `quest_to_world_rpy` parameter carries this.
        # With rpy = (0,0,0) this is identity (preserves the
        # synthetic-debug pipeline, which authors poses in robot-world).
        dp_quest = ctrl.pos - self._ctrl_ref.pos
        dr_quest = ctrl.rot * self._ctrl_ref.rot.inv()
        dp_world = self._R_q2w.apply(dp_quest)
        # Conjugation: WebXR-frame rotation re-expressed in robot-world.
        dr_world = self._R_q2w * dr_quest * self._R_q2w_inv

        if self._debug_freeze_pos:
            dp_world = np.zeros(3)

        # Rotate delta into this arm's base_link frame, because
        # self._ee_ref comes from FK on the single-arm URDF and is
        # expressed in base_link.
        dp_base = self._R_mount_inv.apply(dp_world)
        # Conjugation: world rotation expressed in base_link.
        dr_base = self._R_mount_inv * dr_world * self._R_mount

        target_pos = self._ee_ref.pos + dp_base
        target_rot = dr_base * self._ee_ref.rot

        if self._debug_orient_to_circle:
            # Yaw the calibration EE orientation by α about world Z —
            # no other axis changes. Preserves the calibrated roll
            # and pitch, which keeps the gripper oriented the same
            # way the operator expects; only the direction-of-reach
            # rotates to face the yawed ring.
            target_rot = self._R_yaw_base * self._ee_ref.rot

        if self._debug_freeze_orient:
            # Bypass the controller-orientation channel entirely so any
            # IK-convergence / direction problem must be in the position
            # path. Useful for bisecting frame-mapping bugs.
            target_rot = self._ee_ref.rot

        seed = (self._last_ik_solution
                if self._last_ik_solution is not None
                else np.zeros(N_ARM))
        ik_t0 = time.perf_counter()
        q_sol, ok, err_norm, iters = self._ik(target_pos, target_rot, seed)
        ik_dur_ms = (time.perf_counter() - ik_t0) * 1000.0
        self._prof_ik_ms.append(ik_dur_ms)
        self._prof_iters.append(iters)
        if len(self._prof_ik_ms) > 60:
            self._prof_ik_ms = self._prof_ik_ms[-60:]
            self._prof_iters = self._prof_iters[-60:]

        # Always-on diagnostic (1 Hz). Logs raw dp_quest / dr_quest
        # pre-rotation alongside their re-expressed-in-world forms, so
        # R_q2w can be verified empirically: move/rotate the controller
        # in a known direction and read dp_world / dr_world.
        #
        # Rotations are printed as axis-angle (deg, unit-vector axis).
        # For a single-axis rotation test (e.g. yaw the controller 30°
        # to the operator's right), axis-angle reads off the result
        # directly: |angle| ≈ 30, axis ≈ ±world-Z.
        #
        # Profiling line answers "where is teleop lag coming from?":
        #   ik_ms avg > 33  → IK is the bottleneck (cap iters, raise rate)
        #   eff_hz < ~28    → timer not firing fast enough (executor / GIL)
        #   ctrl_age_ms hi  → upstream Quest/network lag, not us
        now = time.time()
        if now - self._last_ik_ok_log > 1.0:
            r3 = lambda v: np.round(v, 3).tolist()
            def axang(rot: R) -> str:
                rv = rot.as_rotvec()
                ang = float(np.linalg.norm(rv))
                if ang < 1e-6:
                    return "ang=0.0° axis=[0,0,0]"
                axis = rv / ang
                return (
                    f"ang={math.degrees(ang):+.1f}° "
                    f"axis=[{axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}]"
                )

            def stats(buf: list, fmt: str = ".1f") -> str:
                if not buf:
                    return "n/a"
                return (f"avg={sum(buf)/len(buf):{fmt}} "
                        f"max={max(buf):{fmt}} n={len(buf)}")

            eff_hz = (
                1000.0 / (sum(self._prof_tick_gaps_ms) / len(self._prof_tick_gaps_ms))
                if self._prof_tick_gaps_ms else 0.0
            )
            self.get_logger().info(
                f"[{self.side}] {'OK' if ok else 'FAIL'} err={err_norm:.4f}\n"
                f"  POS  dp_quest={r3(dp_quest)} dp_world={r3(dp_world)} "
                f"target_pos={r3(target_pos)}\n"
                f"  ROT  dr_quest:  {axang(dr_quest)}\n"
                f"       dr_world:  {axang(dr_world)}\n"
                f"       dr_base :  {axang(dr_base)}\n"
                f"  PROF ik_ms[{stats(self._prof_ik_ms)}] "
                f"iters[{stats(self._prof_iters, '.0f')}]\n"
                f"       tick_gap_ms[{stats(self._prof_tick_gaps_ms)}] "
                f"eff_hz={eff_hz:.1f} (target={IK_RATE_HZ:.0f})\n"
                f"       ctrl_age_ms[{stats(self._prof_ctrl_age_ms)}]"
            )
            self._last_ik_ok_log = now

        if not ok:
            return

        self._last_ik_solution = q_sol

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(ARM_JOINT_NAMES)
        out.position = q_sol.tolist()
        self._ghost_pub.publish(out)
        if self._follow_on:
            self._cmd_pub.publish(out)

    def _publish_controller_viz(self) -> None:
        """Publish controller pose for the webapp marker.

        Position = dp_world (controller offset from calibration anchor,
        in robot-world frame). Orientation = dr_world (controller
        rotation since calibration, in robot-world frame). The browser
        anchors the marker at the ghost's gripper_flange world position
        captured at calibration, so the marker visually says "this is
        where your hand is, relative to where it started."
        """
        if self._ctrl_ref is None:
            return
        with self._lock:
            ctrl = self._latest_ctrl
        if ctrl is None:
            return

        dp_quest = ctrl.pos - self._ctrl_ref.pos
        dr_quest = ctrl.rot * self._ctrl_ref.rot.inv()
        dp_world = self._R_q2w.apply(dp_quest)
        dr_world = self._R_q2w * dr_quest * self._R_q2w_inv
        q = dr_world.as_quat()  # (x, y, z, w)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_world"
        msg.pose.position.x = float(dp_world[0])
        msg.pose.position.y = float(dp_world[1])
        msg.pose.position.z = float(dp_world[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._ctrl_viz_pub.publish(msg)

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
