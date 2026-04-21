#!/usr/bin/env python3
"""leader_video_test.py — minimal leader-mode test for video submission to AgileX.

Mirrors the upstream two-phase connect pattern (demos/detect_nero_series.py
and agx_arm_ctrl_single_node.py): connect with DEFAULT, read firmware,
reconnect with NeroFW.V111 if firmware >= 1.11 so the V1.11 mode-ctrl
frame layout (0x151) is used. Without this, mode switches may silently
fail on V1.11 arms.

Stop the ROS launch first (nothing else may hold the CAN bus), then run:
  python3 scripts/leader_video_test.py                # default: can_right
  python3 scripts/leader_video_test.py can_left

Video protocol (announce each step aloud for the recording):
  1. Arm is powered, on the bench, in a safe neutral pose.
  2. Start recording.
  3. Run this script. Wait for "LEADER MODE ACTIVE" line.
  4. Let the arm sit for ~3 s untouched — does it hold against gravity?
  5. Gently try to move each joint — should feel zero-force / back-drivable.
  6. Release. If it falls or drifts, that is the defect to show AgileX.
  7. Press Enter to restore normal mode. Stop recording.
"""
import sys
import time

from pyAgxArm import AgxArmFactory, NeroFW, create_agx_arm_config

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
print(f"[leader_video_test] channel={can}")

# Phase 1: connect with default firmware to read the actual version.
cfg = create_agx_arm_config(robot="nero", comm="can", channel=can)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()

deadline = time.time() + 15.0
while arm.get_firmware() is None:
    if time.time() >= deadline:
        raise TimeoutError(f"No firmware response on {can}")
    arm.enable()
    time.sleep(0.5)

sv = arm.get_firmware()["software_version"]
fw = NeroFW.V111 if sv >= "1.11" else NeroFW.DEFAULT
print(f"[leader_video_test] firmware={sv}  driver={fw}")

# Phase 2: reconnect with the matching driver version.
arm.disconnect()
cfg = create_agx_arm_config(
    robot="nero", comm="can", channel=can, firmeware_version=fw
)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()
print("[leader_video_test] connected")

while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("[leader_video_test] arm enabled")

arm.set_leader_mode()
print("[leader_video_test] LEADER MODE ACTIVE — observe gravity comp now")

input("[leader_video_test] press Enter to restore normal mode...\n")

arm.set_normal_mode()
print("[leader_video_test] normal mode restored")
