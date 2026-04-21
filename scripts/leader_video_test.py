#!/usr/bin/env python3
"""leader_video_test.py — minimal leader-mode test for video submission to AgileX.

Mirrors upstream procedure exactly (pyAgxArm/demos/nero/test1.py and
docs/nero/nero_api.md): connect -> enable loop -> set_leader_mode.
No installation_pos write, no pre/post delays — only what AgileX documents.

Stop the ROS launch (nothing else may hold the CAN bus), then run:
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

from pyAgxArm import AgxArmFactory, create_agx_arm_config

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
print(f"[leader_video_test] channel={can}")

cfg = create_agx_arm_config(robot="nero", comm="can", channel=can)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()
print("[leader_video_test] connected")

# Per upstream docs: enable before any mode switch; retry with
# set_normal_mode() so the controller accepts frames.
while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("[leader_video_test] arm enabled")

print(f"[leader_video_test] firmware={arm.get_firmware()}")

arm.set_leader_mode()
print("[leader_video_test] LEADER MODE ACTIVE — observe gravity comp now")

input("[leader_video_test] press Enter to restore normal mode...\n")

arm.set_normal_mode()
print("[leader_video_test] normal mode restored")
