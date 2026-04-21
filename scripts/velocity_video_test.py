#!/usr/bin/env python3
"""velocity_video_test.py — verify AgileX's velocity-feedback workaround.

AgileX reply 2026-04-20: on V1.11 the velocity field reads low; multiply
by 1000 to get rad/s. Fix coming in V1.12.

This script:
  1. Connects, enables, enters leader mode so joints are back-drivable.
  2. Streams per-joint raw velocity and raw*1000 at ~20 Hz.
  3. You manually move joints on camera and watch the printed values.

Stop the ROS launch first (only one process can hold the CAN bus):
  python3 scripts/velocity_video_test.py                # can_right
  python3 scripts/velocity_video_test.py can_left
"""
import sys
import time

from pyAgxArm import AgxArmFactory, create_agx_arm_config

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
print(f"[vel_test] channel={can}")

cfg = create_agx_arm_config(robot="nero", comm="can", channel=can)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()
print("[vel_test] connected")

while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("[vel_test] arm enabled")

print(f"[vel_test] firmware={arm.get_firmware()}")

arm.set_leader_mode()
print("[vel_test] LEADER MODE ACTIVE — move joints by hand to see velocity")
print("[vel_test] columns: J<i> raw=<v> x1000=<v*1000>  (rad/s per AgileX)")
print("[vel_test] Ctrl+C to stop\n")

try:
    while True:
        parts = []
        for i in range(1, 8):
            ms = arm.get_motor_states(i)
            v = ms.msg.velocity if ms is not None else float("nan")
            parts.append(f"J{i} raw={v:+.5f} x1000={v*1000:+.3f}")
        print(" | ".join(parts))
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\n[vel_test] stopping")
finally:
    arm.set_normal_mode()
    print("[vel_test] normal mode restored")
