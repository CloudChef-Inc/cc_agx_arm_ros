#!/usr/bin/env python3
"""test_leader.py — test leader mode with proper delays per AgileX docs.

Stop the ROS launch first, then run directly:
  python3 scripts/test_leader.py                    # default: can_right, ipos=0
  python3 scripts/test_leader.py can_right 1        # horizontal mount
  python3 scripts/test_leader.py can_right 2        # side-left
  python3 scripts/test_leader.py can_right 3        # side-right
"""
import sys, time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
ipos = int(sys.argv[2]) if len(sys.argv) > 2 else 0
print(f"CAN: {can}  installation_pos: {ipos} "
      f"(0=unset, 1=horizontal, 2=side-left, 3=side-right)")

cfg = create_agx_arm_config(robot="nero", comm="can", channel=can)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()

# Enable first (required per docs)
while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("Arm enabled")

# Set installation_pos before entering leader mode
arm._msg_mode.installation_pos = ipos
time.sleep(1)  # 1s delay BEFORE mode switch (per docs)

arm.set_leader_mode()
print(f"Leader mode set (installation_pos={ipos})")
print("Try moving the arm — does it hold against gravity?")

input("\nPress Enter to exit leader mode...\n")

time.sleep(1)
arm.set_normal_mode()
time.sleep(1)
arm.enable()
print("Normal mode restored")
