#!/usr/bin/env python3
"""test_leader.py — test leader mode with proper delays per AgileX docs.

Stop the ROS launch first, then run directly:
  python3 scripts/test_leader.py          # default: can_right
  python3 scripts/test_leader.py can_left # specify CAN interface
"""
import sys, time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, NeroFW

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
print(f"Using CAN interface: {can}")

cfg = create_agx_arm_config(robot="nero", comm="can", channel=can)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()

# Enable first (required per docs)
while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("Arm enabled")

time.sleep(1)  # 1s delay BEFORE mode switch (per docs)

arm.set_leader_mode()
print("Leader mode set — try moving the arm now")
print("It should be zero-force draggable.")

input("\nPress Enter to exit leader mode...\n")

time.sleep(1)
arm.set_normal_mode()
time.sleep(1)
arm.enable()
print("Normal mode restored")
