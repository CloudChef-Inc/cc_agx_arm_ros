#!/usr/bin/env python3
"""leader_loopback_test.py — Agilex-prescribed CAN loopback diagnostic.

Same as leader_video_test.py but opens the CAN channel with
local_loopback=True so candump sees every frame the SDK transmits.
Per Agilex: after set_leader_mode(), TX frame IDs should be 50X.
If they're still 2XX, the firmware never actually entered leader mode.

Usage (stop the ROS launch first; only one process can hold the bus):

  Terminal 1:
    candump can_right
  Terminal 2:
    python3 scripts/leader_loopback_test.py can_right

Watch terminal 1 while terminal 2 prints "LEADER MODE ACTIVE".
Frame IDs that appear after that line are the diagnostic signal.
"""
import sys
import time

from pyAgxArm import AgxArmFactory, NeroFW, create_agx_arm_config

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
print(f"[loopback_test] channel={can}  local_loopback=True")

# Phase 1: detect firmware version with default driver.
cfg = create_agx_arm_config(
    robot="nero", comm="can", channel=can, local_loopback=True
)
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
print(f"[loopback_test] firmware={sv}  driver={fw}")

# Phase 2: reconnect with matching driver, still loopback on.
arm.disconnect()
cfg = create_agx_arm_config(
    robot="nero", comm="can", channel=can,
    firmeware_version=fw, local_loopback=True,
)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()
print("[loopback_test] connected")

while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("[loopback_test] arm enabled (expect 2XX frames in candump)")

time.sleep(2.0)  # leave a clear window of normal-mode frames in candump

arm.set_leader_mode()
print("[loopback_test] LEADER MODE ACTIVE — expect 50X frames in candump")
print("[loopback_test] move arm by hand; watch candump for 50X IDs")

input("[loopback_test] press Enter to restore normal mode...\n")

arm.set_normal_mode()
print("[loopback_test] normal mode restored")
