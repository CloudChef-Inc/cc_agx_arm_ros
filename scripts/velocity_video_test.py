#!/usr/bin/env python3
"""velocity_video_test.py — check whether motor_states velocity ever populates.

Enters leader mode so joints can be back-driven by hand (no commanded
motion). Streams per-joint raw motor-state velocity and raw*1000 at
~20 Hz. You move joints manually and watch the values.

AgileX reply 2026-04-20 prescribed the ×1000 workaround on V1.11 (fix in
V1.12). Our arm is on 1.10, so applicability is unconfirmed — this
script just lets us see what the field actually reports.

Stop the ROS launch first (only one process can hold the CAN bus):
  python3 scripts/velocity_video_test.py                # can_right
  python3 scripts/velocity_video_test.py can_left
"""
import sys
import time

from pyAgxArm import AgxArmFactory, NeroFW, create_agx_arm_config

can = sys.argv[1] if len(sys.argv) > 1 else "can_right"
print(f"[vel_test] channel={can}")

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
print(f"[vel_test] firmware={sv}  driver={fw}")

arm.disconnect()
cfg = create_agx_arm_config(
    robot="nero", comm="can", channel=can, firmeware_version=fw
)
arm = AgxArmFactory.create_arm(cfg)
arm.connect()

while not arm.enable():
    arm.set_normal_mode()
    time.sleep(0.01)
print("[vel_test] arm enabled")

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
