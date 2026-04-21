#!/usr/bin/env python3
"""velocity_video_test.py — verify AgileX's velocity-feedback workaround.

AgileX reply 2026-04-20: on V1.11 the velocity field reads low; multiply
by 1000 to get rad/s. Fix coming in V1.12.

This script mirrors the upstream two-phase connect pattern
(demos/detect_nero_series.py): connect DEFAULT, read firmware, reconnect
with NeroFW.V111 if firmware >= 1.11. Then it commands a small J1 sweep
back and forth so the arm actually moves; during the motion it streams
raw motor-state velocity and raw*1000 per joint.

Stop the ROS launch first (only one process can hold the CAN bus):
  python3 scripts/velocity_video_test.py                # can_right
  python3 scripts/velocity_video_test.py can_left
"""
import sys
import threading
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

arm.set_speed_percent(20)

# Read current pose so we sweep from a safe starting point.
ja = None
for _ in range(100):
    ja = arm.get_joint_angles()
    if ja is not None and ja.hz > 0:
        break
    time.sleep(0.05)
if ja is None or ja.hz <= 0:
    raise RuntimeError("no joint_angles feedback — arm not pushing?")

home = list(ja.msg)
print(f"[vel_test] start pose (rad): {[round(x, 3) for x in home]}")

stop = threading.Event()


def stream():
    print("[vel_test] columns: J<i> raw=<v> x1000=<v*1000>  (rad/s per AgileX)\n")
    while not stop.is_set():
        parts = []
        for i in range(1, 8):
            ms = arm.get_motor_states(i)
            v = ms.msg.velocity if ms is not None else float("nan")
            parts.append(f"J{i} raw={v:+.5f} x1000={v*1000:+.3f}")
        print(" | ".join(parts))
        time.sleep(0.05)


t = threading.Thread(target=stream, daemon=True)
t.start()

try:
    print("[vel_test] commanding small J1 sweep (+/- 0.2 rad) — watch velocity")
    for _ in range(3):
        target = list(home)
        target[0] = home[0] + 0.2
        arm.move_j(target)
        time.sleep(2.0)
        target[0] = home[0] - 0.2
        arm.move_j(target)
        time.sleep(2.0)
    arm.move_j(home)
    time.sleep(2.0)
except KeyboardInterrupt:
    print("\n[vel_test] interrupted")
finally:
    stop.set()
    t.join(timeout=1.0)
    arm.set_normal_mode()
    print("[vel_test] normal mode restored")
