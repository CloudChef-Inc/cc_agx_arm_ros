#!/usr/bin/env python3
"""calibrate_joint_zero.py — re-zero a single Nero joint.

Calls the SDK's calibrate_joint(joint_index), which sends an
arm_joint_config CAN frame with set_motor_current_pos_as_zero=0xAE
and waits for is_set_zero_successfully=0x01 in the firmware response.

DESTRUCTIVE: this rewrites the joint's zero offset in firmware NVM.
Only run with the joint physically at its mechanical zero pose.

Usage (stop the ROS launch first; only one process can hold the bus):

  # Re-zero ONLY joint 1 on the left arm:
  python3 scripts/calibrate_joint_zero.py can_left 1

  # Verify mode (read-only): print firmware + current angles, no write:
  python3 scripts/calibrate_joint_zero.py can_left 1 --dry-run

Procedure:
  1. Power off the arm. Manually pose joint 1 to its mechanical
     zero (visual/painted index, or aligned with the next link as
     defined in the AgileX manual). Other joints can stay where
     they are — calibrate_joint is per-joint.
  2. Power on. Run this script with --dry-run first to confirm
     CAN connectivity and read the current joint angle. Joint 1
     should read close to (but not exactly) 0 if the offset has
     drifted; the magnitude tells you how far off it is.
  3. Re-run without --dry-run to commit. The script will refuse
     unless you type "YES" to confirm.
  4. Power-cycle the arm. Re-run --dry-run; joint 1 should now
     read ~0.000 in the same pose.
"""
import argparse
import sys
import time

from pyAgxArm import AgxArmFactory, NeroFW, create_agx_arm_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", help="CAN channel (e.g. can_left, can_right)")
    ap.add_argument("joint_index", type=int,
                    help="Joint to re-zero (1-7). Use 8 = ALL joints.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Read firmware + joint angles only; do not write.")
    args = ap.parse_args()

    if args.joint_index not in (1, 2, 3, 4, 5, 6, 7, 8):
        print(f"joint_index must be in 1..7 (or 8 for all), got {args.joint_index}")
        sys.exit(2)

    print(f"[set_zero] channel={args.channel}  joint={args.joint_index}  "
          f"dry_run={args.dry_run}")

    cfg = create_agx_arm_config(robot="nero", comm="can", channel=args.channel)
    arm = AgxArmFactory.create_arm(cfg)
    arm.connect()

    deadline = time.time() + 15.0
    while arm.get_firmware() is None:
        if time.time() >= deadline:
            raise TimeoutError(f"No firmware response on {args.channel}")
        arm.enable()
        time.sleep(0.5)

    sv = arm.get_firmware()["software_version"]
    fw = NeroFW.V111 if sv >= "1.11" else NeroFW.DEFAULT
    print(f"[set_zero] firmware={sv}  driver={fw}")

    arm.disconnect()
    cfg = create_agx_arm_config(
        robot="nero", comm="can", channel=args.channel, firmeware_version=fw,
    )
    arm = AgxArmFactory.create_arm(cfg)
    arm.connect()

    while not arm.enable():
        arm.set_normal_mode()
        time.sleep(0.01)
    print("[set_zero] arm enabled, in normal mode")

    time.sleep(0.5)
    js = arm.get_joint_angles()
    if js is not None and js.hz > 0:
        angles = [round(float(v), 4) for v in js.msg[:7]]
        print(f"[set_zero] current angles (rad) = {angles}")
        if 1 <= args.joint_index <= 7:
            print(f"[set_zero] joint {args.joint_index} now reads "
                  f"{angles[args.joint_index - 1]:+.4f} rad — this is "
                  f"the offset that will become the new zero.")
    else:
        print("[set_zero] WARNING: could not read joint angles")

    if args.dry_run:
        print("[set_zero] dry-run, no write performed")
        return

    print()
    print(f"  About to set joint {args.joint_index} current position "
          f"as new zero in firmware NVM.")
    print("  This is persistent across power cycles.")
    print(f"  Confirm joint {args.joint_index} is physically at its "
          f"intended zero pose.")
    confirm = input("  Type YES to commit: ")
    if confirm.strip() != "YES":
        print("[set_zero] aborted")
        return

    if not hasattr(arm, "calibrate_joint"):
        print("[set_zero] ERROR: SDK has no calibrate_joint() — fall back "
              "to AgileX Studio app.")
        sys.exit(3)

    print(f"[set_zero] calling arm.calibrate_joint({args.joint_index})...")
    ok = arm.calibrate_joint(args.joint_index)
    print(f"[set_zero] calibrate_joint returned: {ok}")
    print("[set_zero] power-cycle the arm now, then re-run with --dry-run "
          "to verify the new zero stuck.")


if __name__ == "__main__":
    main()
