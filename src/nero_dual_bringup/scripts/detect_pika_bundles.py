#!/usr/bin/env python3
"""Dynamically discover Pika bundles via D405 serial → label them stably.

Each Pika bundle's USB-C cable carries both USB2 (fisheye + CH340 +
MicroPython) and USB3 (D405) lanes. The CH340 chip and the DECXIN
fisheye both lack unique serial numbers — so udev can't pin them by
serial and port-path pinning breaks if you re-cable.

The D405, however, has a unique serial. We use it as the bundle
anchor:

  1. Find every D405; identify each one's arm side from a known
     serial → side map (edit SIDE_BY_D405_SERIAL below when swapping
     hardware).
  2. From the D405's USB3 sysfs path, find the bundle's USB3 hub
     (D405's parent), then walk up to the closest ancestor hub whose
     idVendor uniquely matches a single hub on the USB2 bus. That
     pair gives us the USB3 → USB2 prefix translation for the same
     physical USB-C connector.
  3. Translate the bundle's USB3 hub path into its USB2 twin path,
     then look at known port assignments inside the bundle (port 1
     = DECXIN fisheye, port 4 = CH340 gripper) to find the actual
     devices.
  4. Symlink /dev/pika_<side> → /dev/ttyUSBN, /dev/fisheye_<side>
     → /dev/videoN.

Runs at boot from systemd, and re-runs whenever a D405 is added by
a udev RUN+= hook. Idempotent — safe to run any number of times.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from typing import Dict, Optional, Tuple

# Serial → side map lives in this config file (one entry per line,
# "<side> <serial>"). Keep it out of code so hardware swaps are a
# one-liner edit with no git push.
SIDES_CONF_PATH = "/etc/nero/d405_sides.conf"

SYSFS_USB = "/sys/bus/usb/devices"
D405_VID, D405_PID = "8086", "0b5b"
CH340_VID, CH340_PID = "1a86", "7522"
DECXIN_VID, DECXIN_PID = "1bcf", "2cd1"
DEVICE_CLASS_HUB = "09"
# Port the CH340 gripper sits on inside the Pika bundle's USB2 hub.
PIKA_GRIPPER_PORT = "4"
# Port the DECXIN fisheye sits on inside the Pika bundle's USB2 hub.
PIKA_FISHEYE_PORT = "1"


def _read(path: str, name: str) -> Optional[str]:
    try:
        with open(f"{path}/{name}") as f:
            return f.read().strip()
    except OSError:
        return None


def _is_hub(usb_path: str) -> bool:
    return _read(f"{SYSFS_USB}/{usb_path}", "bDeviceClass") == DEVICE_CLASS_HUB


def find_d405_paths() -> Dict[str, str]:
    """Return {serial: usb_path} for every D405 enumerated right now."""
    out: Dict[str, str] = {}
    for d in glob.glob(f"{SYSFS_USB}/*"):
        if _read(d, "idVendor") == D405_VID and _read(d, "idProduct") == D405_PID:
            serial = _read(d, "serial")
            if serial:
                out[serial] = os.path.basename(d)
    return out


def load_sides_config() -> Dict[str, str]:
    """Parse /etc/nero/d405_sides.conf — lines like '<side> <serial>'.

    Comments (#) and blank lines ignored. Returns {serial: side}.
    """
    mapping: Dict[str, str] = {}
    try:
        with open(SIDES_CONF_PATH) as f:
            for lineno, line in enumerate(f, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    print(f"nero-detect-pika: {SIDES_CONF_PATH}:{lineno}: "
                          f"expected '<side> <serial>', got {line!r}",
                          file=sys.stderr)
                    continue
                side, serial = parts
                mapping[serial] = side
    except FileNotFoundError:
        pass
    return mapping


def derive_usb2_prefix(bundle_usb3_hub: str
                       ) -> Optional[Tuple[str, str]]:
    """Walk up from the bundle's USB3 hub. At each ancestor, look for
    USB2 hubs on Bus 1 with the same idVendor. Return the first
    (usb3_ancestor, usb2_twin) pair where exactly ONE Bus 1 hub
    matches — that pair is the USB-C cable's host-controller offset.

    Why: the same physical USB-C connector splits internally to USB2
    + USB3 lanes. Both buses see the same downstream port topology
    once you're past the host-controller-specific hub. Finding the
    closest unique twin lets us translate any USB3 path to its
    matching USB2 path without hard-coding hub VIDs.
    """
    parts = bundle_usb3_hub.split(".")
    # Try ancestors deepest-to-shallowest. Deepest unique match wins.
    for depth in range(len(parts), 0, -1):
        anc = ".".join(parts[:depth])
        anc_full = f"{SYSFS_USB}/{anc}"
        if not os.path.isdir(anc_full):
            continue
        anc_vid = _read(anc_full, "idVendor")
        if not anc_vid:
            continue
        # Same VID, on Bus 1, must be a hub.
        candidates = []
        for d in glob.glob(f"{SYSFS_USB}/1-*"):
            name = os.path.basename(d)
            if ":" in name:           # interface path, not a device
                continue
            if not _is_hub(name):
                continue
            if _read(d, "idVendor") == anc_vid:
                candidates.append(name)
        if len(candidates) == 1:
            return anc, candidates[0]
    return None


def child_at_port(hub_path: str, port: str, vid: str, pid: str
                  ) -> Optional[str]:
    """Direct child of `hub_path` at `port` matching VID/PID, or None."""
    candidate = f"{hub_path}.{port}"
    full = f"{SYSFS_USB}/{candidate}"
    if not os.path.isdir(full):
        return None
    if _read(full, "idVendor") == vid and _read(full, "idProduct") == pid:
        return candidate
    return None


def first_video_node(usb_path: str) -> Optional[str]:
    """Lowest-numbered /dev/videoN under a USB device — that's the
    UVC capture node (higher-indexed siblings are metadata)."""
    base = f"{SYSFS_USB}/{usb_path}"
    nodes = []
    # Recursive glob — some cameras expose video* directly under the
    # interface dir, others nest it one level deeper.
    for link in glob.glob(f"{base}/**/video4linux/video*", recursive=True):
        nodes.append(os.path.basename(link))
    if not nodes:
        return None
    nodes.sort(key=lambda s: int(s.replace("video", "")))
    return nodes[0]


def first_tty_node(usb_path: str) -> Optional[str]:
    base = f"{SYSFS_USB}/{usb_path}"
    # Recursive — CH340 exposes tty/ttyUSBN two levels under the
    # interface dir, not one.
    for link in glob.glob(f"{base}/**/tty/tty*", recursive=True):
        return os.path.basename(link)
    return None


def atomic_symlink(target_basename: str, link_path: str) -> None:
    tmp = link_path + ".tmp"
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
    os.symlink(target_basename, tmp)
    os.replace(tmp, link_path)


def main() -> int:
    # Cold boot: USB enumeration may not be done when systemd runs us.
    # Wait up to 15s for at least one D405 to appear.
    deadline = time.monotonic() + 15.0
    d405s: Dict[str, str] = {}
    while time.monotonic() < deadline:
        d405s = find_d405_paths()
        if d405s:
            break
        time.sleep(0.5)
    if not d405s:
        print("nero-detect-pika: no D405s after 15s wait", file=sys.stderr)
        return 1

    sides = load_sides_config()
    if not sides:
        print(f"nero-detect-pika: no config at {SIDES_CONF_PATH}. "
              f"Currently visible D405 serials: "
              f"{sorted(d405s.keys())}. Create the file with one line "
              f"per side, e.g. 'right {next(iter(d405s))}'.",
              file=sys.stderr)
        return 1

    rc = 0
    for serial, d405_path in sorted(d405s.items()):
        side = sides.get(serial)
        if not side:
            print(f"nero-detect-pika: D405 serial {serial!r} not in "
                  f"{SIDES_CONF_PATH} — known sides: {sides}. "
                  f"Visible serials right now: {sorted(d405s.keys())}.",
                  file=sys.stderr)
            rc = max(rc, 2)
            continue
        if "." not in d405_path:
            print(f"nero-detect-pika: unexpected D405 path {d405_path!r}",
                  file=sys.stderr)
            rc = max(rc, 3)
            continue

        bundle_usb3_hub = d405_path.rsplit(".", 1)[0]
        prefix = derive_usb2_prefix(bundle_usb3_hub)
        if not prefix:
            print(f"nero-detect-pika: cannot map USB3→USB2 for "
                  f"{bundle_usb3_hub} (D405 serial {serial}, side {side}). "
                  f"Bundle's USB2 side may not be enumerated.",
                  file=sys.stderr)
            rc = max(rc, 4)
            continue
        usb3_anc, usb2_anc = prefix
        suffix = bundle_usb3_hub[len(usb3_anc):]
        bundle_usb2_hub = usb2_anc + suffix

        ch340 = child_at_port(bundle_usb2_hub, PIKA_GRIPPER_PORT,
                              CH340_VID, CH340_PID)
        decxin = child_at_port(bundle_usb2_hub, PIKA_FISHEYE_PORT,
                               DECXIN_VID, DECXIN_PID)

        if ch340:
            tty = first_tty_node(ch340)
            if tty:
                atomic_symlink(tty, f"/dev/pika_{side}")
                print(f"nero-detect-pika: /dev/pika_{side} -> /dev/{tty}")
            else:
                print(f"nero-detect-pika: CH340 found but no tty node "
                      f"(usb {ch340})", file=sys.stderr)
                rc = max(rc, 5)
        else:
            print(f"nero-detect-pika: no CH340 at "
                  f"{bundle_usb2_hub}.{PIKA_GRIPPER_PORT} for {side}",
                  file=sys.stderr)
            rc = max(rc, 6)

        if decxin:
            video = first_video_node(decxin)
            if video:
                atomic_symlink(video, f"/dev/fisheye_{side}")
                print(f"nero-detect-pika: /dev/fisheye_{side} -> "
                      f"/dev/{video}")
            else:
                print(f"nero-detect-pika: DECXIN found but no video node "
                      f"(usb {decxin})", file=sys.stderr)
                rc = max(rc, 7)
        else:
            print(f"nero-detect-pika: no DECXIN at "
                  f"{bundle_usb2_hub}.{PIKA_FISHEYE_PORT} for {side}",
                  file=sys.stderr)
            rc = max(rc, 8)

    return rc


if __name__ == "__main__":
    sys.exit(main())
