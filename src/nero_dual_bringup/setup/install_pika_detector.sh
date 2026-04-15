#!/usr/bin/env bash
# One-shot installer for the Pika bundle auto-detector.
#
# Drops the detection script into /usr/local/bin, installs the systemd
# unit, removes the brittle port-pinned udev rules for Pika gripper +
# fisheye (script handles those now), keeps the serial-pinned CAN
# rules, and adds a udev RUN+= hook so the script re-runs whenever a
# D405 plugs in.
#
# Idempotent — safe to re-run after pulling repo changes.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Re-running with sudo..." >&2
    exec sudo -E "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT_SRC="$REPO_ROOT/src/nero_dual_bringup/scripts/detect_pika_bundles.py"
SERVICE_SRC="$REPO_ROOT/src/nero_dual_bringup/setup/nero-detect-pika.service"

[ -f "$SCRIPT_SRC" ] || { echo "missing $SCRIPT_SRC" >&2; exit 1; }
[ -f "$SERVICE_SRC" ] || { echo "missing $SERVICE_SRC" >&2; exit 1; }

# 1. Install the script.
install -m 0755 "$SCRIPT_SRC" /usr/local/bin/nero-detect-pika
echo "installed /usr/local/bin/nero-detect-pika"

# 1b. Ensure the D405 serial → side config file exists. When creating
#     a fresh one, drop the currently-visible serials into the file
#     (commented) so assigning sides is a one-edit job.
mkdir -p /etc/nero
if [ ! -f /etc/nero/d405_sides.conf ]; then
    {
        echo "# D405 serial → arm side map for nero-detect-pika."
        echo "# One line per D405: '<side> <serial>'. Edit to match your hardware."
        echo "# Identify which serial is which side by unplugging that arm's"
        echo "# Pika bundle and seeing which serial disappears here:"
        echo "#   python3 -c \"import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number)) for d in rs.context().devices]\""
        echo ""
        echo "# --- currently visible D405 serials (commented) ---"
        if python3 -c "import pyrealsense2" 2>/dev/null; then
            python3 -c "
import pyrealsense2 as rs
for d in rs.context().devices:
    print('# ' + d.get_info(rs.camera_info.serial_number))
"
        fi
        echo ""
        echo "# --- active assignments (uncomment + edit) ---"
        echo "# right <serial>"
        echo "# left  <serial>"
    } > /etc/nero/d405_sides.conf
    echo "wrote template /etc/nero/d405_sides.conf — edit it to assign sides"
else
    echo "keeping existing /etc/nero/d405_sides.conf"
fi

# 2. Install the systemd unit.
install -m 0644 "$SERVICE_SRC" /etc/systemd/system/nero-detect-pika.service
systemctl daemon-reload
systemctl enable nero-detect-pika.service
echo "enabled nero-detect-pika.service"

# 3. Rewrite the udev rules: keep CAN serial pinning (portable across
#    hubs) and replace the fragile Pika/fisheye port pinning with a
#    udev RUN+= that fires the detector script whenever a D405 plugs in.
cat > /etc/udev/rules.d/70-nero-arms.rules <<'EOF'
# CAN adapters (gs_usb, pinned by serial number — portable across any USB hub).
SUBSYSTEM=="net", ACTION=="add", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", ATTRS{serial}=="003D00414148570A20343133", NAME="can_right", RUN+="/sbin/ip link set dev can_right up type can bitrate 1000000", RUN+="/sbin/ip link set dev can_right txqueuelen 1000"
SUBSYSTEM=="net", ACTION=="add", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", ATTRS{serial}=="002E00384148571320343133", NAME="can_left",  RUN+="/sbin/ip link set dev can_left  up type can bitrate 1000000", RUN+="/sbin/ip link set dev can_left  txqueuelen 1000"

# When a D405 enumerates, fire the bundle-detection script. It reads
# the D405 serial, walks the USB topology, and creates stable
# /dev/pika_<side> + /dev/fisheye_<side> symlinks.
SUBSYSTEM=="usb", ACTION=="add", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5b", RUN+="/bin/systemctl --no-block restart nero-detect-pika.service"
EOF
udevadm control --reload-rules
echo "wrote /etc/udev/rules.d/70-nero-arms.rules"

# 4. Run it now so the symlinks come up immediately.
systemctl start nero-detect-pika.service || true
echo
echo "Triggering udev to refresh attached devices..."
udevadm trigger --action=add
sleep 2
systemctl start nero-detect-pika.service || true
sleep 1

echo
echo "=== Resulting symlinks ==="
ls -l /dev/pika_left /dev/pika_right /dev/fisheye_left /dev/fisheye_right 2>&1 || true
ip -br link show type can | grep -E 'can_(left|right)' || true

echo
echo "Done. Future boots and re-cabling will keep these symlinks current."
