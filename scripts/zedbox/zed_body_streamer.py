#!/usr/bin/env python3
"""ZED body tracking streamer for the Nero mocap teleop system.

Runs on the ZedBox (Ubuntu 22.04, no ROS). Opens the ZED camera,
runs BODY_34 body tracking via pyzed, and streams skeleton data as
newline-delimited JSON over a TCP server on port 9090 (configurable).

Each JSON line is one frame for one body:
{
  "body_id": 0,
  "confidence": 0.92,
  "timestamp_ms": 1776400000000,
  "keypoints": {
    "LEFT_SHOULDER": {"pos": [x, y, z], "ori": [qx, qy, qz, qw], "conf": 0.95},
    "LEFT_ELBOW":    {"pos": [x, y, z], "ori": [qx, qy, qz, qw], "conf": 0.93},
    ...
  }
}

The Thor-side teleop_mapper_node connects as a TCP client and
processes these lines. No ROS, no custom messages, no colcon.

Usage:
  python3 zed_body_streamer.py                  # defaults: port 9090
  python3 zed_body_streamer.py --port 9091      # custom port
  python3 zed_body_streamer.py --svo path.svo   # replay from recording
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from typing import Dict, List, Optional

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed not found. Install the ZED SDK Python API.", file=sys.stderr)
    sys.exit(1)

import numpy as np


# ZED BODY_34 keypoint names by index.
BODY_34_NAMES = [
    "PELVIS", "NAVAL_SPINE", "CHEST_SPINE", "NECK",
    "LEFT_CLAVICLE", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "LEFT_HAND", "LEFT_HANDTIP", "LEFT_THUMB",
    "RIGHT_CLAVICLE", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "RIGHT_HAND", "RIGHT_HANDTIP", "RIGHT_THUMB",
    "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT",
    "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT",
    "HEAD", "NOSE", "LEFT_EYE", "LEFT_EAR",
    "RIGHT_EYE", "RIGHT_EAR", "LEFT_HEEL", "RIGHT_HEEL",
]

# Subset we actually stream (arm-relevant + reference points).
# Streaming all 34 wastes bandwidth; the mapper only needs arms + torso.
STREAM_KEYPOINTS = {
    "PELVIS", "NAVAL_SPINE", "CHEST_SPINE", "NECK", "HEAD",
    "LEFT_CLAVICLE", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    "LEFT_HAND", "LEFT_HANDTIP", "LEFT_THUMB",
    "RIGHT_CLAVICLE", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "RIGHT_HAND", "RIGHT_HANDTIP", "RIGHT_THUMB",
}


def body_to_dict(body) -> Optional[Dict]:
    """Convert a ZED Body object to a JSON-serialisable dict."""
    if body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
        return None

    keypoints_3d = body.keypoint
    local_orientations = body.local_orientation_per_joint
    keypoint_confidences = body.keypoint_confidence

    kps = {}
    for idx, name in enumerate(BODY_34_NAMES):
        if name not in STREAM_KEYPOINTS:
            continue
        pos = keypoints_3d[idx]
        ori = local_orientations[idx] if idx < len(local_orientations) else [0, 0, 0, 1]
        conf = float(keypoint_confidences[idx]) / 100.0 if idx < len(keypoint_confidences) else 0.0

        # Skip NaN positions (keypoint not detected).
        if np.isnan(pos[0]):
            continue

        kps[name] = {
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "ori": [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])],
            "conf": round(conf, 3),
        }

    if not kps:
        return None

    return {
        "body_id": int(body.id),
        "confidence": round(float(body.confidence) / 100.0, 3),
        "timestamp_ms": int(time.time() * 1000),
        "keypoints": kps,
    }


class SkeletonServer:
    """TCP server that accepts multiple clients and broadcasts
    skeleton JSON lines to all of them."""

    def __init__(self, port: int):
        self.port = port
        self.clients: List[socket.socket] = []
        self._lock = threading.Lock()
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", port))
        self._server_sock.listen(4)
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True
        )
        self._accept_thread.start()

    def _accept_loop(self):
        while True:
            try:
                client, addr = self._server_sock.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self.clients.append(client)
                print(f"[streamer] client connected: {addr}")
            except OSError:
                break

    def broadcast(self, data: Dict) -> None:
        line = json.dumps(data, separators=(",", ":")) + "\n"
        encoded = line.encode()
        dead: List[socket.socket] = []
        with self._lock:
            for c in self.clients:
                try:
                    c.sendall(encoded)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(c)
            for c in dead:
                self.clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def close(self):
        self._server_sock.close()
        with self._lock:
            for c in self.clients:
                try:
                    c.close()
                except OSError:
                    pass
            self.clients.clear()


def select_primary_body(bodies) -> Optional[object]:
    """Pick the best body to track. Prefers the one closest to
    camera center (smallest |x| position of the pelvis)."""
    best = None
    best_dist = float("inf")
    for body in bodies.body_list:
        if body.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
            continue
        pelvis = body.keypoint[0]
        if np.isnan(pelvis[0]):
            continue
        dist = abs(float(pelvis[0]))
        if dist < best_dist:
            best_dist = dist
            best = body
    return best


def main():
    parser = argparse.ArgumentParser(description="ZED body tracking streamer")
    parser.add_argument("--port", type=int, default=9090,
                        help="TCP server port (default: 9090)")
    parser.add_argument("--svo", type=str, default="",
                        help="Path to SVO file for replay (omit for live camera)")
    parser.add_argument("--resolution", type=str, default="HD720",
                        choices=["HD720", "HD1080", "HD2K"],
                        help="Camera resolution (default: HD720)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target frame rate (default: 30)")
    args = parser.parse_args()

    # --- ZED init ---
    zed = sl.Camera()
    init_params = sl.InitParameters()
    if args.svo:
        init_params.set_from_svo_file(args.svo)
    init_params.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init_params.camera_fps = args.fps
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[streamer] ZED open failed: {status}", file=sys.stderr)
        sys.exit(1)
    print(f"[streamer] ZED opened: {zed.get_camera_information().camera_model}")

    # --- Positional tracking (required by body tracking) ---
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.set_as_static = True  # camera is fixed, not moving
    status = zed.enable_positional_tracking(tracking_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[streamer] positional tracking enable failed: {status}",
              file=sys.stderr)
        sys.exit(1)
    print("[streamer] positional tracking enabled (static mode)")

    # --- Body tracking ---
    bt_params = sl.BodyTrackingParameters()
    bt_params.enable_tracking = True
    bt_params.enable_body_fitting = True
    bt_params.body_format = sl.BODY_FORMAT.BODY_34
    # Confidence threshold attribute name varies across SDK versions.
    for attr in ("detection_confidence_threshold", "minimum_confidence",
                 "confidence_threshold"):
        if hasattr(bt_params, attr):
            setattr(bt_params, attr, 40)
            break

    status = zed.enable_body_tracking(bt_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[streamer] body tracking enable failed: {status}", file=sys.stderr)
        sys.exit(1)
    print("[streamer] body tracking enabled (BODY_34)")

    # --- TCP server ---
    server = SkeletonServer(args.port)
    print(f"[streamer] TCP server listening on :{args.port}")

    # --- Main loop ---
    runtime = sl.RuntimeParameters()
    bodies = sl.Bodies()
    bt_runtime = sl.BodyTrackingRuntimeParameters()
    for attr in ("detection_confidence_threshold", "minimum_confidence",
                 "confidence_threshold"):
        if hasattr(bt_runtime, attr):
            setattr(bt_runtime, attr, 40)
            break

    frame_count = 0
    fps_start = time.monotonic()
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_bodies(bodies, bt_runtime)

            body = select_primary_body(bodies)
            if body is None:
                # No tracked body — send a "lost" marker so the client
                # knows to pause.
                server.broadcast({
                    "body_id": -1,
                    "confidence": 0.0,
                    "timestamp_ms": int(time.time() * 1000),
                    "keypoints": {},
                })
                continue

            data = body_to_dict(body)
            if data is not None:
                server.broadcast(data)

            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= 5.0:
                fps = frame_count / elapsed
                print(f"[streamer] {fps:.1f} fps, "
                      f"{len(server.clients)} client(s), "
                      f"body_id={body.id}")
                frame_count = 0
                fps_start = time.monotonic()

    except KeyboardInterrupt:
        print("\n[streamer] shutting down")
    finally:
        server.close()
        zed.disable_body_tracking()
        zed.close()


if __name__ == "__main__":
    main()
