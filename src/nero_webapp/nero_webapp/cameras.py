"""Camera capture workers for the Nero webapp.

Each camera runs its own blocking capture thread that reads frames from
the physical device and publishes them into a single-slot "latest frame"
holder. Stale frames are dropped on purpose — in a real-time teleop loop,
old frames are pure latency.

Frame format contract: every LatestFrame carries a BGR24 numpy array
ready for av.VideoFrame.from_ndarray(arr, format="bgr24"). RGB sources
(RealSense colorizer) are swapped and copied into a contiguous array
before publishing.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("nero_webapp.cameras")


def _stamp(frame: np.ndarray, label: str) -> np.ndarray:
    """Burn a timestamp overlay into the frame (top-left corner).

    Used for visual latency checks — compare the displayed stamp to the
    browser's matching wall clock. Skip the overlay by passing label="".
    """
    if not label:
        return frame
    # OpenCV is imported lazily so the package doesn't fail to import
    # on systems that only use RealSense / no fisheye (etc).
    import cv2
    if not frame.flags.writeable:
        frame = frame.copy()
    t = time.time()
    text = (
        f"{label} "
        f"{time.strftime('%H:%M:%S', time.localtime(t))}."
        f"{int((t % 1) * 1000):03d}"
    )
    pos = (8, 26)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class LatestFrame:
    """Thread-safe single-slot holder with monotonic sequence numbers.

    Producers call put(); consumers call wait_new(last_seq) which blocks
    until a strictly newer frame exists and returns (frame, seq). Frames
    between consumer polls are silently dropped.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._closed = False
        # Rolling window of put() timestamps — used to report capture fps.
        self._put_times: collections.deque = collections.deque(maxlen=120)

    def put(self, frame: np.ndarray) -> None:
        with self._cv:
            self._frame = frame
            self._seq += 1
            self._put_times.append(time.monotonic())
            self._cv.notify_all()

    def fps(self) -> float:
        with self._cv:
            if len(self._put_times) < 2:
                return 0.0
            dt = self._put_times[-1] - self._put_times[0]
            if dt <= 0:
                return 0.0
            return (len(self._put_times) - 1) / dt

    def info(self) -> dict:
        with self._cv:
            shape = list(self._frame.shape) if self._frame is not None else None
            return {"seq": self._seq, "fps": round(self.fps(), 2), "shape": shape}

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def wait_new(self, last_seq: int, timeout: float = 1.0
                 ) -> Optional[Tuple[np.ndarray, int]]:
        deadline = time.monotonic() + timeout
        with self._cv:
            while not self._closed and self._seq <= last_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            if self._closed or self._frame is None:
                return None
            return self._frame, self._seq


# ---------- Fisheye camera (UVC / V4L2 via OpenCV) -----------------------

class FisheyeCamera:
    """UVC fisheye webcam capture.

    Negotiates MJPEG from V4L2 so we don't pay for full-resolution YUYV
    DMA. OpenCV decodes MJPEG → BGR24 on each read().
    """

    def __init__(self, device_path: str, width: int, height: int,
                 fps: int, overlay_label: str = "fisheye") -> None:
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.overlay_label = overlay_label
        self.frames = LatestFrame()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_ok = threading.Event()

    def start(self) -> bool:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fisheye-capture",
        )
        self._thread.start()
        # Wait briefly for device open so caller can decide to expose
        # the track or not. If we haven't managed to open within 3s,
        # assume it failed.
        return self._started_ok.wait(3.0)

    def stop(self) -> None:
        self._stop.set()
        self.frames.close()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import cv2
        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        # Shrink kernel-side buffer so reads stay on the live frame.
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            logger.error("fisheye: failed to open %s", self.device_path)
            return
        logger.info("fisheye: %s opened at %dx%d@%dfps MJPG",
                    self.device_path, self.width, self.height, self.fps)
        self._started_ok.set()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                frame = _stamp(frame, self.overlay_label)
                self.frames.put(frame)
        finally:
            cap.release()


# ---------- RealSense D405 (color + colorized depth) ---------------------

class RealSenseCamera:
    """Intel RealSense pipeline producing BGR color + BGR colorized depth.

    Two independent LatestFrame slots (self.color, self.depth). Depth is
    NOT aligned to color — alignment costs ~5ms per frame and a second
    GPU/CPU roundtrip; for separate video tracks it's unnecessary.
    """

    def __init__(self, serial: str, color_w: int, color_h: int,
                 depth_w: int, depth_h: int, fps: int,
                 color_label: str = "color", depth_label: str = "depth") -> None:
        self.serial = serial
        self.color_w = color_w
        self.color_h = color_h
        self.depth_w = depth_w
        self.depth_h = depth_h
        self.fps = fps
        self.color_label = color_label
        self.depth_label = depth_label
        self.color = LatestFrame()
        self.depth = LatestFrame()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_ok = threading.Event()

    def start(self) -> bool:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="realsense-capture",
        )
        self._thread.start()
        # RealSense pipeline start typically takes ~1s; give it 5s.
        return self._started_ok.wait(5.0)

    def stop(self) -> None:
        self._stop.set()
        self.color.close()
        self.depth.close()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color,
                             self.color_w, self.color_h,
                             rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth,
                             self.depth_w, self.depth_h,
                             rs.format.z16, self.fps)

        colorizer = rs.colorizer()
        # color_scheme 0 = Jet (bright, high-contrast, good for humans).
        colorizer.set_option(rs.option.color_scheme, 0)

        try:
            pipeline.start(config)
        except Exception as e:  # noqa: BLE001
            logger.error("realsense: pipeline start failed: %s", e)
            return
        logger.info(
            "realsense: started (color %dx%d bgr8, depth %dx%d z16 @ %dfps)",
            self.color_w, self.color_h,
            self.depth_w, self.depth_h, self.fps,
        )
        self._started_ok.set()
        try:
            while not self._stop.is_set():
                try:
                    frames = pipeline.wait_for_frames(1000)
                except RuntimeError:
                    continue
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if color:
                    arr = np.asanyarray(color.get_data())
                    # RealSense frames are read-only views; copy for overlay.
                    arr = arr.copy()
                    arr = _stamp(arr, self.color_label)
                    self.color.put(arr)
                if depth:
                    colorized = colorizer.colorize(depth)
                    rgb = np.asanyarray(colorized.get_data())
                    # Colorizer emits RGB; swap to BGR and force a
                    # contiguous copy so PyAV doesn't choke on the stride.
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    bgr = _stamp(bgr, self.depth_label)
                    self.depth.put(bgr)
        finally:
            pipeline.stop()
