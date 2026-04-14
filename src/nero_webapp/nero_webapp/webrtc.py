"""WebRTC signaling + camera tracks for the Nero webapp.

One RTCPeerConnection per browser client. Each PC gets its own set of
CameraStreamTracks (one per available camera); the tracks share the
underlying LatestFrame holders so the capture threads don't care how
many viewers are watching.

Signaling is a single POST /offer: browser sends SDP offer, server
returns SDP answer + the list of camera names in transceiver order
so the client can label incoming tracks.
"""
from __future__ import annotations

import asyncio
import fractions
import logging
import time
from typing import Dict, List, Optional, Tuple

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from fastapi import FastAPI, HTTPException, Request

from .cameras import LatestFrame

logger = logging.getLogger("nero_webapp.webrtc")


class CameraStreamTrack(VideoStreamTrack):
    """aiortc VideoStreamTrack that pulls frames from a LatestFrame slot.

    recv() blocks via executor thread on the slot's condition variable
    until a newer frame is published. pts is assigned from wall time so
    the RTP clock tracks real capture rate instead of a fixed nominal.
    """

    kind = "video"

    def __init__(self, slot: LatestFrame, label: str) -> None:
        super().__init__()
        self._slot = slot
        self._label = label
        self._last_seq = 0
        self._t0: Optional[float] = None
        self._black_filler: Optional[np.ndarray] = None

    async def recv(self) -> av.VideoFrame:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._slot.wait_new, self._last_seq, 2.0,
        )
        if result is None:
            # Timeout or slot closed. Emit a black frame so the track
            # stays alive; raising here tears down the whole PC.
            if self._black_filler is None:
                self._black_filler = np.zeros((16, 16, 3), dtype=np.uint8)
            arr = self._black_filler
            self._last_seq += 1
        else:
            arr, seq = result
            self._last_seq = seq
        if self._t0 is None:
            self._t0 = time.monotonic()
        pts = int((time.monotonic() - self._t0) * 90000)
        vf = av.VideoFrame.from_ndarray(arr, format="bgr24")
        vf.pts = pts
        vf.time_base = fractions.Fraction(1, 90000)
        return vf


def setup_webrtc_routes(
    app: FastAPI,
    slots_in_order: List[Tuple[str, LatestFrame]],
) -> None:
    """Mount POST /offer.

    slots_in_order is an ordered list of (name, LatestFrame). Tracks are
    added to each new PC in exactly this order — the browser pairs its
    pc.getTransceivers() with the `cameras` list returned in the answer
    to figure out which video element each mid feeds.
    """
    pcs: "set[RTCPeerConnection]" = set()

    @app.post("/offer")
    async def offer(request: Request):
        body = await request.json()
        if "sdp" not in body or "type" not in body:
            raise HTTPException(400, "missing sdp/type")
        sdp = body["sdp"]
        sdp_type = body["type"]
        if not isinstance(sdp, str) or not sdp.strip() or sdp_type != "offer":
            raise HTTPException(400, "sdp must be a non-empty offer")
        remote = RTCSessionDescription(sdp=sdp, type=sdp_type)

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_state() -> None:  # noqa: D401
            logger.info("pc=%s connection=%s", id(pc), pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                try:
                    await pc.close()
                finally:
                    pcs.discard(pc)

        camera_order: List[str] = []
        for name, slot in slots_in_order:
            if slot is None:
                continue
            track = CameraStreamTrack(slot, label=name)
            pc.addTrack(track)
            camera_order.append(name)

        await pc.setRemoteDescription(remote)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "cameras": camera_order,
        }

    @app.on_event("shutdown")
    async def close_peers() -> None:
        await asyncio.gather(*(pc.close() for pc in list(pcs)),
                             return_exceptions=True)
        pcs.clear()
