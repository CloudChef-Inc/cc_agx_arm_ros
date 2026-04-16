"""GStreamer + NVENC WebRTC sender for the Nero webapp.

Replaces aiortc's pure-Python H.264 encoder with NVIDIA's hardware
encoder (nvv4l2h264enc) wrapped in GStreamer's webrtcbin. The browser-
facing wire protocol is identical (POST /offer with an SDP, get back
an SDP answer + the camera-name list), so no client-side change.

Per browser PC we build one Gst.Pipeline:

    appsrc (BGR ndarrays from a LatestFrame slot)
      → videoconvert (BGR → I420 in CPU memory; cheap)
      → nvvidconv (CPU → NVMM DMA, format=NV12)
      → nvv4l2h264enc (Tegra hardware H.264, ~0% CPU)
      → h264parse → rtph264pay
      → webrtcbin sink pad

…repeated per camera, all feeding the same webrtcbin so they share
one ICE connection (BUNDLE).

A small per-camera feeder thread reads frames from the LatestFrame
slot and pushes them into the appsrc — that's the only Python work
in the hot path. Encode + RTP packetisation + UDP send all happen
in native GStreamer threads with no GIL contention.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import List, Optional, Tuple

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import GLib, Gst, GstSdp, GstWebRTC  # noqa: E402

from fastapi import FastAPI, HTTPException, Request

from .cameras import LatestFrame

logger = logging.getLogger("nero_webapp.gstwebrtc")


# ---------- one-time process initialisation ------------------------------

def init_gstreamer() -> None:
    """Idempotent Gst.init wrapper."""
    if not Gst.is_initialized():
        Gst.init(None)


_glib_loop: Optional[GLib.MainLoop] = None
_glib_loop_thread: Optional[threading.Thread] = None


def start_glib_main_loop() -> None:
    """Start a background GLib main loop on a daemon thread.

    webrtcbin uses GLib timers + idle sources for some callbacks; a
    main loop must be running on at least one thread for those to
    fire. Idempotent.
    """
    global _glib_loop, _glib_loop_thread
    if _glib_loop is not None:
        return
    _glib_loop = GLib.MainLoop()
    _glib_loop_thread = threading.Thread(
        target=_glib_loop.run, daemon=True, name="glib-mainloop"
    )
    _glib_loop_thread.start()


# ---------- per-camera GStreamer pipeline string -------------------------

def camera_chain_desc(name: str, width: int, height: int, fps: int,
                      bitrate_kbps: int = 4000,
                      iframe_interval: int = 30) -> str:
    """GStreamer parse string for one camera's source→encode→pay chain.

    Inputs BGR raw video at (width, height, fps); outputs RTP H.264
    packets ready to feed into a webrtcbin sink pad. The encode runs
    on Tegra NVENC, so this chain consumes near-zero CPU.

    Bitrate is per-stream. 4 Mbps is comfortable for 720p H.264 baseline
    at 30fps. iframe_interval = 30 → keyframe every second.
    """
    return (
        # Live source — frames will arrive in real time, push-buffer
        # blocks if downstream is slow. format=time so the encoder gets
        # proper timestamps for rate control + RTP timestamping.
        f"appsrc name={name}_src is-live=true do-timestamp=true format=time "
        f"  caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        f"! queue max-size-buffers=2 leaky=downstream "
        # CPU colour-space convert from BGR to I420 (planar YUV).
        # videoconvert is fast for this — a few % of one core at most.
        f"! videoconvert ! video/x-raw,format=I420 "
        # DMA into NVMM so the hardware encoder can read directly from
        # GPU-accessible memory. Output format NV12 (semi-planar YUV)
        # is what nvv4l2h264enc expects.
        f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
        # NVENC. Property set kept minimal so it works across the
        # nvv4l2h264enc variants on different L4T releases — only
        # bitrate and iframeinterval are universally supported.
        f"! nvv4l2h264enc name={name}_enc "
        f"    iframeinterval={iframe_interval} bitrate={bitrate_kbps * 1000} "
        # Repeat SPS/PPS in-band so a late-joining decoder can sync.
        f"! h264parse config-interval=-1 "
        # RTP packetiser. zero-latency aggregation = send packets
        # immediately, don't bundle multiple frames.
        f"! rtph264pay name={name}_pay pt=96 config-interval=1 "
        f"    aggregate-mode=zero-latency "
        # Trailing queue. We DON'T add a trailing inline caps spec
        # here — gst.parse_bin_from_description rejects
        # `application/x-rtp,...` ('no element "application"'). The
        # caps that webrtcbin requires are set explicitly via
        # add-transceiver in WebRtcSession, and rtph264pay's native
        # output caps already include all the right fields.
        f"! queue"
    )


def _slot_dims(slot: LatestFrame, default: Tuple[int, int] = (1280, 720)
               ) -> Tuple[int, int]:
    """Return (width, height) of the slot's current frame, or default."""
    info = slot.info()
    shape = info.get("shape")
    if shape and len(shape) >= 2:
        h, w = int(shape[0]), int(shape[1])
        return w, h
    return default


# ---------- per-camera frame feeder -------------------------------------

class CameraFeeder:
    """Owns the appsrc element for one camera and the thread that
    pumps frames from a LatestFrame slot into it.

    The element itself is constructed by Gst.parse_launch as part of
    the session pipeline — we just hold a reference to it via
    pipeline.get_by_name() and drive push-buffer.
    """

    def __init__(self, name: str, slot: LatestFrame, appsrc: Gst.Element):
        self.name = name
        self.slot = slot
        self.appsrc = appsrc
        self._feeder_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._feeder_thread = threading.Thread(
            target=self._feed_loop, daemon=True, name=f"{self.name}-feeder",
        )
        self._feeder_thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.appsrc.emit("end-of-stream")
        except Exception:  # noqa: BLE001
            pass
        if self._feeder_thread is not None:
            self._feeder_thread.join(timeout=2.0)

    def _feed_loop(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            res = self.slot.wait_new(last_seq, 1.0)
            if res is None:
                continue
            arr, last_seq = res
            try:
                data = arr.tobytes()
            except Exception as e:  # noqa: BLE001
                logger.error("%s: tobytes failed: %s", self.name, e)
                continue
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            ret = self.appsrc.emit("push-buffer", buf)
            if isinstance(ret, Gst.FlowReturn) and ret != Gst.FlowReturn.OK:
                if ret in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.EOS):
                    logger.info("%s: feeder exiting (%s)",
                                self.name, ret.value_nick)
                    break
                logger.warning("%s: push-buffer returned %s",
                               self.name, ret.value_nick)


# ---------- per-browser session -----------------------------------------

class WebRtcSession:
    """One per browser PeerConnection. Owns a Gst.Pipeline + webrtcbin
    + one CameraFeeder per active camera.

    The pipeline is built in a SINGLE Gst.parse_launch call: gst-launch
    syntax `chain ! sender.` (trailing dot on the named webrtcbin
    element) is the canonical way to link multiple sources to webrtcbin.
    Auto-resolves the sink_%u request pad request that the explicit
    Python API has trouble with on this binding.
    """

    def __init__(self, slots_in_order: List[Tuple[str, LatestFrame]],
                 fps: int = 30):
        self.id = id(self)
        self.pipeline = Gst.Pipeline.new(f"session_{self.id}")

        # Build webrtcbin. webrtcbin doesn't fire pad-added for
        # sender-side sink pads (those are request pads, not signal-
        # auto-created), so we request them explicitly with our RTP
        # caps. This both (a) gives us the pad to link, and (b)
        # registers a sendonly transceiver with the right codec.
        self.webrtcbin = Gst.ElementFactory.make("webrtcbin", "sender")
        if self.webrtcbin is None:
            raise RuntimeError("webrtcbin element not available")
        self.webrtcbin.set_property(
            "bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE,
        )
        self.webrtcbin.set_property(
            "stun-server", "stun://stun.l.google.com:19302",
        )
        if not self.pipeline.add(self.webrtcbin):
            raise RuntimeError("could not add webrtcbin to pipeline")

        # Look up the sink_%u pad template once so we can request
        # pads from it with caps.
        sink_template = self.webrtcbin.get_pad_template("sink_%u")
        if sink_template is None:
            raise RuntimeError("webrtcbin has no sink_%u pad template")

        # Don't pin a specific payload type — browsers use different
        # PT numbers for H.264 (Chrome 102, Firefox 96, etc.) and
        # webrtcbin's create-answer needs to find a match against the
        # offer. clock-rate is the only mandatory non-codec field.
        rtp_caps = Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000"
        )

        self.sources: List[CameraFeeder] = []
        self.camera_names: List[str] = []

        for name, slot in slots_in_order:
            if slot is None:
                continue
            w, h = _slot_dims(slot)
            chain_desc = camera_chain_desc(name, w, h, fps)
            try:
                chain_bin = Gst.parse_bin_from_description(chain_desc, True)
            except GLib.GError as e:
                raise RuntimeError(f"chain {name} parse failed: {e.message}")
            chain_bin.set_property("name", f"{name}_bin")
            if not self.pipeline.add(chain_bin):
                raise RuntimeError(f"could not add {name} bin to pipeline")

            # Request a sink pad on webrtcbin with our exact RTP
            # caps — the modern Gst.Element.request_pad(template,
            # name, caps) signature. Passing the caps tells
            # webrtcbin to set up a sender transceiver matching them
            # and gives us back the pad in one call.
            sink_pad = self.webrtcbin.request_pad(
                sink_template, None, rtp_caps,
            )
            if sink_pad is None:
                raise RuntimeError(
                    f"{name}: webrtcbin.request_pad returned None")

            src_pad = chain_bin.get_static_pad("src")
            if src_pad is None:
                raise RuntimeError(f"{name}: no ghost src pad on chain bin")
            link_ret = src_pad.link(sink_pad)
            if link_ret != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"{name}: chain → webrtcbin link returned {link_ret}")
            logger.info("session %s: linked %s → webrtcbin/%s",
                        self.id, name, sink_pad.get_name())

            appsrc = chain_bin.get_by_name(f"{name}_src")
            if appsrc is None:
                raise RuntimeError(f"appsrc {name}_src missing from chain bin")
            self.sources.append(CameraFeeder(name, slot, appsrc))
            self.camera_names.append(name)

        # Threading.Event signalled when ICE gathering reaches COMPLETE
        # so negotiate() can return the final SDP with all candidates.
        self._ice_done = threading.Event()
        self.webrtcbin.connect(
            "notify::ice-gathering-state", self._on_ice_gathering_state,
        )
        # Threading.Event signalled when the WebRTC connection enters
        # a terminal state, so the FastAPI watcher can tear us down.
        self.connection_closed = threading.Event()
        self.webrtcbin.connect(
            "notify::connection-state", self._on_connection_state,
        )
        # Subscribe to the pipeline bus so element errors surface in
        # logs instead of vanishing into GStreamer's void.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def _on_ice_gathering_state(self, _wrtc, _pspec):
        state = self.webrtcbin.get_property("ice-gathering-state")
        logger.debug("session %s ice-gathering-state=%s",
                     self.id, state.value_nick)
        if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
            self._ice_done.set()

    def _on_connection_state(self, _wrtc, _pspec):
        state = self.webrtcbin.get_property("connection-state")
        logger.info("session %s connection-state=%s",
                    self.id, state.value_nick)
        if state in (
            GstWebRTC.WebRTCPeerConnectionState.DISCONNECTED,
            GstWebRTC.WebRTCPeerConnectionState.FAILED,
            GstWebRTC.WebRTCPeerConnectionState.CLOSED,
        ):
            self.connection_closed.set()

    def _on_bus_message(self, _bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logger.error("session %s pipeline ERROR from %s: %s | %s",
                         self.id, msg.src.get_name() if msg.src else "?",
                         err.message, debug)
            self.connection_closed.set()
        elif t == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            logger.warning("session %s pipeline WARNING from %s: %s | %s",
                           self.id, msg.src.get_name() if msg.src else "?",
                           err.message, debug)
        elif t == Gst.MessageType.EOS:
            logger.info("session %s pipeline EOS", self.id)
            self.connection_closed.set()

    def negotiate(self, offer_sdp_text: str) -> str:
        """Set the browser's offer, create an answer, wait for ICE
        gathering to complete, and return the answer SDP text.

        Blocking — call from a worker thread, not the asyncio loop.
        """
        ret, sdp_msg = GstSdp.SDPMessage.new()
        if ret != GstSdp.SDPResult.OK:
            raise RuntimeError("SDPMessage.new failed")
        ret = GstSdp.sdp_message_parse_buffer(
            offer_sdp_text.encode(), sdp_msg
        )
        if ret != GstSdp.SDPResult.OK:
            raise RuntimeError("offer SDP parse failed")
        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER, sdp_msg
        )

        # Log the offer SDP at INFO so we can see what codecs / PTs /
        # profile-level-ids the browser advertises if create-answer
        # fails downstream. Helpful for debugging codec-mismatch.
        logger.info("session %s offer SDP:\n%s", self.id, offer_sdp_text)

        # 1. set remote description (offer)
        promise = Gst.Promise.new()
        self.webrtcbin.emit("set-remote-description", offer, promise)
        promise.wait()

        # 2. create answer + set local description BEFORE starting
        #    feeders. If feeders push data while create-answer is
        #    in flight, downstream errors / EOS on the chains can
        #    cause webrtcbin to abandon answer creation (we'd see
        #    "create-answer reply missing 'answer' field"). Doing
        #    SDP work on the linked-but-idle pipeline gives webrtcbin
        #    a clean state to negotiate against.
        promise = Gst.Promise.new()
        self.webrtcbin.emit("create-answer", None, promise)
        promise.wait()
        reply = promise.get_reply()
        if reply is None:
            raise RuntimeError("create-answer returned no reply")
        answer = reply.get_value("answer")
        if answer is None:
            raise RuntimeError("create-answer reply missing 'answer' field")

        promise = Gst.Promise.new()
        self.webrtcbin.emit("set-local-description", answer, promise)
        promise.wait()

        # 3. NOW move the pipeline to PLAYING and start the feeders.
        #    set-local-description triggers ICE gathering — for that
        #    to complete the pipeline needs to be active.
        rc = self.pipeline.set_state(Gst.State.PLAYING)
        if rc == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline failed to enter PLAYING")
        for src in self.sources:
            src.start()

        # 4. wait for ICE gathering (vanilla ICE; no trickle to keep
        #    the wire protocol simple — the answer SDP we return
        #    includes all candidates).
        if not self._ice_done.wait(timeout=8.0):
            logger.warning("session %s: ICE gathering timed out at 8s, "
                           "returning SDP with whatever candidates we have",
                           self.id)

        local = self.webrtcbin.get_property("local-description")
        return local.sdp.as_text()

    def stop(self) -> None:
        for src in self.sources:
            src.stop()
        try:
            self.pipeline.set_state(Gst.State.NULL)
        except Exception:  # noqa: BLE001
            pass


# ---------- FastAPI signaling endpoint ----------------------------------

def setup_webrtc_routes(
    app: FastAPI,
    slots_in_order: List[Tuple[str, "LatestFrame | None"]],
) -> None:
    """Mount POST /offer + lifecycle hooks on the FastAPI app.

    slots_in_order is the camera display order, identical contract to
    the previous aiortc backend — slots whose value is None are
    skipped, the ones that remain define the m-line order in the
    answer SDP and the `cameras` list returned to the browser.
    """
    init_gstreamer()
    start_glib_main_loop()

    sessions: "set[WebRtcSession]" = set()

    @app.post("/offer")
    async def offer(request: Request):
        body = await request.json()
        if "sdp" not in body or "type" not in body:
            raise HTTPException(400, "missing sdp/type")
        if body["type"] != "offer" or not body["sdp"].strip():
            raise HTTPException(400, "sdp must be a non-empty offer")

        loop = asyncio.get_event_loop()

        def _build_and_negotiate():
            session = WebRtcSession(slots_in_order)
            try:
                answer_sdp = session.negotiate(body["sdp"])
                return answer_sdp, list(session.camera_names), session
            except Exception:
                session.stop()
                raise

        try:
            answer_sdp, cameras, session = await loop.run_in_executor(
                None, _build_and_negotiate
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("WebRTC negotiation failed")
            raise HTTPException(500, f"negotiation failed: {e}")

        sessions.add(session)
        logger.info("session %s negotiated, cameras=%s",
                    session.id, cameras)

        # Watcher coroutine: poll the connection_closed event (which
        # the GObject signal sets) and tear down on disconnect.
        async def _watch_disconnect(s: WebRtcSession):
            while not s.connection_closed.is_set():
                await asyncio.sleep(1.0)
            await loop.run_in_executor(None, s.stop)
            sessions.discard(s)
            logger.info("session %s torn down", s.id)
        asyncio.create_task(_watch_disconnect(session))

        return {"sdp": answer_sdp, "type": "answer", "cameras": cameras}

    @app.on_event("shutdown")
    async def cleanup():
        loop = asyncio.get_event_loop()
        for s in list(sessions):
            await loop.run_in_executor(None, s.stop)
        sessions.clear()
