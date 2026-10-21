"""The ``stt`` service: a :class:`ServiceHost` turning audio streams into transcripts.

Run it with ``python -m lightfall_voice.stt_service --scope S --nats-url URL --token-stdin``
(Lightfall launches it locally).  Contract ``1.0``:

- rpc ``config.get`` -> the config; ``config.set {model, model_dir, device, language,
  silence_s}`` (any subset) -> the new config.  A change of ``model``, ``model_dir`` or
  ``device`` reloads the model in the background; ``language`` and ``silence_s`` apply at
  once.
- rpc ``utterance.start {rate, format, channels}`` -> ``{stream_id}`` (16000 / s16le / 1);
  ``busy`` while another utterance is active, ``unknown`` "model not ready" before the model
  is loaded.  The client adopts the id and sends kind ``audio`` chunks.
- rpc ``utterance.stop {stream_id}`` finalizes now; ``utterance.cancel {stream_id}`` discards.
- Per-session events: ``utterance.listening {stream_id}``, ``utterance.ended {stream_id,
  reason}`` (``silence|stop|max|no_speech|cancel``), ``utterance.transcript {stream_id,
  text}`` (``""`` when nothing usable was heard) and ``utterance.error {stream_id, code,
  message}``.  ``ended`` is sent before transcribing, so the client can show
  "Transcribing…"; ``no_speech`` and ``cancel`` get no transcript.
- Heartbeat summary ``{model, device, status: loading|downloading|ready|error, error}``.
  The host state stays ``starting`` until the first model load ends (``ready`` or
  ``error``).

Test-only: the environment variable ``LIGHTFALL_STT_FAKE="<text>"`` replaces the real
transcriber with a fake one that returns ``<text>`` for every utterance (used by the
subprocess tests; never set it in production).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lightfall_utils.ipc.services.host import ServiceError, ServiceHost, Session
from lightfall_voice.audio import RATE, pcm16_to_float32
from lightfall_voice.stt_service import filters
from lightfall_voice.stt_service.endpoint import EndpointDetector
from lightfall_voice.stt_service.transcriber import DEVICES, Transcriber

log = logging.getLogger(__name__)

NAME = "stt"
CONTRACT = "1.0"
INSTALL_HINT = "Install lightfall-voice[stt-service]"
FAKE_ENV = "LIGHTFALL_STT_FAKE"
KIND = "audio"
DEFAULT_CONFIG: dict[str, Any] = {
    "model": "small",
    "model_dir": "",
    "device": "auto",
    "language": "en",
    "silence_s": 1.0,
}
_RELOAD_KEYS = ("model", "model_dir", "device")
_END_REASONS = ("silence", "no_speech", "max")
_MISSING_MODULES = ("faster_whisper", "ctranslate2")
_SUMMARY_DEVICES = ("cuda", "cpu", "")

TranscriberFactory = Callable[..., Transcriber]


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("lightfall-voice")
    except Exception:
        return "0.0.0"


def default_factory(
    *,
    model: str,
    device: str,
    model_dir: str,
    language: str,
    progress: Callable[[str], None],
) -> Transcriber:
    """Build and load a :class:`FasterWhisperTranscriber` (blocking)."""
    from lightfall_voice.stt_service.transcriber import FasterWhisperTranscriber

    t = FasterWhisperTranscriber(model, device, model_dir, language, progress)
    t.load()
    return t


class FakeTranscriber:
    """Test-only transcriber returning fixed text (see ``LIGHTFALL_STT_FAKE``)."""

    device = "cpu"

    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio_16k: Any, language: str | None) -> str:
        return self.text


def fake_factory(text: str) -> TranscriberFactory:
    def make(**cfg: Any) -> Transcriber:
        cfg["progress"]("loading")
        return FakeTranscriber(text)

    return make


@dataclass
class _Utterance:
    stream_id: str
    session: Session
    detector: EndpointDetector
    transcriber: Any  # the model current at utterance.start (survives a reload)
    listening: bool = False
    reason: str | None = None  # set once the utterance is ending
    detector_done: bool = False  # the detector ended it: later chunks are not fed


class SttHost(ServiceHost):
    """:class:`ServiceHost` that starts loading the model in the background on start."""

    on_started: Callable[[asyncio.AbstractEventLoop], None] | None = None

    async def start(self, core: Any) -> None:
        await super().start(core)
        if self.on_started is not None:
            self.on_started(asyncio.get_running_loop())


def make_host(
    scope: str = "", *, transcriber_factory: TranscriberFactory | None = None, **kw: Any
) -> ServiceHost:
    """Build the stt :class:`ServiceHost` (not started); ``kw`` go to ``ServiceHost``.

    ``transcriber_factory(model=, device=, model_dir=, language=, progress=)`` returns a
    loaded :class:`Transcriber` (it runs on a background thread and may block).
    """
    if transcriber_factory is None:
        fake = os.environ.get(FAKE_ENV)
        transcriber_factory = (
            fake_factory(fake) if fake is not None else default_factory
        )
    factory = transcriber_factory
    kw.setdefault("contract", CONTRACT)
    kw.setdefault("version", _version())
    kw["auto_ready"] = False
    host = SttHost(NAME, scope, **kw)

    cfg = dict(DEFAULT_CONFIG)
    lock = threading.Lock()  # serializes model loads
    st: dict[str, Any] = {
        "loop": None,
        "gen": 0,  # bumped per requested load; stale loads are discarded
        "transcriber": None,
        "active": None,  # stream id of the active utterance
    }
    utterances: dict[str, _Utterance] = {}

    def summary(status: str, device: str = "", error: str = "") -> dict:
        return {
            "model": cfg["model"],
            "device": device,
            "status": status,
            "error": error,
        }

    host.summary = summary("loading")

    # -- model loading (background thread) ----------------------------------------------

    def post(state: str | None, summ: dict, gen: int, transcriber: Any = None) -> None:
        """Apply a loader update on the loop thread if ``gen`` is still current."""

        def apply() -> None:
            if gen != st["gen"]:
                return
            if transcriber is not None:
                st["transcriber"] = transcriber
            host.set_state(state, summ)

        loop = st["loop"]
        try:
            loop.call_soon_threadsafe(apply)
        except RuntimeError:
            pass  # loop closed: the host is gone

    def load(gen: int, conf: dict) -> None:
        with lock:
            if gen != st["gen"]:
                return
            model = conf["model"]

            def progress(status: str) -> None:
                post(None, {**summary(status), "model": model}, gen)

            try:
                t = factory(
                    model=model,
                    device=conf["device"],
                    model_dir=conf["model_dir"],
                    language=conf["language"],
                    progress=progress,
                )
            except ModuleNotFoundError as e:
                if (e.name or "").split(".")[0] in _MISSING_MODULES:
                    err = INSTALL_HINT
                else:
                    err = f"{type(e).__name__}: {e}"
                log.error("speech model load failed: %s", err)
                post("error", {**summary("error", error=err), "model": model}, gen)
                return
            except Exception as e:
                log.exception("speech model load failed")
                err = str(e) or type(e).__name__
                post("error", {**summary("error", error=err), "model": model}, gen)
                return
            device = getattr(t, "device", "")
            if device not in _SUMMARY_DEVICES:
                log.warning("transcriber reported device %r; summary shows ''", device)
                device = ""
            log.info("speech model %s ready on %s", model, device or "?")
            post("ready", {**summary("ready", device), "model": model}, gen, t)

    def start_load() -> None:
        """Begin a (re)load of the current config; call on the loop thread."""
        st["gen"] += 1
        st["transcriber"] = None  # free the old model before loading the next
        host.set_state(None, summary("loading"))
        threading.Thread(
            target=load, args=(st["gen"], dict(cfg)), name="stt-model-load", daemon=True
        ).start()

    def on_started(loop: asyncio.AbstractEventLoop) -> None:
        st["loop"] = loop
        start_load()

    host.on_started = on_started

    # -- config --------------------------------------------------------------------------

    @host.rpc("config.get")
    async def config_get(session: Session, payload: dict) -> dict:
        return dict(cfg)

    @host.rpc("config.set")
    async def config_set(session: Session, payload: dict) -> dict:
        new = dict(cfg)
        for key in ("model", "model_dir", "language"):
            if key in payload:
                v = payload[key]
                if not isinstance(v, str):
                    raise ServiceError("bad_request", f"{key} must be a string")
                new[key] = v.strip()
        if not new["model"]:
            raise ServiceError("bad_request", "model must not be empty")
        if "device" in payload:
            if payload["device"] not in DEVICES:
                raise ServiceError(
                    "bad_request", f"device must be one of {list(DEVICES)}"
                )
            new["device"] = payload["device"]
        if "silence_s" in payload:
            v = payload["silence_s"]
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0.1 <= v <= 10.0
            ):
                raise ServiceError(
                    "bad_request", "silence_s must be a number in [0.1, 10]"
                )
            new["silence_s"] = float(v)
        reload = any(new[k] != cfg[k] for k in _RELOAD_KEYS)
        cfg.update(new)
        for u in utterances.values():
            u.detector.silence_s = cfg["silence_s"]
        if reload and st["loop"] is not None:
            start_load()
        return {**cfg, "reloading": reload}

    # -- utterances ----------------------------------------------------------------------

    def own(session: Session, payload: dict) -> _Utterance:
        sid = payload.get("stream_id")
        u = utterances.get(sid) if isinstance(sid, str) else None
        if u is None or u.session.token != session.token:
            raise ServiceError("bad_request", f"unknown utterance {sid!r}")
        return u

    def finish(sid: str) -> None:
        utterances.pop(sid, None)
        if st["active"] == sid:
            st["active"] = None

    @host.rpc("utterance.start")
    async def utterance_start(session: Session, payload: dict) -> dict:
        if st["active"] is not None:
            raise ServiceError("busy", "an utterance is already active")
        if st["transcriber"] is None:
            raise ServiceError("unknown", "model not ready")
        rate = payload.get("rate", RATE)
        fmt = payload.get("format", "s16le")
        channels = payload.get("channels", 1)
        if rate != RATE or fmt != "s16le" or channels != 1:
            raise ServiceError(
                "bad_request",
                f"audio must be {RATE} Hz s16le mono, got {rate}/{fmt}/{channels}",
            )
        sid = host.open_stream(
            session, KIND, {"rate": rate, "format": fmt, "channels": channels}
        )
        st["active"] = sid
        utterances[sid] = _Utterance(
            sid,
            session,
            EndpointDetector(silence_s=cfg["silence_s"]),
            st["transcriber"],
        )
        return {"stream_id": sid}

    @host.rpc("utterance.stop")
    async def utterance_stop(session: Session, payload: dict) -> dict:
        u = own(session, payload)
        if u.reason is None:
            u.reason = "stop"
            await host.end_stream(u.stream_id, final=True, reason="stop")
        return {}

    @host.rpc("utterance.cancel")
    async def utterance_cancel(session: Session, payload: dict) -> dict:
        u = own(session, payload)
        if u.reason is not None:
            return {}  # already ending: its ended event is on its way
        u.reason = "cancel"
        finish(u.stream_id)
        await host.end_stream(u.stream_id)
        await host.emit(
            session, "utterance.ended", {"stream_id": u.stream_id, "reason": "cancel"}
        )
        return {}

    def refuse_client_open(session: Session, params: dict) -> None:
        raise ServiceError("bad_request", "open audio streams with utterance.start")

    @host.stream(KIND, on_open=refuse_client_open)
    async def on_audio(
        session: Session, sid: str, seq: int, data: bytes, final: bool
    ) -> None:
        u = utterances.get(sid)
        if u is None:
            return
        # After utterance.stop the chunks already queued (and a final chunk's data) are
        # still fed, so the last word is kept; only the detector's own end stops feeding.
        if data and not u.detector_done:
            ev = u.detector.feed(pcm16_to_float32(data))
            if ev == "listening" and not u.listening:
                u.listening = True
                await host.emit(session, "utterance.listening", {"stream_id": sid})
            elif ev in _END_REASONS and u.reason is None:
                u.reason = ev
                u.detector_done = True
                await host.end_stream(sid, final=True, reason=ev)
        if final:
            await finalize(u)

    async def finalize(u: _Utterance) -> None:
        sid, session = u.stream_id, u.session
        reason = u.reason or "stop"  # a client final chunk / stream.close is a stop
        u.reason = reason
        try:
            await host.emit(
                session, "utterance.ended", {"stream_id": sid, "reason": reason}
            )
            if reason == "no_speech":
                return
            transcriber = u.transcriber
            if transcriber is None:
                await host.emit(
                    session,
                    "utterance.error",
                    {"stream_id": sid, "code": "unknown", "message": "model not ready"},
                )
                return
            audio = u.detector.audio()
            loop = asyncio.get_running_loop()
            try:
                raw = await loop.run_in_executor(
                    None, transcriber.transcribe, audio, cfg["language"] or None
                )
                text = filters.clean(raw or "")
            except Exception as e:
                log.exception("transcription failed")
                await host.emit(
                    session,
                    "utterance.error",
                    {
                        "stream_id": sid,
                        "code": "unknown",
                        "message": str(e) or type(e).__name__,
                    },
                )
                return
            await host.emit(
                session, "utterance.transcript", {"stream_id": sid, "text": text}
            )
        finally:
            finish(sid)

    def on_stream_end(session: Session, sid: str, reason: str) -> None:
        # final/close went through finalize(); cancel through utterance.cancel.  Anything
        # else (idle, session_end, stop) just frees the slot.
        if sid in utterances:
            log.info("utterance %s… dropped (%s)", sid[:6], reason)
            finish(sid)

    host.on_stream_end(on_stream_end)
    return host
