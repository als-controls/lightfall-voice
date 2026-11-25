"""The ``tts`` service: a :class:`ServiceHost` streaming synthesized sentence audio."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading
from typing import Any
import uuid

from lightfall_utils.ipc.services import protocol as P
from lightfall_utils.ipc.services.host import ServiceError, ServiceHost, Session
from lightfall_voice.audio import float32_to_pcm16
from lightfall_voice.tts_service import models
from lightfall_voice.tts_service.synth import RATE, KokoroSynth, split_sentences

log = logging.getLogger(__name__)

P.ERROR_CODES = P.ERROR_CODES | frozenset({"not_ready", "bad_voice"})

NAME = "tts"
CONTRACT = "1.0"
KIND = "audio"
CHUNK_SAMPLES = 4800
CHUNK_BYTES = CHUNK_SAMPLES * 2
DEFAULT_VOICE = "af_heart"
DEFAULT_SPEED = 1.0
MIN_SPEED = 0.5
MAX_SPEED = 2.0
QUALITIES = ("int8", "fp16")
DEFAULT_CONFIG: dict[str, Any] = {
    "model_dir": "",
    "quality": "fp16",
    "lang": "en-us",
}
_RELOAD_KEYS = ("model_dir", "quality")
INITIAL_LOAD_DELAY_S = 2.0


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("lightfall-voice")
    except Exception:
        return "0.0.0"


@dataclass
class _Speech:
    stream_id: str
    session: Session
    sentences: list[str]
    voice: str
    speed: float
    lang: str
    synth: Any
    cancelled: threading.Event = field(default_factory=threading.Event)
    session_ended: threading.Event = field(default_factory=threading.Event)


class TtsHost(ServiceHost):
    """:class:`ServiceHost` that starts loading the model on start and shuts down its worker on stop."""

    on_started: Callable[[asyncio.AbstractEventLoop], None] | None = None
    on_stopping: Callable[[], None] | None = None
    on_stopped: Callable[[], Any] | None = None

    async def start(self, core: Any) -> None:
        await super().start(core)
        if self.on_started is not None:
            self.on_started(asyncio.get_running_loop())

    async def stop(self) -> None:
        if self.on_stopping is not None:
            self.on_stopping()
        try:
            await super().stop()
        finally:
            if self.on_stopped is not None:
                res = self.on_stopped()
                if asyncio.iscoroutine(res):
                    await res


def make_host(
    scope: str = "",
    *,
    synth_factory: Callable[[Path, Path], object] | None = None,
    resolve: Callable[..., tuple[Path, Path]] = models.resolve,
    initial_load_delay_s: float = INITIAL_LOAD_DELAY_S,
    **kw: Any,
) -> ServiceHost:
    """Build the tts :class:`ServiceHost` (not started); ``kw`` go to ``ServiceHost``."""
    factory = synth_factory or KokoroSynth
    kw.setdefault("contract", CONTRACT)
    kw.setdefault("version", _version())
    kw["auto_ready"] = False
    host = TtsHost(NAME, scope, **kw)

    cfg = dict(DEFAULT_CONFIG)
    lock = threading.Lock()  # serializes model loads
    st: dict[str, Any] = {
        "loop": None,
        "gen": 0,  # bumped per requested load; stale loads are discarded
        "synth": None,
        "initial": None,  # TimerHandle of the deferred first load, while pending
        "configured": False,  # a config.set arrived
        "stopping": False,
    }
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-synth")
    speeches: dict[str, _Speech] = {}

    def summary(status: str, error: str = "", *, model: str | None = None) -> dict[str, str]:
        return {
            "model": cfg["quality"] if model is None else model,
            "status": status,
            "error": error,
        }

    host.summary = summary("loading")

    # -- model loading (background thread) ----------------------------------------------

    def post(state: str | None, summ: dict, gen: int, synth: Any = None) -> None:
        """Apply a loader update on the loop thread if ``gen`` is still current."""

        def apply() -> None:
            if gen != st["gen"]:
                return
            if synth is not None:
                st["synth"] = synth
            host.set_state(state, summ)

        loop = st["loop"]
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(apply)
        except RuntimeError:
            pass  # loop closed: the host is gone

    def load(gen: int, conf: dict) -> None:
        with lock:
            if gen != st["gen"] or st["stopping"]:
                return
            quality = conf["quality"]

            def progress(status: str) -> None:
                post(None, summary(status, model=quality), gen)

            try:
                model_path, voices_path = resolve(
                    quality, conf["model_dir"], progress=progress
                )
                if gen != st["gen"] or st["stopping"]:
                    return
                synth = factory(model_path, voices_path)
            except Exception as e:
                log.exception("tts model load failed")
                err = str(e) or type(e).__name__
                post("error", summary("error", err, model=quality), gen)
                return
            log.info("tts model %s ready", quality)
            post("ready", summary("ready", model=quality), gen, synth)

    def start_load() -> None:
        """Begin a (re)load of the current config; call on the loop thread."""
        st["gen"] += 1
        st["synth"] = None
        host.set_state(None, summary("loading"))
        threading.Thread(
            target=load, args=(st["gen"], dict(cfg)), name="tts-model-load", daemon=True
        ).start()

    def initial_load() -> None:
        st["initial"] = None
        start_load()

    def on_started(loop: asyncio.AbstractEventLoop) -> None:
        st["loop"] = loop
        if st["configured"]:
            start_load()
        else:
            st["initial"] = loop.call_later(max(0.0, initial_load_delay_s), initial_load)

    def on_stopping() -> None:
        st["stopping"] = True
        pending = st["initial"]
        if pending is not None:
            pending.cancel()
            st["initial"] = None
        for job in list(speeches.values()):
            job.cancelled.set()

    async def on_stopped() -> None:
        await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    host.on_started = on_started
    host.on_stopping = on_stopping
    host.on_stopped = on_stopped

    # -- config --------------------------------------------------------------------------

    @host.rpc("config.get")
    async def config_get(session: Session, payload: dict) -> dict:
        return dict(cfg)

    @host.rpc("config.set")
    async def config_set(session: Session, payload: dict) -> dict:
        new = dict(cfg)
        for key in ("model_dir", "lang"):
            if key in payload:
                v = payload[key]
                if not isinstance(v, str):
                    raise ServiceError("bad_request", f"{key} must be a string")
                new[key] = v.strip()
        if not new["lang"]:
            raise ServiceError("bad_request", "lang must not be empty")
        if "quality" in payload:
            if payload["quality"] not in QUALITIES:
                raise ServiceError(
                    "bad_request", f"quality must be one of {list(QUALITIES)}"
                )
            new["quality"] = payload["quality"]
        reload = any(new[k] != cfg[k] for k in _RELOAD_KEYS)
        cfg.update(new)
        first = not st["configured"]
        st["configured"] = True
        pending = st["initial"]
        if pending is not None:
            pending.cancel()
            st["initial"] = None
            reload = True
            start_load()
        elif reload and st["loop"] is not None:
            start_load()
        elif first and st["loop"] is None:
            reload = True
        return {**cfg, "reloading": reload}

    # -- synthesis & streaming -----------------------------------------------------------

    async def emit_non_final(job: _Speech, seq: int, data: bytes) -> bool:
        if st["stopping"] or job.cancelled.is_set() or job.session_ended.is_set():
            return False
        await host.emit_chunk(
            job.session, job.stream_id, seq, data, final=False, kind=KIND
        )
        return True

    async def emit_final(
        job: _Speech, seq: int, error: tuple[str, str] | None = None
    ) -> None:
        speeches.pop(job.stream_id, None)
        if st["stopping"]:
            return
        await host.emit_chunk(
            job.session, job.stream_id, seq, b"", final=True, kind=KIND
        )
        if error is not None:
            code, message = error
            await host.emit(
                job.session,
                "speech.error",
                {"stream_id": job.stream_id, "code": code, "message": message},
            )

    def run_speech(job: _Speech) -> None:
        loop = st["loop"]
        if loop is None or st["stopping"]:
            speeches.pop(job.stream_id, None)
            return
        seq = 0
        error: tuple[str, str] | None = None
        try:
            for sentence in job.sentences:
                if st["stopping"] or job.cancelled.is_set() or job.session_ended.is_set():
                    break
                samples = job.synth.synthesize(
                    sentence, voice=job.voice, speed=job.speed, lang=job.lang
                )
                if st["stopping"] or job.cancelled.is_set() or job.session_ended.is_set():
                    break
                pcm = float32_to_pcm16(samples)
                for i in range(0, len(pcm), CHUNK_BYTES):
                    if st["stopping"] or job.cancelled.is_set() or job.session_ended.is_set():
                        break
                    chunk = pcm[i : i + CHUNK_BYTES]
                    sent = asyncio.run_coroutine_threadsafe(
                        emit_non_final(job, seq, chunk), loop
                    ).result()
                    if not sent:
                        break
                    seq += 1
        except Exception as e:
            if not job.cancelled.is_set() or job.session_ended.is_set():
                log.exception("speech synthesis failed")
                error = ("unknown", str(e) or type(e).__name__)
        if error is None and job.session_ended.is_set():
            error = ("session_ended", "session ended")
        try:
            asyncio.run_coroutine_threadsafe(
                emit_final(job, seq, error), loop
            ).result()
        except RuntimeError:
            speeches.pop(job.stream_id, None)

    @host.rpc("speak")
    async def speak(session: Session, payload: dict) -> dict:
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ServiceError("bad_request", "text must not be empty")
        sentences = split_sentences(text)
        if not sentences:
            raise ServiceError("bad_request", "text must not be empty")
        voice = payload.get("voice")
        if voice is None:
            voice = DEFAULT_VOICE
        if not isinstance(voice, str):
            raise ServiceError("bad_request", "voice must be a string")
        raw_speed = payload.get("speed", DEFAULT_SPEED)
        if raw_speed is None:
            raw_speed = DEFAULT_SPEED
        if (
            isinstance(raw_speed, bool)
            or not isinstance(raw_speed, (int, float))
            or not math.isfinite(raw_speed)
        ):
            raise ServiceError("bad_request", "speed must be a finite number")
        speed = min(MAX_SPEED, max(MIN_SPEED, float(raw_speed)))
        synth = st["synth"]
        if synth is None:
            raise ServiceError("not_ready", "model not ready")
        if voice not in synth.voices():
            raise ServiceError("bad_voice", f"unknown voice {voice!r}")

        sid = uuid.uuid4().hex
        job = _Speech(
            stream_id=sid,
            session=session,
            sentences=sentences,
            voice=voice,
            speed=speed,
            lang=cfg["lang"],
            synth=synth,
        )
        speeches[sid] = job
        loop = asyncio.get_running_loop()
        loop.call_soon(executor.submit, run_speech, job)
        return {
            "stream_id": sid,
            "rate": RATE,
            "format": "s16le",
            "channels": 1,
        }

    @host.rpc("cancel")
    async def cancel(session: Session, payload: dict) -> dict:
        sid = payload.get("stream_id")
        if isinstance(sid, str):
            job = speeches.get(sid)
            if job is not None and job.session.token == session.token:
                job.cancelled.set()
        return {}

    def on_session_end(session: Session) -> None:
        for job in list(speeches.values()):
            if job.session.token == session.token:
                job.session_ended.set()
                job.cancelled.set()

    host.on_session_end(on_session_end)
    return host
