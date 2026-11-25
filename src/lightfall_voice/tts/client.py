"""TtsClient: wires SpeechScheduler, the ``tts`` QtServiceClient and Player."""

from __future__ import annotations

import concurrent.futures
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import shiboken6
from PySide6.QtCore import QObject, QThread, Signal

from lightfall.utils.logging import logger
from lightfall_utils.threads import initialize_main_thread_invoker, invoke_in_main_thread

from lightfall_voice.tts.player import Player
from lightfall_voice.tts.scheduler import Priority, SpeechItem, SpeechScheduler

State = Literal["unavailable", "loading", "ready"]

SERVICE = "tts"
EVENTS = ("speech.error",)
DEFAULT_VOICE = "af_heart"
DEFAULT_RATE = 24000
DEFAULT_CHANNELS = 1
MAX_EARLY_CHUNKS = 64

_UNAVAILABLE_HEALTH = frozenset({"down", "incompatible", "unauthorized"})


@dataclass
class OutputSettings:
    """The ``voice.output.*`` settings used by :class:`TtsClient`."""

    enabled: bool = True
    speed: float = 1.0
    quality: str = "fp16"
    model_dir: str = ""
    output_device: str = ""
    default_voice: str = DEFAULT_VOICE
    python: str = ""

    def service_config(self) -> dict[str, Any]:
        """The ``config.set`` payload for the ``tts`` service."""
        return {
            "model_dir": self.model_dir,
            "quality": self.quality,
            "lang": "en-us",
        }


def _alive(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        return shiboken6.isValid(obj)
    except TypeError:  # not a Qt object
        return True


def _reply(future: concurrent.futures.Future) -> tuple[dict | None, str | None]:
    """``(reply, error message)`` of a finished ``QtServiceClient.call`` future."""
    try:
        reply = future.result()
    except Exception as e:  # noqa: BLE001 - any failure is reported the same way
        return None, str(e) or type(e).__name__
    if not isinstance(reply, dict):
        return None, "bad reply from the speech service"
    if reply.get("status") == "error":
        return reply, str(reply.get("message") or reply.get("code") or "speech service error")
    return reply, None


class TtsClient(QObject):
    """Client for the ``tts`` service driving a :class:`SpeechScheduler` and :class:`Player`."""

    state_changed = Signal(str)

    def __init__(
        self,
        settings: OutputSettings,
        *,
        manager: Any = None,
        player_factory: Callable[[str], Any] = Player,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        initialize_main_thread_invoker()
        self._settings = settings
        self._manager = manager
        self._player_factory = player_factory
        self._clock = clock
        self._scheduler = SpeechScheduler(
            start=self._start_item,
            cancel=self._cancel_item,
            clock=clock,
        )

        self._started = False
        self._start_pending = False
        self._client: Any = None
        self._client_conns: list[tuple[Any, Callable]] = []
        self._state: State = "unavailable"
        self._last_error: str = ""
        self._gen = 0
        self._sid: str | None = None
        self._active_item: SpeechItem | None = None
        self._early_chunks: dict[str, deque[tuple[int, bytes, bool]]] = {}
        self._early_errors: set[str] = set()
        self._player: Any = None
        self._replace_player(settings.output_device)

    # -- public API ------------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def settings(self) -> OutputSettings:
        return self._settings

    def start(self) -> None:
        """Connect to the manager and start the ``tts`` service when enabled."""
        if self._started:
            return
        self._started = True
        m = self._mgr()
        m.client_created.connect(self._on_client_created)
        m.services_changed.connect(self._sync_client)
        self._sync_client()
        if self._settings.enabled:
            self._ensure_started()
        self._refresh()

    def stop(self) -> None:
        """Cancel any active speech, disconnect from the manager and stop the player."""
        if not self._started:
            return
        self._reset_playback(cancel=True)
        self._started = False
        self._start_pending = False
        m = self._mgr()
        for sig, slot in (
            (m.client_created, self._on_client_created),
            (m.services_changed, self._sync_client),
        ):
            self._disconnect(sig, slot)
        self._set_client(None)
        self._last_error = ""
        self._set_state("unavailable")

    def speak(
        self,
        message: str,
        voice: str | None = None,
        ttl: float | None = None,
        priority: Priority = "normal",
    ) -> None:
        """Submit ``message`` for synthesis and playback if the service is ready.

        Safe from any thread: agent signals may arrive off the Qt main thread, so a call
        from elsewhere is re-posted to this object's thread (the deadline starts now).
        """
        if not self._on_own_thread():
            invoke_in_main_thread(self.speak, message, voice, ttl, priority)
            return
        if self._state != "ready" or self._client is None:
            logger.debug("lightfall-voice: tts not ready ({}); dropping speech", self._state)
            return
        resolved_voice = voice or self._settings.default_voice or DEFAULT_VOICE
        deadline = (self._clock() + ttl) if ttl is not None else None
        item = SpeechItem(
            text=message,
            voice=resolved_voice,
            speed=self._settings.speed,
            priority=priority,
            deadline=deadline,
        )
        self._scheduler.submit(item)

    def _on_own_thread(self) -> bool:
        return QThread.currentThread() is self.thread()

    def stop_all(self) -> list[SpeechItem]:
        """Stop playback, clear queued speech and return any high-priority items.

        Main thread only (it returns a value); VoiceService.hold() is called from the UI.
        """
        interrupted = self._scheduler.stop_all()
        if self._player is not None:
            self._player.stop()
        return interrupted

    def requeue(self, item: SpeechItem) -> None:
        """Submit ``item`` with its original deadline if the service is ready. Any thread."""
        if not self._on_own_thread():
            invoke_in_main_thread(self.requeue, item)
            return
        if self._state != "ready" or self._client is None:
            logger.debug("lightfall-voice: tts not ready ({}); dropping requeue", self._state)
            return
        self._scheduler.submit(item)

    def apply_settings(self, settings: OutputSettings) -> None:
        """Adopt new output settings: speaker device, service start/stop and ``config.set``."""
        old, self._settings = self._settings, settings
        device_changed = settings.output_device != old.output_device
        interrupted = self._scheduler.current if device_changed else None
        if device_changed:
            if interrupted is not None:
                self._cancel_item(interrupted)
            self._replace_player(settings.output_device)
        if not self._started:
            return
        if old.enabled and not settings.enabled:
            self._start_pending = False
            self._reset_playback(cancel=True)
            try:
                self._mgr().stop_service(SERVICE)
            except KeyError:
                logger.warning("lightfall-voice: the tts service is not registered")
            self._set_client(None)
        elif settings.enabled and not old.enabled:
            before = self._client
            self._ensure_started()
            self._sync_client()
            if self._client is not None and self._client is before:
                self._push_config()
        elif settings.enabled:
            if settings.python != old.python and callable(getattr(self._mgr(), "restart", None)):
                self._reset_playback(cancel=True)
                try:
                    self._mgr().restart(SERVICE)
                except KeyError:
                    logger.warning("lightfall-voice: the tts service is not registered")
            elif settings.service_config() != old.service_config():
                self._push_config()
        if interrupted is not None and settings.enabled:
            self._scheduler.finished(interrupted)
        self._refresh()

    # -- service client --------------------------------------------------------------------

    def _mgr(self):
        if self._manager is None:
            from lightfall.services.manager import ServiceManager

            self._manager = ServiceManager.get_instance()
        return self._manager

    def _ensure_started(self) -> None:
        try:
            self._mgr().ensure_started(SERVICE)
            self._start_pending = False
        except KeyError:
            self._start_pending = True
            logger.debug("lightfall-voice: the tts service is not registered yet")
        except Exception:
            logger.exception("lightfall-voice: could not start the tts service")

    def _on_client_created(self, name: str) -> None:
        if name == SERVICE:
            self._sync_client()

    def _sync_client(self) -> None:
        if not self._started:
            return
        if self._start_pending:
            self._start_pending = False
            if self._settings.enabled:
                self._ensure_started()
        client = self._mgr().client(SERVICE)
        if client is not self._client:
            self._set_client(client)
        else:
            self._refresh()

    def _set_client(self, client) -> None:
        if client is self._client:
            return
        self._reset_playback(cancel=False)
        for sig, slot in self._client_conns:
            self._disconnect(sig, slot)
        self._client_conns = []
        self._client = client
        if client is not None:
            for sig, slot in (
                (client.state_changed, self._on_service_changed),
                (client.info_changed, self._on_service_changed),
                (client.stream_chunk, self._on_stream_chunk),
                (client.event, self._on_event),
                (client.session_reset, self._on_reset),
                (client.restarted, self._on_reset),
            ):
                sig.connect(slot)
                self._client_conns.append((sig, slot))
            for name in EVENTS:
                client.watch_event(name)
            self._push_config()
        self._refresh()

    def _push_config(self) -> None:
        client = self._client
        if client is None or not self._settings.enabled:
            return
        try:
            future = client.call("config.set", self._settings.service_config())
        except Exception:
            logger.exception("lightfall-voice: tts config.set failed")
            return
        future.add_done_callback(self._log_config_reply)

    @staticmethod
    def _log_config_reply(future: concurrent.futures.Future) -> None:
        _reply_dict, error = _reply(future)
        if error:
            logger.warning("lightfall-voice: tts config.set refused: {}", error)

    def _service_status(self) -> tuple[State, str]:
        if not self._settings.enabled:
            return "unavailable", ""
        client = self._client
        if client is None:
            for status in self._statuses():
                if status.get("name") == SERVICE and status.get("health") == "down":
                    return "unavailable", str(status.get("last_error") or "")
            return "loading", ""
        health = client.health
        if health in _UNAVAILABLE_HEALTH:
            return "unavailable", str(client.last_error or "")
        if health not in ("up", "degraded"):
            return "loading", ""
        summary = (client.info or {}).get("summary") or {}
        status = summary.get("status") if isinstance(summary, dict) else None
        if status == "ready":
            return "ready", ""
        if status == "error":
            return "unavailable", str(summary.get("error") or client.last_error or "")
        return "loading", ""

    def _statuses(self) -> list[dict]:
        statuses = getattr(self._mgr(), "statuses", None)
        if not callable(statuses):
            return []
        try:
            return list(statuses())
        except Exception:  # noqa: BLE001
            return []

    def _set_state(self, state: State) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def _refresh(self) -> None:
        status, error = self._service_status()
        self._last_error = error
        self._set_state(status)

    def _on_service_changed(self, *_args) -> None:
        status, error = self._service_status()
        if status == "unavailable" and self._active_item is not None:
            self._reset_playback(cancel=False)
        self._last_error = error
        self._set_state(status)

    def _on_reset(self, *_args) -> None:
        """``session_reset`` / ``restarted``: stop player, clear scheduler, set loading."""
        self._reset_playback(cancel=False)
        self._last_error = ""
        self._set_state("loading")
        self._push_config()

    # -- scheduler & player ----------------------------------------------------------------

    def _replace_player(self, device_id: str) -> None:
        old = self._player
        if old is not None:
            self._disconnect(old.finished, self._on_player_finished)
            self._disconnect(old.failed, self._on_player_failed)
            try:
                old.stop()
            except Exception:
                logger.exception("lightfall-voice: stopping the player failed")
            if hasattr(old, "deleteLater"):
                old.deleteLater()
        player = self._player_factory(device_id)
        player.finished.connect(self._on_player_finished)
        player.failed.connect(self._on_player_failed)
        self._player = player

    def _cancel_stream(self, sid: str | None, client: Any = None) -> None:
        target = self._client if client is None else client
        if not isinstance(sid, str) or not _alive(target):
            return
        try:
            target.call("cancel", {"stream_id": sid}).add_done_callback(self._log_failure)
        except Exception:
            logger.exception("lightfall-voice: cancel failed")

    def _start_item(self, item: SpeechItem) -> None:
        client = self._client
        if client is None:
            self._scheduler.finished(item)
            return
        self._gen += 1
        gen = self._gen
        self._active_item = item
        self._sid = None
        self._early_chunks.clear()
        self._early_errors.clear()
        try:
            future = client.call(
                "speak",
                {"text": item.text, "voice": item.voice, "speed": item.speed},
            )
        except Exception:
            logger.exception("lightfall-voice: speak call failed")
            self._active_item = None
            self._scheduler.finished(item)
            return
        self._on_main_thread(future, lambda f: self._on_speak_reply(gen, item, client, f))

    def _on_speak_reply(
        self,
        gen: int,
        item: SpeechItem,
        client: Any,
        future: concurrent.futures.Future,
    ) -> None:
        reply, error = _reply(future)
        sid = reply.get("stream_id") if reply is not None and error is None else None
        if (
            gen != self._gen
            or self._scheduler.current is not item
            or client is not self._client
        ):
            if isinstance(sid, str):
                self._early_chunks.pop(sid, None)
                self._early_errors.discard(sid)
            if self._active_item is None or self._sid is not None:
                self._early_chunks.clear()
                self._early_errors.clear()
            self._cancel_stream(sid, client)
            return
        if error is not None or not isinstance(sid, str):
            logger.warning("lightfall-voice: speak failed: {}", error or "no stream id")
            self._active_item = None
            self._sid = None
            self._early_chunks.clear()
            self._early_errors.clear()
            self._scheduler.finished(item)
            return
        early_chunks = sorted(self._early_chunks.pop(sid, ()), key=lambda c: c[0])
        early_error = sid in self._early_errors
        self._early_chunks.clear()
        self._early_errors.clear()
        rate = int(reply.get("rate") or DEFAULT_RATE)
        channels = int(reply.get("channels") or DEFAULT_CHANNELS)
        self._sid = sid
        if early_error:
            self._finish_failed_item(cancel_remote=True)
            return
        self._player.begin(sid, rate, channels)
        if self._sid != sid or self._active_item is not item:
            return
        for _seq, chunk_data, final in early_chunks:
            if self._sid != sid or self._active_item is not item:
                return
            self._player.feed(sid, chunk_data, final)

    def _cancel_item(self, _item: SpeechItem) -> None:
        self._gen += 1
        sid, self._sid = self._sid, None
        self._active_item = None
        self._early_chunks.clear()
        self._early_errors.clear()
        self._cancel_stream(sid)
        if self._player is not None:
            self._player.stop()

    def _reset_playback(self, *, cancel: bool) -> None:
        if not cancel:
            self._gen += 1
            self._sid = None
            self._active_item = None
            self._early_chunks.clear()
            self._early_errors.clear()
        self._scheduler.stop_all()
        self._early_chunks.clear()
        self._early_errors.clear()
        if self._player is not None:
            self._player.stop()

    def _finish_failed_item(self, *, cancel_remote: bool) -> None:
        item = self._active_item
        if item is None:
            return
        self._gen += 1
        sid, self._sid = self._sid, None
        self._active_item = None
        self._early_chunks.clear()
        self._early_errors.clear()
        if cancel_remote:
            self._cancel_stream(sid)
        if self._player is not None:
            self._player.stop()
        self._scheduler.finished(item)

    def _on_stream_chunk(self, sid: str, seq: int, data: object, final: bool) -> None:
        if self._active_item is not None and self._sid is None:
            if sid in self._early_errors:
                return
            buf = self._early_chunks.get(sid)
            if buf is None:
                buf = deque()
                self._early_chunks[sid] = buf
            if len(buf) >= MAX_EARLY_CHUNKS:
                logger.warning("lightfall-voice: early chunk buffer overflow for {}", sid)
                self._early_errors.add(sid)
                return
            buf.append((int(seq), bytes(data), bool(final)))
            return
        if sid != self._sid or self._sid is None or self._player is None:
            return
        self._player.feed(sid, bytes(data), bool(final))

    def _on_event(self, name: str, payload: dict) -> None:
        if name != "speech.error":
            return
        payload = payload if isinstance(payload, dict) else {}
        logger.warning(
            "lightfall-voice: speech.error ({}): {}",
            payload.get("code"),
            payload.get("message"),
        )
        sid = payload.get("stream_id")
        if not isinstance(sid, str):
            return
        if self._active_item is not None and self._sid is None:
            self._early_errors.add(sid)
            return
        if sid == self._sid and self._active_item is not None:
            self._finish_failed_item(cancel_remote=False)

    def _on_player_finished(self, sid: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._player:
            return
        if sid != self._sid or self._active_item is None:
            return
        item = self._active_item
        self._gen += 1
        self._active_item = None
        self._sid = None
        self._early_chunks.clear()
        self._early_errors.clear()
        self._scheduler.finished(item)

    def _on_player_failed(self, reason: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._player:
            return
        logger.warning("lightfall-voice: player failed: {}", reason)
        self._finish_failed_item(cancel_remote=True)

    # -- helpers ---------------------------------------------------------------------------

    def _on_main_thread(self, future: concurrent.futures.Future, fn: Callable) -> None:
        gen = self._gen

        def deliver(f: concurrent.futures.Future) -> None:
            invoke_in_main_thread(self._run_guarded, fn, f, gen)

        future.add_done_callback(deliver)

    def _run_guarded(
        self, fn: Callable, future: concurrent.futures.Future, gen: int | None = None
    ) -> None:
        if not _alive(self):
            return
        try:
            fn(future)
        except Exception:
            logger.exception("lightfall-voice: tts callback failed")
            if self._active_item is not None and (gen is None or gen == self._gen):
                self._finish_failed_item(cancel_remote=True)

    @staticmethod
    def _log_failure(future: concurrent.futures.Future) -> None:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.debug("lightfall-voice: tts request failed: {}", exc)
            return
        result = future.result()
        if isinstance(result, dict) and result.get("status") == "error":
            logger.debug("lightfall-voice: tts request refused: {}", result.get("message"))

    @staticmethod
    def _disconnect(signal: Any, slot: Callable) -> None:
        try:
            signal.disconnect(slot)
        except (RuntimeError, TypeError):
            pass
