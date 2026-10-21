"""VoiceInputController: mic buttons, the hotkey and the push-to-talk state machine.

The controller owns one :class:`~lightfall_voice.stt.recorder.Recorder` per utterance and
talks to the ``stt`` service through the ``QtServiceClient`` that Lightfall's
``ServiceManager`` creates (lazily, and anew after a restart).  Its public ``state`` is one of
``unavailable``, ``loading``, ``idle``, ``listening`` and ``transcribing``:

- ``unavailable``/``loading``/``idle`` come from the service: voice input disabled, client
  health and the heartbeat ``summary.status``.
- ``idle`` -> ``listening``: :meth:`toggle` holds spoken notifications, calls
  ``utterance.start``, adopts the returned stream and starts the recorder; every chunk is
  forwarded with ``seq`` 0, 1, 2, ...
- ``listening`` -> ``transcribing``: a second :meth:`toggle` (``utterance.stop``), or the
  service's ``utterance.ended`` (silence / max).
- back to ``idle``: the transcript (submitted to the widget the recording *started* on),
  ``no_speech``, an error, :meth:`cancel` (Esc), a session reset or restart, or a recorder
  failure.  Notifications are released on every exit.

Futures returned by the client complete on its loop thread; their results are marshalled
back with :func:`lightfall_utils.threads.invoke_in_main_thread` before any state or widget
is touched.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import shiboken6
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QWidget

from lightfall.utils.logging import logger
from lightfall_utils.threads import initialize_main_thread_invoker, invoke_in_main_thread

from lightfall_voice.audio import RATE
from lightfall_voice.stt.mic_button import MicButton
from lightfall_voice.stt.recorder import Recorder

State = Literal["unavailable", "loading", "idle", "listening", "transcribing"]

SERVICE = "stt"
STREAM_KIND = "audio"
EVENTS = ("utterance.listening", "utterance.ended", "utterance.transcript", "utterance.error")
START_PAYLOAD = {"rate": RATE, "format": "s16le", "channels": 1}
CLAUDE_PANEL_ID = "lightfall.panels.claude"

# UI copy (exact, see the design's Global Constraints)
LISTENING = "Listening…"
TRANSCRIBING = "Transcribing…"
NOTHING_HEARD = "Didn't catch that"
LOADING = "Loading speech model…"
DISABLED = "Voice input is disabled"
RESTARTED = "Speech service restarted"
ELSEWHERE = "Voice input is recording in another session"
NO_RESPONSE = "Speech service did not respond"

_UNAVAILABLE_HEALTH = frozenset({"down", "incompatible", "unauthorized"})
_SHOW_MS = 120_000  # placeholder "flash" long enough to outlast an utterance
_TRANSCRIBE_TIMEOUT_MS = 120_000  # give up waiting for a transcript after this long
# give up listening after the service's 60 s utterance cap plus a 10 s margin
_LISTEN_TIMEOUT_MS = 70_000

_PREFIX = "voice.input."


@dataclass
class VoiceInputSettings:
    """The ``voice.input.*`` preferences."""

    enabled: bool = False
    model_dir: str = ""
    hotkey: str = "F9"
    model: str = "small"
    device: str = "auto"
    input_device: str = ""
    silence_s: float = 1.0
    language: str = "en"
    python: str = ""

    @classmethod
    def from_preferences(cls, prefs) -> VoiceInputSettings:
        defaults = cls()
        values: dict[str, Any] = {}
        for key, default in asdict(defaults).items():
            raw = prefs.get(_PREFIX + key, default)
            try:
                values[key] = default if raw is None else type(default)(raw)
            except (TypeError, ValueError):
                values[key] = default
        return cls(**values)

    def save(self, prefs) -> None:
        for key, value in asdict(self).items():
            prefs.set(_PREFIX + key, value)

    def service_config(self) -> dict:
        """The ``config.set`` payload."""
        return {
            "model": self.model,
            "model_dir": self.model_dir,
            "device": self.device,
            "language": self.language,
            "silence_s": float(self.silence_s),
        }


def _alive(obj) -> bool:
    if obj is None:
        return False
    try:
        return shiboken6.isValid(obj)
    except TypeError:  # not a Qt object
        return True


def _default_window_lookup():
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance()
    if app is None:
        return None
    windows = [w for w in app.topLevelWidgets() if isinstance(w, QMainWindow)]
    for w in windows:
        if hasattr(w, "get_panel"):
            return w
    return windows[0] if windows else None


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


class _EscFilter(QObject):
    """While listening, a bare Esc anywhere in ``window`` cancels the recording.

    An application event filter rather than a ``QShortcut``: with another Esc shortcut in
    the same window (TheaterOverlay's), Qt treats both as ambiguous and delivers
    ``activatedAmbiguously`` to only one of them per press, in rotation, so a shortcut of
    ours could miss the first Esc.  Accepting the ``ShortcutOverride`` keeps Esc out of the
    shortcut map for that window, and the following ``KeyPress`` is consumed here.  Other
    windows (dialogs included) are untouched, and the filter exists only while listening.
    """

    def __init__(self, window, on_escape: Callable[[], None]) -> None:
        super().__init__()
        self._window = window
        self._on_escape = on_escape
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        etype = event.type()
        if not self._armed or etype not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return False
        if not isinstance(obj, QWidget) or not _alive(self._window):
            return False
        if event.key() != Qt.Key.Key_Escape or event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        if event.isAutoRepeat():
            return False  # a held Esc: only the first press counts; stay armed
        if obj.window() is not self._window:
            return False
        if etype == QEvent.Type.ShortcutOverride:
            event.accept()  # deliver Esc as a key press instead of a shortcut
            return True
        self._armed = False
        self._on_escape()
        return True


class VoiceInputController(QObject):
    """Push-to-talk voice input for agent sessions (see the module docstring)."""

    state_changed = Signal(str)

    def __init__(
        self,
        settings: VoiceInputSettings,
        voice_service,
        *,
        manager=None,
        recorder_factory: Callable[..., Any] = Recorder,
        panel_lookup: Callable[[], Any] | None = None,
        widget_scanner: Callable[[], list] | None = None,
        window_lookup: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        initialize_main_thread_invoker()
        self._settings = settings
        self._voice = voice_service
        self._manager = manager
        self._recorder_factory = recorder_factory
        self._window_lookup = window_lookup or _default_window_lookup
        self._panel_lookup = panel_lookup or self._default_panel_lookup
        if widget_scanner is None:
            from lightfall_voice.service import default_widget_scanner

            widget_scanner = default_widget_scanner
        self._scan = widget_scanner

        self._started = False
        self._start_pending = False  # enabled, but "stt" was not registered yet
        self._bus = None
        # Keyed by id(widget); the entry also keeps the widget, so a reused id (the old
        # widget was destroyed without _forget_widget running) is detected in
        # attach_widget by the identity check, and destroyed widgets are popped.
        self._buttons: dict[int, tuple[QObject, MicButton]] = {}
        self._client = None
        self._client_conns: list[tuple[Any, Callable]] = []
        self._hotkey_shortcut: QShortcut | None = None
        self._esc_filter: _EscFilter | None = None

        self._state: State = "unavailable"
        self._tooltip = DISABLED
        # utterance: phase is idle | starting | listening | transcribing
        self._phase = "idle"
        self._gen = 0  # bumped on every exit, so stale replies are dropped
        self._sid: str | None = None
        self._seq = 0
        self._target = None
        self._target_key: int | None = None
        self._recorder = None
        self._held = False
        self._shown: str | None = None  # placeholder currently shown on the target
        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_watchdog)

    # -- public API ------------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def settings(self) -> VoiceInputSettings:
        return self._settings

    def start(self) -> None:
        """Connect to the manager and AgentBus, add mic buttons, install the hotkey.

        Starts the ``stt`` service when voice input is enabled.
        """
        if self._started:
            return
        self._started = True
        m = self._mgr()
        m.client_created.connect(self._on_client_created)
        m.services_changed.connect(self._sync_client)
        try:
            from lightfall.agents.bus import AgentBus

            self._bus = AgentBus.get_instance()
            self._bus.agents_changed.connect(self.rescan)
        except Exception as e:  # noqa: BLE001
            logger.warning("lightfall-voice: AgentBus unavailable ({}); scan-once only", e)
            self._bus = None
        self.rescan()
        self._sync_client()
        if self._settings.enabled:
            self._ensure_started()
        self._refresh()

    def stop(self) -> None:
        """Cancel any utterance, disconnect everything and remove the mic buttons."""
        if not self._started:
            return
        self._abort(None, cancel=True)
        self._started = False
        self._start_pending = False
        m = self._mgr()
        for sig, slot in (
            (m.client_created, self._on_client_created),
            (m.services_changed, self._sync_client),
        ):
            self._disconnect(sig, slot)
        if self._bus is not None:
            self._disconnect(self._bus.agents_changed, self.rescan)
            self._bus = None
        self._set_client(None)
        self._remove_esc()
        if _alive(self._hotkey_shortcut):
            self._hotkey_shortcut.setEnabled(False)
            self._hotkey_shortcut.deleteLater()
        self._hotkey_shortcut = None
        for _w, button in self._buttons.values():
            if _alive(button):
                button.setParent(None)
                button.deleteLater()
        self._buttons.clear()

    def toggle(self, widget=None) -> None:
        """Start recording on ``widget`` (default: the visible agent tab), or stop it."""
        if self._phase == "listening":
            self._stop_listening()
            return
        if self._phase != "idle":
            return
        if widget is None:
            widget = self._current_widget()
            if widget is None:
                return
        if self._service_status()[0] != "idle":
            self._refresh()
            return
        self._begin(widget)

    def cancel(self) -> None:
        """Discard the current utterance (Esc): nothing is transcribed or sent."""
        self._abort(None, cancel=True)

    def rescan(self) -> None:
        self._install_shortcuts()
        try:
            widgets = self._scan()
        except Exception:
            logger.exception("lightfall-voice: widget scan failed")
            return
        for widget in widgets:
            self.attach_widget(widget)

    def attach_widget(self, widget) -> None:
        """Add a mic button to ``widget``'s input row (once)."""
        key = id(widget)
        entry = self._buttons.get(key)
        if entry is not None and entry[0] is widget:
            return
        if not callable(getattr(widget, "add_input_accessory", None)):
            return
        button = MicButton()
        button.clicked.connect(lambda _checked=False, w=widget: self._on_button(w))
        try:
            widget.add_input_accessory(button)
        except Exception:
            logger.exception("lightfall-voice: could not add the mic button")
            button.deleteLater()
            return
        self._buttons[key] = (widget, button)
        destroyed = getattr(widget, "destroyed", None)
        if destroyed is not None:
            destroyed.connect(lambda _=None, k=key: self._forget_widget(k))
        self._refresh()

    def apply_settings(self, settings: VoiceInputSettings) -> None:
        """Adopt new settings: hotkey, service start/stop and ``config.set``."""
        old, self._settings = self._settings, settings
        if _alive(self._hotkey_shortcut):
            if settings.hotkey != old.hotkey:
                self._hotkey_shortcut.setKey(QKeySequence(settings.hotkey))
            self._hotkey_shortcut.setEnabled(settings.enabled)  # free the key when off
        if not self._started:
            return
        if old.enabled and not settings.enabled:
            self._start_pending = False
            self._abort(DISABLED, cancel=True)
            try:
                self._mgr().stop_service(SERVICE)
            except KeyError:
                logger.warning("lightfall-voice: the stt service is not registered")
            self._set_client(None)
        elif settings.enabled and not old.enabled:
            before = self._client
            self._ensure_started()
            self._sync_client()
            if self._client is not None and self._client is before:
                self._push_config()
        elif settings.enabled:
            if settings.python != old.python and callable(getattr(self._mgr(), "restart", None)):
                self._abort(RESTARTED, cancel=True)
                try:
                    self._mgr().restart(SERVICE)  # new interpreter: a new child
                except KeyError:
                    logger.warning("lightfall-voice: the stt service is not registered")
            else:
                self._push_config()
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
            # Preload plugins run before service plugins register their descriptors:
            # retry on the next services_changed (see _sync_client).
            self._start_pending = True
            logger.debug("lightfall-voice: the stt service is not registered yet")
        except Exception:
            logger.exception("lightfall-voice: could not start the stt service")

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
        if self._phase != "idle":
            self._abort(RESTARTED, cancel=False)  # the old session's stream is gone
        for sig, slot in self._client_conns:
            self._disconnect(sig, slot)
        self._client_conns = []
        self._client = client
        if client is not None:
            for sig, slot in (
                (client.state_changed, self._on_service_changed),
                (client.info_changed, self._on_service_changed),
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
            logger.exception("lightfall-voice: config.set failed")
            return
        future.add_done_callback(self._log_config_reply)

    @staticmethod
    def _log_config_reply(future: concurrent.futures.Future) -> None:
        # Runs on the client's loop thread: logging only, no Qt objects.
        _reply_dict, error = _reply(future)
        if error:
            logger.warning("lightfall-voice: config.set refused: {}", error)

    def _service_status(self) -> tuple[State, str]:
        if not self._settings.enabled:
            return "unavailable", DISABLED
        client = self._client
        if client is None:
            for status in self._statuses():
                if status.get("name") == SERVICE and status.get("health") == "down":
                    return "unavailable", status.get("last_error") or "Speech service is down"
            return "loading", LOADING
        health = client.health
        if health in _UNAVAILABLE_HEALTH:
            return "unavailable", client.last_error or f"Speech service is {health}"
        if health != "up":
            return "loading", LOADING
        summary = (client.info or {}).get("summary") or {}
        status = summary.get("status") if isinstance(summary, dict) else None
        if status == "ready":
            return "idle", f"Voice input ({self._settings.hotkey})"
        if status == "error":
            return "unavailable", str(summary.get("error") or "Speech model failed to load")
        return "loading", LOADING

    def _statuses(self) -> list[dict]:
        statuses = getattr(self._mgr(), "statuses", None)
        if not callable(statuses):
            return []
        try:
            return list(statuses())
        except Exception:  # noqa: BLE001
            return []

    def _on_service_changed(self, *_args) -> None:
        status, reason = self._service_status()
        if self._phase != "idle" and status == "unavailable":
            self._abort(reason, cancel=False)
        else:
            self._refresh()

    def _on_reset(self, *_args) -> None:
        """``session_reset`` / ``restarted``: re-push config, abandon the utterance."""
        self._push_config()
        if self._phase != "idle":
            self._abort(RESTARTED, cancel=False)

    # -- utterance -------------------------------------------------------------------------

    def _begin(self, widget) -> None:
        client = self._client
        self._gen += 1
        gen = self._gen
        self._phase = "starting"
        self._target = widget
        self._target_key = id(widget)
        self._seq = 0
        self._hold()
        self._refresh()
        try:
            future = client.call("utterance.start", dict(START_PAYLOAD))
        except Exception as e:  # noqa: BLE001
            self._finish(str(e) or type(e).__name__)
            return
        self._on_main_thread(future, lambda f: self._on_start_reply(gen, client, f))

    def _on_start_reply(self, gen: int, client, future: concurrent.futures.Future) -> None:
        reply, error = _reply(future)
        sid = reply.get("stream_id") if reply is not None and error is None else None
        if gen != self._gen or self._phase != "starting" or client is not self._client:
            if isinstance(sid, str) and _alive(client):
                try:  # cancelled meanwhile: free the service's slot
                    client.call("utterance.cancel", {"stream_id": sid}).add_done_callback(
                        self._log_failure
                    )
                except Exception:
                    logger.exception("lightfall-voice: utterance.cancel failed")
            return
        if error is not None or not isinstance(sid, str):
            self._finish(error or "Speech service returned no stream id")
            return
        self._sid = sid
        client.adopt_stream(sid, STREAM_KIND).add_done_callback(self._log_failure)
        recorder = self._recorder_factory(self._settings.input_device)
        recorder.chunk.connect(self._on_chunk)
        recorder.failed.connect(self._on_recorder_failed)
        self._recorder = recorder
        self._phase = "listening"
        self._show(LISTENING)
        self._install_esc(self._target)
        self._watchdog.start(_LISTEN_TIMEOUT_MS)
        self._refresh()
        recorder.start()  # may emit failed synchronously

    def _stop_listening(self) -> None:
        sid = self._sid
        self._enter_transcribing()
        client = self._client
        if client is not None and sid is not None:
            # An error here (e.g. the utterance already ended on silence) is harmless: the
            # ended/transcript events still arrive, and the watchdog covers a lost service.
            client.call("utterance.stop", {"stream_id": sid}).add_done_callback(
                self._log_failure
            )

    def _enter_transcribing(self) -> None:
        self._stop_recorder()
        self._remove_esc()
        self._phase = "transcribing"
        self._show(TRANSCRIBING)
        self._watchdog.start(_TRANSCRIBE_TIMEOUT_MS)
        self._refresh()

    def _on_chunk(self, data: bytes) -> None:
        client = self._client
        if self._phase != "listening" or self._sid is None or client is None:
            return
        seq = self._seq
        self._seq += 1
        client.send_chunk(self._sid, seq, bytes(data)).add_done_callback(self._log_failure)

    def _on_event(self, name: str, payload: dict) -> None:
        # Events are dropped while "starting": the stream id is unknown until the
        # utterance.start reply arrives, and the service sends nothing for a stream before
        # it receives audio, which only flows once we are listening.
        if self._phase not in ("listening", "transcribing"):
            return
        payload = payload if isinstance(payload, dict) else {}
        sid = payload.get("stream_id")
        if sid != self._sid and not (name == "utterance.error" and sid is None):
            return
        if name == "utterance.ended":
            reason = payload.get("reason")
            if reason == "no_speech":
                self._finish(NOTHING_HEARD)
            elif reason == "cancel":
                self._finish(None)
            elif self._phase == "listening":
                self._enter_transcribing()
        elif name == "utterance.transcript":
            text = str(payload.get("text") or "").strip()
            target = self._target
            self._finish(None if text else NOTHING_HEARD)
            if text and _alive(target):
                try:
                    target.submit_user_prompt(text, source="voice")
                except Exception:
                    logger.exception("lightfall-voice: submitting the transcript failed")
        elif name == "utterance.error":
            self._finish(str(payload.get("message") or "Speech recognition failed"))

    def _on_recorder_failed(self, reason: str) -> None:
        if self.sender() is not self._recorder:
            return
        self._abort(reason or "Microphone failed", cancel=True)

    def _on_watchdog(self) -> None:
        if self._phase in ("listening", "transcribing"):
            self._abort(NO_RESPONSE, cancel=True)

    def _abort(self, flash: str | None, *, cancel: bool) -> None:
        if self._phase == "idle":
            return
        client, sid = self._client, self._sid
        if cancel and client is not None and sid is not None:
            try:
                client.call("utterance.cancel", {"stream_id": sid}).add_done_callback(
                    self._log_failure
                )
            except Exception:
                logger.exception("lightfall-voice: utterance.cancel failed")
        self._finish(flash)

    def _finish(self, flash: str | None) -> None:
        """Leave the utterance: stop recording, release notifications, restore the UI."""
        self._gen += 1
        self._watchdog.stop()
        self._stop_recorder()
        self._remove_esc()
        target, shown = self._target, self._shown
        self._phase = "idle"
        self._sid = None
        self._seq = 0
        self._target = None
        self._target_key = None
        self._shown = None
        self._release()
        if _alive(target):
            try:
                if flash:
                    target.placeholder_flash(flash)
                elif shown:
                    target.placeholder_flash(shown, 0)  # end the long "flash" now
            except Exception:
                logger.exception("lightfall-voice: placeholder update failed")
        self._refresh()

    def _stop_recorder(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        self._disconnect(recorder.chunk, self._on_chunk)
        self._disconnect(recorder.failed, self._on_recorder_failed)
        try:
            recorder.stop()
        except Exception:
            logger.exception("lightfall-voice: stopping the recorder failed")
        if hasattr(recorder, "deleteLater"):
            recorder.deleteLater()

    def _show(self, text: str) -> None:
        target = self._target
        if not _alive(target):
            return
        self._shown = text
        try:
            target.placeholder_flash(text, _SHOW_MS)
        except Exception:
            logger.exception("lightfall-voice: placeholder update failed")

    def _hold(self) -> None:
        if self._held or self._voice is None:
            return
        try:
            self._voice.hold()
            self._held = True
        except Exception:
            logger.exception("lightfall-voice: hold() failed")

    def _release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            self._voice.release()
        except Exception:
            logger.exception("lightfall-voice: release() failed")

    # -- UI --------------------------------------------------------------------------------

    def _default_panel_lookup(self):
        window = self._window_lookup()
        get_panel = getattr(window, "get_panel", None)
        return get_panel(CLAUDE_PANEL_ID) if callable(get_panel) else None

    def _current_widget(self):
        try:
            panel = self._panel_lookup()
            return panel.current_assistant_widget() if panel is not None else None
        except Exception:
            logger.exception("lightfall-voice: active agent tab lookup failed")
            return None

    def _install_shortcuts(self) -> None:
        if _alive(self._hotkey_shortcut):
            return
        self._hotkey_shortcut = None
        window = self._window_lookup()
        if window is None:
            return
        hotkey = QShortcut(QKeySequence(self._settings.hotkey), window)
        hotkey.setContext(Qt.ShortcutContext.ApplicationShortcut)
        hotkey.setEnabled(self._settings.enabled)  # don't take the key while disabled
        hotkey.activated.connect(self._on_hotkey)
        self._hotkey_shortcut = hotkey

    def _install_esc(self, target) -> None:
        """Esc cancels while listening, in the target's own window only (see _EscFilter)."""
        self._remove_esc()
        window_of = getattr(target, "window", None)
        window = window_of() if callable(window_of) else None
        if window is None:
            window = self._window_lookup()
        app = QApplication.instance()
        if window is None or app is None:
            return
        esc = _EscFilter(window, self.cancel)
        app.installEventFilter(esc)
        self._esc_filter = esc

    def _remove_esc(self) -> None:
        esc, self._esc_filter = self._esc_filter, None
        if not _alive(esc):
            return
        esc.disarm()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(esc)
        esc.deleteLater()  # may be inside its own eventFilter call

    def _on_hotkey(self) -> None:
        self.toggle()

    def _on_button(self, widget) -> None:
        self.toggle(widget)
        self._refresh()  # re-sync the checkable button if nothing changed

    def _forget_widget(self, key: int) -> None:
        self._buttons.pop(key, None)
        if key == self._target_key and self._phase != "idle":
            self._target = None
            self._abort(None, cancel=True)

    def _refresh(self) -> None:
        status, tip = self._service_status()
        hotkey = self._settings.hotkey
        phase = self._phase
        if phase == "listening":
            state, tip = "listening", f"Stop listening ({hotkey}); Esc cancels"
        elif phase == "transcribing":
            state, tip = "transcribing", TRANSCRIBING
        else:
            state = status
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)
        self._tooltip = tip
        for key, (widget, button) in list(self._buttons.items()):
            if not _alive(button):
                self._buttons.pop(key, None)
                continue
            if phase == "idle":
                button.set_state(status, tip)
            elif widget is self._target:
                if phase == "starting":
                    button.set_state("idle", tip, enabled=False)
                else:
                    button.set_state(state, tip)
            else:
                button.set_state("idle", ELSEWHERE, enabled=False)

    # -- helpers ---------------------------------------------------------------------------

    def _on_main_thread(self, future: concurrent.futures.Future, fn: Callable) -> None:
        gen = self._gen  # the utterance this callback belongs to

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
        except Exception as e:
            logger.exception("lightfall-voice: voice input callback failed")
            # Never leave an utterance half-started: a raise after the stream id is known
            # (recorder creation, signal wiring) would otherwise leak the hold.  Only the
            # callback's own utterance is aborted, never a newer one.
            if self._phase != "idle" and (gen is None or gen == self._gen):
                self._abort(f"Voice input failed: {e}", cancel=True)

    @staticmethod
    def _log_failure(future: concurrent.futures.Future) -> None:
        # Runs on the client's loop thread: logging only.
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.debug("lightfall-voice: stt request failed: {}", exc)
            return
        result = future.result()
        if isinstance(result, dict) and result.get("status") == "error":
            logger.debug("lightfall-voice: stt request refused: {}", result.get("message"))

    @staticmethod
    def _disconnect(signal, slot) -> None:
        try:
            signal.disconnect(slot)
        except (RuntimeError, TypeError):
            pass
