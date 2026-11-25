"""VoiceService: discovers Claude sessions, wires signals, speaks via TtsClient.

Discovery is external to lightfall core: on AgentBus.agents_changed we scan
QApplication.allWidgets() for ClaudeAssistantWidget instances and attach to
their .agent signals. Everything downstream (policy, phrases, TTS client)
is injected and unit-testable without lightfall widgets.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import threading
import time
from typing import Callable

from PySide6.QtCore import QObject, QThread, Signal

from lightfall.utils.logging import logger
from lightfall_utils.threads import initialize_main_thread_invoker, invoke_in_main_thread

from lightfall_voice import phrases
from lightfall_voice.events import (
    VoiceEvent,
    error_event,
    permission_event,
    summary_event,
    tool_event,
)
from lightfall_voice.policy import (
    VoicePolicy,
    VoiceSettings,
    muted_by_default,
    with_agent_muted,
)
from lightfall_voice.tts.client import TtsClient
from lightfall_voice.tts.scheduler import SpeechItem
from lightfall_voice.voices import DEFAULT_VOICE


@dataclass(frozen=True)
class _RequeuedSpeech:
    item: SpeechItem


MAX_HELD = 50


def _is_high_held(entry: tuple[VoiceEvent, float] | _RequeuedSpeech) -> bool:
    if isinstance(entry, _RequeuedSpeech):
        return entry.item.priority == "high"
    event, _ = entry
    return event.priority == "high"


# Name for a session whose widget carries none. Spoken phrases omit it, so
# the main assistant's errors and permission requests aren't name-prefixed.
PRIMARY_AGENT = "lightfall"

_service: "VoiceService | None" = None


def get_service() -> "VoiceService | None":
    return _service


def set_service(service: "VoiceService | None") -> None:
    global _service
    _service = service


def default_widget_scanner() -> list:
    from PySide6.QtWidgets import QApplication

    from lightfall.claude.widget import ClaudeAssistantWidget

    app = QApplication.instance()
    if app is None:
        return []
    return [w for w in app.allWidgets() if isinstance(w, ClaudeAssistantWidget)]


class VoiceService(QObject):
    # Emitted after apply_settings(), e.g. a mute toggled from the panel's
    # title bar or the preferences page.
    settings_changed = Signal()

    def __init__(
        self,
        settings: VoiceSettings,
        client: TtsClient | None = None,
        widget_scanner: Callable[[], list] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        initialize_main_thread_invoker()
        self._policy = VoicePolicy(settings)
        self._client = client if client is not None else TtsClient(settings.to_output())
        self._scan = widget_scanner or default_widget_scanner
        self._clock = clock
        # Track attached agents by id() so rescans are idempotent. Keep a
        # real reference too: Qt disconnects automatically on destruction.
        self._attached: dict[int, QObject] = {}
        self._attached_names: dict[int, str] = {}
        self._bus = None
        # Fallback for backends that don't put the turn's final text on
        # result_received (info["text"]): the latest non-empty
        # message_received text per agent (by name). Claude Code sessions
        # stream, so message_received never fires for them.
        self._last_message: dict[str, str] = {}
        # hold()/release(): while the counter is > 0 events queue instead of
        # speaking. The lock guards the counter, queue and drain flag because
        # agent signals may be delivered off the Qt main thread.
        self._hold_lock = threading.Lock()
        self._hold_count = 0
        self._held: list[tuple[VoiceEvent, float] | _RequeuedSpeech] = []
        self._draining = False

    # -- hold / release ----------------------------------------------------

    def hold(self) -> None:
        """Suspend speaking; events queue until the matching release(). Nests."""
        with self._hold_lock:
            self._hold_count += 1
            first = self._hold_count == 1
        if first:
            interrupted = self._client.stop_all()
            if interrupted:
                with self._hold_lock:
                    self._held[:0] = [_RequeuedSpeech(item) for item in interrupted]
                    self._trim_held()

    def _trim_held(self) -> None:
        while len(self._held) > MAX_HELD:
            idx = next((i for i, e in enumerate(self._held) if not _is_high_held(e)), None)
            if idx is None:
                break
            self._held.pop(idx)

    def release(self) -> None:
        """Undo one hold(); at zero, speak queued events in order, dropping stale ones."""
        with self._hold_lock:
            if self._hold_count == 0:
                logger.debug("lightfall-voice: release() without matching hold(); ignored")
                return
            self._hold_count -= 1
            if self._hold_count > 0 or self._draining:
                return
            self._draining = True
        # Events arriving mid-drain are appended to the same queue (see
        # _dispatch), so they are spoken after the older ones, never before.
        while True:
            with self._hold_lock:
                if self._hold_count > 0 or not self._held:
                    self._draining = False
                    return
                queued = self._held.pop(0)
            try:
                if isinstance(queued, _RequeuedSpeech):
                    self._client.requeue(queued.item)
                    continue
                event, enqueued_at = queued
                now = self._clock()
                ttl = self._ttl_for(event)
                if ttl is not None and now - enqueued_at > ttl:
                    continue
                self._speak_event(event, now)
            except Exception:
                logger.exception("lightfall-voice: failed to speak held event")

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._client.start()
        try:
            from lightfall.agents.bus import AgentBus

            self._bus = AgentBus.get_instance()
            self._bus.agents_changed.connect(self.rescan)
        except Exception as e:
            logger.warning("lightfall-voice: AgentBus unavailable ({}); scan-once only", e)
        self.rescan()

    def stop(self) -> None:
        if self._bus is not None:
            try:
                self._bus.agents_changed.disconnect(self.rescan)
            except (RuntimeError, TypeError):
                pass
            self._bus = None
        with self._hold_lock:
            self._held.clear()
            self._hold_count = 0
            self._draining = False
        self._client.stop()
        self._attached.clear()
        self._attached_names.clear()

    def apply_settings(self, settings: VoiceSettings) -> None:
        old_last_tool_time = self._policy._last_tool_time
        self._policy = VoicePolicy(settings)
        self._policy._last_tool_time = old_last_tool_time
        self._client.apply_settings(settings.to_output())
        self.settings_changed.emit()

    @property
    def settings(self) -> VoiceSettings:
        return self._policy.settings

    @property
    def tts_state(self) -> str:
        return getattr(self._client, "state", "ready")

    @property
    def tts_error(self) -> str:
        return getattr(self._client, "last_error", "")

    def is_agent_muted(self, agent_name: str) -> bool:
        override = self.settings.agent_overrides.get(agent_name)
        return override.muted if override is not None else muted_by_default(agent_name)

    def set_agent_muted(self, agent_name: str, muted: bool) -> VoiceSettings:
        """Mute or unmute one agent live; returns the new settings to persist."""
        settings = dataclasses.replace(
            self.settings,
            agent_overrides=with_agent_muted(
                self.settings.agent_overrides, agent_name, muted
            ),
        )
        self.apply_settings(settings)
        return settings

    def agent_name_of(self, widget) -> str:
        """The name this service speaks for ``widget``'s session under."""
        return self._resolve_agent_name(widget)

    def attached_agent_names(self) -> list[str]:
        """Names of the live sessions this service speaks for, sorted."""
        return sorted(set(self._attached_names.values()))

    def speak_test(self, message: str = "Voice notifications are working") -> None:
        """Speak a test message directly, bypassing policy, in the default voice."""
        self._client.speak(message, voice=self.settings.default_voice or DEFAULT_VOICE)

    # -- discovery ---------------------------------------------------------

    def rescan(self) -> None:
        try:
            widgets = self._scan()
        except Exception:
            logger.exception("lightfall-voice: widget scan failed")
            return
        for widget in widgets:
            self.attach_widget(widget)

    def attach_widget(self, widget) -> None:
        agent = getattr(widget, "agent", None)
        if agent is None or id(agent) in self._attached:
            return
        agent_name = self._resolve_agent_name(widget)
        try:
            agent.tool_called.connect(
                lambda tool, tool_input, n=agent_name: self._on_tool(n, tool, tool_input)
            )
            agent.message_received.connect(
                lambda text, n=agent_name: self._on_message(n, text)
            )
            agent.result_received.connect(
                lambda result, n=agent_name: self._on_result(n, result)
            )
            agent.error_occurred.connect(
                lambda _msg, n=agent_name: self._on_error(n)
            )
            agent.permission_requested.connect(
                lambda _rid, tool, _ti, n=agent_name: self._on_permission(n, tool)
            )
            agent.destroyed.connect(
                lambda _=None, key=id(agent): self._forget(key)
            )
        except AttributeError:
            logger.debug("lightfall-voice: widget agent lacks expected signals; skipped")
            return
        self._attached[id(agent)] = agent
        self._attached_names[id(agent)] = agent_name
        logger.info("lightfall-voice: attached to agent session '{}'", agent_name)

    def _forget(self, key: int) -> None:
        self._attached.pop(key, None)
        self._attached_names.pop(key, None)

    def _resolve_agent_name(self, widget) -> str:
        endpoint = getattr(widget, "bus_endpoint", None)
        name = getattr(endpoint, "name", None) if endpoint is not None else None
        if name:
            return name
        spec = getattr(widget, "spec", None)
        name = getattr(spec, "name", None) if spec is not None else None
        if name:
            return name
        return PRIMARY_AGENT

    # -- signal handlers (never raise) ---------------------------------------

    @staticmethod
    def _ttl_for(event: VoiceEvent) -> float | None:
        return None if event.priority == "high" else 10.0

    def _speak_event(self, event: VoiceEvent, now: float) -> None:
        decision = self._policy.decide(event, now=now)
        if decision is None:
            logger.debug(
                "lightfall-voice: dropped {} event from '{}' (policy)",
                event.kind, event.agent_name,
            )
            return
        text, voice = decision
        priority = "high" if event.priority == "high" else "normal"
        self._client.speak(
            text,
            voice=voice,
            ttl=self._ttl_for(event),
            priority=priority,
        )

    def _dispatch(
        self, event: VoiceEvent, received_at: float | None = None
    ) -> None:
        now = self._clock() if received_at is None else received_at
        if QThread.currentThread() is not self.thread():
            invoke_in_main_thread(self._dispatch, event, now)
            return
        with self._hold_lock:
            if self._hold_count > 0 or self._draining:
                self._held.append((event, now))
                self._trim_held()
                return
        self._speak_event(event, now)

    def _on_tool(self, agent_name: str, tool: str, tool_input: dict) -> None:
        try:
            text = phrases.tool_announcement(tool, tool_input or {})
            if text is None:
                logger.debug("lightfall-voice: tool '{}' is silent", tool)
                return
            self._dispatch(tool_event(agent_name, tool, text, self._clock()))
        except Exception:
            logger.exception("lightfall-voice: tool handler failed")

    def _on_message(self, agent_name: str, text: str) -> None:
        try:
            if text:
                self._last_message[agent_name] = text
        except Exception:
            logger.exception("lightfall-voice: message handler failed")

    def _on_result(self, agent_name: str, result: dict) -> None:
        try:
            buffered = self._last_message.pop(agent_name, None)
            final = result.get("text") if isinstance(result, dict) else None
            source = final or buffered
            if not source:
                logger.debug("lightfall-voice: turn from '{}' ended with no text", agent_name)
                return
            text = phrases.completion_phrase(source)
            if text is None:
                return
            self._dispatch(summary_event(agent_name, text, self._clock()))
        except Exception:
            logger.exception("lightfall-voice: result handler failed")

    def _on_error(self, agent_name: str) -> None:
        try:
            name = None if agent_name == PRIMARY_AGENT else agent_name
            self._dispatch(error_event(agent_name, phrases.error_phrase(name), self._clock()))
        except Exception:
            logger.exception("lightfall-voice: error handler failed")

    def _on_permission(self, agent_name: str, tool: str) -> None:
        try:
            name = None if agent_name == PRIMARY_AGENT else agent_name
            self._dispatch(
                permission_event(agent_name, phrases.permission_phrase(tool, name), self._clock())
            )
        except Exception:
            logger.exception("lightfall-voice: permission handler failed")
