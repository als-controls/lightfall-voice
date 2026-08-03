"""VoiceService: discovers Claude sessions, wires signals, speaks via daemon.

Discovery is external to lightfall core: on AgentBus.agents_changed we scan
QApplication.allWidgets() for ClaudeAssistantWidget instances and attach to
their .agent signals. Everything downstream (policy, phrases, daemon client)
is injected and unit-testable without lightfall widgets.
"""

from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QObject

from lightfall.utils.logging import logger

from lightfall_voice import phrases
from lightfall_voice.daemon_client import DaemonClient
from lightfall_voice.events import (
    VoiceEvent,
    error_event,
    permission_event,
    summary_event,
    tool_event,
)
from lightfall_voice.policy import VoicePolicy, VoiceSettings

PRIMARY_AGENT = "lightfall"

_service: "VoiceService | None" = None


def get_service() -> "VoiceService | None":
    return _service


def set_service(service: "VoiceService | None") -> None:
    global _service
    _service = service


def _default_widget_scanner() -> list:
    from PySide6.QtWidgets import QApplication

    from lightfall.claude.widget import ClaudeAssistantWidget

    app = QApplication.instance()
    if app is None:
        return []
    return [w for w in app.allWidgets() if isinstance(w, ClaudeAssistantWidget)]


class VoiceService(QObject):
    def __init__(
        self,
        settings: VoiceSettings,
        client: DaemonClient | None = None,
        widget_scanner: Callable[[], list] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self._policy = VoicePolicy(settings, primary_agent=PRIMARY_AGENT)
        self._client = client or DaemonClient(
            host=settings.daemon_host, port=settings.daemon_port
        )
        self._scan = widget_scanner or _default_widget_scanner
        self._clock = clock
        # Track attached agents by id() so rescans are idempotent. Keep a
        # real reference too: Qt disconnects automatically on destruction.
        self._attached: dict[int, QObject] = {}
        self._bus = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
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
        self._client.stop()
        self._attached.clear()

    def apply_settings(self, settings: VoiceSettings) -> None:
        self._policy = VoicePolicy(settings, primary_agent=PRIMARY_AGENT)
        self._client.configure(settings.daemon_host, settings.daemon_port)

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
        spec = getattr(widget, "spec", None)
        agent_name = getattr(spec, "name", None) or PRIMARY_AGENT
        try:
            agent.tool_called.connect(
                lambda tool, tool_input, n=agent_name: self._on_tool(n, tool, tool_input)
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
                lambda _=None, key=id(agent): self._attached.pop(key, None)
            )
        except AttributeError:
            logger.debug("lightfall-voice: widget agent lacks expected signals; skipped")
            return
        self._attached[id(agent)] = agent
        logger.info("lightfall-voice: attached to agent session '{}'", agent_name)

    # -- signal handlers (never raise) ---------------------------------------

    def _dispatch(self, event: VoiceEvent) -> None:
        decision = self._policy.decide(event, now=self._clock())
        if decision is None:
            return
        text, voice = decision
        self._client.speak(text, voice=voice)

    def _on_tool(self, agent_name: str, tool: str, tool_input: dict) -> None:
        try:
            text = phrases.tool_announcement(tool, tool_input or {})
            if text is None:
                return
            self._dispatch(tool_event(agent_name, tool, text, self._clock()))
        except Exception:
            logger.exception("lightfall-voice: tool handler failed")

    def _on_result(self, agent_name: str, result: dict) -> None:
        try:
            raw = result.get("result") if isinstance(result, dict) else None
            text = phrases.completion_phrase(raw if isinstance(raw, str) else None)
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
