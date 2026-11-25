"""Normalized voice events: high = non-interruptible (permission, error);
low = interruptible chatter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceEvent:
    agent_name: str
    kind: str  # "tool" | "summary" | "permission" | "error"
    text: str
    priority: str  # "high" | "low"
    timestamp: float
    tool_name: str | None = None


def tool_event(agent_name: str, tool_name: str, text: str, now: float) -> VoiceEvent:
    return VoiceEvent(agent_name, "tool", text, "low", now, tool_name)


def summary_event(agent_name: str, text: str, now: float) -> VoiceEvent:
    return VoiceEvent(agent_name, "summary", text, "low", now)


def permission_event(agent_name: str, text: str, now: float) -> VoiceEvent:
    return VoiceEvent(agent_name, "permission", text, "high", now)


def error_event(agent_name: str, text: str, now: float) -> VoiceEvent:
    return VoiceEvent(agent_name, "error", text, "high", now)
