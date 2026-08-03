"""Professional spoken-phrase set. Pure functions, no state.

Copy rules: neutral tone; permission phrases contain "permission" and error
phrases contain "attention" -- the daemon assigns HIGH (non-interruptible)
priority based on those words.
"""

from __future__ import annotations

import re
from pathlib import PurePath

_MAX_SUMMARY = 200

# Tool -> announcement. None (or absence) = silent.
_TOOL_ACTIONS: dict[str, str | None] = {
    "Read": "Reading {stem}",
    "Edit": "Editing {stem}",
    "MultiEdit": "Editing {stem}",
    "Write": "Writing {stem}",
    "NotebookEdit": "Editing {stem}",
    "Bash": "Running a command",
    "PowerShell": "Running a command",
    "Grep": "Searching",
    "Glob": "Searching",
    "WebSearch": "Searching the web",
    "WebFetch": "Fetching a web page",
    "TodoWrite": None,
    "Task": None,
    "Agent": None,
}

# Tool -> natural-language object for permission phrases.
_TOOL_OBJECTS = {
    "Bash": "run a command",
    "PowerShell": "run a command",
    "Edit": "edit a file",
    "MultiEdit": "edit a file",
    "Write": "write a file",
    "Read": "read a file",
    "WebFetch": "access the web",
    "WebSearch": "search the web",
}


def _stem(tool_input: dict) -> str | None:
    path = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path")
    if not path:
        return None
    return PurePath(str(path)).stem.replace("_", " ").replace("-", " ")


def tool_announcement(tool_name: str, tool_input: dict) -> str | None:
    """Short announcement for a tool call, or None to stay silent."""
    template = _TOOL_ACTIONS.get(tool_name)
    if template is None:
        return None
    if "{stem}" in template:
        stem = _stem(tool_input)
        if not stem:
            return template.replace(" {stem}", " a file")
        return template.format(stem=stem)
    return template


def permission_phrase(tool_name: str, agent_name: str | None = None) -> str:
    obj = _TOOL_OBJECTS.get(tool_name, f"use {tool_name}")
    if agent_name:
        return f"Agent {agent_name}: permission needed to {obj}"
    return f"Permission needed to {obj}"


def completion_phrase(result_text: str | None) -> str | None:
    """Spoken turn summary from the final assistant text."""
    text = (result_text or "").strip()
    if not text:
        return "Task complete"
    # Strip markdown emphasis/code markers and collapse whitespace.
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Task complete"
    if len(text) > _MAX_SUMMARY:
        text = text[:_MAX_SUMMARY].rstrip() + "..."
    return text


def error_phrase(agent_name: str | None = None) -> str:
    if agent_name:
        return f"Agent {agent_name} error. Attention needed"
    return "Agent error. Attention needed"
