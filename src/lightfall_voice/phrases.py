"""Professional spoken-phrase set. Pure functions, no state.

Copy rules: neutral tone; permission phrases contain "permission" and error
phrases contain "attention".
"""

from __future__ import annotations

import re
from pathlib import PurePath

_MAX_SUMMARY = 200

# Tool -> announcement. None = silent. Tools absent from this table are
# announced from their humanized name (see _humanized_announcement), so new
# MCP tools speak by default; the per-(agent, tool) rate limit in policy.py
# bounds the chatter.
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
    "ToolSearch": None,
    "AskUserQuestion": "Asking a question",
    # Cosmetic avatar state, called on most turns.
    "lightfall_set_emotion": None,
}

_MCP_PREFIX = re.compile(r"^mcp__.+?__")

# Leading/trailing words treated as verbs when humanizing a tool name.
_VERBS = frozenset(
    "abort activate check clear click close create delete discover edit execute fetch "
    "find fit get invoke launch list load manage move open pause plot push query read "
    "record reload remove reset resume run save scan search send set show start stop "
    "submit type unload update wait write".split()
)
# Verbs whose gerund doubles the final consonant.
_DOUBLED = frozenset("get plot run set stop".split())

# Server/namespace words dropped from the front of a tool name.
_NAMESPACES = frozenset({"lightfall"})

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


def _base_name(tool_name: str) -> str:
    """Strip an ``mcp__<server>__`` prefix: ``mcp__lf__lightfall_read`` -> ``lightfall_read``."""
    return _MCP_PREFIX.sub("", tool_name)


def _words(name: str) -> list[str]:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    words = [w.lower() for w in re.split(r"[_\-\s]+", name) if w]
    while len(words) > 1 and words[0] in _NAMESPACES:
        words.pop(0)
    return words


def _gerund(verb: str) -> str:
    if verb in _DOUBLED:
        return verb + verb[-1] + "ing"
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    return verb + "ing"


def _humanized_announcement(base: str) -> str | None:
    """Lead with the first verb found: ``read_device`` -> "Reading device",
    ``ipython_execute`` -> "Executing ipython"; no verb -> "Using <words>"."""
    words = _words(base)
    if not words:
        return None
    i = next((i for i, w in enumerate(words) if w in _VERBS), None)
    if i is None:
        return "Using " + " ".join(words)
    rest = words[:i] + words[i + 1:]
    return " ".join([_gerund(words[i]).capitalize(), *rest])


def tool_announcement(tool_name: str, tool_input: dict) -> str | None:
    """Short announcement for a tool call, or None to stay silent."""
    base = _base_name(tool_name)
    if base not in _TOOL_ACTIONS:
        return _humanized_announcement(base)
    template = _TOOL_ACTIONS[base]
    if template is None:
        return None
    if "{stem}" in template:
        stem = _stem(tool_input)
        if not stem:
            return template.replace(" {stem}", " a file")
        return template.format(stem=stem)
    return template


def permission_phrase(tool_name: str, agent_name: str | None = None) -> str:
    base = _base_name(tool_name)
    obj = _TOOL_OBJECTS.get(base) or "use " + (" ".join(_words(base)) or tool_name)
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
