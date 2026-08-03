# lightfall-voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Lightfall plugin that speaks agent events (tool use, turn summaries, permission requests, errors) through the existing Kokoro voice daemon, with subagents muted by default.

**Architecture:** A single `SettingsPlugin` (preload) whose `on_loaded()` starts a `VoiceService`. The service discovers live Claude sessions via `AgentBus.agents_changed` + a widget scan, normalizes Qt signals into `VoiceEvent`s, applies policy (enable flags, subagent mute, 3 s/tool rate limit, 10 s TTL), renders professional phrases, and fire-and-forgets JSON to the daemon on `127.0.0.1:52849` from a worker thread. One small patch to the daemon (hooks repo) adds a direct-message path.

**Tech Stack:** Python ≥3.11, PySide6 (via lightfall), hatchling + hatch-vcs, pytest + pytest-qt. No TTS/torch/numpy deps in the plugin.

## Global Constraints

- Repo: `C:\Users\rp\PycharmProjects\ncs\lightfall-voice`, branch `master`. Commit after every task; explicit `git add` paths, never `git add -A`.
- Package: src-layout `src/lightfall_voice/`. Runtime dependency: `lightfall` only. Dev extras: `pytest`, `pytest-qt`.
- Run tests with the venv python: create `.venv` in the repo (`py -3.11 -m venv .venv` or `python -m venv .venv`), `.venv\Scripts\python -m pip install -e .[dev]` plus `-e C:\Users\rp\PycharmProjects\ncs\lightfall`; always `.venv\Scripts\python -m pytest`, never bare `pytest`.
- Tone rule: all spoken copy is neutral and professional — no nicknames, no "Very well", no jokes. Permission phrases MUST contain the word "permission" and error phrases the word "attention" (the daemon derives HIGH priority from those words).
- All signal handlers must be exception-proof: wrap bodies in try/except that logs and returns. Never let an exception propagate into `QtClaudeAgent`'s emit path.
- GUI thread rule: no socket I/O on the GUI thread — all daemon traffic goes through the `DaemonClient` worker thread.
- Task 1 is the ONLY task that touches `C:\Users\rp\.claude\hooks\voice_notifications` (a separate git repo — commit there separately). No task touches the main `lightfall` repo.
- Preferences keys (exact): `voice.enabled`, `voice.events.tools`, `voice.events.summary`, `voice.events.permission`, `voice.events.errors`, `voice.daemon_host`, `voice.daemon_port`, `voice.agent_overrides` (dict `{agent_name: {"muted": bool, "voice": str | None}}`).

---

### Task 1: Daemon direct-message path (hooks repo)

**Files:**
- Modify: `C:\Users\rp\.claude\hooks\voice_notifications\voice_daemon.py` (`handle_speak_request`, currently line 229)
- Test: `C:\Users\rp\.claude\hooks\voice_notifications\test_direct_message.py`

**Interfaces:**
- Produces: daemon accepts `{"action": "speak", "args": {"message": <str>, "voice": <str|null>}}` with **no** `hook` key, and queues it verbatim (bypassing `should_announce`, transcript logic, and personality). Response `{"status": "ok", "action": "queued", ...}`. Later tasks (DaemonClient) rely on exactly this payload shape.

- [ ] **Step 1: Write the failing test**

```python
# C:\Users\rp\.claude\hooks\voice_notifications\test_direct_message.py
"""Unit tests for the hookless direct-message path in handle_speak_request."""
from unittest.mock import MagicMock

from voice_daemon import VoiceDaemon


def make_daemon():
    daemon = VoiceDaemon.__new__(VoiceDaemon)  # skip __init__ (sockets, files)
    daemon.handler = MagicMock()
    daemon.stats = {"spoken": 0, "errors": 0}
    daemon.log = lambda msg: None
    daemon.queue_speech = MagicMock()
    return daemon


def test_direct_message_without_hook_is_queued():
    daemon = make_daemon()
    resp = daemon.handle_speak_request(
        {"action": "speak", "args": {"message": "Task complete", "voice": "af_sky"}}
    )
    assert resp["status"] == "ok"
    assert resp["action"] == "queued"
    daemon.queue_speech.assert_called_once_with("Task complete", voice="af_sky")


def test_direct_message_skips_should_announce():
    daemon = make_daemon()
    daemon.handler.should_announce = MagicMock(return_value=False)
    resp = daemon.handle_speak_request({"action": "speak", "args": {"message": "hi"}})
    assert resp["action"] == "queued"
    daemon.handler.should_announce.assert_not_called()


def test_hookless_request_without_message_is_skipped():
    daemon = make_daemon()
    resp = daemon.handle_speak_request({"action": "speak", "args": {}})
    assert resp["status"] == "ok"
    assert resp["action"] == "skipped"
    daemon.queue_speech.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\rp\.claude\hooks\voice_notifications && .venv\Scripts\python -m pytest test_direct_message.py -v`
Expected: FAIL — first test gets `action == "skipped"` (reason `should_announce=False`, since `hook_type` is None), third may pass incidentally.
(If pytest is not installed in that venv: `.venv\Scripts\python -m pip install pytest`.)

- [ ] **Step 3: Implement the direct-message branch**

In `voice_daemon.py`, `handle_speak_request`, immediately after the three `request.get(...)` lines (`hook_type`, `stdin_data`, `args`) and **before** the `tool_name` block:

```python
            # Direct message path (no hook): used by in-process clients such as
            # the lightfall-voice plugin. Queue verbatim, no hook heuristics.
            if not hook_type:
                message = args.get("message")
                if not message:
                    return {"status": "ok", "action": "skipped", "reason": "no message"}
                self.queue_speech(message, voice=args.get("voice"))
                self.stats["spoken"] += 1
                return {"status": "ok", "action": "queued", "message": message[:100]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:\Users\rp\.claude\hooks\voice_notifications && .venv\Scripts\python -m pytest test_direct_message.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit (hooks repo)**

```bash
cd C:\Users\rp\.claude\hooks\voice_notifications
git add voice_daemon.py test_direct_message.py
git commit -m "Add hookless direct-message path to speak action for in-process clients"
```

---

### Task 2: Package scaffolding + phrases module

**Files:**
- Create: `pyproject.toml`, `README.md`, `.gitignore`, `src/lightfall_voice/__init__.py`, `src/lightfall_voice/manifest.py`, `src/lightfall_voice/phrases.py`
- Test: `tests/test_phrases.py`, `tests/test_manifest.py`

**Interfaces:**
- Produces: `phrases.tool_announcement(tool_name: str, tool_input: dict) -> str | None`; `phrases.permission_phrase(tool_name: str, agent_name: str | None = None) -> str`; `phrases.completion_phrase(result_text: str | None) -> str | None`; `phrases.error_phrase(agent_name: str | None = None) -> str`. `manifest.manifest` is a `lightfall.plugins.manifest.PluginManifest`.

- [ ] **Step 1: Scaffolding files**

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "lightfall-voice"
description = "Voice notifications plugin for Lightfall via the local Kokoro voice daemon"
readme = "README.md"
requires-python = ">=3.11"
dynamic = ["version"]
dependencies = ["lightfall"]

[project.optional-dependencies]
dev = ["pytest", "pytest-qt"]

[project.entry-points."lightfall.plugins"]
lightfall_voice = "lightfall_voice.manifest:manifest"

[tool.hatch.version]
source = "vcs"
fallback-version = "0.0.0"

[tool.hatch.build.targets.wheel]
packages = ["src/lightfall_voice"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
dist/
.pytest_cache/
```

`README.md` (initial stub — expanded in Task 7):

```markdown
# lightfall-voice

Voice notifications plugin for Lightfall. Speaks agent events (tool use, turn
summaries, permission requests, errors) through the local Kokoro voice daemon
(`~/.claude/hooks/voice_notifications`, TCP 127.0.0.1:52849).
```

`src/lightfall_voice/__init__.py`:

```python
"""Voice notifications plugin for Lightfall."""
```

`src/lightfall_voice/manifest.py`:

```python
"""Plugin manifest for lightfall-voice."""

from lightfall.plugins.manifest import PluginEntry, PluginManifest

manifest = PluginManifest(
    name="lightfall-voice",
    version="0.1.0",
    description="Spoken notifications for Claude agent sessions via the local voice daemon",
    plugins=[
        # preload=True so on_loaded() runs at startup and starts the VoiceService.
        PluginEntry(
            type_name="settings",
            name="voice",
            import_path="lightfall_voice.settings:VoiceSettingsPlugin",
            preload=True,
        ),
    ],
)
```

- [ ] **Step 2: Create venv and install**

```bash
cd C:\Users\rp\PycharmProjects\ncs\lightfall-voice
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev] -e C:\Users\rp\PycharmProjects\ncs\lightfall
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_phrases.py
from lightfall_voice import phrases


def test_read_announcement_names_the_file_stem():
    msg = phrases.tool_announcement("Read", {"file_path": "C:/proj/src/loader.py"})
    assert msg == "Reading loader"


def test_edit_announcement():
    msg = phrases.tool_announcement("Edit", {"file_path": "C:/proj/config.json"})
    assert msg == "Editing config"


def test_bash_announcement_is_generic():
    assert phrases.tool_announcement("Bash", {"command": "rm -rf /"}) == "Running a command"


def test_search_tools():
    assert phrases.tool_announcement("Grep", {}) == "Searching"
    assert phrases.tool_announcement("Glob", {}) == "Searching"


def test_silent_tools_return_none():
    assert phrases.tool_announcement("TodoWrite", {}) is None
    assert phrases.tool_announcement("Task", {}) is None


def test_unknown_tool_returns_none():
    assert phrases.tool_announcement("SomeMcpTool", {}) is None


def test_permission_phrase_contains_permission_keyword():
    msg = phrases.permission_phrase("Bash")
    assert msg == "Permission needed to run a command"
    assert "permission" in msg.lower()


def test_permission_phrase_with_agent_prefix():
    msg = phrases.permission_phrase("Edit", agent_name="alignment")
    assert msg == "Agent alignment: permission needed to edit a file"


def test_completion_phrase_trims_long_text():
    msg = phrases.completion_phrase("x" * 500)
    assert len(msg) <= 203  # 200 + ellipsis
    assert msg.startswith("x")


def test_completion_phrase_empty_falls_back():
    assert phrases.completion_phrase(None) == "Task complete"
    assert phrases.completion_phrase("   ") == "Task complete"


def test_completion_phrase_strips_markdown_noise():
    assert phrases.completion_phrase("**Done.** See `foo.py`") == "Done. See foo.py"


def test_error_phrase_contains_attention_keyword():
    assert phrases.error_phrase() == "Agent error. Attention needed"
    assert "attention" in phrases.error_phrase().lower()


def test_error_phrase_with_agent():
    assert phrases.error_phrase("alignment") == "Agent alignment error. Attention needed"
```

```python
# tests/test_manifest.py
def test_manifest_declares_preloaded_settings_plugin():
    from lightfall_voice.manifest import manifest

    entries = manifest.get_plugins_by_type("settings")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.name == "voice"
    assert entry.preload is True
    assert entry.import_path == "lightfall_voice.settings:VoiceSettingsPlugin"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_phrases.py tests/test_manifest.py -v`
Expected: FAIL — `lightfall_voice.phrases` has no attributes / import errors.

- [ ] **Step 5: Implement phrases.py**

```python
# src/lightfall_voice/phrases.py
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_phrases.py tests/test_manifest.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml README.md .gitignore src/lightfall_voice tests/test_phrases.py tests/test_manifest.py
git commit -m "feat: package scaffolding, manifest, and professional phrase set"
```

---

### Task 3: VoiceEvent model

**Files:**
- Create: `src/lightfall_voice/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces: `VoiceEvent` frozen dataclass with fields `agent_name: str`, `kind: str` (one of `"tool" | "summary" | "permission" | "error"`), `text: str`, `priority: str` (`"high" | "low"`), `timestamp: float`, plus optional `tool_name: str | None = None`. Constructors: `tool_event(agent_name, tool_name, text, now)`, `summary_event(agent_name, text, now)`, `permission_event(agent_name, text, now)`, `error_event(agent_name, text, now)`. Permission/error are `"high"`, tool/summary `"low"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events.py
from lightfall_voice.events import (
    VoiceEvent,
    error_event,
    permission_event,
    summary_event,
    tool_event,
)


def test_tool_event_is_low_priority():
    e = tool_event("lightfall", "Read", "Reading loader", now=100.0)
    assert e == VoiceEvent(
        agent_name="lightfall", kind="tool", text="Reading loader",
        priority="low", timestamp=100.0, tool_name="Read",
    )


def test_summary_event_is_low_priority():
    e = summary_event("lightfall", "Task complete", now=1.0)
    assert e.kind == "summary" and e.priority == "low" and e.tool_name is None


def test_permission_event_is_high_priority():
    e = permission_event("lightfall", "Permission needed to run a command", now=1.0)
    assert e.kind == "permission" and e.priority == "high"


def test_error_event_is_high_priority():
    e = error_event("alignment", "Agent alignment error. Attention needed", now=1.0)
    assert e.kind == "error" and e.priority == "high"


def test_events_are_frozen():
    import dataclasses, pytest
    e = summary_event("a", "b", now=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.text = "mutated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_events.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement events.py**

```python
# src/lightfall_voice/events.py
"""Normalized voice events. Priority follows daemon semantics:
high = non-interruptible (permission, error); low = interruptible chatter."""

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_events.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/lightfall_voice/events.py tests/test_events.py
git commit -m "feat: VoiceEvent model with daemon-aligned priorities"
```

---

### Task 4: Daemon client (worker thread, fire-and-forget)

**Files:**
- Create: `src/lightfall_voice/daemon_client.py`
- Test: `tests/test_daemon_client.py`, `tests/conftest.py` (fake daemon fixture)

**Interfaces:**
- Consumes: daemon payload contract from Task 1.
- Produces: `DaemonClient(host: str = "127.0.0.1", port: int = 52849, connect_timeout: float = 2.0)` with methods `speak(message: str, voice: str | None = None) -> None` (non-blocking enqueue), `stop() -> None` (drains/joins the worker), and `configure(host: str, port: int) -> None` (applies to subsequent sends). Failures are swallowed (logged at debug).

- [ ] **Step 1: Write the fake-daemon fixture and failing tests**

```python
# tests/conftest.py
"""Shared fixtures: a fake voice daemon capturing JSON-line requests."""

from __future__ import annotations

import json
import socket
import threading

import pytest


class FakeDaemon:
    def __init__(self):
        self.requests: list[dict] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        self._sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except socket.timeout:
                continue
            with client:
                data = b""
                client.settimeout(1.0)
                try:
                    while not data.endswith(b"\n"):
                        chunk = client.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    if data.strip():
                        self.requests.append(json.loads(data.decode("utf-8")))
                    client.sendall(b'{"status": "ok", "action": "queued"}\n')
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._sock.close()


@pytest.fixture
def fake_daemon():
    daemon = FakeDaemon()
    yield daemon
    daemon.close()
```

```python
# tests/test_daemon_client.py
import time

from lightfall_voice.daemon_client import DaemonClient


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_speak_sends_hookless_payload(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        client.speak("Task complete", voice="af_sky")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
        req = fake_daemon.requests[0]
        assert req == {
            "action": "speak",
            "args": {"message": "Task complete", "voice": "af_sky"},
        }
    finally:
        client.stop()


def test_speak_without_voice_omits_voice_key_value(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        client.speak("Searching")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
        assert fake_daemon.requests[0]["args"]["voice"] is None
    finally:
        client.stop()


def test_daemon_down_is_silent():
    client = DaemonClient(port=1, connect_timeout=0.2)  # nothing listens on port 1
    try:
        client.speak("hello")  # must not raise
        time.sleep(0.5)
    finally:
        client.stop()


def test_speak_never_blocks_caller(fake_daemon):
    client = DaemonClient(port=fake_daemon.port)
    try:
        start = time.monotonic()
        for _ in range(20):
            client.speak("msg")
        assert time.monotonic() - start < 0.1
    finally:
        client.stop()


def test_configure_redirects(fake_daemon):
    client = DaemonClient(port=1)
    try:
        client.configure("127.0.0.1", fake_daemon.port)
        client.speak("after reconfigure")
        assert wait_for(lambda: len(fake_daemon.requests) == 1)
    finally:
        client.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_daemon_client.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement daemon_client.py**

```python
# src/lightfall_voice/daemon_client.py
"""Fire-and-forget TCP client for the voice daemon's JSON line protocol.

One background worker drains a queue; every send opens a short-lived
connection. Any failure drops the message -- a missed notification is
always better than a blocked or crashing GUI.
"""

from __future__ import annotations

import json
import queue
import socket
import threading

from lightfall.utils.logging import logger

_SENTINEL = object()


class DaemonClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 52849,
        connect_timeout: float = 2.0,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = connect_timeout
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="lightfall-voice-daemon-client", daemon=True
        )
        self._thread.start()

    def configure(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)

    def speak(self, message: str, voice: str | None = None) -> None:
        self._queue.put({"action": "speak", "args": {"message": message, "voice": voice}})

    def stop(self) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=3)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            try:
                self._send(item)
            except Exception as e:
                logger.debug("voice daemon unreachable, dropping message: {}", e)

    def _send(self, request: dict) -> None:
        with socket.create_connection((self._host, self._port), timeout=self._timeout) as sock:
            sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
            sock.settimeout(self._timeout)
            try:
                sock.recv(4096)  # read-and-ignore the response line
            except OSError:
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_daemon_client.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/lightfall_voice/daemon_client.py tests/test_daemon_client.py tests/conftest.py
git commit -m "feat: worker-thread daemon client with fake-daemon test fixture"
```

---

### Task 5: Policy engine (settings model, mute/rate-limit/TTL)

**Files:**
- Create: `src/lightfall_voice/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `VoiceEvent` (Task 3).
- Produces:
  - `AgentOverride` dataclass: `muted: bool = True`, `voice: str | None = None`.
  - `VoiceSettings` dataclass: `enabled: bool = True`, `tools: bool = True`, `summary: bool = True`, `permission: bool = True`, `errors: bool = True`, `daemon_host: str = "127.0.0.1"`, `daemon_port: int = 52849`, `agent_overrides: dict[str, AgentOverride]` (default empty). Class method `from_preferences(prefs) -> VoiceSettings` and method `to_preferences(prefs) -> None` using the exact `voice.*` keys from Global Constraints (`prefs` is any object with `.get(key, default)` / `.set(key, value)`).
  - `VoicePolicy(settings: VoiceSettings, primary_agent: str = "lightfall")` with `decide(event: VoiceEvent, now: float) -> tuple[str, str | None] | None` — returns `(text_to_speak, voice_or_None)` or `None` to stay silent. Constants `TTL_SECONDS = 10.0`, `TOOL_RATE_LIMIT_SECONDS = 3.0`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_policy.py
from lightfall_voice.events import (
    error_event,
    permission_event,
    summary_event,
    tool_event,
)
from lightfall_voice.policy import AgentOverride, VoicePolicy, VoiceSettings


def make_policy(**settings_kwargs):
    return VoicePolicy(VoiceSettings(**settings_kwargs))


def test_primary_agent_tool_event_speaks():
    p = make_policy()
    assert p.decide(tool_event("lightfall", "Read", "Reading loader", 100.0), now=100.0) == (
        "Reading loader", None,
    )


def test_master_disable_silences_everything():
    p = make_policy(enabled=False)
    assert p.decide(permission_event("lightfall", "Permission needed", 0.0), now=0.0) is None


def test_event_kind_toggles():
    p = make_policy(tools=False)
    assert p.decide(tool_event("lightfall", "Read", "Reading x", 0.0), now=0.0) is None
    assert p.decide(summary_event("lightfall", "Task complete", 0.0), now=0.0) is not None


def test_subagent_low_priority_muted_by_default():
    p = make_policy()
    assert p.decide(tool_event("alignment", "Read", "Reading x", 0.0), now=0.0) is None
    assert p.decide(summary_event("alignment", "Done", 0.0), now=0.0) is None


def test_subagent_high_priority_speaks_by_default():
    p = make_policy()
    out = p.decide(error_event("alignment", "Agent alignment error. Attention needed", 0.0), now=0.0)
    assert out == ("Agent alignment error. Attention needed", None)


def test_subagent_unmuted_override_speaks_with_voice():
    p = make_policy(agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")})
    out = p.decide(tool_event("alignment", "Read", "Reading x", 0.0), now=0.0)
    assert out == ("Reading x", "af_sky")


def test_explicitly_muted_agent_still_speaks_high_priority():
    p = make_policy(agent_overrides={"alignment": AgentOverride(muted=True)})
    assert p.decide(permission_event("alignment", "Permission needed", 0.0), now=0.0) is not None


def test_primary_agent_voice_override():
    p = make_policy(agent_overrides={"lightfall": AgentOverride(muted=False, voice="af_nicole")})
    out = p.decide(summary_event("lightfall", "Task complete", 0.0), now=0.0)
    assert out == ("Task complete", "af_nicole")


def test_ttl_drops_stale_low_priority():
    p = make_policy()
    assert p.decide(tool_event("lightfall", "Read", "Reading x", 0.0), now=11.0) is None


def test_ttl_never_drops_high_priority():
    p = make_policy()
    assert p.decide(permission_event("lightfall", "Permission needed", 0.0), now=60.0) is not None


def test_tool_rate_limit_per_agent_and_tool():
    p = make_policy()
    assert p.decide(tool_event("lightfall", "Read", "Reading a", 0.0), now=0.0) is not None
    assert p.decide(tool_event("lightfall", "Read", "Reading b", 1.0), now=1.0) is None
    assert p.decide(tool_event("lightfall", "Grep", "Searching", 1.0), now=1.0) is not None
    assert p.decide(tool_event("lightfall", "Read", "Reading c", 4.0), now=4.0) is not None


def test_settings_preferences_roundtrip():
    class FakePrefs:
        def __init__(self):
            self.data = {}
        def get(self, key, default=None):
            return self.data.get(key, default)
        def set(self, key, value):
            self.data[key] = value

    prefs = FakePrefs()
    s = VoiceSettings(
        enabled=False, tools=False, daemon_port=9999,
        agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")},
    )
    s.to_preferences(prefs)
    loaded = VoiceSettings.from_preferences(prefs)
    assert loaded == s
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_policy.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement policy.py**

```python
# src/lightfall_voice/policy.py
"""Decides whether/how a VoiceEvent is spoken. Pure logic, injectable time."""

from __future__ import annotations

from dataclasses import dataclass, field

from lightfall_voice.events import VoiceEvent

TTL_SECONDS = 10.0
TOOL_RATE_LIMIT_SECONDS = 3.0

_KEYS = {
    "enabled": "voice.enabled",
    "tools": "voice.events.tools",
    "summary": "voice.events.summary",
    "permission": "voice.events.permission",
    "errors": "voice.events.errors",
    "daemon_host": "voice.daemon_host",
    "daemon_port": "voice.daemon_port",
    "agent_overrides": "voice.agent_overrides",
}


@dataclass(frozen=True)
class AgentOverride:
    muted: bool = True
    voice: str | None = None


@dataclass
class VoiceSettings:
    enabled: bool = True
    tools: bool = True
    summary: bool = True
    permission: bool = True
    errors: bool = True
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 52849
    agent_overrides: dict[str, AgentOverride] = field(default_factory=dict)

    @classmethod
    def from_preferences(cls, prefs) -> "VoiceSettings":
        raw_overrides = prefs.get(_KEYS["agent_overrides"], {}) or {}
        overrides = {
            name: AgentOverride(
                muted=bool(o.get("muted", True)), voice=o.get("voice") or None
            )
            for name, o in raw_overrides.items()
        }
        return cls(
            enabled=bool(prefs.get(_KEYS["enabled"], True)),
            tools=bool(prefs.get(_KEYS["tools"], True)),
            summary=bool(prefs.get(_KEYS["summary"], True)),
            permission=bool(prefs.get(_KEYS["permission"], True)),
            errors=bool(prefs.get(_KEYS["errors"], True)),
            daemon_host=str(prefs.get(_KEYS["daemon_host"], "127.0.0.1")),
            daemon_port=int(prefs.get(_KEYS["daemon_port"], 52849)),
            agent_overrides=overrides,
        )

    def to_preferences(self, prefs) -> None:
        prefs.set(_KEYS["enabled"], self.enabled)
        prefs.set(_KEYS["tools"], self.tools)
        prefs.set(_KEYS["summary"], self.summary)
        prefs.set(_KEYS["permission"], self.permission)
        prefs.set(_KEYS["errors"], self.errors)
        prefs.set(_KEYS["daemon_host"], self.daemon_host)
        prefs.set(_KEYS["daemon_port"], self.daemon_port)
        prefs.set(
            _KEYS["agent_overrides"],
            {
                name: {"muted": o.muted, "voice": o.voice}
                for name, o in self.agent_overrides.items()
            },
        )


class VoicePolicy:
    def __init__(self, settings: VoiceSettings, primary_agent: str = "lightfall") -> None:
        self.settings = settings
        self.primary_agent = primary_agent
        self._last_tool_time: dict[tuple[str, str], float] = {}

    def decide(self, event: VoiceEvent, now: float) -> tuple[str, str | None] | None:
        s = self.settings
        if not s.enabled:
            return None

        kind_enabled = {
            "tool": s.tools,
            "summary": s.summary,
            "permission": s.permission,
            "error": s.errors,
        }.get(event.kind, False)
        if not kind_enabled:
            return None

        override = s.agent_overrides.get(event.agent_name)
        is_primary = event.agent_name == self.primary_agent
        # Subagents are muted by default; explicit overrides win either way.
        # High-priority events (permission, error) always break through mutes.
        if event.priority != "high":
            muted = override.muted if override is not None else not is_primary
            if muted:
                return None
            if now - event.timestamp > TTL_SECONDS:
                return None
            if event.kind == "tool" and event.tool_name:
                key = (event.agent_name, event.tool_name)
                last = self._last_tool_time.get(key, float("-inf"))
                if now - last < TOOL_RATE_LIMIT_SECONDS:
                    return None
                self._last_tool_time[key] = now

        voice = override.voice if override is not None else None
        return (event.text, voice)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_policy.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/lightfall_voice/policy.py tests/test_policy.py
git commit -m "feat: voice policy engine with subagent muting, TTL, and rate limiting"
```

---

### Task 6: VoiceService (discovery + signal wiring)

**Files:**
- Create: `src/lightfall_voice/service.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `VoicePolicy`, `VoiceSettings` (Task 5); `DaemonClient` (Task 4); `phrases` (Task 2); event constructors (Task 3).
- Produces: `VoiceService(settings: VoiceSettings, client: DaemonClient | None = None, widget_scanner: Callable[[], list] | None = None, clock: Callable[[], float] = time.monotonic)` — QObject with methods `start() -> None` (connects to `AgentBus.agents_changed`, runs one initial scan), `stop() -> None`, `rescan() -> None`, `attach_widget(widget) -> None` (idempotent), `apply_settings(settings: VoiceSettings) -> None`. Module function `get_service() -> VoiceService | None` / `set_service(service)` for the settings plugin.
- Signal signatures consumed from `QtClaudeAgent` (lightfall `claude/agent.py:201-222`): `tool_called(str, dict)`, `result_received(dict)`, `error_occurred(str)`, `permission_requested(str, str, dict)`, `task_finished(str, str, str, str, dict)`.
- Agent name resolution: `widget.spec.name` if present, else `"lightfall"`. Primary agent is `"lightfall"`.
- Turn summary text: `result_received` dict — use `result.get("result")` if a non-empty string, else `None` (→ "Task complete").

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_service.py
"""Service tests drive stub QObjects with QtClaudeAgent-shaped signals."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from PySide6.QtCore import QObject, Signal

from lightfall_voice.policy import AgentOverride, VoiceSettings
from lightfall_voice.service import VoiceService


class StubAgent(QObject):
    tool_called = Signal(str, dict)
    result_received = Signal(dict)
    error_occurred = Signal(str)
    permission_requested = Signal(str, str, dict)
    task_finished = Signal(str, str, str, str, dict)


class StubWidget(QObject):
    def __init__(self, agent_name="lightfall"):
        super().__init__()
        self.agent = StubAgent()
        self.spec = SimpleNamespace(name=agent_name)


def make_service(qtbot, widgets, settings=None, t0=1000.0):
    clock = {"t": t0}
    client = MagicMock()
    service = VoiceService(
        settings=settings or VoiceSettings(),
        client=client,
        widget_scanner=lambda: widgets,
        clock=lambda: clock["t"],
    )
    service.rescan()
    return service, client, clock


def test_tool_call_speaks_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/loader.py"})
    client.speak.assert_called_once_with("Reading loader", voice=None)


def test_permission_request_speaks_high_priority_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.permission_requested.emit("req1", "Bash", {})
    client.speak.assert_called_once_with("Permission needed to run a command", voice=None)


def test_result_speaks_summary(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.result_received.emit({"result": "Fixed the flaky test."})
    client.speak.assert_called_once_with("Fixed the flaky test.", voice=None)


def test_result_without_text_says_task_complete(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.result_received.emit({})
    client.speak.assert_called_once_with("Task complete", voice=None)


def test_error_speaks_attention_phrase(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.error_occurred.emit("boom")
    client.speak.assert_called_once_with("Agent error. Attention needed", voice=None)


def test_subagent_session_is_muted_for_low_priority(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("Read", {"file_path": "C:/x/a.py"})
    w.agent.result_received.emit({"result": "done"})
    client.speak.assert_not_called()


def test_subagent_error_is_name_prefixed(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.error_occurred.emit("boom")
    client.speak.assert_called_once_with("Agent alignment error. Attention needed", voice=None)


def test_subagent_permission_is_name_prefixed(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    w.agent.permission_requested.emit("req1", "Edit", {})
    client.speak.assert_called_once_with(
        "Agent alignment: permission needed to edit a file", voice=None
    )


def test_rescan_is_idempotent(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    service.rescan()
    service.rescan()
    w.agent.tool_called.emit("Bash", {"command": "ls"})
    client.speak.assert_called_once()  # exactly one connection despite 3 scans


def test_widget_without_agent_attr_is_skipped(qtbot):
    class Degraded(QObject):
        pass
    service, client, _ = make_service(qtbot, [Degraded()])  # must not raise


def test_apply_settings_updates_policy_and_client(qtbot):
    w = StubWidget(agent_name="alignment")
    service, client, _ = make_service(qtbot, [w])
    new = VoiceSettings(
        daemon_host="127.0.0.1", daemon_port=6000,
        agent_overrides={"alignment": AgentOverride(muted=False, voice="af_sky")},
    )
    service.apply_settings(new)
    client.configure.assert_called_with("127.0.0.1", 6000)
    w.agent.tool_called.emit("Grep", {})
    client.speak.assert_called_once_with("Searching", voice="af_sky")


def test_silent_tools_do_not_speak(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    w.agent.tool_called.emit("TodoWrite", {})
    w.agent.tool_called.emit("Task", {})
    client.speak.assert_not_called()


def test_handler_exceptions_do_not_propagate(qtbot):
    w = StubWidget()
    service, client, _ = make_service(qtbot, [w])
    client.speak.side_effect = RuntimeError("client exploded")
    w.agent.tool_called.emit("Bash", {})  # must not raise through emit
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_service.py -v`
Expected: FAIL — module not found. (pytest-qt provides `qtbot`; PySide6 comes with lightfall.)

- [ ] **Step 3: Implement service.py**

```python
# src/lightfall_voice/service.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_service.py -v`
Expected: all PASS. Then run the whole suite: `.venv\Scripts\python -m pytest -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightfall_voice/service.py tests/test_service.py
git commit -m "feat: VoiceService with AgentBus-driven session discovery"
```

---

### Task 7: Settings plugin + README

**Files:**
- Create: `src/lightfall_voice/settings.py`
- Modify: `README.md`
- Test: `tests/test_settings_plugin.py`

**Interfaces:**
- Consumes: `VoiceSettings.from_preferences/to_preferences` (Task 5); `VoiceService`, `get_service`, `set_service` (Task 6).
- Produces: `VoiceSettingsPlugin(SettingsPlugin)` — `name = "voice"`, `display_name = "Voice Notifications"`, `category = "plugins"`. `on_loaded()` builds `VoiceSettings` from `PreferencesManager`, creates + starts a `VoiceService`, registers it via `set_service`. `save_settings()` persists the widget state and calls `service.apply_settings(...)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_settings_plugin.py
from unittest.mock import MagicMock, patch

from lightfall_voice.policy import VoiceSettings
from lightfall_voice import service as service_module
from lightfall_voice.settings import VoiceSettingsPlugin


class FakePrefs:
    def __init__(self):
        self.data = {}
    def get(self, key, default=None):
        return self.data.get(key, default)
    def set(self, key, value):
        self.data[key] = value


def make_plugin(prefs=None):
    plugin = VoiceSettingsPlugin()
    plugin._prefs = prefs or FakePrefs()  # test seam, see implementation
    return plugin


def test_identity():
    p = make_plugin()
    assert p.name == "voice"
    assert p.display_name == "Voice Notifications"
    assert p.category == "plugins"


def test_on_loaded_starts_service():
    p = make_plugin()
    with patch.object(service_module, "VoiceService") as service_cls:
        p.on_loaded()
        service_cls.assert_called_once()
        service_cls.return_value.start.assert_called_once()
    assert service_module.get_service() is service_cls.return_value
    service_module.set_service(None)


def test_widget_roundtrip_saves_settings(qtbot):
    prefs = FakePrefs()
    p = make_plugin(prefs)
    widget = p.create_widget()
    qtbot.addWidget(widget)
    p.load_settings()

    p._enabled_check.setChecked(False)
    p._tools_check.setChecked(False)
    p._port_spin.setValue(6000)

    service = MagicMock()
    service_module.set_service(service)
    try:
        p.save_settings()
    finally:
        service_module.set_service(None)

    loaded = VoiceSettings.from_preferences(prefs)
    assert loaded.enabled is False
    assert loaded.tools is False
    assert loaded.daemon_port == 6000
    service.apply_settings.assert_called_once()


def test_validate_rejects_bad_port(qtbot):
    p = make_plugin()
    widget = p.create_widget()
    qtbot.addWidget(widget)
    p._host_edit.setText("")
    errors = p.validate()
    assert errors  # empty host is invalid


def test_test_button_speaks(qtbot):
    p = make_plugin()
    widget = p.create_widget()
    qtbot.addWidget(widget)
    service = MagicMock()
    service_module.set_service(service)
    try:
        p._on_test_clicked()
        service._client.speak.assert_called_once()
    finally:
        service_module.set_service(None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_settings_plugin.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement settings.py**

```python
# src/lightfall_voice/settings.py
"""Voice Notifications settings page. Owns the VoiceService lifecycle:
on_loaded() (preload) starts the service; save_settings() re-applies policy."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lightfall.plugins.settings_plugin import SettingsPlugin
from lightfall.utils.logging import logger

from lightfall_voice import service as service_module
from lightfall_voice.policy import VoiceSettings


class VoiceSettingsPlugin(SettingsPlugin):
    def __init__(self) -> None:
        super().__init__()
        self._prefs = None  # test seam; resolved lazily via _preferences()
        self._widget: QWidget | None = None

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "voice"

    @property
    def display_name(self) -> str:
        return "Voice Notifications"

    @property
    def description(self) -> str:
        return "Spoken notifications for Claude agent sessions via the local voice daemon"

    @property
    def category(self) -> str:
        return "plugins"

    # -- lifecycle -------------------------------------------------------------

    def _preferences(self):
        if self._prefs is None:
            from lightfall.ui.preferences.manager import PreferencesManager

            self._prefs = PreferencesManager.get_instance()
        return self._prefs

    def on_loaded(self) -> None:
        try:
            settings = VoiceSettings.from_preferences(self._preferences())
            service = service_module.VoiceService(settings)
            service.start()
            service_module.set_service(service)
        except Exception:
            logger.exception("lightfall-voice: failed to start VoiceService")

    # -- widget ---------------------------------------------------------------

    def create_widget(self, parent: QWidget | None = None) -> QWidget:
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)

        self._enabled_check = QCheckBox("Enable voice notifications")

        events_box = QGroupBox("Spoken events")
        events_form = QVBoxLayout(events_box)
        self._tools_check = QCheckBox("Tool announcements")
        self._summary_check = QCheckBox("Turn summaries")
        self._permission_check = QCheckBox("Permission requests")
        self._errors_check = QCheckBox("Errors")
        for cb in (self._tools_check, self._summary_check,
                   self._permission_check, self._errors_check):
            events_form.addWidget(cb)

        daemon_box = QGroupBox("Voice daemon")
        daemon_form = QFormLayout(daemon_box)
        self._host_edit = QLineEdit()
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        daemon_form.addRow("Host", self._host_edit)
        daemon_form.addRow("Port", self._port_spin)

        self._test_button = QPushButton("Test voice")
        self._test_button.clicked.connect(self._on_test_clicked)

        layout.addWidget(self._enabled_check)
        layout.addWidget(events_box)
        layout.addWidget(daemon_box)
        layout.addWidget(self._test_button)
        layout.addStretch()

        # Plugin outlives the dialog: drop widget refs on destruction
        # (same pattern as lightfall-vibe's settings plugin).
        widget.destroyed.connect(self._on_widget_destroyed)
        self._widget = widget
        return widget

    def _on_widget_destroyed(self) -> None:
        self._widget = None

    def _on_test_clicked(self) -> None:
        service = service_module.get_service()
        if service is not None:
            service._client.speak("Voice notifications are working")

    # -- load/save/validate -----------------------------------------------------

    def load_settings(self) -> None:
        s = VoiceSettings.from_preferences(self._preferences())
        self._enabled_check.setChecked(s.enabled)
        self._tools_check.setChecked(s.tools)
        self._summary_check.setChecked(s.summary)
        self._permission_check.setChecked(s.permission)
        self._errors_check.setChecked(s.errors)
        self._host_edit.setText(s.daemon_host)
        self._port_spin.setValue(s.daemon_port)

    def validate(self) -> list[str]:
        errors = []
        if self._widget is not None and not self._host_edit.text().strip():
            errors.append("Voice daemon host must not be empty")
        return errors

    def save_settings(self) -> None:
        prefs = self._preferences()
        existing = VoiceSettings.from_preferences(prefs)
        s = VoiceSettings(
            enabled=self._enabled_check.isChecked(),
            tools=self._tools_check.isChecked(),
            summary=self._summary_check.isChecked(),
            permission=self._permission_check.isChecked(),
            errors=self._errors_check.isChecked(),
            daemon_host=self._host_edit.text().strip(),
            daemon_port=self._port_spin.value(),
            agent_overrides=existing.agent_overrides,
        )
        s.to_preferences(prefs)
        service = service_module.get_service()
        if service is not None:
            service.apply_settings(s)
```

Note: per-agent overrides persist through `voice.agent_overrides` and are honored by the policy (Task 5), but v1 exposes no table UI for them — they're editable via preferences. Adding the table is a straightforward follow-up.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_settings_plugin.py -v`
Expected: all PASS.

- [ ] **Step 5: Expand README**

Replace `README.md` with:

```markdown
# lightfall-voice

Voice notifications plugin for Lightfall. Speaks Claude agent events through
the local Kokoro voice daemon from `~/.claude/hooks/voice_notifications`
(JSON line protocol, TCP `127.0.0.1:52849`).

## What it speaks

- **Permission requests** — "Permission needed to run a command" (high
  priority, non-interruptible)
- **Turn summaries** — trimmed final assistant text, or "Task complete"
- **Tool announcements** — "Reading loader", "Running a command"
  (rate-limited to one per tool per 3 s)
- **Errors** — "Agent error. Attention needed"

Subagent sessions are muted by default; their permission requests and errors
still speak, prefixed with the agent name. Low-priority events older than
10 s are dropped, never spoken late.

## Install

    pip install -e .

Registers via the `lightfall.plugins` entry point; the settings page appears
under Preferences → Plugins → Voice Notifications.

## Requirements

- The voice daemon must accept hookless direct messages
  (`voice_daemon.py` commit "Add hookless direct-message path"). The plugin
  does not start the daemon; if it is unreachable, notifications are
  silently dropped.

## Settings keys

`voice.enabled`, `voice.events.{tools,summary,permission,errors}`,
`voice.daemon_host`, `voice.daemon_port`, `voice.agent_overrides`
(`{agent_name: {"muted": bool, "voice": "af_sky"}}` — no UI yet, edit via
preferences).

## Tests

    .venv\Scripts\python -m pytest
```

- [ ] **Step 6: Run the full suite**

Run: `.venv\Scripts\python -m pytest -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lightfall_voice/settings.py tests/test_settings_plugin.py README.md
git commit -m "feat: Voice Notifications settings plugin owning the service lifecycle"
```

---

### Task 8: Live smoke test

**Files:** none created (manual verification).

- [ ] **Step 1: Verify daemon accepts direct messages**

```bash
cd C:\Users\rp\PycharmProjects\ncs\lightfall-voice
.venv\Scripts\python -c "from lightfall_voice.daemon_client import DaemonClient; import time; c = DaemonClient(); c.speak('Voice notifications are working'); time.sleep(2); c.stop()"
```

Expected: the message is spoken aloud (daemon must be running — it autostarts with any Claude Code hook activity, or run `voice_daemon.py foreground` from the hooks repo). If the daemon was already running before Task 1, restart it so the patched code is loaded: send `{"action": "shutdown"}` or kill the `voice_daemon.py` process; the next hook event or a manual `python voice_daemon.py foreground` brings it back.

- [ ] **Step 2: Launch Lightfall and verify end-to-end**

Launch Lightfall from its own checkout with this plugin installed
(`C:\Users\rp\PycharmProjects\ncs\lightfall\.venv\Scripts\python -m pip install -e C:\Users\rp\PycharmProjects\ncs\lightfall-voice`, then start Lightfall as usual). Confirm:
1. Preferences shows "Voice Notifications" under Plugins.
2. "Test voice" button speaks.
3. Asking the main assistant to read a file produces "Reading <stem>" and a spoken turn summary.
4. A permission prompt speaks the permission phrase.

Report any failures rather than patching lightfall core; if discovery misses
sessions, capture logs (`lightfall-voice:` prefix) for follow-up.
```

---

## Self-review notes

- Spec coverage: daemon reuse (T1/T4), manifest/settings preload (T2/T7), phrases (T2), events+priority (T3), policy incl. subagent mute/TTL/rate-limit (T5), discovery via AgentBus + widget scan (T6), tests-without-audio (conftest fake daemon), README (T7), smoke (T8). Spec's "per-agent overrides table" UI is deferred to a follow-up (persistence + policy support are implemented); noted in Task 7 and README.
- Type consistency: `decide()` returns `tuple[str, str | None] | None` used identically in T5/T6; preference keys identical in Global Constraints, T5, T7.
