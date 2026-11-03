"""Shared fixtures: a fake voice daemon capturing JSON-line requests."""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# Preferences form harness (modelled on lightfall's tests/ui/preferences/test_devices_form.py)
# ---------------------------------------------------------------------------

USER, LOCAL, BRANCH, BEAMLINE = "user:u1", "local", "branch:7.0.1.1", "beamline:7.0.1"
WRITABLE = (USER, LOCAL, BRANCH)
ORDER = (BEAMLINE, BRANCH, LOCAL, USER)  # broadest first


def _scoped_prefs_class():
    from PySide6.QtCore import QObject, Signal

    class ScopedPrefs(QObject):
        """One layer per scope; a key's value is its narrowest layer. Unscoped writes go to USER."""

        scopes_changed = Signal(object)
        scope_info = SimpleNamespace(mode="server")

        def __init__(self):
            super().__init__()
            self.layers: dict = {}
            self.calls: list[tuple] = []
            self.slots: dict = {}

        def scope_choices(self):
            from lightfall.ui.preferences.scope_api import ScopeChoice

            return [
                ScopeChoice(USER, "user", "User", None, ""),
                ScopeChoice(LOCAL, "local", "This console", None, ""),
                ScopeChoice(BRANCH, "branch", "Branch", "maintainer", ""),
                ScopeChoice(BEAMLINE, "beamline", "Beamline", "viewer", "Read-only"),
            ]

        def default_scope(self):
            return USER

        def allowed_at(self, key, scope):
            from lightfall.ui.preferences.scope_api import allowed_at

            return allowed_at(self, key, scope)

        def writable_at(self, key, scope):
            return scope in WRITABLE and self.allowed_at(key, scope)

        def lock_at(self, key, scope):
            return None

        def layer_value(self, key, scope):
            return self.layers.get((scope, key))

        def inherited_items(self, key, scope):
            return {}

        def resolved_items(self, key):
            return {}

        def get(self, key, default=None):
            for s in reversed(ORDER):
                if (s, key) in self.layers:
                    return self.layers[(s, key)]
            return default

        def set(self, key, value, *, scope=None):
            self.calls.append((key, value, scope))
            self.layers[(scope or USER, key)] = value

        def remove(self, key, *, scope=None):
            self.calls.append((key, None, scope))
            self.layers.pop((scope or USER, key), None)

        def subscribe(self, key, slot):
            pass

        def unsubscribe(self, key, slot):
            pass

    return ScopedPrefs


@pytest.fixture(autouse=True)
def _fresh_spec_registry():
    """Pages register specs and console keys process-wide; start each test clean."""
    from lightfall.scopes.keys import _reset_client_registry

    _reset_client_registry()
    yield
    _reset_client_registry()


@pytest.fixture
def _no_real_audio_inputs(monkeypatch):
    """Never enumerate the machine's microphones."""
    from PySide6.QtMultimedia import QMediaDevices

    monkeypatch.setattr(QMediaDevices, "audioInputs", staticmethod(lambda: []))


@pytest.fixture
def voice_prefs(monkeypatch, qapp, _no_real_audio_inputs):
    """A scoped fake store installed as ``PreferencesManager.get_instance()``."""
    prefs = _scoped_prefs_class()()
    monkeypatch.setattr(
        "lightfall.ui.preferences.manager.PreferencesManager.get_instance",
        classmethod(lambda cls: prefs),
    )
    return prefs


@pytest.fixture
def voice_env(monkeypatch, voice_prefs):
    """The Voice page alone in a PreferencesDialog over ``voice_prefs``; no real service/controller."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from lightfall_voice import service as service_module
    from lightfall_voice import settings as settings_module
    from lightfall_voice.settings import VoiceSettingsPlugin

    service_module.set_service(None)
    settings_module.set_controller(None)
    pages = [VoiceSettingsPlugin()]
    plugins = SimpleNamespace(
        get_ready_by_type=lambda _t: [SimpleNamespace(instance=p) for p in pages],
        get_all=lambda: [],
    )
    services = SimpleNamespace(get=lambda _cls: plugins)
    monkeypatch.setattr(
        "lightfall.core.services.ServiceRegistry.get_instance", classmethod(lambda cls: services)
    )
    boxes: list[str] = []
    for name in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda _p, _t, text, *a, **k: boxes.append(text))
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    dialogs: list[tuple] = []
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: dialogs.append(a) or "")
    )
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: dialogs.append(a) or ("", ""))
    )
    yield SimpleNamespace(prefs=voice_prefs, page=pages[0], boxes=boxes, file_dialogs=dialogs)
    service_module.set_service(None)
    settings_module.set_controller(None)
