"""Voice page on the scope-aware form: notifications, agents and daemon.

The old hand-built page (characterized in the commit before the rewrite) wrote every
voice key unscoped on every save, even untouched. The form page writes only what the
user edited, at the scope shown: the daemon and this machine's voice input keys at
This console, everything else at User. after_save() re-applies the saved settings to
the running VoiceService and VoiceInputController.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QComboBox, QKeySequenceEdit, QLineEdit

from lightfall_voice import service as service_module
from lightfall_voice import settings as settings_module
from lightfall_voice.policy import VoiceSettings
from lightfall_voice.settings import VoiceSettingsPlugin
from lightfall_voice.stt.controller import VoiceInputSettings

USER, LOCAL, BRANCH, BEAMLINE = "user:u1", "local", "branch:7.0.1.1", "beamline:7.0.1"  # as in conftest
OVERRIDES = "voice.agent_overrides"
DEFAULT_VOICE = "voice.default_voice"
CONSOLE_KEYS = {
    "voice.daemon_host",
    "voice.daemon_port",
    "voice.input.model_dir",
    "voice.input.python",
    "voice.input.input_device",
    "voice.input.device",
}
USER_KEYS = {
    "voice.enabled",
    "voice.events.tools",
    "voice.events.summary",
    "voice.events.permission",
    "voice.events.errors",
    DEFAULT_VOICE,
    OVERRIDES,
    "voice.input.enabled",
    "voice.input.hotkey",
    "voice.input.model",
    "voice.input.language",
    "voice.input.silence_s",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dialog(qtbot, scope=USER):
    from lightfall.ui.preferences.dialog import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog._prompt_restart = lambda: None
    if scope is not None:
        dialog.select_scope(scope)
    return dialog


def _form(dialog):
    from lightfall.ui.preferences.form.view import SettingsForm

    return next(f for f in dialog.findChildren(SettingsForm) if f._plugin.name == "voice")


def _editor(dialog, key):
    return _form(dialog)._rows[key].editor


def _pick(dialog, key, text):
    box = _editor(dialog, key).findChild(QComboBox)
    index = box.findText(text)
    assert index >= 0, (key, text, [box.itemText(i) for i in range(box.count())])
    box.setCurrentIndex(index)


def _watch(editor):
    seen = []
    editor.changed.connect(lambda: seen.append(1))
    return seen


# ---------------------------------------------------------------------------
# Identity, lifecycle (kept from the old page)
# ---------------------------------------------------------------------------


def test_identity(voice_prefs):
    p = VoiceSettingsPlugin()
    assert p.name == "voice"
    assert p.display_name == "Voice Notifications"
    assert p.category == "plugins"


def test_on_loaded_starts_service(voice_prefs):
    p = VoiceSettingsPlugin()
    with (
        patch.object(service_module, "VoiceService") as service_cls,
        patch.object(settings_module.controller_module, "VoiceInputController"),
    ):
        p.on_loaded()
        service_cls.assert_called_once()
        service_cls.return_value.start.assert_called_once()
    assert service_module.get_service() is service_cls.return_value
    service_module.set_service(None)
    settings_module.set_controller(None)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_console_keys_and_user_specs(voice_prefs):
    from lightfall.ui.preferences.form.schema import Action, section_fields

    page = VoiceSettingsPlugin()
    assert page.is_form_page()
    for name in ("create_widget", "load_settings", "save_settings", "validate", "_prefs"):
        assert name not in vars(VoiceSettingsPlugin), name
    assert not hasattr(page, "_prefs")
    assert set(page.console_keys()) == CONSOLE_KEYS
    specs = {s.key: s for s in page.specs()}
    assert set(specs) == USER_KEYS
    assert all(s.scopes == frozenset({"user"}) for s in specs.values())
    assert specs[OVERRIDES].type is dict
    assert specs["voice.enabled"].type is bool
    assert specs["voice.input.silence_s"].type is float
    assert page.validate_values(None) == []

    for key in CONSOLE_KEYS:
        assert voice_prefs.allowed_at(key, LOCAL), key
        for scope in (USER, BRANCH, BEAMLINE, "facility"):
            assert not voice_prefs.allowed_at(key, scope), (key, scope)
    for key in USER_KEYS:
        assert voice_prefs.allowed_at(key, USER), key
        for scope in (LOCAL, BRANCH, BEAMLINE, "facility"):
            assert not voice_prefs.allowed_at(key, scope), (key, scope)

    sections = page.sections()
    assert [s.title for s in sections] == [
        "Voice notifications", "Agents", "Voice daemon", "Voice input", "Voice input on this machine"
    ]
    notif, agents, daemon, voice_input, machine = sections
    assert [(f.key, f.editor.kind) for f in section_fields(notif)] == [
        ("voice.enabled", "toggle"),
        ("voice.events.tools", "toggle"),
        ("voice.events.summary", "toggle"),
        ("voice.events.permission", "toggle"),
        ("voice.events.errors", "toggle"),
        (DEFAULT_VOICE, "custom"),
    ]
    [test] = [i for i in notif.items if isinstance(i, Action)]
    assert test.label == "Test voice"
    [ov] = section_fields(agents)
    assert (ov.key, ov.editor.kind) == (OVERRIDES, "custom")
    assert ov.description == (
        "Checked agents are muted; permission requests and errors always speak. "
        "Pick a voice per agent, or leave it on the default voice."
    )
    host, port = section_fields(daemon)
    assert (host.key, host.editor.kind, host.editor.blank_is_unset, host.placeholder) == (
        "voice.daemon_host", "text", True, "127.0.0.1")
    assert (port.key, port.editor.kind, port.editor.minimum, port.editor.maximum, port.placeholder) == (
        "voice.daemon_port", "integer", 1, 65535, "52849")
    enabled, hotkey, model, language, silence = section_fields(voice_input)
    assert (enabled.key, enabled.editor.kind) == ("voice.input.enabled", "toggle")
    assert (hotkey.key, hotkey.editor.kind, hotkey.placeholder) == ("voice.input.hotkey", "custom", "F9")
    assert (model.key, model.editor.kind) == ("voice.input.model", "choice")
    assert [v for v, _ in model.editor.options] == ["base", "small", "medium", "large-v3", "distil-large-v3"]
    assert (language.key, language.editor.kind) == ("voice.input.language", "choice")
    assert list(language.editor.options)[0] == ("", "Auto-detect")
    assert (silence.key, silence.editor.kind, silence.unit) == ("voice.input.silence_s", "number", "s")
    assert (silence.editor.minimum, silence.editor.maximum, silence.editor.step) == (0.3, 5.0, 0.1)
    device, input_device, model_dir, python = section_fields(machine)
    assert (device.key, device.editor.kind) == ("voice.input.device", "choice")
    assert list(device.editor.options) == [
        ("auto", "Auto (GPU if available)"), ("gpu", "GPU"), ("cpu", "CPU")]
    assert (input_device.key, input_device.editor.kind) == ("voice.input.input_device", "choice")
    assert callable(input_device.editor.options)
    assert (model_dir.key, model_dir.editor.kind, model_dir.editor.mode, model_dir.placeholder) == (
        "voice.input.model_dir", "path", "dir", "Empty: the Hugging Face cache")
    assert (python.key, python.editor.kind, python.editor.mode, python.placeholder) == (
        "voice.input.python", "path", "file", "Empty: Lightfall's own interpreter")


def test_page_shows_user_keys_at_user_and_console_keys_at_this_console(qtbot, voice_env):
    dialog = _dialog(qtbot)
    form = _form(dialog)
    assert set(form._rows) == USER_KEYS
    dialog.select_scope(LOCAL)
    assert set(form._rows) == CONSOLE_KEYS


# ---------------------------------------------------------------------------
# Editors
# ---------------------------------------------------------------------------


def test_agent_voices_editor_round_trip_and_drops_defaults(qtbot, voice_prefs):
    from lightfall_voice.settings_editors import AgentVoicesEditor

    service = MagicMock()
    service.attached_agent_names.return_value = ["xpcs-analysis"]
    service_module.set_service(service)
    try:
        editor = AgentVoicesEditor()
    finally:
        service_module.set_service(None)
    qtbot.addWidget(editor)
    assert editor.full_width
    tree = editor._tree
    assert [tree.headerItem().text(c) for c in range(2)] == ["Agent (checked = muted)", "Voice"]
    seen = _watch(editor)
    stored = {
        "old-agent": {"muted": True, "voice": None},
        "observer": {"muted": True, "voice": "af_sky"},
    }
    editor.set_value(stored)
    assert seen == []  # set_value never emits
    rows = editor.items()
    assert {"lightfall", "old-agent", "observer"} <= set(rows)
    assert editor.value() == stored
    assert rows["old-agent"][0].checkState(0) == Qt.CheckState.Checked
    assert rows["observer"][1].voice() == "af_sky"
    assert rows["lightfall"][0].checkState(0) == Qt.CheckState.Unchecked
    assert rows["lightfall"][1].currentText() == "Default voice"

    # Mute lightfall and give it a voice: both land in its entry.
    rows["lightfall"][0].setCheckState(0, Qt.CheckState.Checked)
    assert seen == [1]
    rows["lightfall"][1].select_voice("am_michael")
    assert len(seen) == 2
    assert editor.value()["lightfall"] == {"muted": True, "voice": "am_michael"}

    # Unmuting keeps the agent's voice.
    rows["observer"][0].setCheckState(0, Qt.CheckState.Unchecked)
    assert editor.value()["observer"] == {"muted": False, "voice": "af_sky"}

    # Back to the default (unmuted, default voice): the entry is dropped.
    rows["observer"][1].select_voice("")
    rows["lightfall"][0].setCheckState(0, Qt.CheckState.Unchecked)
    rows["lightfall"][1].select_voice("")
    assert editor.value() == {"old-agent": {"muted": True, "voice": None}}
    rows["old-agent"][0].setCheckState(0, Qt.CheckState.Unchecked)
    assert editor.value() is None  # nothing left to store

    # A typed (uncatalogued) voice id is kept.
    combo = rows["lightfall"][1]
    combo.setEditText("zf_xiaoxiao")
    combo.lineEdit().editingFinished.emit()
    assert editor.value() == {"lightfall": {"muted": False, "voice": "zf_xiaoxiao"}}

    # Reloading the stored dict shows it again without emitting.
    count = len(seen)
    editor.set_value({"lightfall": {"muted": False, "voice": "zf_xiaoxiao"}})
    assert len(seen) == count
    assert editor.items()["lightfall"][1].voice() == "zf_xiaoxiao"
    editor.set_value(None)
    assert editor.value() is None


def test_external_mute_picked_up_and_not_overwritten(qtbot, voice_env):
    """Review Focus 2: SessionMuteAction writes voice.agent_overrides while the dialog is open."""
    prefs = voice_env.prefs
    prefs.layers[(USER, OVERRIDES)] = {"observer": {"muted": True, "voice": "af_sky"}}
    dialog = _dialog(qtbot)
    editor = _editor(dialog, OVERRIDES)
    assert editor.items()["observer"][0].checkState(0) == Qt.CheckState.Checked

    external = {
        "observer": {"muted": True, "voice": "af_sky"},
        "lightfall": {"muted": True, "voice": None},
    }
    prefs.set(OVERRIDES, external)  # unscoped, as the title-bar mute button saves
    prefs.calls.clear()
    _form(dialog).refresh_from_store()
    assert editor.items()["lightfall"][0].checkState(0) == Qt.CheckState.Checked

    _pick(dialog, "voice.enabled", "Off")
    assert dialog._apply() is True
    assert prefs.calls == [("voice.enabled", False, USER)]
    assert prefs.layers[(USER, OVERRIDES)] == external


def test_voice_choice_not_set_placeholder_and_typed_id(qtbot, voice_env):
    from lightfall.ui.preferences.form.editors import NOT_SET

    from lightfall_voice.settings_editors import VoiceChoiceEditor

    editor = VoiceChoiceEditor()
    qtbot.addWidget(editor)
    seen = _watch(editor)
    combo = editor.findChild(QComboBox)
    line = combo.lineEdit()
    assert combo.isEditable()
    assert combo.itemText(1) == "Voice daemon's own default"
    assert combo.itemText(2) == "af_heart — Heart (American female, warm)"

    editor.set_value(None)
    assert (line.text(), line.placeholderText()) == ("", NOT_SET)  # no literal "Not set" text
    assert editor.value() is None
    editor.set_value("")
    assert editor.value() == ""
    assert combo.currentText() == "Voice daemon's own default"
    editor.set_value("bf_emma")
    assert editor.value() == "bf_emma"
    assert combo.currentText() == "bf_emma — Emma (British female)"
    editor.set_value("zf_xiaoxiao")
    assert editor.value() == "zf_xiaoxiao"
    assert seen == []

    editor.set_value(None)
    line.setText("zf_other")
    line.editingFinished.emit()
    assert seen == [1]
    assert editor.value() == "zf_other"
    line.editingFinished.emit()  # nothing new
    assert seen == [1]

    # Through the page: a typed id is saved at User.
    dialog = _dialog(qtbot)
    page_line = _editor(dialog, DEFAULT_VOICE).findChild(QComboBox).lineEdit()
    assert page_line.text() == ""
    page_line.setText("zf_xiaoxiao")
    page_line.editingFinished.emit()
    assert dialog._apply() is True
    assert voice_env.prefs.calls == [(DEFAULT_VOICE, "zf_xiaoxiao", USER)]


def test_hotkey_editor_round_trip(qtbot, voice_env):
    from lightfall.ui.preferences.form.schema import Custom, SettingField

    from lightfall_voice.settings_editors import hotkey_editor

    editor = hotkey_editor(SettingField("voice.input.hotkey", Custom(hotkey_editor), placeholder="F9"))
    qtbot.addWidget(editor)
    seen = _watch(editor)
    edit = editor.findChild(QKeySequenceEdit)
    assert edit.findChild(QLineEdit).placeholderText() == "F9"
    assert editor.value() is None
    editor.set_value("Ctrl+F8")
    assert editor.value() == "Ctrl+F8"
    assert edit.keySequence().toString() == "Ctrl+F8"
    editor.set_value(None)
    assert editor.value() is None
    assert seen == []

    edit.setKeySequence(QKeySequence("F10"))
    assert seen == [1]
    assert editor.value() == "F10"
    edit.clear()
    assert len(seen) == 2
    assert editor.value() is None

    dialog = _dialog(qtbot)
    _editor(dialog, "voice.input.hotkey").findChild(QKeySequenceEdit).setKeySequence(QKeySequence("Ctrl+F8"))
    assert dialog._apply() is True
    assert voice_env.prefs.calls == [("voice.input.hotkey", "Ctrl+F8", USER)]


# ---------------------------------------------------------------------------
# Save, apply, actions
# ---------------------------------------------------------------------------


def test_after_save_applies_to_service_and_controller(qtbot, voice_env):
    prefs = voice_env.prefs
    page = voice_env.page
    page.after_save(USER, {"voice.enabled": False})  # no service, no controller: nothing to do

    service, controller = MagicMock(), MagicMock()
    service_module.set_service(service)
    settings_module.set_controller(controller)
    prefs.layers[(USER, "voice.events.tools")] = False
    prefs.layers[(LOCAL, "voice.daemon_port")] = 6000
    prefs.layers[(USER, "voice.input.silence_s")] = 1.5
    page.after_save(USER, {"voice.events.tools": False})
    page.after_save(LOCAL, {"voice.daemon_port": 6000})  # same Apply, second scope: same result
    assert service.apply_settings.call_count == 2
    assert controller.apply_settings.call_count == 2
    expected = VoiceSettings.from_preferences(prefs)
    assert expected.tools is False and expected.daemon_port == 6000
    for call in service.apply_settings.call_args_list:
        assert call.args[0] == expected
    expected_input = VoiceInputSettings.from_preferences(prefs)
    assert expected_input.silence_s == 1.5
    for call in controller.apply_settings.call_args_list:
        assert call.args[0] == expected_input

    service.reset_mock()
    controller.reset_mock()
    page.after_save(USER, {"something.else": 1})
    service.apply_settings.assert_not_called()
    controller.apply_settings.assert_not_called()

    # Through the dialog: edits at User and This console are both applied.
    dialog = _dialog(qtbot)
    _pick(dialog, "voice.events.summary", "Off")
    dialog.select_scope(LOCAL)
    _pick(dialog, "voice.input.device", "CPU")
    assert dialog._apply() is True
    assert sorted(prefs.calls) == [
        ("voice.events.summary", False, USER),
        ("voice.input.device", "cpu", LOCAL),
    ]
    applied = service.apply_settings.call_args.args[0]
    assert applied.summary is False
    assert controller.apply_settings.call_args.args[0].device == "cpu"


def test_test_voice_action(qtbot, voice_env):
    from lightfall.ui.preferences.form.items import ActionItem

    dialog = _dialog(qtbot)
    item = next(a for a in _form(dialog).findChildren(ActionItem) if a.action.label == "Test voice")
    item.button.click()
    qtbot.waitUntil(lambda: item.status.text() == "Voice service not running", timeout=5000)

    service = MagicMock()
    service_module.set_service(service)
    item.button.click()
    qtbot.waitUntil(lambda: item.status.text() == "Spoken", timeout=5000)
    service.speak_test.assert_called_once_with()
    assert voice_env.prefs.calls == []


def test_untouched_page_writes_nothing(qtbot, voice_env):
    service, controller = MagicMock(), MagicMock()
    service_module.set_service(service)
    settings_module.set_controller(controller)
    dialog = _dialog(qtbot)
    assert dialog._apply() is True
    dialog.select_scope(LOCAL)
    assert dialog._apply() is True
    dialog.select_scope(USER)
    assert dialog._apply() is True
    assert voice_env.prefs.calls == []
    service.apply_settings.assert_not_called()
    controller.apply_settings.assert_not_called()
    service.speak_test.assert_not_called()
    assert voice_env.file_dialogs == []


def test_agent_voices_read_only_survives_reload(qtbot, voice_prefs):
    """refresh_from_store calls set_value without set_read_only: a locked list stays locked."""
    from lightfall_voice.settings_editors import AgentVoicesEditor

    editor = AgentVoicesEditor()
    qtbot.addWidget(editor)
    editor.set_read_only(True)
    editor.set_value({"observer": {"muted": True, "voice": "af_sky"}, "lightfall": {"muted": True, "voice": None}})
    rows = editor.items()
    assert {"observer", "lightfall"} <= set(rows)
    for item, combo in rows.values():
        assert not item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert not combo.isEnabled()
    editor.set_read_only(False)
    for item, combo in editor.items().values():
        assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable
        assert combo.isEnabled()
