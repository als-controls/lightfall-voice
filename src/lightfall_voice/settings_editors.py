"""Custom form editors for Settings -> Voice Notifications.

None of them writes preferences: the form reads ``value()`` after ``changed``
and writes it at the scope shown. ``set_value()`` never emits ``changed``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lightfall.ui.preferences.form.editors import NOT_SET, ChoiceEditor, FieldEditor, muted_color
from lightfall.ui.preferences.form.schema import EditorSpec, SettingField
from lightfall.utils.logging import logger

from lightfall_voice import service as service_module
from lightfall_voice.policy import (
    AgentOverride,
    muted_by_default,
    with_agent_muted,
    with_agent_voice,
)
from lightfall_voice.voices import KOKORO_VOICES

DAEMON_DEFAULT_LABEL = "Voice daemon's own default"
AGENT_DEFAULT_LABEL = "Default voice"
AGENT_HEADERS = ("Agent (checked = muted)", "Voice")


def _voice_options() -> list[tuple[str, str]]:
    return [(voice_id, f"{voice_id} — {label}") for voice_id, label in KOKORO_VOICES]


def _typed_voice(text: str) -> str:
    """A typed voice id: the first word of the text (as the old page took it)."""
    return text.split()[0] if text.strip() else ""


# -- default voice -----------------------------------------------------------


class VoiceChoiceEditor(ChoiceEditor):
    """Editable voice combo: "Not set", the daemon's own default (""), then the Kokoro voices.

    While "Not set" is current the line is empty with a muted "Not set"
    placeholder, so typing starts a fresh id. A typed id that is not listed
    is the value (its first word); finishing an edit emits ``changed``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        options = [("", DAEMON_DEFAULT_LABEL), *_voice_options()]
        super().__init__(EditorSpec("choice", options=options), parent)
        self._editable = True
        self._combo.setEditable(True)
        self._combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        line = self._combo.lineEdit()
        line.setPlaceholderText(NOT_SET)
        line.editingFinished.connect(self._on_edit_finished)
        line.textChanged.connect(lambda _t: self._mute_not_set())
        self._blank_not_set()
        self._mute_not_set()
        self._last: Any = self.value()
        self.changed.connect(self._remember)

    def _remember(self) -> None:
        self._last = self.value()

    def _on_edit_finished(self) -> None:
        if not self._loading and self.value() != self._last:
            self.changed.emit()

    def _shows_not_set(self) -> bool:
        return self._combo.currentIndex() == 0 and self._combo.currentText() in ("", NOT_SET)

    def _apply_mute(self) -> None:
        super()._apply_mute()
        if getattr(self, "_editable", False):  # set after ChoiceEditor.__init__
            line = self._combo.lineEdit()
            pal = line.palette()
            pal.setColor(QPalette.ColorRole.PlaceholderText, muted_color(self._combo))
            line.setPalette(pal)

    def _blank_not_set(self) -> None:
        """On "Not set": empty the line (QComboBox copied the entry's text into it)."""
        if getattr(self, "_editable", False) and self._combo.currentIndex() == 0:
            if self._combo.currentText() == NOT_SET:
                self._combo.lineEdit().clear()

    def _on_index(self, index: int) -> None:
        self._blank_not_set()
        super()._on_index(index)

    def value(self) -> str | None:
        text = self._combo.currentText().strip()
        if not text or text == NOT_SET:
            return None
        index = self._combo.findText(text)
        if index > 0:
            return self._values[index]
        return _typed_voice(text)

    def set_value(self, value: Any) -> None:
        if value is None or value in self._values[1:]:
            super().set_value(value)
            self._blank_not_set()
            self._mute_not_set()
        else:  # an id the list does not offer is shown as typed text
            self._loading = True
            try:
                self._combo.setCurrentIndex(-1)
                self._combo.setEditText(str(value))
                self._mute_not_set()
            finally:
                self._loading = False
        self._last = self.value()


def voice_choice_editor(field: SettingField) -> VoiceChoiceEditor:
    """``Custom`` factory for ``voice.default_voice``."""
    return VoiceChoiceEditor()


# -- hotkey ------------------------------------------------------------------


class HotkeyEditor(FieldEditor):
    """A ``QKeySequenceEdit``; the value is the PortableText, None when empty."""

    def __init__(self, placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edit = QKeySequenceEdit(self)
        self._edit.setClearButtonEnabled(True)
        self._edit.setMaximumSequenceLength(1)
        line = self._edit.findChild(QLineEdit)
        if line is not None:
            line.setPlaceholderText(placeholder or NOT_SET)
        layout.addWidget(self._edit)
        self._loading = False
        self._last: str | None = None
        self._edit.keySequenceChanged.connect(self._on_changed)

    def _on_changed(self, _seq: QKeySequence) -> None:
        if self._loading:
            return
        value = self.value()
        if value != self._last:
            self._last = value
            self.changed.emit()

    def value(self) -> str | None:
        text = self._edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        return text or None

    def set_value(self, value: Any) -> None:
        self._loading = True
        try:
            self._edit.setKeySequence(QKeySequence(str(value)) if value else QKeySequence())
        finally:
            self._loading = False
        self._last = self.value()

    def set_read_only(self, read_only: bool) -> None:
        self._edit.setEnabled(not read_only)


def hotkey_editor(field: SettingField) -> HotkeyEditor:
    """``Custom`` factory for ``voice.input.hotkey``."""
    return HotkeyEditor(field.placeholder)


# -- per-agent mute and voice --------------------------------------------------


def known_agent_names() -> set[str]:
    """Agent specs plus live sessions: every name a session tab can carry."""
    names: set[str] = {service_module.PRIMARY_AGENT}
    try:
        from lightfall.agents.registry import AgentSpecRegistry

        names.update(spec.name for spec in AgentSpecRegistry.get_instance().specs())
    except Exception:
        logger.debug("lightfall-voice: agent registry unavailable for the mute list")
    service = service_module.get_service()
    if service is not None:
        try:
            names.update(service.attached_agent_names())
        except Exception:
            logger.debug("lightfall-voice: live sessions unavailable for the mute list")
    return names


def _overrides_from(raw: Any) -> dict[str, AgentOverride]:
    """Stored ``voice.agent_overrides`` as AgentOverrides (as VoiceSettings reads it)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, AgentOverride] = {}
    for name, o in raw.items():
        o = o if isinstance(o, dict) else {}
        out[str(name)] = AgentOverride(muted=bool(o.get("muted", False)), voice=o.get("voice") or None)
    return out


class _AgentVoiceCombo(QComboBox):
    """Per-row editable voice picker: "Default voice" (""), then the Kokoro voices."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.addItem(AGENT_DEFAULT_LABEL, "")
        for voice_id, label in _voice_options():
            self.addItem(label, voice_id)

    def voice(self) -> str:
        """The chosen voice id ("" = the default voice); a typed id is taken verbatim."""
        text = self.currentText().strip()
        index = self.findText(text)
        if index >= 0:
            return self.itemData(index) or ""
        return _typed_voice(text)

    def select_voice(self, voice: str | None) -> None:
        index = self.findData(voice or "")
        if index < 0:
            self.addItem(voice, voice)  # an id the catalogue doesn't list
            index = self.count() - 1
        self.setCurrentIndex(index)


class AgentVoicesEditor(FieldEditor):
    """``voice.agent_overrides`` as a list: checked agents are muted, one voice picker each.

    Rows are the known agents plus every agent already stored. ``value()``
    starts from the stored dict, so entries for agents not listed survive;
    entries that restate the default (not muted, default voice) are dropped,
    and an empty result is None (not set).
    """

    full_width = True

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._tree = QTreeWidget(self)
        self._tree.setHeaderLabels(list(AGENT_HEADERS))
        self._tree.setRootIsDecorated(False)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self._tree)
        self._stored: dict[str, AgentOverride] = {}
        self._loading = False
        self._read_only = False
        self._last: Any = None
        self._tree.itemChanged.connect(lambda _item, column: column == 0 and self._on_edit())
        self.set_value(None)

    def _rows(self):
        for row in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(row)
            yield item, self._tree.itemWidget(item, 1)

    def items(self) -> dict[str, tuple[QTreeWidgetItem, _AgentVoiceCombo]]:
        """Rows by agent name (for tests and callers)."""
        return {item.text(0): (item, combo) for item, combo in self._rows()}

    def _on_edit(self) -> None:
        if self._loading:
            return
        value = self.value()
        if value != self._last:
            self._last = value
            self.changed.emit()

    def value(self) -> dict[str, dict[str, Any]] | None:
        overrides = dict(self._stored)
        for item, combo in self._rows():
            name = item.text(0)
            overrides = with_agent_voice(overrides, name, combo.voice())
            overrides = with_agent_muted(overrides, name, item.checkState(0) == Qt.CheckState.Checked)
        if not overrides:
            return None
        return {name: {"muted": o.muted, "voice": o.voice} for name, o in sorted(overrides.items())}

    def set_value(self, value: Any) -> None:
        self._loading = True
        try:
            self._stored = _overrides_from(value)
            self._tree.clear()
            for name in sorted(known_agent_names() | set(self._stored)):
                item = QTreeWidgetItem([name])
                checkable = Qt.ItemFlag.ItemIsUserCheckable
                # A reload (refresh_from_store) must not unlock a read-only list.
                item.setFlags(item.flags() & ~checkable if self._read_only else item.flags() | checkable)
                override = self._stored.get(name)
                muted = override.muted if override is not None else muted_by_default(name)
                item.setCheckState(0, Qt.CheckState.Checked if muted else Qt.CheckState.Unchecked)
                self._tree.addTopLevelItem(item)
                combo = _AgentVoiceCombo()
                combo.select_voice(override.voice if override is not None else None)
                combo.setEnabled(not self._read_only)
                combo.currentIndexChanged.connect(lambda _i: self._on_edit())
                combo.lineEdit().editingFinished.connect(self._on_edit)
                self._tree.setItemWidget(item, 1, combo)
        finally:
            self._loading = False
        self._last = self.value()

    def set_read_only(self, read_only: bool) -> None:
        self._read_only = read_only
        for item, combo in self._rows():
            flags = item.flags()
            if read_only:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsUserCheckable)
            else:
                item.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
            combo.setEnabled(not read_only)


def agent_voices_editor(field: SettingField) -> AgentVoicesEditor:
    """``Custom`` factory for ``voice.agent_overrides``."""
    return AgentVoicesEditor()
