"""FasterWhisperTranscriber with fake ``faster_whisper`` / ``ctranslate2`` modules.

Never loads a real model: both modules are replaced in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lightfall_voice.stt_service.transcriber import FasterWhisperTranscriber


class _Fakes:
    def __init__(self, *, cuda_fails=False, cached=True, cuda_devices=1):
        self.models: list[dict] = []
        self.downloads: list[dict] = []
        self.cuda_fails = cuda_fails
        self.cached = cached
        fakes = self

        class WhisperModel:
            def __init__(self, model_size_or_path, **kw):
                rec = {"model": model_size_or_path, **kw}
                fakes.models.append(rec)
                if kw.get("device") == "cuda" and fakes.cuda_fails:
                    raise RuntimeError("CUDA driver missing")
                self.kw = kw
                self.transcribe_calls: list[dict] = []

            def transcribe(self, audio, **kw):
                self.transcribe_calls.append(kw)
                segs = [SimpleNamespace(text=" hello"), SimpleNamespace(text=" world")]
                return iter(segs), SimpleNamespace(language="en")

        def download_model(size_or_id, **kw):
            fakes.downloads.append({"model": size_or_id, **kw})
            if kw.get("local_files_only") and not fakes.cached:
                raise RuntimeError("not in cache")
            return f"/cache/{size_or_id}"

        fw = types.ModuleType("faster_whisper")
        fw.WhisperModel = WhisperModel
        fw.download_model = download_model
        ct2 = types.ModuleType("ctranslate2")
        ct2.get_cuda_device_count = lambda: cuda_devices
        self.fw, self.ct2 = fw, ct2


@pytest.fixture
def fakes(monkeypatch):
    def install(**kw):
        f = _Fakes(**kw)
        monkeypatch.setitem(sys.modules, "faster_whisper", f.fw)
        monkeypatch.setitem(sys.modules, "ctranslate2", f.ct2)
        return f

    return install


def test_auto_falls_back_to_cpu(fakes):
    f = fakes(cuda_fails=True)
    t = FasterWhisperTranscriber("small", "auto")
    t.load()
    assert t.device == "cpu"
    assert [(m["device"], m["compute_type"]) for m in f.models] == [
        ("cuda", "float16"),
        ("cpu", "int8"),
    ]


def test_auto_uses_cuda_when_it_loads(fakes):
    f = fakes()
    t = FasterWhisperTranscriber("small", "auto")
    t.load()
    assert t.device == "cuda"
    assert [(m["device"], m["compute_type"]) for m in f.models] == [("cuda", "float16")]


def test_forced_gpu_error_raises(fakes):
    f = fakes(cuda_fails=True)
    t = FasterWhisperTranscriber("small", "gpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        t.load()
    assert [m["device"] for m in f.models] == ["cuda"]
    assert t.device == ""


def test_cpu_uses_int8(fakes):
    f = fakes()
    t = FasterWhisperTranscriber("base", "cpu")
    t.load()
    assert t.device == "cpu"
    assert [(m["device"], m["compute_type"]) for m in f.models] == [("cpu", "int8")]


def test_model_dir_offline_args(fakes):
    f = fakes()
    t = FasterWhisperTranscriber("small", "cpu", model_dir="D:/models")
    t.load()
    (m,) = f.models
    assert m["download_root"] == "D:/models"
    assert m["local_files_only"] is True


def test_no_model_dir_allows_download(fakes):
    f = fakes()
    FasterWhisperTranscriber("small", "cpu").load()
    (m,) = f.models
    assert m.get("download_root") is None
    assert m.get("local_files_only", False) is False


def test_progress_downloading_vs_loading(fakes):
    seen: list[str] = []
    fakes(cached=False)
    FasterWhisperTranscriber("small", "cpu", progress=seen.append).load()
    assert seen == ["downloading"]

    seen.clear()
    fakes(cached=True)
    FasterWhisperTranscriber("small", "cpu", progress=seen.append).load()
    assert seen == ["loading"]

    seen.clear()
    fakes(cached=False)  # model_dir is offline: never a download
    FasterWhisperTranscriber(
        "small", "cpu", model_dir="D:/m", progress=seen.append
    ).load()
    assert seen == ["loading"]


def test_transcribe_joins_segments_with_vad(fakes):
    fakes()
    t = FasterWhisperTranscriber("small", "cpu")
    t.load()
    text = t.transcribe(np.zeros(16000, dtype=np.float32), "en")
    assert text == "hello world"
    call = t._model.transcribe_calls[-1]
    assert call["language"] == "en"
    assert call["vad_filter"] is True


def test_transcribe_empty_language_is_autodetect(fakes):
    fakes()
    t = FasterWhisperTranscriber("small", "cpu")
    t.load()
    t.transcribe(np.zeros(16000, dtype=np.float32), "")
    assert t._model.transcribe_calls[-1]["language"] is None


def test_windows_adds_nvidia_dll_dirs(fakes, monkeypatch, tmp_path):
    from lightfall_voice.stt_service import transcriber as tr

    fakes()
    for pkg in ("cublas", "cudnn"):
        (tmp_path / "nvidia" / pkg / "bin").mkdir(parents=True)
    added: list[str] = []
    monkeypatch.setattr(tr, "_ADDED_DLL_DIRS", {})
    monkeypatch.setattr(tr.os, "name", "nt")
    monkeypatch.setattr(tr.os, "add_dll_directory", added.append, raising=False)

    def find_spec(name):
        pkg = name.rsplit(".", 1)[-1]
        return SimpleNamespace(
            submodule_search_locations=[str(tmp_path / "nvidia" / pkg)]
        )

    monkeypatch.setattr(tr.importlib.util, "find_spec", find_spec)
    tr.FasterWhisperTranscriber("small", "cpu").load()
    assert sorted(added) == sorted(
        str(tmp_path / "nvidia" / p / "bin") for p in ("cublas", "cudnn")
    )


def test_nvidia_dll_dirs_idempotent(monkeypatch, tmp_path):
    from lightfall_voice.stt_service import transcriber as tr

    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    added: list[str] = []
    monkeypatch.setattr(tr, "_ADDED_DLL_DIRS", {})
    monkeypatch.setenv("PATH", "base")
    monkeypatch.setattr(tr.os, "name", "nt")
    monkeypatch.setattr(tr.os, "add_dll_directory", added.append, raising=False)

    def find_spec(name):
        if not name.endswith("cublas"):
            return None
        return SimpleNamespace(
            submodule_search_locations=[str(tmp_path / "nvidia" / "cublas")]
        )

    monkeypatch.setattr(tr.importlib.util, "find_spec", find_spec)
    first = tr.add_nvidia_dll_dirs()
    path_after_first = tr.os.environ["PATH"]
    second = tr.add_nvidia_dll_dirs()
    assert first == [str(tmp_path / "nvidia" / "cublas" / "bin")]
    assert second == []
    assert added == first
    assert tr.os.environ["PATH"] == path_after_first


def test_import_without_faster_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    t = FasterWhisperTranscriber("small", "cpu")
    with pytest.raises(ImportError):
        t.load()
