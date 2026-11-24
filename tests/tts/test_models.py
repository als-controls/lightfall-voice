from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading

import pytest

from lightfall_voice.tts_service import models
from lightfall_voice.tts_service.models import (
    BASE_URL,
    MODEL_FILES,
    ModelDownloadError,
    ModelFilesMissing,
    default_cache_dir,
    resolve,
)

INT8_BYTES = b"fake-int8-onnx-model-payload"
FP16_BYTES = b"fake-fp16-onnx-model-payload-longer"
VOICES_BYTES = b"fake-voices-bin-payload"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    routes: dict[str, bytes | int] = {
        "/kokoro-v1.0.int8.onnx": INT8_BYTES,
        "/kokoro-v1.0.fp16.onnx": FP16_BYTES,
        "/voices-v1.0.bin": VOICES_BYTES,
    }
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            payload = routes.get(self.path, 404)
            if isinstance(payload, int):
                self.send_error(payload)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()

    cache_dir = tmp_path / "cache" / "kokoro"
    fake_files = {
        "int8": ("kokoro-v1.0.int8.onnx", len(INT8_BYTES), _sha256(INT8_BYTES)),
        "fp16": ("kokoro-v1.0.fp16.onnx", len(FP16_BYTES), _sha256(FP16_BYTES)),
        "voices": ("voices-v1.0.bin", len(VOICES_BYTES), _sha256(VOICES_BYTES)),
    }
    base_url = f"http://127.0.0.1:{server.server_port}/"
    monkeypatch.setattr(models, "MODEL_FILES", fake_files)
    monkeypatch.setattr(models, "BASE_URL", base_url)
    monkeypatch.setattr(models, "default_cache_dir", lambda: cache_dir)

    try:
        yield {
            "base_url": base_url,
            "cache_dir": cache_dir,
            "routes": routes,
            "requests": requests,
            "files": fake_files,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_downloads_and_verifies(fake_env):
    assert BASE_URL == (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    )
    assert MODEL_FILES == {
        "int8": (
            "kokoro-v1.0.int8.onnx",
            92361271,
            "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
        ),
        "fp16": (
            "kokoro-v1.0.fp16.onnx",
            177464787,
            "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
        ),
        "voices": (
            "voices-v1.0.bin",
            28214398,
            "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
        ),
    }
    assert default_cache_dir().name == "kokoro"

    model_path, voices_path = resolve("int8", base_url=fake_env["base_url"])

    assert model_path == fake_env["cache_dir"] / "kokoro-v1.0.int8.onnx"
    assert voices_path == fake_env["cache_dir"] / "voices-v1.0.bin"
    assert model_path.read_bytes() == INT8_BYTES
    assert voices_path.read_bytes() == VOICES_BYTES
    assert not list(fake_env["cache_dir"].glob("*.part"))
    assert fake_env["requests"] == ["/kokoro-v1.0.int8.onnx", "/voices-v1.0.bin"]


def test_reuses_cached_by_size(fake_env):
    cache_dir: Path = fake_env["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = cache_dir / "kokoro-v1.0.int8.onnx"
    cached_voices = cache_dir / "voices-v1.0.bin"
    cached_model.write_bytes(b"x" * len(INT8_BYTES))
    cached_voices.write_bytes(b"y" * len(VOICES_BYTES))

    model_path, voices_path = resolve("int8", base_url=fake_env["base_url"])

    assert model_path == cached_model
    assert voices_path == cached_voices
    assert fake_env["requests"] == []
    assert model_path.read_bytes() == b"x" * len(INT8_BYTES)

    cached_model.write_bytes(b"wrong-size")
    resolve("int8", base_url=fake_env["base_url"])
    assert fake_env["requests"] == ["/kokoro-v1.0.int8.onnx"]
    assert model_path.read_bytes() == INT8_BYTES


def test_checksum_mismatch_removes_part(fake_env):
    fake_env["routes"]["/kokoro-v1.0.int8.onnx"] = b"corrupted-bytes-same-or-diff-len"

    with pytest.raises(ModelDownloadError, match=r"Checksum mismatch for kokoro-v1\.0\.int8\.onnx"):
        resolve("int8", base_url=fake_env["base_url"])

    cache_dir: Path = fake_env["cache_dir"]
    assert not (cache_dir / "kokoro-v1.0.int8.onnx.part").exists()
    assert not (cache_dir / "kokoro-v1.0.int8.onnx").exists()


def test_network_error_leaves_no_part(fake_env):
    fake_env["routes"]["/kokoro-v1.0.int8.onnx"] = 500
    cache_dir: Path = fake_env["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    stale_part = cache_dir / "kokoro-v1.0.int8.onnx.part"
    stale_part.write_bytes(b"stale")

    with pytest.raises(ModelDownloadError) as exc_info:
        resolve("int8", base_url=fake_env["base_url"])

    msg = str(exc_info.value)
    assert "kokoro-v1.0.int8.onnx" in msg
    assert "500" in msg or "Internal Server Error" in msg
    assert not stale_part.exists()
    assert not list(cache_dir.glob("*.part"))


def test_model_dir_used_without_download(fake_env, tmp_path: Path):
    custom_dir = tmp_path / "custom_models"
    custom_dir.mkdir()
    m_file = custom_dir / "kokoro-v1.0.int8.onnx"
    v_file = custom_dir / "voices-v1.0.bin"
    m_file.write_bytes(b"unverified-local-model")
    v_file.write_bytes(b"unverified-local-voices")

    model_path, voices_path = resolve(
        "int8", model_dir=str(custom_dir), base_url=fake_env["base_url"]
    )

    assert model_path == m_file
    assert voices_path == v_file
    assert fake_env["requests"] == []


def test_model_dir_missing_file_names_it(fake_env, tmp_path: Path):
    custom_dir = tmp_path / "custom_models"
    custom_dir.mkdir()

    with pytest.raises(ModelFilesMissing) as exc_info:
        resolve("int8", model_dir=str(custom_dir), base_url=fake_env["base_url"])
    assert str(exc_info.value) == f"Missing kokoro-v1.0.int8.onnx in {custom_dir}"

    (custom_dir / "kokoro-v1.0.int8.onnx").write_bytes(b"only-model")
    with pytest.raises(ModelFilesMissing) as exc_info:
        resolve("int8", model_dir=str(custom_dir), base_url=fake_env["base_url"])
    assert str(exc_info.value) == f"Missing voices-v1.0.bin in {custom_dir}"
    assert fake_env["requests"] == []


def test_unknown_quality(fake_env):
    with pytest.raises(ValueError):
        resolve("fp32", base_url=fake_env["base_url"])
    with pytest.raises(ValueError):
        resolve("voices", base_url=fake_env["base_url"])


def test_progress_called_once_only_when_downloading(fake_env):
    events: list[str] = []

    resolve("int8", base_url=fake_env["base_url"], progress=events.append)
    assert events == ["downloading"]

    events.clear()
    resolve("int8", base_url=fake_env["base_url"], progress=events.append)
    assert events == []
