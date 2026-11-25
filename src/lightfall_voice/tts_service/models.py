"""Pinned Kokoro model files with verified first-use download."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import http.client
import os
from pathlib import Path
import urllib.error
import urllib.request

_DEFAULT_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
)
BASE_URL = _DEFAULT_BASE_URL

MODEL_FILES: dict[str, tuple[str, int, str]] = {
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

_CHUNK_BYTES = 1024 * 1024
_TIMEOUT_S = 30


class ModelFilesMissing(RuntimeError):
    """Raised when an explicit model_dir is missing required Kokoro files."""


class ModelDownloadError(RuntimeError):
    """Raised when downloading or verifying a Kokoro model file fails."""


def default_cache_dir() -> Path:
    import platformdirs

    return Path(platformdirs.user_cache_dir("lightfall-voice")) / "kokoro"


def _download(url: str, filename: str, expected_sha256: str, dest: Path) -> None:
    part = dest.with_name(f"{filename}.part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as resp, open(part, "wb") as out:
            while chunk := resp.read(_CHUNK_BYTES):
                digest.update(chunk)
                out.write(chunk)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        part.unlink(missing_ok=True)
        reason = getattr(exc, "reason", exc)
        raise ModelDownloadError(f"Failed to download {url}: {reason}") from exc

    if digest.hexdigest().lower() != expected_sha256.lower():
        part.unlink(missing_ok=True)
        raise ModelDownloadError(f"Checksum mismatch for {filename}")

    os.replace(part, dest)


def resolve(
    quality: str,
    model_dir: str = "",
    *,
    base_url: str = BASE_URL,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, Path]:
    if quality == "voices" or quality not in MODEL_FILES:
        raise ValueError(f"Unknown quality: {quality}")

    model_entry = MODEL_FILES[quality]
    voices_entry = MODEL_FILES["voices"]
    entries = (model_entry, voices_entry)

    if model_dir:
        root = Path(model_dir)
        for filename, _, _ in entries:
            if not (root / filename).is_file():
                raise ModelFilesMissing(f"Missing {filename} in {root}")
        return root / model_entry[0], root / voices_entry[0]

    cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    effective_base_url = BASE_URL if base_url == _DEFAULT_BASE_URL else base_url
    prefix = effective_base_url.rstrip("/")

    notified = False
    for filename, expected_size, expected_sha256 in entries:
        dest = cache_dir / filename
        if dest.is_file() and dest.stat().st_size == expected_size:
            continue
        if progress is not None and not notified:
            progress("downloading")
            notified = True
        url = f"{prefix}/{filename}"
        _download(url, filename, expected_sha256, dest)

    return cache_dir / model_entry[0], cache_dir / voices_entry[0]
