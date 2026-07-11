"""Перенос модели между PlatformAPI-чекпоинтом и локальной директорией.

По умолчанию файлы читаются/пишутся через read/write presigned URL.
Stream API оставлен opt-in, потому что в PlatformAPI 2.0.0.1a0
checkpoint stream methods в InternalClient ещё не реализованы.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["download_checkpoint", "upload_directory"]

# PlatformAPI 2.0.0.1a0 exposes stream objects, but checkpoint stream
# methods are not wired in InternalClient yet. Keep streaming opt-in.
STREAM_THRESHOLD = 64 * 1024 * 1024  # 64 MiB
CHUNK = 16 * 1024 * 1024  # 16 MiB
USE_STREAMS = os.environ.get("QUANT_USE_STREAMS", "").lower() in {"1", "true", "yes"}


def download_checkpoint(checkpoint: Any, dest: str | Path) -> Path:
    """Скачать все файлы входного чекпоинта в локальную директорию `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    keys = list(checkpoint.get_file_iterator(prefix=None))
    logger.info("checkpoint: %d files to download -> %s", len(keys), dest)

    for key in keys:
        target = dest / key
        target.parent.mkdir(parents=True, exist_ok=True)
        size = _file_size(checkpoint, key)
        if USE_STREAMS and size is not None and size > STREAM_THRESHOLD:
            _stream_read_to_file(checkpoint, key, target, size)
        else:
            target.write_bytes(checkpoint.read(key))
        logger.debug("downloaded %s (%s bytes)", key, size)
    return dest


def upload_directory(mutable_checkpoint: Any, src: str | Path) -> int:
    """Залить все файлы директории `src` в изменяемый чекпоинт. Возвращает число файлов."""
    src = Path(src)
    files = [p for p in src.rglob("*") if p.is_file()]
    logger.info("uploading %d files from %s", len(files), src)

    for path in files:
        key = str(path.relative_to(src))
        size = path.stat().st_size
        if USE_STREAMS and size > STREAM_THRESHOLD:
            _stream_write_file(mutable_checkpoint, key, path)
        else:
            mutable_checkpoint.write(key, file=path.read_bytes())
        logger.debug("uploaded %s (%d bytes)", key, size)
    return len(files)


def _file_size(checkpoint: Any, key: str) -> int | None:
    try:
        meta = checkpoint.get_file_metadata(key)
        return int(getattr(meta, "size", None) or getattr(meta, "size_bytes", 0)) or None
    except Exception:  # noqa: BLE001 — размер не критичен, упадём на whole-read
        return None


def _stream_read_to_file(checkpoint: Any, key: str, target: Path, size: int) -> None:
    stream = checkpoint.create_stream(key)
    with target.open("wb") as fh:
        start = 0
        while start < size:
            end = min(start + CHUNK, size)
            fh.write(stream.read(start=start, end=end))
            start = end


def _stream_write_file(mutable_checkpoint: Any, key: str, path: Path) -> None:
    stream = mutable_checkpoint.create_stream(key)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            stream.write(data=chunk)
