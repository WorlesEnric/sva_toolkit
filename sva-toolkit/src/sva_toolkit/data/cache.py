from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, BinaryIO, Iterator

from sva_toolkit.runtime.atomic_io import atomic_write_json
from sva_toolkit.runtime.diagnostics import LOGGER

CACHE_SCHEMA_VERSION = 1
_CACHE_SCHEMA_FIELD = "__cache_schema"


def load_cached_result(cache_dir: str | None, cache_key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None

    cache_path = Path(cache_dir) / f"{cache_key}.json"
    if not cache_path.exists():
        return None

    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(cached, dict):
        return None
    if cached.get(_CACHE_SCHEMA_FIELD) != CACHE_SCHEMA_VERSION:
        return None

    result = dict(cached)
    result.pop(_CACHE_SCHEMA_FIELD, None)
    result["from_cache"] = True
    return result


def write_cached_result(cache_dir: str | None, cache_key: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return

    cache_path = Path(cache_dir) / f"{cache_key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with advisory_cache_lock(cache_path):
        cache_payload = dict(payload)
        cache_payload[_CACHE_SCHEMA_FIELD] = CACHE_SCHEMA_VERSION
        atomic_write_json(cache_path, cache_payload)


@contextmanager
def advisory_cache_lock(cache_path: Path) -> Iterator[None]:
    lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as handle:
        locked = _acquire_advisory_lock(handle, lock_path)
        try:
            yield
        finally:
            if locked:
                _release_advisory_lock(handle)


def _acquire_advisory_lock(handle: BinaryIO, lock_path: Path) -> bool:
    if os.name == "nt":
        return _acquire_windows_lock(handle, lock_path)

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        LOGGER.warning("cache lock unavailable for %s; continuing without a lock", lock_path)
        return False

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return True


def _release_advisory_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        _release_windows_lock(handle)
        return

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        return

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire_windows_lock(handle: BinaryIO, lock_path: Path) -> bool:
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - POSIX test environment
        LOGGER.warning("cache lock unavailable for %s; continuing without a lock", lock_path)
        return False

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()

    while True:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            time.sleep(0.05)


def _release_windows_lock(handle: BinaryIO) -> None:
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - POSIX test environment
        return

    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        return


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "advisory_cache_lock",
    "load_cached_result",
    "write_cached_result",
]
