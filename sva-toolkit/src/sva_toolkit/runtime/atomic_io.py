"""Atomic file writers used by runtime and higher-level workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def atomic_write_text(
    path: str | os.PathLike[str],
    content: str,
    *,
    encoding: str = "utf-8",
) -> None:
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {destination.parent}")

    temp_path = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with _lock_for_path(destination):
        try:
            with temp_path.open("w", encoding=encoding, newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, destination)
            _fsync_directory(destination.parent)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=False) + "\n"
    atomic_write_text(path, text)


def atomic_write_jsonl(
    path: str | os.PathLike[str],
    rows: Iterable[Any],
) -> None:
    text = "\n".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False)
        for row in rows
    )
    if text:
        text += "\n"
    atomic_write_text(path, text)


def _lock_for_path(path: Path) -> threading.Lock:
    key = os.path.abspath(os.fspath(path))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


def _fsync_directory(path: Path) -> None:
    directory_flags = getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, os.O_RDONLY | directory_flags)
    except OSError:
        return

    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


__all__ = [
    "atomic_write_text",
    "atomic_write_json",
    "atomic_write_jsonl",
]
