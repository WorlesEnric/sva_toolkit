"""Shared diagnostic counters and CLI logging configuration."""

from __future__ import annotations

import logging
import sys
from types import MappingProxyType
import threading
from typing import Final, Mapping

DIAGNOSTIC_KINDS: Final[tuple[str, ...]] = (
    "opaque_property",
    "translator_fallback",
    "lossy_extraction",
    "cache_miss",
    "retry_exhausted",
)

LOGGER = logging.getLogger("sva_toolkit")
if not any(isinstance(handler, logging.NullHandler) for handler in LOGGER.handlers):
    LOGGER.addHandler(logging.NullHandler())

_CLI_HANDLER: logging.Handler | None = None
_CLI_HANDLER_LOCK = threading.Lock()


class _DynamicStderrHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


class Diagnostics:
    def __init__(self, kinds: tuple[str, ...] = DIAGNOSTIC_KINDS) -> None:
        self._kinds = tuple(kinds)
        self._counts = {kind: 0 for kind in self._kinds}
        self._lock = threading.Lock()

    def record(self, kind: str, *, detail: str | None = None) -> None:
        _ = detail
        with self._lock:
            if kind not in self._counts:
                raise ValueError(f"Unsupported diagnostic kind: {kind}")
            self._counts[kind] += 1

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(dict(sorted(self._counts.items())))

    def reset(self) -> None:
        with self._lock:
            for kind in self._counts:
                self._counts[kind] = 0

    def render_summary(self) -> str:
        with self._lock:
            non_zero = [
                f"{kind}={count}"
                for kind, count in sorted(self._counts.items())
                if count > 0
            ]
        if not non_zero:
            return ""
        return "Diagnostics summary: " + ", ".join(non_zero)


DEFAULT_DIAGNOSTICS = Diagnostics()


def configure_cli_logging(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    LOGGER.setLevel(level)
    LOGGER.propagate = False

    global _CLI_HANDLER
    with _CLI_HANDLER_LOCK:
        if _CLI_HANDLER is None:
            handler = _DynamicStderrHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            LOGGER.addHandler(handler)
            _CLI_HANDLER = handler
    return LOGGER


__all__ = [
    "DIAGNOSTIC_KINDS",
    "LOGGER",
    "DEFAULT_DIAGNOSTICS",
    "Diagnostics",
    "configure_cli_logging",
]
