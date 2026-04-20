from __future__ import annotations

from types import MappingProxyType
import threading
from typing import Mapping

from sva_toolkit.runtime.diagnostics import LOGGER
from sva_toolkit.sva.ast import SourceSpan


_KINDS = (
    "opaque_property",
    "opaque_sequence",
    "opaque_expr",
    "fallback_recover_used",
)


class _ParserDiagnostics:
    def __init__(self) -> None:
        self._counts = {kind: 0 for kind in _KINDS}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            for kind in self._counts:
                self._counts[kind] = 0

    def record(self, kind: str) -> None:
        with self._lock:
            if kind not in self._counts:
                raise ValueError(f"Unsupported parser diagnostic kind: {kind}")
            self._counts[kind] += 1

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(dict(sorted(self._counts.items())))

    def opaque_count(self) -> int:
        snapshot = self.snapshot()
        return snapshot["opaque_property"] + snapshot["opaque_sequence"] + snapshot["opaque_expr"]

    def emit_warning(self, kind: str, text: str, span: SourceSpan | None) -> None:
        self.record(kind)
        self.record("fallback_recover_used")
        location = ""
        if span is not None:
            location = f" at {span.start}:{span.end}"
        preview = " ".join(text.strip().split())
        if len(preview) > 160:
            preview = preview[:157] + "..."
        LOGGER.warning("parser recover=True downgraded to %s%s: %s", kind, location, preview)


ParserDiagnostics = _ParserDiagnostics()


__all__ = ["ParserDiagnostics"]
