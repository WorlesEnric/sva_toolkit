from __future__ import annotations

from pathlib import Path
import sys

import pytest

from sva_toolkit.runtime.tools import ToolRegistry


def test_register_marks_tools_as_available_or_unavailable() -> None:
    registry = ToolRegistry()

    available = registry.register(
        "python-runtime",
        search_names=("definitely-not-a-real-tool", Path(sys.executable).name),
    )
    missing = registry.register("missing-tool", search_names=("definitely-not-a-real-tool",))

    assert available.available is True
    assert available.path is not None
    assert missing.available is False
    assert missing.path is None


def test_get_unknown_tool_raises_key_error() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


def test_require_unavailable_tool_raises_runtime_error() -> None:
    registry = ToolRegistry()
    registry.register("missing-tool", search_names=("definitely-not-a-real-tool",))

    with pytest.raises(RuntimeError, match="Required tool 'missing-tool' is not available"):
        registry.require("missing-tool")


def test_available_tools_returns_only_available_entries() -> None:
    registry = ToolRegistry()
    registry.register("python-runtime", search_names=(Path(sys.executable).name,))
    registry.register("missing-tool", search_names=("definitely-not-a-real-tool",))

    assert set(registry.available_tools) == {"python-runtime"}
